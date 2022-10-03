use alloc::vec::Vec;

use crate::utils::ModuleInfo;
use anyhow::{anyhow, Result};
use wasm_encoder::SectionId;
use wasmparser::{BlockType, CodeSectionReader, Type};

// The cost in stack items that should be charged per call of a function. This is
// is a static cost that is added to each function call. This makes sense because even
// if a function does not use any parameters or locals some stack space on the host
// machine might be consumed to hold some context.
const ACTIVATION_FRAME_COST: u32 = 2;

/// Control stack frame.
#[derive(Debug)]
struct Frame {
	/// Stack becomes polymorphic only after an instruction that
	/// never passes control further was executed.
	is_polymorphic: bool,

	/// Count of values which will be pushed after the exit
	/// from the current block.
	end_arity: u32,

	/// Count of values which should be poped upon a branch to
	/// this frame.
	///
	/// This might be diffirent from `end_arity` since branch
	/// to the loop header can't take any values.
	branch_arity: u32,

	/// Stack height before entering in the block.
	start_height: u32,
}

/// This is a compound stack that abstracts tracking height of the value stack
/// and manipulation of the control stack.
struct Stack {
	height: u32,
	control_stack: Vec<Frame>,
}

impl Stack {
	fn new() -> Stack {
		Stack { height: ACTIVATION_FRAME_COST, control_stack: Vec::new() }
	}

	/// Returns current height of the value stack.
	fn height(&self) -> u32 {
		self.height
	}

	/// Returns a reference to a frame by specified depth relative to the top of
	/// control stack.
	fn frame(&self, rel_depth: u32) -> Result<&Frame> {
		let control_stack_height: usize = self.control_stack.len();
		let last_idx = control_stack_height
			.checked_sub(1)
			.ok_or_else(|| anyhow!("control stack is empty"))?;
		let idx = last_idx
			.checked_sub(rel_depth as usize)
			.ok_or_else(|| anyhow!("control stack out-of-bounds"))?;
		Ok(&self.control_stack[idx])
	}

	/// Mark successive instructions as unreachable.
	///
	/// This effectively makes stack polymorphic.
	fn mark_unreachable(&mut self) -> Result<()> {
		let top_frame = self
			.control_stack
			.last_mut()
			.ok_or_else(|| anyhow!("stack must be non-empty"))?;
		top_frame.is_polymorphic = true;
		Ok(())
	}

	/// Push control frame into the control stack.
	fn push_frame(&mut self, frame: Frame) {
		self.control_stack.push(frame);
	}

	/// Pop control frame from the control stack.
	///
	/// Returns `Err` if the control stack is empty.
	fn pop_frame(&mut self) -> Result<Frame> {
		self.control_stack.pop().ok_or_else(|| anyhow!("stack must be non-empty"))
	}

	/// Truncate the height of value stack to the specified height.
	fn trunc(&mut self, new_height: u32) {
		self.height = new_height;
	}

	/// Push specified number of values into the value stack.
	///
	/// Returns `Err` if the height overflow usize value.
	fn push_values(&mut self, value_count: u32) -> Result<()> {
		self.height =
			self.height.checked_add(value_count).ok_or_else(|| anyhow!("stack overflow"))?;
		Ok(())
	}

	/// Pop specified number of values from the value stack.
	///
	/// Returns `Err` if the stack happen to be negative value after
	/// values popped.
	fn pop_values(&mut self, value_count: u32) -> Result<()> {
		if value_count == 0 {
			return Ok(())
		}
		{
			let top_frame = self.frame(0)?;
			if self.height == top_frame.start_height {
				// It is an error to pop more values than was pushed in the current frame
				// (ie pop values pushed in the parent frame), unless the frame became
				// polymorphic.
				return if top_frame.is_polymorphic {
					Ok(())
				} else {
					return Err(anyhow!("trying to pop more values than pushed"))
				}
			}
		}

		self.height =
			self.height.checked_sub(value_count).ok_or_else(|| anyhow!("stack underflow"))?;

		Ok(())
	}
}

/// This function expects the function to be validated. func_idx have sub import funcs num
pub fn compute(func_idx: u32, module: &ModuleInfo) -> Result<u32> {
	use wasmparser::Operator::*;

	let code_section = CodeSectionReader::new(
		&module
			.raw_sections
			.get(&SectionId::Code.into())
			.ok_or_else(|| anyhow!("No code section"))?
			.data,
		0,
	)?;
	// Get a signature and a body of the specified function.
	let wasmparser::Type::Func(func_signature) =
		module.get_functype_idx(module.imported_functions_count + func_idx)?;
	let body = code_section
		.into_iter()
		.nth(func_idx as usize)
		.ok_or_else(|| anyhow!("Function body for the index isn't found"))??;
	let body_reader = body.get_operators_reader()?;
	let mut stack = Stack::new();
	let mut max_height: u32 = 0;

	// Add implicit frame for the function. Breaks to this frame and execution of
	// the last end should deal with this frame.
	let func_arity = func_signature.results().len() as u32;
	stack.push_frame(Frame {
		is_polymorphic: false,
		end_arity: func_arity,
		branch_arity: func_arity,
		start_height: 0,
	});

	for opcode_r in body_reader {
		// If current value stack is higher than maximal height observed so far,
		// save the new height.
		// However, we don't increase maximal value in unreachable code.
		if stack.height() > max_height && !stack.frame(0)?.is_polymorphic {
			max_height = stack.height();
		}
		let opcode = opcode_r?; //todo how to optimize this?
		match opcode {
				Nop => {},
				Block{ty} | Loop{ty} | If{ty} => {
					let end_arity = if ty == BlockType::Empty { 0 } else { 1 };
					let branch_arity = if let Loop{..} = opcode { 0 } else { end_arity };
					if let If{..} = opcode {
						stack.pop_values(1)?;
					}
					let height = stack.height();
					stack.push_frame(Frame {
						is_polymorphic: false,
						end_arity,
						branch_arity,
						start_height: height,
					});
				},
				Else => {
					// The frame at the top should be pushed by `If`. So we leave
					// it as is.
				},
				End => {
					let frame = stack.pop_frame()?;
					stack.trunc(frame.start_height);
					stack.push_values(frame.end_arity)?;
				},
				Unreachable => {
					stack.mark_unreachable()?;
				},
				Br{relative_depth} => {
					// Pop values for the destination block result.
					let target_arity = stack.frame(relative_depth)?.branch_arity;
					stack.pop_values(target_arity)?;

					// This instruction unconditionally transfers control to the specified block,
					// thus all instruction until the end of the current block is deemed unreachable
					stack.mark_unreachable()?;
				},
				BrIf{relative_depth} => {
					// Pop values for the destination block result.
					let target_arity = stack.frame(relative_depth)?.branch_arity;
					stack.pop_values(target_arity)?;

					// Pop condition value.
					stack.pop_values(1)?;

					// Push values back.
					stack.push_values(target_arity)?;
				},
				BrTable{table} => {
					let arity_of_default = stack.frame(table.default())?.branch_arity;

					// Check that all jump targets have an equal arities.
					for target in table.targets() {
						let arity = stack.frame(target?)?.branch_arity;
						if arity != arity_of_default {
							return Err(anyhow!("Arity of all jump-targets must be equal"))
						}
					}

					// Because all jump targets have an equal arities, we can just take arity of
					// the default branch.
					stack.pop_values(arity_of_default)?;

					// This instruction doesn't let control flow to go further, since the control flow
					// should take either one of branches depending on the value or the default branch.
					stack.mark_unreachable()?;
				},
				Return => {
					// Pop return values of the function. Mark successive instructions as unreachable
					// since this instruction doesn't let control flow to go further.
					stack.pop_values(func_arity)?;
					stack.mark_unreachable()?;
				},
				Call{function_index} => {
					let Type::Func(ty) = module.get_functype_idx(function_index)?;

					// Pop values for arguments of the function.
					stack.pop_values(ty.params().len() as u32)?;

					// Push result of the function execution to the stack.
					let callee_arity = ty.results().len() as u32;
					stack.push_values(callee_arity)?;
				},
				CallIndirect{ type_index,..} => {
					let Type::Func(ty) =  module.types_map.get(type_index as usize).ok_or_else(||anyhow!("Type not found"))?;
					// Pop the offset into the function table.
					stack.pop_values(1)?;

					// Pop values for arguments of the function.
					stack.pop_values(ty.params().len() as u32)?;

					// Push result of the function execution to the stack.
					let callee_arity = ty.results().len() as u32;
					stack.push_values(callee_arity)?;
				},
				Drop => {
					stack.pop_values(1)?;
				},
				Select => {
					// Pop two values and one condition.
					stack.pop_values(2)?;
					stack.pop_values(1)?;

					// Push the selected value.
					stack.push_values(1)?;
				},
				LocalGet{..} => {
					stack.push_values(1)?;
				},
				LocalSet{..} => {
					stack.pop_values(1)?;
				},
				LocalTee{..} => {
					// This instruction pops and pushes the value, so
					// effectively it doesn't modify the stack height.
					stack.pop_values(1)?;
					stack.push_values(1)?;
				},
				GlobalGet{..} => {
					stack.push_values(1)?;
				},
				GlobalSet{..} => {
					stack.pop_values(1)?;
				},
				I32Load{..} |
				I64Load{..} |
				F32Load{..} |
				F64Load{..} |
				I32Load8S{..} |
				I32Load8U{..} |
				I32Load16S{..} |
				I32Load16U{..} |
				I64Load8S{..} |
				I64Load8U{..} |
				I64Load16S{..} |
				I64Load16U{..} |
				I64Load32S{..} |
				I64Load32U{..} => {
					// These instructions pop the address and pushes the result,
					// which effictively don't modify the stack height.
					stack.pop_values(1)?;
					stack.push_values(1)?;
				},

				I32Store{..} |
				I64Store{..} |
				F32Store{..} |
				F64Store{..} |
				I32Store8{..} |
				I32Store16{..} |
				I64Store8{..} |
				I64Store16{..} |
				I64Store32{..} => {
					// These instructions pop the address and the value.
					stack.pop_values(2)?;
				},

				MemorySize{..} => {
					// Pushes current memory size
					stack.push_values(1)?;
				},
				MemoryGrow{..} => {
					// Grow memory takes the value of pages to grow and pushes
					stack.pop_values(1)?;
					stack.push_values(1)?;
				},

				I32Const{..} | I64Const{..} | F32Const{..} | F64Const{..} => {
					// These instructions just push the single literal value onto the stack.
					stack.push_values(1)?;
				},

				I32Eqz | I64Eqz => {
					// These instructions pop the value and compare it against zero, and pushes
					// the result of the comparison.
					stack.pop_values(1)?;
					stack.push_values(1)?;
				},

				I32Eq | I32Ne | I32LtS | I32LtU | I32GtS | I32GtU | I32LeS | I32LeU | I32GeS |
				I32GeU | I64Eq | I64Ne | I64LtS | I64LtU | I64GtS | I64GtU | I64LeS | I64LeU |
				I64GeS | I64GeU | F32Eq | F32Ne | F32Lt | F32Gt | F32Le | F32Ge | F64Eq | F64Ne |
				F64Lt | F64Gt | F64Le | F64Ge => {
					// Comparison operations take two operands and produce one result.
					stack.pop_values(2)?;
					stack.push_values(1)?;
				},

				I32Clz | I32Ctz | I32Popcnt | I64Clz | I64Ctz | I64Popcnt | F32Abs | F32Neg |
				F32Ceil | F32Floor | F32Trunc | F32Nearest | F32Sqrt | F64Abs | F64Neg | F64Ceil |
				F64Floor | F64Trunc | F64Nearest | F64Sqrt => {
					// Unary operators take one operand and produce one result.
					stack.pop_values(1)?;
					stack.push_values(1)?;
				},

				I32Add | I32Sub | I32Mul | I32DivS | I32DivU | I32RemS | I32RemU | I32And | I32Or |
				I32Xor | I32Shl | I32ShrS | I32ShrU | I32Rotl | I32Rotr | I64Add | I64Sub |
				I64Mul | I64DivS | I64DivU | I64RemS | I64RemU | I64And | I64Or | I64Xor | I64Shl |
				I64ShrS | I64ShrU | I64Rotl | I64Rotr | F32Add | F32Sub | F32Mul | F32Div |
				F32Min | F32Max | F32Copysign | F64Add | F64Sub | F64Mul | F64Div | F64Min |
				F64Max | F64Copysign => {
					// Binary operators take two operands and produce one result.
					stack.pop_values(2)?;
					stack.push_values(1)?;
				},

				I32WrapI64 | I32TruncSatF32S | I32TruncSatF32U | I32TruncSatF64S | I32TruncSatF64U |
				I64TruncSatF32S | I64TruncSatF32U | I64TruncSatF64S | I64TruncSatF64U |
				I64ExtendI32U|I64ExtendI32S|
				F32ConvertI32S | F32ConvertI32U | F32ConvertI64S | F32ConvertI64U |
				F64ConvertI32S|F64ConvertI32U|F64ConvertI64S|F64ConvertI64U|
				F32DemoteF64 | F64PromoteF32 | I32ReinterpretF32 | I64ReinterpretF64 | F32ReinterpretI32 |
				F64ReinterpretI64 => {
					// Conversion operators take one value and produce one result.
					stack.pop_values(1)?;
					stack.push_values(1)?;
				},

				//#[cfg(feature = "sign_ext")]
				I32Extend8S|
				I32Extend16S|
				I64Extend8S|
				I64Extend16S|
				I64Extend32S  => {
					stack.pop_values(1)?;
					stack.push_values(1)?;
				},

				//#[cfg(feature = "bulk")]
				MemoryInit{..} |			//todo there are 12 instructions in wasmparser
				MemoryCopy{..} |
				MemoryFill{..} |
				TableInit{..} |
				TableCopy{..} => {
					stack.pop_values(3)?;
				},
				ElemDrop{..}|DataDrop{..}=> {},
				_=>panic!("not support"),
			}
	}

	Ok(max_height)
}

#[cfg(test)]
mod tests {
	use super::*;

	fn parse_wat(source: &str) -> ModuleInfo {
		let module_bytes = wat::parse_str(source).unwrap();
		ModuleInfo::new(&module_bytes).unwrap()
	}

	#[test]
	fn simple_test() {
		let module = parse_wat(
			r#"
(module
	(func
		i32.const 1
			i32.const 2
				i32.const 3
				drop
			drop
		drop
	)
)
"#,
		);

		let height = compute(0, &module).unwrap();
		assert_eq!(height, 3 + ACTIVATION_FRAME_COST);
	}

	#[test]
	fn implicit_and_explicit_return() {
		let module = parse_wat(
			r#"
(module
	(func (result i32)
		i32.const 0
		return
	)
)
"#,
		);

		let height = compute(0, &module).unwrap();
		assert_eq!(height, 1 + ACTIVATION_FRAME_COST);
	}

	#[test]
	fn dont_count_in_unreachable() {
		let module = parse_wat(
			r#"
(module
  (memory 0)
  (func (result i32)
	unreachable
	grow_memory
  )
)
"#,
		);

		let height = compute(0, &module).unwrap();
		assert_eq!(height, ACTIVATION_FRAME_COST);
	}

	#[test]
	fn yet_another_test() {
		let module = parse_wat(
			r#"
(module
  (memory 0)
  (func
	;; Push two values and then pop them.
	;; This will make max depth to be equal to 2.
	i32.const 0
	i32.const 1
	drop
	drop

	;; Code after `unreachable` shouldn't have an effect
	;; on the max depth.
	unreachable
	i32.const 0
	i32.const 1
	i32.const 2
  )
)
"#,
		);

		let height = compute(0, &module).unwrap();
		assert_eq!(height, 2 + ACTIVATION_FRAME_COST);
	}

	#[test]
	fn call_indirect() {
		let module = parse_wat(
			r#"
(module
	(table $ptr 1 1 funcref)
	(elem $ptr (i32.const 0) func 1)
	(func $main
		(call_indirect (i32.const 0))
		(call_indirect (i32.const 0))
		(call_indirect (i32.const 0))
	)
	(func $callee
		i64.const 42
		drop
	)
)
"#,
		);

		let height = compute(0, &module).unwrap();
		assert_eq!(height, 1 + ACTIVATION_FRAME_COST);
	}

	#[test]
	fn breaks() {
		let module = parse_wat(
			r#"
(module
	(func $main
		block (result i32)
			block (result i32)
				i32.const 99
				br 1
			end
		end
		drop
	)
)
"#,
		);

		let height = compute(0, &module).unwrap();
		assert_eq!(height, 1 + ACTIVATION_FRAME_COST);
	}

	#[test]
	fn if_else_works() {
		let module = parse_wat(
			r#"
(module
	(func $main
		i32.const 7
		i32.const 1
		if (result i32)
			i32.const 42
		else
			i32.const 99
		end
		i32.const 97
		drop
		drop
		drop
	)
)
"#,
		);

		let height = compute(0, &module).unwrap();
		assert_eq!(height, 3 + ACTIVATION_FRAME_COST);
	}
}

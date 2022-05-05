//! This module is used to instrument a Wasm module with gas metering code.
//!
//! The primary public interface is the [`inject`] function which transforms a given
//! module into one that charges gas for code to be executed. See function documentation for usage
//! and details.

#[cfg(test)]
mod validation;

use alloc::{vec, vec::Vec};
use core::{cmp::min, mem, num::NonZeroU64};
use std::collections::{HashMap, HashSet};
use parity_wasm::{
	builder,
	elements::{
		self, BlockType, Instruction,
		Instruction::{End, I32Const},
		ValueType,
	},
};
#[cfg(feature = "bulk")]
use parity_wasm::elements::BulkInstruction;

/// An interface that describes instruction costs.
pub trait Rules {
	/// Returns the cost for the passed `instruction`.
	///
	/// Returning an error can be used as a way to indicat that an instruction
	/// is forbidden
	fn instruction_cost(&self, instruction: &Instruction) -> Result<InstructionCost, ()>;
}

/// Dynamic costs instructions.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum InstructionCost {
	/// Charge fixed amount per instruction.
	Fixed(u64),

	/// Charge the specified amount of base miligas, plus miligas based on the
	/// last item on the stack. For memory.grow this is the number of pages.
	/// For memory.copy this is the number of bytes to copy.
	Linear(u64, NonZeroU64),
}

/// A type that implements [`Rules`] so that every instruction costs the same.
///
/// This is a simplification that is mostly useful for development and testing.
///
/// # Note
///
/// In a production environment it usually makes no sense to assign every instruction
/// the same cost. A proper implemention of [`Rules`] should be prived that is probably
/// created by benchmarking.
pub struct ConstantCostRules {
	instruction_cost: u64,
	memory_grow_cost: u64,
}

impl ConstantCostRules {
	/// Create a new [`ConstantCostRules`].
	///
	/// Uses `instruction_cost` for every instruction and `memory_grow_cost` to dynamically
	/// meter the memory growth instruction.
	pub fn new(instruction_cost: u64, memory_grow_cost: u64) -> Self {
		Self { instruction_cost, memory_grow_cost }
	}
}

impl Default for ConstantCostRules {
	/// Uses instruction cost of `1` and disables memory growth instrumentation.
	fn default() -> Self {
		Self { instruction_cost: 1, memory_grow_cost: 0 }
	}
}

impl Rules for ConstantCostRules {
	fn instruction_cost(&self, i: &Instruction) -> Result<InstructionCost, ()> {
		match i {
			Instruction::GrowMemory(_) => Ok(NonZeroU64::new(self.memory_grow_cost).map_or(InstructionCost::Fixed(0), |c|InstructionCost::Linear(0, c))),
			_ => Ok(InstructionCost::Fixed(self.instruction_cost)),
		}
	}
}

pub const GAS_COUNTER_NAME: &str = "gas_counter";

/// Transforms a given module into one that charges gas for code to be executed by proxy of an
/// imported gas metering function.
///
/// The output module imports a mutable global i64 $GAS_COUNTER_NAME ("gas_couner") from the
/// specified module. The value specifies the amount of available units of gas. A new function
/// doing gas accounting using this global is added to the module. Having the accounting logic
/// in WASM lets us avoid the overhead of external calls.
///
/// The body of each function is divided into metered blocks, and the calls to charge gas are
/// inserted at the beginning of every such block of code. A metered block is defined so that,
/// unless there is a trap, either all of the instructions are executed or none are. These are
/// similar to basic blocks in a control flow graph, except that in some cases multiple basic
/// blocks can be merged into a single metered block. This is the case if any path through the
/// control flow graph containing one basic block also contains another.
///
/// Charging gas is at the beginning of each metered block ensures that 1) all instructions
/// executed are already paid for, 2) instructions that will not be executed are not charged for
/// unless execution traps, and 3) the number of calls to "gas" is minimized. The corollary is that
/// modules instrumented with this metering code may charge gas for instructions not executed in
/// the event of a trap.
///
/// Additionally, each `memory.grow` instruction found in the module is instrumented to first make
/// a call to charge gas for the additional pages requested. This cannot be done as part of the
/// block level gas charges as the gas cost is not static and depends on the stack argument to
/// `memory.grow`.
///
/// The above transformations are performed for every function body defined in the module. This
/// function also rewrites all function indices references by code, table elements, etc., since
/// the addition of an imported functions changes the indices of module-defined functions.
///
/// This routine runs in time linear in the size of the input module.
///
/// The function fails if the module contains any operation forbidden by gas rule set, returning
/// the original module as an Err. Only one imported global is allowed per `gas_module_name`, the one corresponding to the gas spending measurement
pub fn inject<R: Rules>(
	module: elements::Module,
	rules: &R,
	gas_module_name: &str,
) -> Result<elements::Module, ()> {
	// Injecting gas counting external

	let mut mbuilder = builder::from_module(module);
	mbuilder.push_import(
		builder::import()
			.module(gas_module_name)
			.field(GAS_COUNTER_NAME)
			.external()
			.global(ValueType::I64, true)
			.build(),
	);

	// back to plain module
	let mut module = mbuilder.build();

	// calculate actual global index of the imported definition
	//    (subtract all imports that are NOT globals)

	let gas_global = module.import_count(elements::ImportCountType::Global) as u32 - 1;

	let total_func = module.functions_space() as u32;

	// We'll push the gas counter fuction after all other functions
	let gas_func = total_func;

	// dynamic counter funcs come last (note: this gets incremented before it's used)
	let mut next_dyncnt_func = total_func;
	let mut dyn_funcs = HashMap::new();

	let mut error = false;

	// Updating calling addresses (all calls to function index >= `gas_func` should be incremented)
	for section in module.sections_mut() {
		match section {
			elements::Section::Code(code_section) =>
				for func_body in code_section.bodies_mut() {
					for instruction in func_body.code_mut().elements_mut().iter_mut() {
						match instruction {
							Instruction::GetGlobal(global_index) |
							Instruction::SetGlobal(global_index)
								if *global_index >= gas_global =>
								*global_index += 1,
							_ => {},
						}
					}

					match inject_counter(func_body.code_mut(), rules, gas_func) {
						Ok(dyn_instrs) => {
							// create indexes for instructions with dynamic gas charges
							for instr in dyn_instrs {
								dyn_funcs.entry(instr).or_insert_with(||{
									next_dyncnt_func+=1;
									next_dyncnt_func
								});
							}
						},
						Err(_) => {
							error = true;
							break
						}
					}

					inject_dynamic_counters(func_body.code_mut(), &mut dyn_funcs);
				},

			// adjust global exports
			elements::Section::Export(export_section) => {
				for export in export_section.entries_mut() {
					if let elements::Internal::Global(global_index) = export.internal_mut() {
						if *global_index >= gas_global {
							*global_index += 1;
						}
					}
				}
			},

			elements::Section::Import(import_section) => {
				// Take the imports for the gasglobal
				let gas_globals = import_section.entries().iter().filter(|&p| match p.external() {
					elements::External::Global(g) =>
						p.module() == gas_module_name &&
							p.field() == GAS_COUNTER_NAME &&
							g.is_mutable(),
					_ => false,
				});
				// Ensure there is only one gas global import
				if gas_globals.count() != 1 {
					error = true;
					break
				}
			},
			elements::Section::Element(elements_section) => {
				for segment in elements_section.entries_mut() {
					if let Some(inst) = segment.offset() {
						if !check_offset_code(inst.code()) {
							error = true;
							break
						}
					}
				}
			},
			elements::Section::Data(data_section) =>
				for segment in data_section.entries_mut() {
					if let Some(inst) = segment.offset() {
						if !check_offset_code(inst.code()) {
							error = true;
							break
						}
					}
				},

			_ => {},
		}
	}

	if error {
		return Err(())
	}

	module = add_gas_counter(module, gas_global);

	if !dyn_funcs.is_empty() {
		add_dynamic_counters(module, rules, gas_func, &dyn_funcs)
	} else {
		Ok(module)
	}
}

fn check_offset_code(code: &[Instruction]) -> bool {
	matches!(code, [I32Const(_), End])
}

/// A control flow block is opened with the `block`, `loop`, and `if` instructions and is closed
/// with `end`. Each block implicitly defines a new label. The control blocks form a stack during
/// program execution.
///
/// An example of block:
///
/// ```ignore
/// loop
///   i32.const 1
///   get_local 0
///   i32.sub
///   tee_local 0
///   br_if 0
/// end
/// ```
///
/// The start of the block is `i32.const 1`.
#[derive(Debug)]
struct ControlBlock {
	/// The lowest control stack index corresponding to a forward jump targeted by a br, br_if, or
	/// br_table instruction within this control block. The index must refer to a control block
	/// that is not a loop, meaning it is a forward jump. Given the way Wasm control flow is
	/// structured, the lowest index on the stack represents the furthest forward branch target.
	///
	/// This value will always be at most the index of the block itself, even if there is no
	/// explicit br instruction targeting this control block. This does not affect how the value is
	/// used in the metering algorithm.
	lowest_forward_br_target: usize,

	/// The active metering block that new instructions contribute a gas cost towards.
	active_metered_block: MeteredBlock,

	/// Whether the control block is a loop. Loops have the distinguishing feature that branches to
	/// them jump to the beginning of the block, not the end as with the other control blocks.
	is_loop: bool,
}

/// A block of code that metering instructions will be inserted at the beginning of. Metered blocks
/// are constructed with the property that, in the absence of any traps, either all instructions in
/// the block are executed or none are.
#[derive(Debug)]
struct MeteredBlock {
	/// Index of the first instruction (aka `Opcode`) in the block.
	start_pos: usize,
	/// Sum of costs of all instructions until end of the block.
	cost: u64,
}

/// Counter is used to manage state during the gas metering algorithm implemented by
/// `inject_counter`.
struct Counter {
	/// A stack of control blocks. This stack grows when new control blocks are opened with
	/// `block`, `loop`, and `if` and shrinks when control blocks are closed with `end`. The first
	/// block on the stack corresponds to the function body, not to any labelled block. Therefore
	/// the actual Wasm label index associated with each control block is 1 less than its position
	/// in this stack.
	stack: Vec<ControlBlock>,

	/// A list of metered blocks that have been finalized, meaning they will no longer change.
	finalized_blocks: Vec<MeteredBlock>,
}

impl Counter {
	fn new() -> Counter {
		Counter { stack: Vec::new(), finalized_blocks: Vec::new() }
	}

	/// Open a new control block. The cursor is the position of the first instruction in the block.
	fn begin_control_block(&mut self, cursor: usize, is_loop: bool) {
		let index = self.stack.len();
		self.stack.push(ControlBlock {
			lowest_forward_br_target: index,
			active_metered_block: MeteredBlock { start_pos: cursor, cost: 0 },
			is_loop,
		})
	}

	/// Close the last control block. The cursor is the position of the final (pseudo-)instruction
	/// in the block.
	fn finalize_control_block(&mut self, cursor: usize) -> Result<(), ()> {
		// This either finalizes the active metered block or merges its cost into the active
		// metered block in the previous control block on the stack.
		self.finalize_metered_block(cursor)?;

		// Pop the control block stack.
		let closing_control_block = self.stack.pop().ok_or(())?;
		let closing_control_index = self.stack.len();

		if self.stack.is_empty() {
			return Ok(())
		}

		// Update the lowest_forward_br_target for the control block now on top of the stack.
		{
			let control_block = self.stack.last_mut().ok_or(())?;
			control_block.lowest_forward_br_target = min(
				control_block.lowest_forward_br_target,
				closing_control_block.lowest_forward_br_target,
			);
		}

		// If there may have been a branch to a lower index, then also finalize the active metered
		// block for the previous control block. Otherwise, finalize it and begin a new one.
		let may_br_out = closing_control_block.lowest_forward_br_target < closing_control_index;
		if may_br_out {
			self.finalize_metered_block(cursor)?;
		}

		Ok(())
	}

	/// Finalize the current active metered block.
	///
	/// Finalized blocks have final cost which will not change later.
	fn finalize_metered_block(&mut self, cursor: usize) -> Result<(), ()> {
		let closing_metered_block = {
			let control_block = self.stack.last_mut().ok_or(())?;
			mem::replace(
				&mut control_block.active_metered_block,
				MeteredBlock { start_pos: cursor + 1, cost: 0 },
			)
		};

		// If the block was opened with a `block`, then its start position will be set to that of
		// the active metered block in the control block one higher on the stack. This is because
		// any instructions between a `block` and the first branch are part of the same basic block
		// as the preceding instruction. In this case, instead of finalizing the block, merge its
		// cost into the other active metered block to avoid injecting unnecessary instructions.
		let last_index = self.stack.len() - 1;
		if last_index > 0 {
			let prev_control_block = self
				.stack
				.get_mut(last_index - 1)
				.expect("last_index is greater than 0; last_index is stack size - 1; qed");
			let prev_metered_block = &mut prev_control_block.active_metered_block;
			if closing_metered_block.start_pos == prev_metered_block.start_pos {
				prev_metered_block.cost += closing_metered_block.cost;
				return Ok(())
			}
		}

		if closing_metered_block.cost > 0 {
			self.finalized_blocks.push(closing_metered_block);
		}
		Ok(())
	}

	/// Handle a branch instruction in the program. The cursor is the index of the branch
	/// instruction in the program. The indices are the stack positions of the target control
	/// blocks. Recall that the index is 0 for a `return` and relatively indexed from the top of
	/// the stack by the label of `br`, `br_if`, and `br_table` instructions.
	fn branch(&mut self, cursor: usize, indices: &[usize]) -> Result<(), ()> {
		self.finalize_metered_block(cursor)?;

		// Update the lowest_forward_br_target of the current control block.
		for &index in indices {
			let target_is_loop = {
				let target_block = self.stack.get(index).ok_or(())?;
				target_block.is_loop
			};
			if target_is_loop {
				continue
			}

			let control_block = self.stack.last_mut().ok_or(())?;
			control_block.lowest_forward_br_target =
				min(control_block.lowest_forward_br_target, index);
		}

		Ok(())
	}

	/// Returns the stack index of the active control block. Returns None if stack is empty.
	fn active_control_block_index(&self) -> Option<usize> {
		self.stack.len().checked_sub(1)
	}

	/// Get a reference to the currently active metered block.
	fn active_metered_block(&mut self) -> Result<&mut MeteredBlock, ()> {
		let top_block = self.stack.last_mut().ok_or(())?;
		Ok(&mut top_block.active_metered_block)
	}

	/// Increment the cost of the current block by the specified value.
	fn increment(&mut self, val: u64) -> Result<(), ()> {
		let top_block = self.active_metered_block()?;
		top_block.cost = top_block.cost.checked_add(val).ok_or(())?;
		Ok(())
	}
}

fn inject_dynamic_counters(instructions: &mut elements::Instructions, dyn_funcs: &mut HashMap<Instruction, u32>) -> usize {
	use parity_wasm::elements::Instruction::*;
	let mut counter = 0;
	for instruction in instructions.elements_mut() {
		// TODO CHECK FOR CONST!!!!

		if let Some(func_idx ) = dyn_funcs.get(instruction) {
			*instruction = Call(*func_idx);
			counter += 1;
		}
	}
	counter
}

fn add_dynamic_counters<R: Rules>(
	module: elements::Module,
	rules: &R,
	gas_func: u32,
	dyn_funcs: &HashMap<Instruction, u32>,
) -> Result<elements::Module, ()> {
	use parity_wasm::elements::Instruction::*;

	let mut funcs_sorted: Vec<(&Instruction, &u32)> = dyn_funcs.into_iter().collect();
	funcs_sorted.sort_by(|(_, lk), (_, rk)| lk.cmp(rk));

	let mut b = builder::from_module(module);

	for (instr, _) in funcs_sorted {
		let cost = match rules.instruction_cost(instr)? {
			InstructionCost::Linear(_, val) => val.get(),
			_ => return Err(()) // "dynamic instruction cost wasn't linear" todo anyhow errs
		};

		let (params, rets) = instruction_signature(instr)?;

		let mut counter_body = Vec::new();

		// first get all params back onto the stack
		for (i, _) in params.into_iter().enumerate() {
			counter_body.push(GetGlobal(i as u32))
		}

		// get the dynamic param back onto the stack
		counter_body.push(GetLocal(params.len() as u32-1));

		// cast the dynamic param if needed
		match params.last().ok_or(())? { // "dynamic functions must have params" todo anyhow errs
			ValueType::I32 => counter_body.push(I64ExtendUI32),
			_ => return Err(()), // "unsupported dynamic param type" todo anyhow
		}

		// calculate and charge gas, then call the original instruction!
		counter_body.append(&mut vec![
			I64Const(cost as i64),
			I64Mul,
			Call(gas_func),

			instr.clone(),

			End,
		]);

		b.push_function(
			builder::function().signature()
				.with_params(params.to_vec())
				.with_results(rets.to_vec())
				.build()
				.body()
				.with_instructions(elements::Instructions::new(counter_body))
				.build()
				.build(),
		);
	}

	Ok(b.build())
}

fn instruction_signature(instr: &Instruction) -> Result<(&'static[ValueType], &'static[ValueType]), ()> {
	use parity_wasm::elements::Instruction::*;

	match instr {
		GrowMemory(_) => Ok((&[ValueType::I32], &[ValueType::I32])),

		#[cfg(feature = "bulk")]
		Bulk(BulkInstruction::MemoryInit(_)) |
		Bulk(BulkInstruction::MemoryCopy) |
		Bulk(BulkInstruction::MemoryFill) |
		Bulk(BulkInstruction::TableInit(_)) |
		Bulk(BulkInstruction::TableCopy) => Ok((&[ValueType::I32, ValueType::I32, ValueType::I32], ())),

		_ => Err(()) // "instruction not supported" todo anyhow
	}
}

fn add_gas_counter(module: elements::Module, gas_global: u32) -> elements::Module {
	use parity_wasm::elements::Instruction::*;

	let mut b = builder::from_module(module);
	b.push_function(
		builder::function()
			.signature()
			.with_param(ValueType::I64)
			.build()
			.body()
			.with_instructions(elements::Instructions::new(vec![
				GetGlobal(gas_global), // (oldgas)
				GetLocal(0),           // (oldgas) (used)
				I64Sub,                // (newgas)
				SetGlobal(gas_global), //
				GetGlobal(gas_global), // (newgas)
				I64Const(0),           // (newgas) (0)
				I64LtS,                // (newgas ltz)
				If(BlockType::NoResult),
				Unreachable,
				End,
				End,
			]))
			.build()
			.build(),
	);

	b.build()
}

fn determine_metered_blocks<R: Rules>(
	instructions: &elements::Instructions,
	rules: &R,
) -> Result<(Vec<MeteredBlock>, HashSet<Instruction>), ()> {
	use parity_wasm::elements::Instruction::*;

	let mut counter = Counter::new();

	// Begin an implicit function (i.e. `func...end`) block.
	counter.begin_control_block(0, false);

	let mut last_const: Option<i32> = None;
	let mut liniear_cost_instructions = HashSet::new();

	for cursor in 0..instructions.elements().len() {
		let instruction = &instructions.elements()[cursor];
		let instruction_cost = match rules.instruction_cost(instruction)? {
			InstructionCost::Fixed(c) => c,
			InstructionCost::Linear(base, cost_per) => {
				if let Some(stack_top) = last_const {
					base + (stack_top as u64 * cost_per.get())
				} else {
					liniear_cost_instructions.insert(instruction.clone());
					// linear part will get charged at runtime (this instruction will get replaced
					// with a call to gas-charging func)
					base
				}
			}
		};

		match instruction {
			Block(_) => {
				counter.increment(instruction_cost)?;

				// Begin new block. The cost of the following opcodes until `end` or `else` will
				// be included into this block. The start position is set to that of the previous
				// active metered block to signal that they should be merged in order to reduce
				// unnecessary metering instructions.
				let top_block_start_pos = counter.active_metered_block()?.start_pos;
				counter.begin_control_block(top_block_start_pos, false);
			},
			If(_) => {
				counter.increment(instruction_cost)?;
				counter.begin_control_block(cursor + 1, false);
			},
			Loop(_) => {
				counter.increment(instruction_cost)?;
				counter.begin_control_block(cursor + 1, true);
			},
			End => {
				counter.finalize_control_block(cursor)?;
			},
			Else => {
				counter.finalize_metered_block(cursor)?;
			},
			Br(label) | BrIf(label) => {
				counter.increment(instruction_cost)?;

				// Label is a relative index into the control stack.
				let active_index = counter.active_control_block_index().ok_or(())?;
				let target_index = active_index.checked_sub(*label as usize).ok_or(())?;
				counter.branch(cursor, &[target_index])?;
			},
			BrTable(br_table_data) => {
				counter.increment(instruction_cost)?;

				let active_index = counter.active_control_block_index().ok_or(())?;
				let target_indices = [br_table_data.default]
					.iter()
					.chain(br_table_data.table.iter())
					.map(|label| active_index.checked_sub(*label as usize))
					.collect::<Option<Vec<_>>>()
					.ok_or(())?;
				counter.branch(cursor, &target_indices)?;
			},
			Return => {
				counter.increment(instruction_cost)?;
				counter.branch(cursor, &[0])?;
			},
			I32Const(v) => {
				last_const = Some(*v);
				counter.increment(instruction_cost)?;
				continue;
			},
			_ => {
				// An ordinal non control flow instruction increments the cost of the current block.
				counter.increment(instruction_cost)?;
			},
		}

		last_const = None;
	}

	counter.finalized_blocks.sort_unstable_by_key(|block| block.start_pos);
	Ok((counter.finalized_blocks, liniear_cost_instructions))
}

fn inject_counter<R: Rules>(
	instructions: &mut elements::Instructions,
	rules: &R,
	gas_func: u32,
) -> Result<HashSet<Instruction>, ()> {
	let (blocks, dynamic_instrs) = determine_metered_blocks(instructions, rules)?;
	insert_metering_calls(instructions, blocks, gas_func).map(|_|dynamic_instrs)
}

// Then insert metering calls into a sequence of instructions given the block locations and costs.
fn insert_metering_calls(
	instructions: &mut elements::Instructions,
	blocks: Vec<MeteredBlock>,
	gas_func: u32,
) -> Result<(), ()> {
	use parity_wasm::elements::Instruction::*;

	// To do this in linear time, construct a new vector of instructions, copying over old
	// instructions one by one and injecting new ones as required.
	let new_instrs_len = instructions.elements().len() + 2 * blocks.len();
	let original_instrs =
		mem::replace(instructions.elements_mut(), Vec::with_capacity(new_instrs_len));
	let new_instrs = instructions.elements_mut();

	let mut block_iter = blocks.into_iter().peekable();
	for (original_pos, instr) in original_instrs.into_iter().enumerate() {
		// If there the next block starts at this position, inject metering instructions.
		let used_block = if let Some(block) = block_iter.peek() {
			if block.start_pos == original_pos {
				new_instrs.push(I64Const(block.cost as i64));
				new_instrs.push(Call(gas_func));
				true
			} else {
				false
			}
		} else {
			false
		};

		if used_block {
			block_iter.next();
		}

		// Copy over the original instruction.
		new_instrs.push(instr);
	}

	if block_iter.next().is_some() {
		return Err(())
	}

	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;
	use parity_wasm::{builder, elements, elements::Instruction::*, serialize};

	fn get_function_body(
		module: &elements::Module,
		index: usize,
	) -> Option<&[elements::Instruction]> {
		module
			.code_section()
			.and_then(|code_section| code_section.bodies().get(index))
			.map(|func_body| func_body.code().elements())
	}

	#[test]
	fn simple_grow() {
		let module = parse_wat(
			r#"(module
			(func (result i32)
			  global.get 0
			  memory.grow)
			(global i32 (i32.const 42))
			(memory 0 1)
			)"#,
		);

		let injected_module = inject(module, &ConstantCostRules::new(1, 10_000), "env").unwrap();

		// global0 - gas
		// global1 - orig global0
		// func0 - main
		// func1 - gas_counter
		// func2 - grow_counter

		assert_eq!(
			get_function_body(&injected_module, 0).unwrap(),
			&vec![I64Const(2), Call(1), GetGlobal(1), Call(2), End][..]
		);
		// 1 is gas counter
		assert_eq!(
			get_function_body(&injected_module, 2).unwrap(),
			&vec![
				GetLocal(0),
				GetLocal(0),
				I64ExtendUI32,
				I64Const(10000),
				I64Mul,
				Call(1),
				GrowMemory(0),
				End,
			][..]
		);

		let binary = serialize(injected_module).expect("serialization failed");
		wasmparser::validate(&binary).unwrap();
	}

	#[test]
	fn grow_no_gas_no_track() {
		let module = parse_wat(
			r"(module
			(func (result i32)
			  global.get 0
			  memory.grow)
			(global i32 (i32.const 42))
			(memory 0 1)
			)",
		);

		let injected_module = inject(module, &ConstantCostRules::default(), "env").unwrap();

		assert_eq!(
			get_function_body(&injected_module, 0).unwrap(),
			&vec![I64Const(2), Call(1), GetGlobal(1), GrowMemory(0), End][..]
		);

		assert_eq!(injected_module.functions_space(), 2);

		let binary = serialize(injected_module).expect("serialization failed");
		wasmparser::validate(&binary).unwrap();
	}

	#[test]
	fn call_index() {
		let module = builder::module()
			.global()
			.value_type()
			.i32()
			.build()
			.function()
			.signature()
			.param()
			.i32()
			.build()
			.body()
			.build()
			.build()
			.function()
			.signature()
			.param()
			.i32()
			.build()
			.body()
			.with_instructions(elements::Instructions::new(vec![
				Call(0),
				If(elements::BlockType::NoResult),
				Call(0),
				Call(0),
				Call(0),
				Else,
				Call(0),
				Call(0),
				End,
				Call(0),
				End,
			]))
			.build()
			.build()
			.build();

		let injected_module = inject(module, &ConstantCostRules::default(), "env").unwrap();

		assert_eq!(
			get_function_body(&injected_module, 1).unwrap(),
			&vec![
				I64Const(3),
				Call(2),
				Call(0),
				If(elements::BlockType::NoResult),
				I64Const(3),
				Call(2),
				Call(0),
				Call(0),
				Call(0),
				Else,
				I64Const(2),
				Call(2),
				Call(0),
				Call(0),
				End,
				Call(0),
				End
			][..]
		);
	}

	fn parse_wat(source: &str) -> elements::Module {
		let module_bytes = wat::parse_str(source).unwrap();
		elements::deserialize_buffer(module_bytes.as_ref()).unwrap()
	}

	macro_rules! test_gas_counter_injection {
		(name = $name:ident; input = $input:expr; expected = $expected:expr) => {
			#[test]
			fn $name() {
				let input_module = parse_wat($input);
				let expected_module = parse_wat($expected);

				let injected_module = inject(input_module, &ConstantCostRules::default(), "env")
					.expect("inject_gas_counter call failed");

				let actual_func_body = get_function_body(&injected_module, 0)
					.expect("injected module must have a function body");
				let expected_func_body = get_function_body(&expected_module, 0)
					.expect("post-module must have a function body");

				assert_eq!(actual_func_body, expected_func_body);
			}
		};
	}

	test_gas_counter_injection! {
		name = simple;
		input = r#"
		(module
			(func (result i32)
				(get_global 0)))
		"#;
		expected = r#"
		(module
			(func (result i32)
				(call 1 (i64.const 1))
				(get_global 1)))
		"#
	}

	#[test]
	fn test_gas_error_fvm_fuzzin_5() {
		let input = r#"
		(module
			(type (;0;) (func (result i32)))
			(type (;1;) (func (param i32)))
			(type (;2;) (func (param i32) (result i32)))
			(type (;3;) (func (param i32 i32)))
			(type (;4;) (func (param i32) (result i64)))
			(type (;5;) (func (param i32 i32 i32) (result i64)))
			(type (;6;) (func (param i32 i32 i32)))
			(type (;7;) (func (param i32 i32) (result i32)))
			(type (;8;) (func (param i32 i32 i32 i32)))
			(type (;9;) (func (param i64 i32) (result i64)))
			(type (;10;) (func (param i32 i64)))
			(import "env" "memory" (memory (;0;) 256 256))
			(import "env" "DYNAMICTOP_PTR" (global (;0;) i32))
			(import "env" "STACKTOP" (global (;1;) i32))
			(import "env" "enlargeMemory" (func (;0;) (type 0)))
			(import "env" "getTotalMemory" (func (;1;) (type 0)))
			(import "env" "abortOnCannotGrowMemory" (func (;2;) (type 0)))
			(import "env" "___setErrNo" (func (;3;) (type 1)))
			(func (;4;) (type 9) (param i64 i32) (result i64)
			  local.get 0
			  i32.const 64
			  local.get 1
			  i32.sub
			  i64.extend_i32_u
			  i64.shl
			  local.get 0
			  local.get 1
			  i64.extend_i32_u
			  i64.shr_u
			  i64.or
			)
		  )
		"#;
		let input_module = parse_wat(input);
		let injected_module = inject(input_module, &ConstantCostRules::default(), "other")
			.expect("inject_gas_counter call failed");
	}

	test_gas_counter_injection! {
		name = nested;
		input = r#"
		(module
			(func (result i32)
				(get_global 0)
				(block
					(get_global 0)
					(get_global 0)
					(get_global 0))
				(get_global 0)))
		"#;
		expected = r#"
		(module
			(func (result i32)
				(call 1 (i64.const 6))
				(get_global 1)
				(block
					(get_global 1)
					(get_global 1)
					(get_global 1))
				(get_global 1)))
		"#
	}

	test_gas_counter_injection! {
		name = ifelse;
		input = r#"
		(module
			(func (result i32)
				(get_global 0)
				(if
					(then
						(get_global 0)
						(get_global 0)
						(get_global 0))
					(else
						(get_global 0)
						(get_global 0)))
				(get_global 0)))
		"#;
		expected = r#"
		(module
			(func (result i32)
				(call 1 (i64.const 3))
				(get_global 1)
				(if
					(then
						(call 1 (i64.const 3))
						(get_global 1)
						(get_global 1)
						(get_global 1))
					(else
						(call 1 (i64.const 2))
						(get_global 1)
						(get_global 1)))
				(get_global 1)))
		"#
	}

	test_gas_counter_injection! {
		name = branch_innermost;
		input = r#"
		(module
			(func (result i32)
				(get_global 0)
				(block
					(get_global 0)
					(drop)
					(br 0)
					(get_global 0)
					(drop))
				(get_global 0)))
		"#;
		expected = r#"
		(module
			(func (result i32)
				(call 1 (i64.const 6))
				(get_global 1)
				(block
					(get_global 1)
					(drop)
					(br 0)
					(call 1 (i64.const 2))
					(get_global 1)
					(drop))
				(get_global 1)))
		"#
	}

	test_gas_counter_injection! {
		name = branch_outer_block;
		input = r#"
		(module
			(func (result i32)
				(get_global 0)
				(block
					(get_global 0)
					(if
						(then
							(get_global 0)
							(get_global 0)
							(drop)
							(br_if 1)))
					(get_global 0)
					(drop))
				(get_global 0)))
		"#;
		expected = r#"
		(module
			(func (result i32)
				(call 1 (i64.const 5))
				(get_global 1)
				(block
					(get_global 1)
					(if
						(then
							(call 1 (i64.const 4))
							(get_global 1)
							(get_global 1)
							(drop)
							(br_if 1)))
					(call 1 (i64.const 2))
					(get_global 1)
					(drop))
				(get_global 1)))
		"#
	}

	test_gas_counter_injection! {
		name = branch_outer_loop;
		input = r#"
		(module
			(func (result i32)
				(get_global 0)
				(loop
					(get_global 0)
					(if
						(then
							(get_global 0)
							(br_if 0))
						(else
							(get_global 0)
							(get_global 0)
							(drop)
							(br_if 1)))
					(get_global 0)
					(drop))
				(get_global 0)))
		"#;
		expected = r#"
		(module
			(func (result i32)
				(call 1 (i64.const 3))
				(get_global 1)
				(loop
					(call 1 (i64.const 4))
					(get_global 1)
					(if
						(then
							(call 1 (i64.const 2))
							(get_global 1)
							(br_if 0))
						(else
							(call 1 (i64.const 4))
							(get_global 1)
							(get_global 1)
							(drop)
							(br_if 1)))
					(get_global 1)
					(drop))
				(get_global 1)))
		"#
	}

	test_gas_counter_injection! {
		name = return_from_func;
		input = r#"
		(module
			(func (result i32)
				(get_global 0)
				(if
					(then
						(return)))
				(get_global 0)))
		"#;
		expected = r#"
		(module
			(func (result i32)
				(call 1 (i64.const 2))
				(get_global 1)
				(if
					(then
						(call 1 (i64.const 1))
						(return)))
				(call 1 (i64.const 1))
				(get_global 1)))
		"#
	}

	test_gas_counter_injection! {
		name = branch_from_if_not_else;
		input = r#"
		(module
			(func (result i32)
				(get_global 0)
				(block
					(get_global 0)
					(if
						(then (br 1))
						(else (br 0)))
					(get_global 0)
					(drop))
				(get_global 0)))
		"#;
		expected = r#"
		(module
			(func (result i32)
				(call 1 (i64.const 5))
				(get_global 1)
				(block
					(get_global 1)
					(if
						(then
							(call 1 (i64.const 1))
							(br 1))
						(else
							(call 1 (i64.const 1))
							(br 0)))
					(call 1 (i64.const 2))
					(get_global 1)
					(drop))
				(get_global 1)))
		"#
	}

	test_gas_counter_injection! {
		name = empty_loop;
		input = r#"
		(module
			(func
				(loop
					(br 0)
				)
				unreachable
			)
		)
		"#;
		expected = r#"
		(module
			(func
				(call 1 (i64.const 2))
				(loop
					(call 1 (i64.const 1))
					(br 0)
				)
				unreachable
			)
		)
		"#
	}
}

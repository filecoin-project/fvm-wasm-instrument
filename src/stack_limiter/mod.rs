//! Contains the code for the stack height limiter instrumentation.

use crate::utils::{
	copy_locals,
	translator::{DefaultTranslator, Translator},
	ModuleInfo,
};
use alloc::vec::Vec;
use anyhow::{anyhow, Result};
use wasm_encoder::{
	CodeSection, ConstExpr, Function, GlobalSection, GlobalType, SectionId, ValType,
};
use wasmparser::{CodeSectionReader, FunctionBody, GlobalSectionReader, Operator};

/// Macro to generate preamble and postamble.
macro_rules! instrument_call {
	($callee_idx: expr, $callee_stack_cost: expr, $stack_height_global_idx: expr, $stack_limit: expr) => {{
		use wasm_encoder::Instruction::*;
		[
			// stack_height += stack_cost(F)
			GlobalGet($stack_height_global_idx),
			I32Const($callee_stack_cost),
			I32Add,
			GlobalSet($stack_height_global_idx),
			// if stack_counter > LIMIT: unreachable
			GlobalGet($stack_height_global_idx),
			I32Const($stack_limit as i32),
			I32GtU,
			If(wasm_encoder::BlockType::Empty),
			Unreachable,
			End,
			// Original call
			Call($callee_idx),
			// stack_height -= stack_cost(F)
			GlobalGet($stack_height_global_idx),
			I32Const($callee_stack_cost),
			I32Sub,
			GlobalSet($stack_height_global_idx),
		]
	}};
}

mod max_height;
mod thunk;

pub struct Context {
	stack_height_global_idx: u32,
	func_stack_costs: Vec<u32>,
	stack_limit: u32,
}

impl Context {
	/// Returns index in a global index space of a stack_height global variable.
	fn stack_height_global_idx(&self) -> u32 {
		self.stack_height_global_idx
	}

	/// Returns `stack_cost` for `func_idx`.
	fn stack_cost(&self, func_idx: u32) -> Option<u32> {
		self.func_stack_costs.get(func_idx as usize).cloned()
	}

	/// Returns stack limit specified by the rules.
	fn stack_limit(&self) -> u32 {
		self.stack_limit
	}
}

/// Inject the instumentation that makes stack overflows deterministic, by introducing
/// an upper bound of the stack size.
///
/// This pass introduces a global mutable variable to track stack height,
/// and instruments all calls with preamble and postamble.
///
/// Stack height is increased prior the call. Otherwise, the check would
/// be made after the stack frame is allocated.
///
/// The preamble is inserted before the call. It increments
/// the global stack height variable with statically determined "stack cost"
/// of the callee. If after the increment the stack height exceeds
/// the limit (specified by the `rules`) then execution traps.
/// Otherwise, the call is executed.
///
/// The postamble is inserted after the call. The purpose of the postamble is to decrease
/// the stack height by the "stack cost" of the callee function.
///
/// Note, that we can't instrument all possible ways to return from the function. The simplest
/// example would be a trap issued by the host function.
/// That means stack height global won't be equal to zero upon the next execution after such trap.
///
/// # Thunks
///
/// Because stack height is increased prior the call few problems arises:
///
/// - Stack height isn't increased upon an entry to the first function, i.e. exported function.
/// - Start function is executed externally (similar to exported functions).
/// - It is statically unknown what function will be invoked in an indirect call.
///
/// The solution for this problems is to generate a intermediate functions, called 'thunks', which
/// will increase before and decrease the stack height after the call to original function, and
/// then make exported function and table entries, start section to point to a corresponding thunks.
///
/// # Stack cost
///
/// Stack cost of the function is calculated as a sum of it's locals
/// and the maximal height of the value stack.
///
/// All values are treated equally, as they have the same size.
///
/// The rationale is that this makes it possible to use the following very naive wasm executor:
///
/// - values are implemented by a union, so each value takes a size equal to the size of the largest
///   possible value type this union can hold. (In MVP it is 8 bytes)
/// - each value from the value stack is placed on the native stack.
/// - each local variable and function argument is placed on the native stack.
/// - arguments pushed by the caller are copied into callee stack rather than shared between the
///   frames.
/// - upon entry into the function entire stack frame is allocated.
pub fn inject(raw_wasm: &[u8], stack_limit: u32) -> Result<Vec<u8>> {
	let mut module_info = ModuleInfo::new(raw_wasm)?;
	let mut ctx = Context {
		stack_height_global_idx: generate_stack_height_global(&mut module_info)?,
		func_stack_costs: compute_stack_costs(&module_info)?,
		stack_limit,
	};

	instrument_functions(&mut ctx, &mut module_info)?;
	thunk::generate_thunks(&mut ctx, &mut module_info)?;

	Ok(module_info.bytes())
}

/// Generate a new global that will be used for tracking current stack height.
fn generate_stack_height_global(module: &mut ModuleInfo) -> Result<u32> {
	let mut global_sec_builder = GlobalSection::new();
	let index = if let Some(global_sec) = &module.raw_sections.get(&SectionId::Global.into()) {
		let reader = GlobalSectionReader::new(&global_sec.data, 0)?;
		let count = reader.get_count();
		for global in reader {
			DefaultTranslator.translate_global(global?, &mut global_sec_builder)?;
		}
		count
	} else {
		0
	};

	global_sec_builder
		.global(GlobalType { val_type: ValType::I32, mutable: true }, &ConstExpr::i32_const(0));
	module.replace_section(SectionId::Global.into(), &global_sec_builder)?;
	Ok(index)
}

/// Calculate stack costs for all functions.
///
/// Returns a vector with a stack cost for each function, including imports.
fn compute_stack_costs(module: &ModuleInfo) -> Result<Vec<u32>> {
	let func_imports = module.num_imported_functions();

	// TODO: optimize!
	(0..module.num_functions())
		.map(|func_idx| {
			if func_idx < func_imports {
				// We can't calculate stack_cost of the import functions.
				Ok(0)
			} else {
				compute_stack_cost(func_idx as u32, module)
			}
		})
		.collect()
}

/// Stack cost of the given *defined* function is the sum of it's locals count (that is,
/// number of arguments plus number of local variables) and the maximal stack
/// height.
fn compute_stack_cost(func_idx: u32, module: &ModuleInfo) -> Result<u32> {
	// To calculate the cost of a function we need to convert index from
	// function index space to defined function spaces.
	let func_imports = module.num_imported_functions();
	let defined_func_idx = func_idx
		.checked_sub(func_imports)
		.ok_or_else(|| anyhow!("this should be a index of a defined function"))?;

	let code_section_reader = CodeSectionReader::new(
		&module
			.raw_sections
			.get(&SectionId::Code.into())
			.ok_or_else(|| anyhow!("not find code section"))?
			.data,
		0,
	)?;

	let local_reader = code_section_reader
		.into_iter()
		.collect::<wasmparser::Result<Vec<FunctionBody>>>()?
		.get(defined_func_idx as usize)
		.ok_or_else(|| anyhow!("function body is out of bounds"))?
		.get_locals_reader()?;

	let locals_count: u32 = local_reader.get_count();
	let max_stack_height = max_height::compute(defined_func_idx, module)?;

	locals_count
		.checked_add(max_stack_height)
		.ok_or_else(|| anyhow!("overflow in adding locals_count and max_stack_height"))
}

fn instrument_functions(ctx: &mut Context, module: &mut ModuleInfo) -> Result<()> {
	let mut code_builder = CodeSection::new();
	if let Some(code_sec) = module.raw_sections.get(&SectionId::Code.into()) {
		let function_sec_reader = CodeSectionReader::new(&code_sec.data, 0)?;
		for body in function_sec_reader {
			let body_encoder = instrument_function(ctx, body?)?;
			code_builder.function(&body_encoder);
		}
	}
	module.replace_section(SectionId::Code.into(), &code_builder)
}

/// This function searches `call` instructions and wrap each call
/// with preamble and postamble.
///
/// Before:
///
/// ```text
/// get_local 0
/// get_local 1
/// call 228
/// drop
/// ```
///
/// After:
///
/// ```text
/// get_local 0
/// get_local 1
///
/// < ... preamble ... >
///
/// call 228
///
/// < .. postamble ... >
///
/// drop
/// ```
#[allow(clippy::needless_collect)]
fn instrument_function(ctx: &mut Context, func: FunctionBody) -> Result<wasm_encoder::Function> {
	struct InstrumentCall {
		offset: usize,
		callee: u32,
		cost: u32,
	}
	let mut func_code_builder = Function::new(copy_locals(&func)?);
	let reader = func.get_operators_reader()?;
	let operators = reader.into_iter().collect::<wasmparser::Result<Vec<Operator>>>()?;

	let calls: Vec<_> = operators
		.iter()
		.enumerate()
		.filter_map(|(offset, operator)| {
			if let Operator::Call { function_index: callee } = operator {
				//todo CallDirect
				ctx.stack_cost(*callee).and_then(|cost| {
					if cost > 0 {
						Some(InstrumentCall { callee: *callee, offset, cost })
					} else {
						None
					}
				})
			} else {
				None
			}
		})
		.collect();
	// The `instrumented_call!` contains the call itself. This is why we need to subtract one.
	let mut call_peeker = calls.into_iter().peekable();
	for (original_pos, instr) in operators.into_iter().enumerate() {
		// whether there is some call instruction at this position that needs to be instrumented
		let did_instrument = if let Some(call) = call_peeker.peek() {
			if call.offset == original_pos {
				instrument_call!(
					call.callee,
					call.cost as i32,
					ctx.stack_height_global_idx(),
					ctx.stack_limit()
				)
				.iter()
				.for_each(|instr| {
					func_code_builder.instruction(instr);
				});
				true
			} else {
				false
			}
		} else {
			false
		};

		if did_instrument {
			call_peeker.next();
		} else {
			func_code_builder.instruction(&DefaultTranslator.translate_op(&instr)?);
		}
	}

	if call_peeker.next().is_some() {
		return Err(anyhow!("not all calls were used"))
	}

	Ok(func_code_builder)
}

#[cfg(test)]
mod tests {
	use super::*;

	fn parse_wat(source: &str) -> ModuleInfo {
		let module_bytes = wat::parse_str(source).unwrap();
		ModuleInfo::new(&module_bytes).unwrap()
	}

	#[test]
	fn test_with_params_and_result() {
		let raw_wasm = parse_wat(
			r#"(module
						(func (export "i32.add") (param i32 i32) (result i32)
							get_local 0
							get_local 1
							i32.add
						)
					)"#,
		)
		.bytes();

		let inject_raw_wasm = inject(&raw_wasm, 1024).expect("Failed to inject stack counter");
		wasmparser::validate(&inject_raw_wasm).expect("Invalid module");
	}
}

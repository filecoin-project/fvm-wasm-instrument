//! This module is used to instrument a Wasm module with gas metering code.
//!
//! The primary public interface is the [`inject`] function which transforms a given
//! module into one that charges gas for code to be executed. See function documentation for usage
//! and details.

#[cfg(test)]
pub mod validation;

use crate::utils::{
    copy_locals,
    translator::{DefaultTranslator, Translator},
    truncate_len_from_encoder, ModuleInfo,
};
use alloc::{vec, vec::Vec};
use anyhow::{anyhow, Result};
use core::{cmp::min, mem};
use std::num::NonZeroU32;
use wasm_encoder::{
    BlockType, ExportSection, Function, ImportSection, Instruction, SectionId, ValType,
};
use wasmparser::{
    CodeSectionReader, DataKind, DataSectionReader, ElementItem, ElementSectionReader,
    ExportSectionReader, ExternalKind, FuncType, FunctionBody, FunctionSectionReader,
    ImportSectionReader, Operator, SectionReader, Type, TypeRef, TypeSectionReader,
};

pub const GAS_COUNTER_NAME: &str = "gas_counter";

/// An interface that describes instruction costs.
pub trait Rules {
    /// Returns the cost for the passed `instruction`.
    ///
    /// Returning an error can be used as a way to indicate that an instruction
    /// is forbidden
    fn instruction_cost(&self, instruction: &Operator) -> Result<InstructionCost>;

    /// Returns cost for each call to the gas charging function
    fn gas_charge_cost(&self) -> u64;

    /// Returns cost of calculating linear cost at runtime. Does not apply to
    /// instructions cost of which can be statically determined (linearly priced
    /// ops proceded by a const). Added to gas_charge_cost on dynamic charges
    fn linear_calc_cost(&self) -> u64;
}

/// Dynamic costs instructions.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum InstructionCost {
    /// Charge fixed amount per instruction.
    Fixed(u64),

    /// Charge the specified amount of base miligas, plus miligas based on the
    /// last item on the stack. For memory.grow this is the number of pages.
    /// For memory.copy this is the number of bytes to copy.
    ///
    /// Note: in order to make overflows impossible, the second (cost per unit)
    /// value must be in range 0x1 ~ 0x7fff_ffff.
    Linear(u64, NonZeroU32),
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
    memory_grow_cost: u32,
}

impl ConstantCostRules {
    /// Create a new [`ConstantCostRules`].
    ///
    /// Uses `instruction_cost` for every instruction and `memory_grow_cost` to dynamically
    /// meter the memory growth instruction.
    pub fn new(instruction_cost: u64, memory_grow_cost: u32) -> Self {
        Self {
            instruction_cost,
            memory_grow_cost,
        }
    }
}

impl Default for ConstantCostRules {
    /// Uses instruction cost of `1` and disables memory growth instrumentation.
    fn default() -> Self {
        Self {
            instruction_cost: 1,
            memory_grow_cost: 0,
        }
    }
}

impl Rules for ConstantCostRules {
    fn instruction_cost(&self, i: &Operator) -> Result<InstructionCost> {
        match i {
            Operator::MemoryGrow { .. } => Ok(NonZeroU32::new(self.memory_grow_cost)
                .map_or(InstructionCost::Fixed(1), |c| InstructionCost::Linear(1, c))),
            _ => Ok(InstructionCost::Fixed(self.instruction_cost)),
        }
    }

    fn gas_charge_cost(&self) -> u64 {
        0
    }

    fn linear_calc_cost(&self) -> u64 {
        0
    }
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

/// An instruction that requires Linear gas charge to be applied
#[derive(Debug, Clone)]
struct MeteredInstruction {
    /// Index of the instruction.
    pos: usize,
    /// Cost per unit. Multiplied by top of stack to get the actual cost
    unit_cost: u32,
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
        Counter {
            stack: Vec::new(),
            finalized_blocks: Vec::new(),
        }
    }

    /// Open a new control block. The cursor is the position of the first instruction in the block.
    fn begin_control_block(&mut self, cursor: usize, is_loop: bool) {
        let index = self.stack.len();
        self.stack.push(ControlBlock {
            lowest_forward_br_target: index,
            active_metered_block: MeteredBlock {
                start_pos: cursor,
                cost: 0,
            },
            is_loop,
        })
    }

    /// Close the last control block. The cursor is the position of the final (pseudo-)instruction
    /// in the block.
    fn finalize_control_block(&mut self, cursor: usize) -> Result<()> {
        // This either finalizes the active metered block or merges its cost into the active
        // metered block in the previous control block on the stack.
        self.finalize_metered_block(cursor)?;

        // Pop the control block stack.
        let closing_control_block = self.stack.pop().ok_or_else(|| anyhow!("stack not found"))?;
        let closing_control_index = self.stack.len();

        if self.stack.is_empty() {
            return Ok(());
        }

        // Update the lowest_forward_br_target for the control block now on top of the stack.
        {
            let control_block = self
                .stack
                .last_mut()
                .ok_or_else(|| anyhow!("stack not found"))?;
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
    fn finalize_metered_block(&mut self, cursor: usize) -> Result<()> {
        let closing_metered_block = {
            let control_block = self
                .stack
                .last_mut()
                .ok_or_else(|| anyhow!("stack not found"))?;
            mem::replace(
                &mut control_block.active_metered_block,
                MeteredBlock {
                    start_pos: cursor + 1,
                    cost: 0,
                },
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
                return Ok(());
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
    fn branch(&mut self, cursor: usize, indices: &[usize]) -> Result<()> {
        self.finalize_metered_block(cursor)?;

        // Update the lowest_forward_br_target of the current control block.
        for &index in indices {
            let target_is_loop = {
                let target_block = self
                    .stack
                    .get(index)
                    .ok_or_else(|| anyhow!("unable to find stack index"))?;
                target_block.is_loop
            };
            if target_is_loop {
                continue;
            }

            let control_block = self
                .stack
                .last_mut()
                .ok_or_else(|| anyhow!("unable to find stack index"))?;
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
    fn active_metered_block(&mut self) -> Result<&mut MeteredBlock> {
        let top_block = self
            .stack
            .last_mut()
            .ok_or_else(|| anyhow!("stack not exit"))?;
        Ok(&mut top_block.active_metered_block)
    }

    /// Increment the cost of the current block by the specified value.
    fn increment(&mut self, val: u64) -> Result<()> {
        let top_block = self.active_metered_block()?;
        top_block.cost = top_block
            .cost
            .checked_add(val)
            .ok_or_else(|| anyhow!("add cost overflow"))?;
        Ok(())
    }
}

fn determine_metered_blocks<R: Rules>(
    func_body: &wasmparser::FunctionBody,
    rules: &R,
) -> Result<(Vec<MeteredBlock>, Vec<MeteredInstruction>)> {
    use wasmparser::Operator::*;

    let mut counter = Counter::new();
    // Begin an implicit function (i.e. `func...end`) block.
    counter.begin_control_block(0, false);

    // set if the previous instruction was a I32Const. Used to precompute linear
    // instruction costs where there are sequences of instructions like
    // `[..]; i32.const 123; memory.copy`
    let mut last_const: Option<i32> = None;

    let mut metered_instrs = Vec::new();

    let operators = func_body
        .get_operators_reader()
        .unwrap()
        .into_iter()
        .collect::<wasmparser::Result<Vec<Operator>>>()
        .unwrap();
    for (cursor, instruction) in operators.iter().enumerate() {
        let instruction_cost = match rules.instruction_cost(instruction)? {
            InstructionCost::Fixed(c) => c,
            InstructionCost::Linear(base, cost_per) => {
                // Enforce that cost per unit fits in 31 bits
                if cost_per.get() >= 0x8000_0000 {
                    return Err(anyhow!("cost per unit excedes the 0x80000000 limit"));
                }

                if let Some(stack_top) = last_const {
                    if stack_top < 0 {
                        // See "NOTE(negative bulk instruction arg)" below
                        return Err(anyhow!(
                            "linearly priced instructions are illegal with negative arguments"
                        ));
                    }

                    // note: doesn't overflow because both sides are u32
                    base + ((stack_top as u64) * (cost_per.get() as u64))
                } else {
                    // Code in insert_metering_calls below needs to create temporary locals in order
                    // to be able to duplicate stack items. For simplicity/performance that code
                    // hard-codes i32 valtype. This is fine as all instructions for which dynamic
                    // pricing makes sense have i32 stack top.
                    //
                    // If the need ever arises to support more value types, this check will need to
                    // be removed, and the logic creating temporary locals in insert_metering_calls
                    // will need to be made smarter.
                    match instruction_stack_top_type(instruction)? {
                        ValType::I32 => {},
                        _ => return Err(anyhow!("linearly priced instructions with non-i32 stack top aren't supported yet")),
                    };

                    metered_instrs.push(MeteredInstruction {
                        pos: cursor,
                        unit_cost: cost_per.get(),
                    });
                    // linear part will get charged at runtime (this instruction will get replaced
                    // with a call to gas-charging func)
                    base + rules.gas_charge_cost() + rules.linear_calc_cost()
                }
            }
        };

        match instruction {
            Block { ty: _ } => {
                counter.increment(instruction_cost)?;

                // Begin new block. The cost of the following opcodes until `end` or `else` will
                // be included into this block. The start position is set to that of the previous
                // active metered block to signal that they should be merged in order to reduce
                // unnecessary metering instructions.
                let top_block_start_pos = counter.active_metered_block()?.start_pos;
                counter.begin_control_block(top_block_start_pos, false);
            }
            If { ty: _ } => {
                counter.increment(instruction_cost)?;
                counter.begin_control_block(cursor + 1, false);
            }
            wasmparser::Operator::Loop { ty: _ } => {
                counter.increment(instruction_cost)?;
                counter.begin_control_block(cursor + 1, true);
            }
            End => {
                counter.finalize_control_block(cursor)?;
            }
            Else => {
                counter.finalize_metered_block(cursor)?;
            }
            wasmparser::Operator::Br {
                relative_depth: label,
            }
            | wasmparser::Operator::BrIf {
                relative_depth: label,
            } => {
                counter.increment(instruction_cost)?;

                // Label is a relative index into the control stack.
                let active_index = counter
                    .active_control_block_index()
                    .ok_or_else(|| anyhow!("active control block not exit"))?;
                let target_index = active_index
                    .checked_sub(*label as usize)
                    .ok_or_else(|| anyhow!("index not found"))?;
                counter.branch(cursor, &[target_index])?;
            }
            wasmparser::Operator::BrTable {
                table: br_table_data,
            } => {
                counter.increment(instruction_cost)?;

                let active_index = counter
                    .active_control_block_index()
                    .ok_or_else(|| anyhow!("index not found"))?;
                let r = br_table_data
                    .targets()
                    .collect::<wasmparser::Result<Vec<u32>>>()
                    .unwrap();
                let target_indices = Vec::with_capacity(br_table_data.default() as usize)
                    .iter()
                    .chain(r.iter())
                    .map(|label| active_index.checked_sub(*label as usize))
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| anyhow!("to do check this error"))?;
                counter.branch(cursor, &target_indices)?;
            }
            wasmparser::Operator::Return => {
                counter.increment(instruction_cost)?;
                counter.branch(cursor, &[0])?;
            }
            wasmparser::Operator::I32Const { value } => {
                last_const = Some(*value);
                counter.increment(instruction_cost)?;

                // continue so that we don't clear last_const below
                continue;
            }
            _ => {
                // An ordinal non control flow instruction increments the cost of the current block.
                counter.increment(instruction_cost)?;
            }
        }

        last_const = None;
    }

    counter
        .finalized_blocks
        .sort_unstable_by_key(|block| block.start_pos);
    Ok((counter.finalized_blocks, metered_instrs))
}

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
/// the original module as an Err. Only one imported global is allowed per `gas_module_name`, the
/// one corresponding to the gas spending measurement
pub fn inject<R: Rules>(raw_wasm: &[u8], rules: &R, gas_module_name: &str) -> Result<Vec<u8>> {
    // Injecting gas counting external
    let mut module_info = ModuleInfo::new(raw_wasm)?;
    add_gas_global_import(&mut module_info, gas_module_name)?;

    // calculate actual global index of the imported definition
    //    (subtract all imports that are NOT globals)
    let gas_global = module_info.imported_globals_count - 1;
    let total_func = module_info.function_map.len() as u32;

    // We'll push the gas counter fuction after all other functions
    let gas_func = total_func;

    let mut error = false;

    // Read types which are needed in later steps
    let mut functype_param_counts = Vec::new();
    if let Some(type_section) = module_info.raw_sections.get_mut(&SectionId::Type.into()) {
        let type_sec_reader = TypeSectionReader::new(&type_section.data, 0)?;

        for t in type_sec_reader {
            let Type::Func(ft) = t?;
            let count = ft.params().len() as u32;

            functype_param_counts.push(count);
        }
    }

    // we will need function parameter counts when adding temporary locals for dynamic
    // gas charges
    let mut func_param_counts: Vec<u32> = Vec::new();
    if let Some(func_section) = module_info
        .raw_sections
        .get_mut(&SectionId::Function.into())
    {
        let func_sec_reader = FunctionSectionReader::new(&func_section.data, 0)?;

        for type_res in func_sec_reader {
            let type_idx = type_res?;
            let params = *functype_param_counts
                .get(type_idx as usize)
                .ok_or_else(|| anyhow!("functype missing"))?;
            func_param_counts.push(params);
        }
    }

    // Updating calling addresses (all calls to function index >= `gas_func` should be incremented)
    if let Some(code_section) = module_info.raw_sections.get_mut(&SectionId::Code.into()) {
        let mut code_section_builder = wasm_encoder::CodeSection::new();
        let mut code_sec_reader = CodeSectionReader::new(&code_section.data, 0)?;

        let mut param_counts = func_param_counts.into_iter();

        // For each function
        while !code_sec_reader.eof() {
            let func_body = code_sec_reader.read()?;
            let mut func_builder = wasm_encoder::Function::new(copy_locals(&func_body)?);

            // Go through instructions, increment all global gets
            // todo this is wrong, can have imports at lower index??
            let mut operator_reader = func_body.get_operators_reader()?;
            while !operator_reader.eof() {
                let op = operator_reader.read()?;
                match op {
                    // todo if > gas global??
                    Operator::GlobalGet { global_index } => {
                        func_builder.instruction(&Instruction::GlobalGet(global_index + 1))
                    }
                    Operator::GlobalSet { global_index } => {
                        func_builder.instruction(&Instruction::GlobalSet(global_index + 1))
                    }
                    op => func_builder.instruction(&DefaultTranslator.translate_op(&op)?),
                };
            }

            let param_count = param_counts
                .next()
                .ok_or_else(|| anyhow!("out of func defs for param counts"))?;

            // Determine metered blocks and dynamically priced instructions
            // Rewrite function bodies with code block gas tracking instrumented
            // TODO: merge the second step into the loop above which is already rewriting functions
            match inject_counter(
                &FunctionBody::new(0, &truncate_len_from_encoder(&func_builder)?),
                rules,
                param_count,
                gas_func,
            ) {
                Ok(new_builder) => func_builder = new_builder,
                Err(_) => {
                    error = true;
                    break;
                }
            }

            code_section_builder.function(&func_builder);
        }
        module_info.replace_section(SectionId::Code.into(), &code_section_builder)?;
    }

    if let Some(export_section) = module_info.raw_sections.get_mut(&SectionId::Export.into()) {
        let mut export_sec_builder = ExportSection::new();
        let mut export_sec_reader = ExportSectionReader::new(&export_section.data, 0)?;
        while !export_sec_reader.eof() {
            let export = export_sec_reader.read()?;
            let mut global_index = export.index;
            if let ExternalKind::Global = export.kind {
                if global_index >= gas_global {
                    global_index += 1;
                }
            }
            export_sec_builder.export(
                export.name,
                DefaultTranslator.translate_export_kind(export.kind)?,
                global_index,
            );
        }
    }

    if let Some(import_section) = module_info.raw_sections.get_mut(&SectionId::Import.into()) {
        // Take the imports for the gasglobal
        let import_sec_reader = ImportSectionReader::new(&import_section.data, 0)?;
        let gas_globals = import_sec_reader.into_iter().filter(|r| match r {
            Ok(p) => match p.ty {
                TypeRef::Global(g) => {
                    if p.module == gas_module_name && p.name == GAS_COUNTER_NAME {
                        if !g.mutable {
                            error = true
                        }

                        true
                    } else {
                        false
                    }
                }

                _ => false,
            },
            _ => false,
        });

        // Ensure there is only one gas global import
        if gas_globals.count() != 1 {
            return Err(anyhow!("expected 1 gas global"));
        }
    }

    if let Some(ele_section) = module_info.raw_sections.get_mut(&SectionId::Element.into()) {
        let ele_sec_reader = ElementSectionReader::new(&ele_section.data, 0)?;
        for segment in ele_sec_reader {
            let element_reader = segment?.items.get_items_reader()?;
            for ele in element_reader {
                if let ElementItem::Expr(expr) = ele? {
                    let operators = expr
                        .get_operators_reader()
                        .into_iter()
                        .collect::<wasmparser::Result<Vec<Operator>>>()
                        .unwrap();
                    if !check_offset_code(&operators) {
                        error = true;
                        break;
                    }
                }
            }
        }
    }

    if let Some(data_section) = module_info.raw_sections.get_mut(&SectionId::Data.into()) {
        let data_sec_reader = DataSectionReader::new(&data_section.data, 0)?;
        for data in data_sec_reader {
            if let DataKind::Active {
                memory_index: _,
                offset_expr: expr,
            } = data?.kind
            {
                let operators = expr
                    .get_operators_reader()
                    .into_iter()
                    .collect::<wasmparser::Result<Vec<Operator>>>()
                    .unwrap();
                if !check_offset_code(&operators) {
                    error = true;
                    break;
                }
            }
        }
    }

    if error {
        return Err(anyhow!("inject fail"));
    }

    let (func_t, gas_counter_func) = generate_gas_counter(gas_global);
    module_info.add_func(func_t, &gas_counter_func)?;

    Ok(module_info.bytes())
}

fn generate_gas_counter(gas_global: u32) -> (Type, Function) {
    use wasm_encoder::Instruction::*;
    let mut func = wasm_encoder::Function::new(None);
    func.instruction(&GlobalGet(gas_global));
    func.instruction(&LocalGet(0));
    func.instruction(&I64Sub);
    func.instruction(&GlobalSet(gas_global));
    func.instruction(&GlobalGet(gas_global));
    func.instruction(&I64Const(0));
    func.instruction(&I64LtS);
    func.instruction(&If(BlockType::Empty));
    func.instruction(&Unreachable);
    func.instruction(&End);
    func.instruction(&End);
    (
        Type::Func(FuncType::new(vec![wasmparser::ValType::I64], vec![])),
        func,
    )
}

fn inject_counter<R: Rules>(
    instructions: &wasmparser::FunctionBody,
    rules: &R,
    param_count: u32,
    gas_func: u32,
) -> Result<wasm_encoder::Function> {
    let (blocks, metered_instrs) = determine_metered_blocks(instructions, rules)?;
    let charge_cost = rules.gas_charge_cost();

    insert_metering_calls(
        instructions,
        blocks,
        metered_instrs,
        param_count,
        gas_func,
        charge_cost,
    )
}

// Then insert metering calls into a sequence of instructions given the block locations and costs.
fn insert_metering_calls(
    func_body: &wasmparser::FunctionBody,
    blocks: Vec<MeteredBlock>,
    instructions: Vec<MeteredInstruction>,
    param_count: u32,
    gas_func: u32,
    charge_cost: u64,
) -> Result<wasm_encoder::Function> {
    // collect value types on which we will be doing dynamic gas math.
    // We need those for temp locals because wasm has no other way to duplicate stack items..

    // todo: if in the future when we want to do linear gas cost on instructions where
    //  the last parameter doesn't happen to always be i32 this will need to be smarter
    // note: code in determine_metered_blocks already enforces that all `instructions`
    // have i32 stack top.
    let has_i32_temp = !instructions.is_empty();

    let mut locals = copy_locals(func_body)?;
    let temp_local_idx = param_count + (&locals).iter().fold(0, |acc, (count, _)| acc + count);

    if has_i32_temp {
        locals.push((1, ValType::I32));
    }

    // To do this in linear time, construct a new vector of instructions, copying over old
    // instructions one by one and injecting new ones as required.
    let mut new_func = wasm_encoder::Function::new(locals);

    let mut block_iter = blocks.into_iter().peekable();
    let mut instr_iter = instructions.into_iter().peekable();
    let operators = func_body
        .get_operators_reader()
        .unwrap()
        .into_iter()
        .collect::<wasmparser::Result<Vec<Operator>>>()
        .unwrap();
    for (original_pos, instr) in operators.iter().enumerate() {
        // If there the next block starts at this position, inject metering func_body.
        if let Some(block) = block_iter.peek() {
            if block.start_pos == original_pos {
                new_func.instruction(&wasm_encoder::Instruction::I64Const(
                    (charge_cost + block.cost) as i64,
                ));
                new_func.instruction(&wasm_encoder::Instruction::Call(gas_func));

                block_iter.next();
            }
        }

        // if this instruction requires dynamic gas charge calculation, inject that code
        if let Some(metered_instr) = instr_iter.peek() {
            if metered_instr.pos == original_pos {
                // duplicate stack top
                // save into temp local
                new_func.instruction(&wasm_encoder::Instruction::LocalTee(temp_local_idx));

                // one copy to do math for gas charge
                new_func.instruction(&wasm_encoder::Instruction::LocalGet(temp_local_idx));

                // NOTE(negative bulk instruction arg):
                // right now this instrumentation is mostly meant for bulk memory instructions
                // In the formal spec instructions are NOT required to trap when the "count" argument
                // is negative, and if the spec is followed exactly, those instructions may take
                // very long to trap with a negative argument.
                //
                // e.g. see https://webassembly.github.io/spec/core/exec/instructions.html#xref-syntax-instructions-syntax-instr-memory-mathsf-memory-init-x
                // To guard against this we use unsigned extend instructions.
                // This means that e.g. -1_i32 becomes 0x0000_0000_ffff_ffff

                // cast to I64 if needed
                // note: today we only expect i32, so cast always needed
                new_func.instruction(&wasm_encoder::Instruction::I64ExtendI32U);

                // calculate gas charge
                new_func.instruction(&wasm_encoder::Instruction::I64Const(
                    metered_instr.unit_cost as i64,
                ));
                new_func.instruction(&wasm_encoder::Instruction::I64Mul);

                // charge gas!
                new_func.instruction(&wasm_encoder::Instruction::Call(gas_func));

                instr_iter.next();
            }
        }
        // Copy over the original instruction.
        new_func.instruction(&DefaultTranslator.translate_op(instr)?);
    }

    if block_iter.next().is_some() {
        return Err(anyhow!("metered blocks should be all consumed"));
    }
    if instr_iter.next().is_some() {
        return Err(anyhow!("metered instructions should be all consumed"));
    }

    Ok(new_func)
}

fn add_gas_global_import(module: &mut ModuleInfo, gas_module_name: &str) -> Result<()> {
    let mut import_decoder = ImportSection::new();
    if let Some(import_sec) = module.raw_sections.get_mut(&SectionId::Import.into()) {
        let import_sec_reader = ImportSectionReader::new(&import_sec.data, 0)?;
        for import in import_sec_reader {
            DefaultTranslator.translate_import(import?, &mut import_decoder)?;
        }
    }

    import_decoder.import(
        gas_module_name,
        GAS_COUNTER_NAME,
        wasm_encoder::GlobalType {
            val_type: ValType::I64,
            mutable: true,
        },
    );
    module.imported_globals_count += 1;
    module.replace_section(SectionId::Import.into(), &import_decoder)
}

fn check_offset_code(code: &[Operator]) -> bool {
    matches!(code, [Operator::I32Const { value: _ }, Operator::End])
}

fn instruction_stack_top_type(instr: &Operator<'_>) -> Result<ValType> {
    use wasmparser::Operator::*;

    match instr {
        // Note: may not trap on negative arg
        // https://webassembly.github.io/spec/core/exec/instructions.html#xref-syntax-instructions-syntax-instr-memory-mathsf-memory-grow
        MemoryGrow { .. }

        | TableGrow { .. }

        // Note: may not trap on negative arg, and/or may be very expensive
        // https://webassembly.github.io/spec/core/exec/instructions.html#xref-syntax-instructions-syntax-instr-memory-mathsf-memory-init-x
        | MemoryInit { .. }

        // Note: may not trap on negative arg
        // https://webassembly.github.io/spec/core/exec/instructions.html#xref-syntax-instructions-syntax-instr-memory-mathsf-memory-grow
        | MemoryCopy { .. }

        // Note: may not trap on negative arg, and/or may be very expensive
        // https://webassembly.github.io/spec/core/exec/instructions.html#xref-syntax-instructions-syntax-instr-memory-mathsf-memory-fill
        | MemoryFill { .. }

        | TableInit { .. }
        | TableCopy { .. }
        | TableFill { .. } => Ok(ValType::I32),

        _ => Err(anyhow!("instruction not supported")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_encoder::{Encode, Instruction::*};
    use wasmparser::FunctionBody;

    fn check_expect_function_body(
        raw_wasm: &[u8],
        index: usize,
        ops2: &[wasm_encoder::Instruction],
    ) -> bool {
        let mut body_raw = vec![];
        ops2.iter().for_each(|v| v.encode(&mut body_raw));
        get_function_body(raw_wasm, index).eq(&body_raw)
    }

    fn get_function_body(raw_wasm: &[u8], index: usize) -> Vec<u8> {
        let mut module = ModuleInfo::new(raw_wasm).unwrap();
        let func_sec = module
            .raw_sections
            .get_mut(&SectionId::Code.into())
            .unwrap();
        let func_bodies = wasmparser::CodeSectionReader::new(&func_sec.data, 0)
            .unwrap()
            .into_iter()
            .collect::<wasmparser::Result<Vec<FunctionBody>>>()
            .unwrap();

        let func_body = func_bodies
            .get(index)
            .unwrap_or_else(|| panic!("module don't have function {}body", index));

        let list = func_body
            .get_operators_reader()
            .unwrap()
            .into_iter()
            .map(|op| DefaultTranslator.translate_op(&op.unwrap()).unwrap())
            .collect::<Vec<Instruction>>();

        println!("{:?}", list);

        let start = func_body
            .get_operators_reader()
            .unwrap()
            .original_position();
        func_sec.data[start..func_body.range().end].to_vec()
    }

    fn parse_wat(source: &str) -> ModuleInfo {
        let module_bytes = wat::parse_str(source).unwrap();
        ModuleInfo::new(&module_bytes).unwrap()
    }

    #[test]
    fn gas_charge_charge() {
        pub struct TestRules {}
        impl Rules for TestRules {
            fn instruction_cost(&self, i: &Operator) -> Result<InstructionCost> {
                Ok(match i {
                    _ => InstructionCost::Fixed(1),
                })
            }

            fn gas_charge_cost(&self) -> u64 {
                13
            }
            fn linear_calc_cost(&self) -> u64 {
                99
            }
        }

        let module = parse_wat(
            r#"(module
			(func (result i32)
			  i32.const 10)
			(memory 0 1)
			)"#,
        );

        let raw_wasm = module.bytes();
        let injected_raw_wasm = inject(&raw_wasm, &TestRules {}, "env").unwrap();

        assert!(check_expect_function_body(
            &injected_raw_wasm,
            0,
            &[
                I64Const(14), // 1 + 13
                Call(1),      // gas charge
                I32Const(10),
                End,
            ]
        ));

        wasmparser::validate(&injected_raw_wasm).unwrap();
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

        let raw_wasm = module.bytes();
        let injected_raw_wasm =
            inject(&raw_wasm, &ConstantCostRules::new(1, 10_000), "env").unwrap();

        // global0 - gas
        // global1 - orig global0
        // func0 - main
        // func1 - gas_counter
        assert!(check_expect_function_body(
            &injected_raw_wasm,
            0,
            &[
                I64Const(2),
                Call(1),      // gas charge
                GlobalGet(1), // original code
                // <dynamic charge>
                LocalTee(0),
                LocalGet(0),
                I64ExtendI32U,
                I64Const(10000),
                I64Mul,
                Call(1),
                // </dynamic charge>
                MemoryGrow(0), // original code
                End,
            ]
        ));

        wasmparser::validate(&injected_raw_wasm).unwrap();
    }

    #[test]
    fn gas_charge_charge_const_linear() {
        pub struct TestRules {}
        impl Rules for TestRules {
            fn instruction_cost(&self, i: &Operator) -> Result<InstructionCost> {
                Ok(match i {
                    Operator::MemoryGrow { .. } => {
                        InstructionCost::Linear(17, NonZeroU32::new(7).unwrap())
                    }
                    _ => InstructionCost::Fixed(3),
                })
            }

            fn gas_charge_cost(&self) -> u64 {
                13
            }
            fn linear_calc_cost(&self) -> u64 {
                5
            }
        }

        let module = parse_wat(
            r#"(module
			(func (result i32)
			  i32.const 10
			  memory.grow 0)
			(memory 0 1)
			)"#,
        );

        let raw_wasm = module.bytes();
        let injected_raw_wasm = inject(&raw_wasm, &TestRules {}, "env").unwrap();

        assert!(check_expect_function_body(
            &injected_raw_wasm,
            0,
            &[
                I64Const(103), // 3*1 + 17 + 10*7 + 13*1
                Call(1),       // gas charge
                I32Const(10),
                MemoryGrow(0),
                End,
            ]
        ));

        wasmparser::validate(&injected_raw_wasm).unwrap();
    }

    #[test]
    fn gas_charge_charge_dyn_linear() {
        pub struct TestRules {}
        impl Rules for TestRules {
            fn instruction_cost(&self, i: &Operator) -> Result<InstructionCost> {
                Ok(match i {
                    Operator::MemoryGrow { .. } => {
                        InstructionCost::Linear(17, NonZeroU32::new(7).unwrap())
                    }
                    _ => InstructionCost::Fixed(3),
                })
            }

            fn gas_charge_cost(&self) -> u64 {
                13
            }
            fn linear_calc_cost(&self) -> u64 {
                5
            }
        }

        let module = parse_wat(
            r#"(module
			(func (result i32)
			  i32.const 10
			  i32.const 1
			  i32.mul
			  memory.grow 0)
			(memory 0 1)
			)"#,
        );

        let raw_wasm = module.bytes();
        let injected_raw_wasm = inject(&raw_wasm, &TestRules {}, "env").unwrap();

        assert!(check_expect_function_body(
            &injected_raw_wasm,
            0,
            &[
                I64Const(57), // 3*3 + 17 + 13*2 + 5
                Call(1),      // gas charge
                I32Const(10),
                I32Const(1),
                I32Mul,
                LocalTee(0),
                LocalGet(0),
                LocalGet(0),
                I32Const(0),
                I32LtS,
                If(BlockType::Empty),
                Unreachable,
                End,
                I64ExtendI32U,
                I64Const(7),
                I64Mul,
                Call(1),
                MemoryGrow(0),
                End,
            ]
        ));

        wasmparser::validate(&injected_raw_wasm).unwrap();
    }

    #[test]
    fn grow_const() {
        let module = parse_wat(
            r#"(module
            (func (result i32)
              i32.const 10
              memory.grow)
            (memory 0 1)
            )"#,
        );

        let raw_wasm = module.bytes();
        let injected_raw_wasm =
            inject(&raw_wasm, &ConstantCostRules::new(1, 10_000), "env").unwrap();

        // global0 - gas
        // global1 - orig global0
        // func0 - main
        // func1 - gas_counter
        assert!(check_expect_function_body(
            &injected_raw_wasm,
            0,
            &[
                I64Const(100002),
                Call(1), // gas charge
                I32Const(10),
                MemoryGrow(0),
                End,
            ]
        ));

        wasmparser::validate(&injected_raw_wasm).unwrap();
    }

    #[test]
    fn grow_const_neg_fail() {
        let module = parse_wat(
            r#"(module
            (func (result i32)
              i32.const -10
              memory.grow)
            (memory 0 1)
            )"#,
        );

        let raw_wasm = module.bytes();
        wasmparser::validate(&raw_wasm).unwrap();

        assert!(inject(&raw_wasm, &ConstantCostRules::new(1, 10_000), "env").is_err())
    }

    #[test]
    fn simple_grow_two() {
        // this test checks dynamic counter for instructions with different const param

        pub struct TestRules {}
        impl Rules for TestRules {
            fn instruction_cost(&self, i: &Operator) -> Result<InstructionCost> {
                Ok(match i {
                    Operator::MemoryGrow { .. } => {
                        InstructionCost::Linear(1, NonZeroU32::new(10).unwrap())
                    }
                    Operator::MemoryInit { .. } => {
                        InstructionCost::Linear(3, NonZeroU32::new(12).unwrap())
                    }
                    _ => InstructionCost::Fixed(1),
                })
            }

            fn gas_charge_cost(&self) -> u64 {
                0
            }
            fn linear_calc_cost(&self) -> u64 {
                0
            }
        }

        let module = parse_wat(
            r#"(module
            (global i32 (i32.const 42))
            (memory 0 1)
            (func (param i32) (result i32)
              local.get 0
              global.get 0
              i32.mul
              memory.grow
              (memory.init 1
                  (i32.const 16)
                  (i32.const 0)
                  (i32.mul (i32.const 7) (i32.const 1)))
              (memory.init 0
                  (i32.const 8)
                  (i32.const 0)
                  (i32.mul (i32.const 2) (i32.const 1))))
            (data "gm")
            (data "goodbye"))"#,
        );

        let raw_wasm = module.bytes();
        wasmparser::validate(&raw_wasm).unwrap();

        let injected_raw_wasm = inject(&raw_wasm, &TestRules {}, "env").unwrap();

        // global0 - gas
        // global1 - orig global0
        // func0 - main
        // func1 - gas_counter

        assert!(check_expect_function_body(
            &injected_raw_wasm,
            0,
            &[
                I64Const(20),
                Call(1),
                LocalGet(0),
                GlobalGet(1),
                I32Mul,
                LocalTee(1),
                LocalGet(1),
                I64ExtendI32U,
                I64Const(10),
                I64Mul,
                Call(1),
                MemoryGrow(0),
                I32Const(16),
                I32Const(0),
                I32Const(7),
                I32Const(1),
                I32Mul,
                LocalTee(1),
                LocalGet(1),
                I64ExtendI32U,
                I64Const(12),
                I64Mul,
                Call(1),
                MemoryInit { mem: 0, data: 1 },
                I32Const(8),
                I32Const(0),
                I32Const(2),
                I32Const(1),
                I32Mul,
                LocalTee(1),
                LocalGet(1),
                I64ExtendI32U,
                I64Const(12),
                I64Mul,
                Call(1),
                MemoryInit { mem: 0, data: 0 },
                End
            ]
        ));

        wasmparser::validate(&injected_raw_wasm).unwrap();
    }

    #[test]
    fn grow_no_gas_no_track() {
        let raw_wasm = parse_wat(
            r"(module
            (func (result i32)
              global.get 0
              memory.grow)
            (global i32 (i32.const 42))
            (memory 0 1)
            )",
        )
        .bytes();
        let injected_raw_wasm = inject(&raw_wasm, &ConstantCostRules::default(), "env").unwrap();

        assert!(check_expect_function_body(
            &injected_raw_wasm,
            0,
            &[I64Const(2), Call(1), GlobalGet(1), MemoryGrow(0), End,]
        ));

        let injected_module = ModuleInfo::new(&injected_raw_wasm).unwrap();
        assert_eq!(injected_module.num_functions(), 2);
        wasmparser::validate(&injected_raw_wasm).unwrap();
    }

    #[test]
    fn call_index() {
        let raw_wasm = parse_wat(
            r"(module
                  (type (;0;) (func (result i32)))
                  (func (;0;) (type 0) (result i32))
                  (func (;1;) (type 0) (result i32)
                    call 0
                    if  ;; label = @1
                      call 0
                      call 0
                      call 0
                    else
                      call 0
                      call 0
                    end
                    call 0
                  )
                  (global (;0;) i32 )
                )",
        )
        .bytes();
        let injected_raw_wasm = inject(&raw_wasm, &ConstantCostRules::default(), "env").unwrap();

        assert!(check_expect_function_body(
            &injected_raw_wasm,
            1,
            &vec![
                I64Const(3),
                Call(2),
                Call(0),
                If(BlockType::Empty),
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
            ]
        ));
    }

    macro_rules! test_gas_counter_injection {
        (name = $name:ident; input = $input:expr; expected = $expected:expr) => {
            #[test]
            fn $name() {
                let input_wasm = parse_wat($input).bytes();
                let expected_wasm = parse_wat($expected).bytes();

                let injected_wasm = inject(&input_wasm, &ConstantCostRules::default(), "env")
                    .expect("inject_gas_counter call failed");

                let actual_func_body = get_function_body(&injected_wasm, 0);

                let expected_func_body = get_function_body(&expected_wasm, 0);

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
        let raw_wasm = parse_wat(input).bytes();
        let injected_raw_wasm = inject(&raw_wasm, &ConstantCostRules::default(), "other")
            .expect("inject_gas_counter call failed");
        wasmparser::validate(&injected_raw_wasm).unwrap();

        assert!(check_expect_function_body(
            &injected_raw_wasm,
            0,
            &[
                I64Const(11),
                Call(5),
                LocalGet(0),
                I32Const(64),
                LocalGet(1),
                I32Sub,
                I64ExtendI32U,
                I64Shl,
                LocalGet(0),
                LocalGet(1),
                I64ExtendI32U,
                I64ShrU,
                I64Or,
                End,
            ]
        ));

        // 1 is gas counter
        assert!(check_expect_function_body(
            &injected_raw_wasm,
            1,
            &[
                GlobalGet(2), // 2 imported globals, so gas one is third
                LocalGet(0),
                I64Sub,
                GlobalSet(2),
                GlobalGet(2),
                I64Const(0),
                I64LtS,
                If(BlockType::Empty),
                Unreachable,
                End,
                End
            ]
        ));
    }

    #[test]
    fn test_user_gas_global_fails() {
        let input = r#"
        (module
            (type (;0;) (func (param i64 i32) (result i64)))
            (import "other" "gas_counter" (global (;0;) i64))
            (func (;0;) (type 0) (param i64 i32) (result i64)
              local.get 0
              local.get 1
              i64.or
            )
          )
        "#;
        let raw_wasm = parse_wat(input).bytes();
        let estr = inject(&raw_wasm, &ConstantCostRules::default(), "other")
            .unwrap_err()
            .to_string();
        assert!(estr == "expected 1 gas global", "error was {}", estr);
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

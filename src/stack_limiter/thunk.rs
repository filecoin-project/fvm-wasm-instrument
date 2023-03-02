use super::Context;
use crate::utils::{
    translator::{ConstExprKind, DefaultTranslator, Translator},
    ModuleInfo,
};
#[cfg(not(features = "std"))]
use alloc::collections::BTreeMap as Map;
use alloc::vec::Vec;
use anyhow::{anyhow, Result};
#[cfg(features = "std")]
use std::collections::HashMap as Map;
use wasm_encoder::{
    CodeSection, ElementMode, ElementSection, Elements, ExportSection, FunctionSection, SectionId,
};
use wasmparser::{
    CodeSectionReader, ElementItems, ElementKind, ElementSectionReader, ExportSectionReader,
    ExternalKind, FunctionSectionReader, Type,
};

struct Thunk {
    signature: wasmparser::FuncType,
    // Index in function space of this thunk.
    idx: Option<u32>,
    callee_stack_cost: u32,
}

pub(super) fn generate_thunks(ctx: &mut Context, module: &mut ModuleInfo) -> Result<()> {
    // First, we need to collect all function indices that should be replaced by thunks
    let exports = match module.raw_sections.get(&SectionId::Export.into()) {
        Some(raw_sec) => ExportSectionReader::new(&raw_sec.data, 0)?
            .into_iter()
            .collect::<wasmparser::Result<Vec<wasmparser::Export>>>()?,
        None => vec![],
    };

    //element maybe null
    let elem_segments = match module.raw_sections.get(&SectionId::Element.into()) {
        Some(v) => ElementSectionReader::new(&v.data, 0)?
            .into_iter()
            .collect::<wasmparser::Result<Vec<wasmparser::Element>>>()?,
        None => vec![],
    };

    let mut replacement_map: Map<u32, Thunk> = {
        let exported_func_indices = exports.iter().filter_map(|entry| match entry.kind {
            ExternalKind::Func => Some(entry.index),
            _ => None,
        });

        let mut table_func_indices = vec![];
        for segment in elem_segments.clone() {
            match segment.items {
                ElementItems::Functions(segment) => {
                    for func_idx in segment {
                        table_func_indices.push(func_idx?);
                    }
                }
                ElementItems::Expressions(_) => {}
            }
        }

        // Replacement map is at least export section size.
        let mut replacement_map: Map<u32, Thunk> = Map::new();

        for func_idx in exported_func_indices
            .chain(table_func_indices)
            .chain(module.start_function.into_iter())
        {
            let callee_stack_cost = ctx
                .stack_cost(func_idx)
                .ok_or_else(|| anyhow!("function index isn't found"))?;

            // Don't generate a thunk if stack_cost of a callee is zero.
            if callee_stack_cost != 0 {
                replacement_map.insert(
                    func_idx,
                    Thunk {
                        signature: match module.get_functype_idx(func_idx)?.clone() {
                            Type::Func(ft) => ft,
                        },
                        idx: None,
                        callee_stack_cost,
                    },
                );
            }
        }

        replacement_map
    };

    // Then, we generate a thunk for each original function.

    // Save current func_idx
    let mut func_body_sec_builder = CodeSection::new();
    let func_body_sec_data = &module
        .raw_sections
        .get(&SectionId::Code.into())
        .ok_or_else(|| anyhow!("no function body"))?
        .data;

    let code_sec_reader = CodeSectionReader::new(func_body_sec_data, 0)?;
    for func_body in code_sec_reader {
        DefaultTranslator.translate_code(func_body?, &mut func_body_sec_builder)?;
    }

    let mut func_sec_builder = FunctionSection::new();
    let func_sec_data = &module
        .raw_sections
        .get(&SectionId::Function.into())
        .ok_or_else(|| anyhow!("no function section"))? //todo allow empty function file?
        .data;
    for func_body in FunctionSectionReader::new(func_sec_data, 0)? {
        func_sec_builder.function(func_body?);
    }

    let mut next_func_idx = module.function_map.len() as u32;
    for (func_idx, thunk) in replacement_map.iter_mut() {
        // Thunk body consist of:
        //  - argument pushing
        //  - instrumented call
        //  - end
        let mut thunk_body = wasm_encoder::Function::new(None);

        for (arg_idx, _) in thunk.signature.params().iter().enumerate() {
            thunk_body.instruction(&wasm_encoder::Instruction::LocalGet(arg_idx as u32));
        }

        instrument_call!(
            *func_idx,
            thunk.callee_stack_cost as i32,
            ctx.stack_height_global_idx(),
            ctx.stack_limit()
        )
        .iter()
        .for_each(|v| {
            thunk_body.instruction(v);
        });
        thunk_body.instruction(&wasm_encoder::Instruction::End);

        let func_type = module
            .resolve_type_idx(&Type::Func(thunk.signature.clone()))
            .ok_or_else(|| anyhow!("signature not exit"))?; //resolve thunk func type, this signature should exit
        func_sec_builder.function(func_type); //add thunk function
        func_body_sec_builder.function(&thunk_body); //add thunk body

        thunk.idx = Some(next_func_idx);
        next_func_idx += 1;
    }

    // And finally, fixup thunks in export and table sections.

    // Fixup original function index to a index of a thunk generated earlier.
    let mut export_sec_builder = ExportSection::new();
    for export in exports {
        let mut function_idx = export.index;
        if let ExternalKind::Func = export.kind {
            if let Some(thunk) = replacement_map.get(&function_idx) {
                function_idx = thunk
                    .idx
                    .expect("at this point an index must be assigned to each thunk");
            }
        }
        export_sec_builder.export(
            export.name,
            DefaultTranslator.translate_export_kind(export.kind)?,
            function_idx,
        );
    }

    let mut ele_sec_builder = ElementSection::new();
    for ele in elem_segments {
        match ele.items {
            ElementItems::Functions(section) => {
                let mut functions = Vec::new();
                for func_idx in section {
                    let mut func_idx: u32 = func_idx?;
                    if let Some(thunk) = replacement_map.get(&func_idx) {
                        func_idx = thunk
                            .idx
                            .expect("at this point an index must be assigned to each thunk");
                    }

                    functions.push(func_idx);
                }

                //todo edit element is little complex,
                let offset;
                let mode = match ele.kind {
                    ElementKind::Active {
                        table_index,
                        offset_expr,
                    } => {
                        offset = DefaultTranslator
                            .translate_const_expr(&offset_expr, ConstExprKind::ElementOffset)?;
                        ElementMode::Active {
                            table: Some(table_index),
                            offset: &offset,
                        }
                    }
                    ElementKind::Passive => ElementMode::Passive,
                    ElementKind::Declared => ElementMode::Declared,
                };

                ele_sec_builder.segment(wasm_encoder::ElementSegment {
                    mode,
                    /// The element segment's type.
                    element_type: DefaultTranslator.translate_ref_ty(&ele.ty)?,
                    /// This segment's elements.
                    elements: Elements::Functions(&functions),
                });
            }
            ElementItems::Expressions(_) => {
                DefaultTranslator.translate_element(ele, &mut ele_sec_builder)?;
            }
        }
    }

    module.replace_section(SectionId::Function.into(), &func_sec_builder)?;
    module.replace_section(SectionId::Code.into(), &func_body_sec_builder)?;
    module.replace_section(SectionId::Export.into(), &export_sec_builder)?;
    module.replace_section(SectionId::Element.into(), &ele_sec_builder)?;
    if let Some(start_idx) = module.start_function {
        let mut new_func_idx = start_idx;
        if let Some(thunk) = replacement_map.get(&start_idx) {
            new_func_idx = thunk
                .idx
                .expect("at this point an index must be assigned to each thunk");
        }

        module.replace_section(
            SectionId::Start.into(),
            &wasm_encoder::StartSection {
                function_index: new_func_idx,
            },
        )?;
    }
    Ok(())
}

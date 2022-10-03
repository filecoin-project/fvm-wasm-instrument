use crate::utils::{
	translator::{DefaultTranslator, Translator},
	ModuleInfo,
};
use alloc::{format, vec::Vec};
use anyhow::Result;
use wasm_encoder::{ExportKind, ExportSection, SectionId};
use wasmparser::{ExportSectionReader, Global};

/// Export all declared mutable globals as `prefix_index`.
///
/// This will export all internal mutable globals under the name of
/// concat(`prefix`, `"_"`, `i`) where i is the index inside the range of
/// [0..total number of internal mutable globals].
pub fn export_mutable_globals(module: &[u8], prefix: &str) -> Result<Vec<u8>> {
	let mut module = ModuleInfo::new(module)?;
	let globals = match module.raw_sections.get(&SectionId::Global.into()) {
		Some(global_sec) => wasmparser::GlobalSectionReader::new(&global_sec.data, 0)?
			.into_iter()
			.collect::<wasmparser::Result<Vec<Global>>>()?,
		None => vec![],
	};

	let mut export_sec_builder = ExportSection::new();
	if let Some(export_sec) = module.raw_sections.get(&SectionId::Export.into()) {
		for export in ExportSectionReader::new(&export_sec.data, 0)?.into_iter() {
			DefaultTranslator.translate_export(&export?, &mut export_sec_builder)?;
		}
	}

	let mutable_global_idx =
		globals
			.iter()
			.enumerate()
			.filter_map(|(index, global)| if global.ty.mutable { Some(index) } else { None });
	for (symbol_index, export) in mutable_global_idx.into_iter().enumerate() {
		export_sec_builder.export(
			format!("{}_{}", prefix, symbol_index).as_str(),
			ExportKind::Global,
			module.num_imported_globals() + export as u32,
		);
	}
	module.replace_section(SectionId::Export.into(), &export_sec_builder)?;
	Ok(module.bytes())
}

#[cfg(test)]
mod tests {

	use super::export_mutable_globals;
	use crate::utils::ModuleInfo;

	fn parse_wat(source: &str) -> ModuleInfo {
		let module_bytes = wat::parse_str(source).unwrap();
		ModuleInfo::new(&module_bytes).unwrap()
	}

	macro_rules! test_export_global {
		(name = $name:ident; input = $input:expr; expected = $expected:expr) => {
			#[test]
			fn $name() {
				let input_wasm = parse_wat($input).bytes();
				let expected_bytes = parse_wat($expected).bytes();

				let actual_bytes = export_mutable_globals(&input_wasm, "exported_internal_global")
					.expect("injected module must have a function body");

				let actual_wat = wasmprinter::print_bytes(actual_bytes).unwrap();
				let expected_wat = wasmprinter::print_bytes(expected_bytes).unwrap();

				if actual_wat != expected_wat {
					for diff in diff::lines(&expected_wat, &actual_wat) {
						match diff {
							diff::Result::Left(l) => println!("-{}", l),
							diff::Result::Both(l, _) => println!(" {}", l),
							diff::Result::Right(r) => println!("+{}", r),
						}
					}
					panic!()
				}
			}
		};
	}

	test_export_global! {
		name = simple;
		input = r#"
		(module
			(global (;0;) (mut i32) (i32.const 1))
			(global (;1;) (mut i32) (i32.const 0)))
		"#;
		expected = r#"
		(module
			(global (;0;) (mut i32) (i32.const 1))
			(global (;1;) (mut i32) (i32.const 0))
			(export "exported_internal_global_0" (global 0))
			(export "exported_internal_global_1" (global 1)))
		"#
	}

	test_export_global! {
		name = with_import;
		input = r#"
		(module
			(import "env" "global" (global $global i64))
			(global (;0;) (mut i32) (i32.const 1))
			(global (;1;) (mut i32) (i32.const 0)))
		"#;
		expected = r#"
		(module
			(import "env" "global" (global $global i64))
			(global (;0;) (mut i32) (i32.const 1))
			(global (;1;) (mut i32) (i32.const 0))
			(export "exported_internal_global_0" (global 1))
			(export "exported_internal_global_1" (global 2)))
		"#
	}

	test_export_global! {
		name = with_import_and_some_are_immutable;
		input = r#"
		(module
			(import "env" "global" (global $global i64))
			(global (;0;) i32 (i32.const 1))
			(global (;1;) (mut i32) (i32.const 0)))
		"#;
		expected = r#"
		(module
			(import "env" "global" (global $global i64))
			(global (;0;) i32 (i32.const 1))
			(global (;1;) (mut i32) (i32.const 0))
			(export "exported_internal_global_0" (global 2)))
		"#
	}
}

pub mod translator;
use crate::utils::translator::{DefaultTranslator, Translator};
use anyhow::{anyhow, Result};
use std::{
	collections::{BTreeMap, HashSet},
	ops::Range,
};

use wasm_encoder::{Encode, SectionId};
use wasmparser::{
	Chunk, ExternalKind, GlobalType, MemoryType, Parser, Payload, SectionReader, TableType, Type,
};

#[derive(Clone, Debug)]
pub struct RawSection {
	/// The id for this section.
	pub id: u8,
	/// The raw data for this section.
	pub data: Vec<u8>,
}

impl RawSection {
	pub fn new(id: u8, data: Vec<u8>) -> Self {
		RawSection { id, data }
	}
}

impl Encode for RawSection {
	fn encode(&self, sink: &mut Vec<u8>) {
		self.data.encode(sink);
	}
}

impl wasm_encoder::Section for RawSection {
	fn id(&self) -> u8 {
		self.id
	}
}

impl wasm_encoder::ComponentSection for RawSection {
	fn id(&self) -> u8 {
		self.id
	}
}

/// Provides module information for future usage during mutation
/// an instance of ModuleInfo could be user to determine which mutation could be applied
#[derive(Default, Clone, Debug)]
#[warn(dead_code)]
pub struct ModuleInfo {
	// The following fields are offsets inside the `raw_sections` field.
	// The main idea is to maintain the order of the sections in the input Wasm.
	pub export_names: HashSet<String>,

	pub exports_count: u32,
	pub exports_global_count: u32,

	pub elements_count: u32,
	pub data_segments_count: u32,
	pub start_function: Option<u32>,
	pub memory_count: u32,
	pub table_count: u32,
	pub tag_count: u32,

	pub imported_functions_count: u32,
	pub imported_globals_count: u32,
	pub imported_memories_count: u32,
	pub imported_tables_count: u32,
	pub imported_tags_count: u32,

	// types for inner functions
	pub types_map: Vec<Type>,

	// function idx to type idx
	pub function_map: Vec<u32>,
	pub global_types: Vec<GlobalType>,
	pub table_elem_types: Vec<TableType>,
	pub memory_types: Vec<MemoryType>,

	// raw_sections
	pub raw_sections: BTreeMap<u8, RawSection>,
}

impl ModuleInfo {
	/// Parse the given Wasm bytes and fill out a `ModuleInfo` AST for it.
	pub fn new(input_wasm: &[u8]) -> Result<ModuleInfo> {
		let mut parser = Parser::new(0);
		let mut info = ModuleInfo::default();
		let mut wasm = input_wasm;

		loop {
			let (payload, consumed) = match parser.parse(wasm, true)? {
				Chunk::NeedMoreData(hint) => {
					panic!("Invalid Wasm module {:?}", hint);
				},
				Chunk::Parsed { consumed, payload } => (payload, consumed),
			};
			match payload {
				Payload::CodeSectionStart { count: _, range, size: _ } => {
					info.section(SectionId::Code.into(), range.clone(), input_wasm);
					parser.skip_section();
					// update slice, bypass the section
					wasm = &input_wasm[range.end..];

					continue
				},
				Payload::TypeSection(mut reader) => {
					info.section(SectionId::Type.into(), reader.range(), input_wasm);

					// Save function types
					for _ in 0..reader.get_count() {
						reader.read().map(|ty| {
							info.types_map.push(ty);
						})?;
					}
				},
				Payload::ImportSection(mut reader) => {
					info.section(SectionId::Import.into(), reader.range(), input_wasm);

					for _ in 0..reader.get_count() {
						let ty = reader.read()?;
						match ty.ty {
							wasmparser::TypeRef::Func(ty) => {
								// Save imported functions
								info.function_map.push(ty);
								info.imported_functions_count += 1;
							},
							wasmparser::TypeRef::Global(ty) => {
								info.global_types.push(ty);
								info.imported_globals_count += 1;
							},
							wasmparser::TypeRef::Memory(ty) => {
								info.memory_count += 1;
								info.imported_memories_count += 1;
								info.memory_types.push(ty);
							},
							wasmparser::TypeRef::Table(ty) => {
								info.table_count += 1;
								info.imported_tables_count += 1;
								info.table_elem_types.push(ty);
							},
							wasmparser::TypeRef::Tag(_ty) => {
								info.tag_count += 1;
								info.imported_tags_count += 1;
							},
						}
					}
				},
				Payload::FunctionSection(mut reader) => {
					info.section(SectionId::Function.into(), reader.range(), input_wasm);

					for _ in 0..reader.get_count() {
						reader.read().map(|ty| {
							info.function_map.push(ty);
						})?;
					}
				},
				Payload::TableSection(mut reader) => {
					info.table_count += reader.get_count();
					info.section(SectionId::Table.into(), reader.range(), input_wasm);

					for _ in 0..reader.get_count() {
						let ty = reader.read()?;
						info.table_elem_types.push(ty);
					}
				},
				Payload::MemorySection(mut reader) => {
					info.memory_count += reader.get_count();
					info.section(SectionId::Memory.into(), reader.range(), input_wasm);

					for _ in 0..reader.get_count() {
						let ty = reader.read()?;
						info.memory_types.push(ty);
					}
				},
				Payload::GlobalSection(mut reader) => {
					info.section(SectionId::Global.into(), reader.range(), input_wasm);

					for _ in 0..reader.get_count() {
						info.global_types.push(reader.read()?.ty);
					}
				},
				Payload::ExportSection(mut reader) => {
					info.exports_count = reader.get_count();

					for _ in 0..reader.get_count() {
						let entry = reader.read()?;
						if let ExternalKind::Global = entry.kind {
							info.exports_global_count += 1;
						}
						info.export_names.insert(entry.name.into());
					}

					info.section(SectionId::Export.into(), reader.range(), input_wasm);
				},
				Payload::StartSection { func, range } => {
					info.start_function = Some(func);
					info.section(SectionId::Start.into(), range, input_wasm);
				},
				Payload::ElementSection(reader) => {
					info.elements_count = reader.get_count();
					info.section(SectionId::Element.into(), reader.range(), input_wasm);
				},
				Payload::DataSection(reader) => {
					info.data_segments_count = reader.get_count();
					info.section(SectionId::Data.into(), reader.range(), input_wasm);
				},
				Payload::CustomSection(c) => {
					info.section(SectionId::Custom.into(), c.range(), input_wasm);
				},
				Payload::UnknownSection { id, contents: _, range } => {
					info.section(id, range, input_wasm);
				},
				Payload::DataCountSection { count: _, range } => {
					info.section(SectionId::DataCount.into(), range, input_wasm);
				},
				Payload::Version { .. } => {},
				Payload::End(_) => break,
				_ => todo!("{:?} not implemented", payload),
			}
			wasm = &wasm[consumed..];
		}

		Ok(info)
	}

	/// Registers a new raw_section in the ModuleInfo
	pub fn section(&mut self, id: u8, range: Range<usize>, full_wasm: &[u8]) {
		self.raw_sections.insert(id, RawSection::new(id, full_wasm[range].to_vec()));
	}

	/// Returns the function type based on the index of the function type
	/// `types[functions[idx]]`
	pub fn get_functype_idx(&self, idx: u32) -> Result<&Type> {
		if idx >= self.function_map.len() as u32 {
			return Err(anyhow!("function {} not exit", idx))
		}
		let functpeindex = self.function_map[idx as usize] as usize;
		if functpeindex >= self.types_map.len() {
			return Err(anyhow!("type {} not exit", functpeindex))
		}
		Ok(&self.types_map[functpeindex])
	}

	pub fn resolve_type_idx(&self, t: &Type) -> Option<u32> {
		for (index, ty) in self.types_map.iter().enumerate() {
			let Type::Func(ot) = ty;
			let Type::Func(dt) = t;
			if ot.eq(dt) {
				return Some(index as u32)
			}
		}
		None
	}

	pub fn add_func_type(&mut self, func_type: &Type) -> Result<u32> {
		let func_type_index = match self.resolve_type_idx(func_type) {
			None => self.types_map.len() as u32,
			Some(index) => return Ok(index),
		};
		//add new type
		let mut type_builder = wasm_encoder::TypeSection::new();
		for t in &self.types_map {
			DefaultTranslator.translate_type_def(t.clone(), &mut type_builder)?;
		}
		self.types_map.push(func_type.clone());
		DefaultTranslator.translate_type_def(func_type.clone(), &mut type_builder)?;
		self.replace_section(SectionId::Type.into(), &type_builder)?;
		Ok(func_type_index)
	}

	/// Returns the number of globals used by the Wasm binary including imported
	/// glboals
	#[allow(dead_code)]
	pub fn get_global_count(&self) -> usize {
		self.global_types.len()
	}

	/// Replace the `i`th section in this module with the given new section.
	pub fn replace_section(
		&mut self,
		sec_type: u8,
		new_section: &impl wasm_encoder::Section,
	) -> Result<()> {
		self.raw_sections
			.insert(sec_type, RawSection::new(sec_type, truncate_len_from_encoder(new_section)?));
		Ok(())
	}

	pub fn add_func(&mut self, func_type: Type, func_body: &wasm_encoder::Function) -> Result<()> {
		let func_type_index = self.add_func_type(&func_type)?;

		let mut func_sec_builder = wasm_encoder::FunctionSection::new();
		let func_sec_reader = wasmparser::FunctionSectionReader::new(
			&self
				.raw_sections
				.get(&SectionId::Function.into())
				.ok_or_else(|| anyhow!("code not exit"))?
				.data,
			0,
		)?;
		for func in func_sec_reader {
			func_sec_builder.function(func?);
		}
		self.function_map.push(func_type_index);
		func_sec_builder.function(func_type_index);
		self.replace_section(SectionId::Function.into(), &func_sec_builder)?;

		let mut code_sec_builder = wasm_encoder::CodeSection::new();
		let code_sec_reader = wasmparser::CodeSectionReader::new(
			&self
				.raw_sections
				.get(&SectionId::Code.into())
				.ok_or_else(|| anyhow!("code not exit"))?
				.data,
			0,
		)?;
		for code in code_sec_reader {
			DefaultTranslator.translate_code(code?, &mut code_sec_builder)?
		}
		code_sec_builder.function(func_body);
		self.replace_section(SectionId::Code.into(), &code_sec_builder)
	}

	pub fn bytes(&self) -> Vec<u8> {
		let mut module = wasm_encoder::Module::new();
		for s in self.raw_sections.values() {
			module.section(s);
		}
		module.finish()
	}

	#[allow(dead_code)]
	pub fn num_functions(&self) -> u32 {
		self.function_map.len() as u32
	}

	#[allow(dead_code)]
	pub fn num_local_functions(&self) -> u32 {
		self.num_functions() - self.num_imported_functions()
	}

	#[allow(dead_code)]
	pub fn num_imported_functions(&self) -> u32 {
		self.imported_functions_count
	}

	#[allow(dead_code)]
	pub fn num_tables(&self) -> u32 {
		self.table_count
	}

	#[allow(dead_code)]
	pub fn num_imported_tables(&self) -> u32 {
		self.imported_tables_count
	}

	#[allow(dead_code)]
	pub fn num_memories(&self) -> u32 {
		self.memory_count
	}

	#[allow(dead_code)]
	pub fn num_imported_memories(&self) -> u32 {
		self.imported_memories_count
	}

	#[allow(dead_code)]
	pub fn num_globals(&self) -> u32 {
		self.global_types.len() as u32
	}

	#[allow(dead_code)]
	pub fn num_imported_globals(&self) -> u32 {
		self.imported_globals_count
	}

	#[allow(dead_code)]
	pub fn num_local_globals(&self) -> u32 {
		self.global_types.len() as u32 - self.imported_globals_count
	}

	#[allow(dead_code)]
	pub fn num_tags(&self) -> u32 {
		self.tag_count
	}

	#[allow(dead_code)]
	pub fn num_imported_tags(&self) -> u32 {
		self.imported_tags_count
	}

	#[allow(dead_code)]
	pub fn num_data(&self) -> u32 {
		self.data_segments_count
	}

	#[allow(dead_code)]
	pub fn num_elements(&self) -> u32 {
		self.elements_count
	}

	#[allow(dead_code)]
	pub fn num_types(&self) -> u32 {
		self.types_map.len() as u32
	}

	#[allow(dead_code)]
	pub fn num_export_global(&self) -> u32 {
		self.exports_global_count
	}
}

// Then insert metering calls into a sequence of instructions given the block locations and costs.
pub fn copy_locals(
	func_body: &wasmparser::FunctionBody,
) -> Result<Vec<(u32, wasm_encoder::ValType)>> {
	let mut local_reader = func_body.get_locals_reader()?;
	// Get current locals and map to encoder types
	let current_locals: Vec<(u32, wasm_encoder::ValType)> = (0..local_reader.get_count())
		.map(|_| {
			let (count, ty) = local_reader.read().unwrap();
			(count, DefaultTranslator.translate_ty(&ty).unwrap())
		})
		.collect::<Vec<(u32, wasm_encoder::ValType)>>();

	Ok(current_locals)
}

//todo unable to get function encoder body directly, remove this after option wasmparser
pub fn truncate_len_from_encoder(func_builder: &dyn wasm_encoder::Encode) -> Result<Vec<u8>> {
	let mut d = vec![];
	func_builder.encode(&mut d);
	let mut r = wasmparser::BinaryReader::new(&d);
	let size = r.read_var_u32()?;
	Ok(r.read_bytes(size as usize)?.to_vec())
}

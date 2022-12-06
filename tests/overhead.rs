use fvm_wasm_instrument::{gas_metering, inject_stack_limiter};
use std::{
	fs::{read, read_dir},
	path::PathBuf,
};

fn fixture_dir() -> PathBuf {
	let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
	path.push("benches");
	path.push("fixtures");
	path
}

/// Print the overhead of applying gas metering, stack height limiting or both.
///
/// Use `cargo test print_overhead -- --nocapture`.
#[test]
fn print_size_overhead() {
	let mut results: Vec<_> = read_dir(fixture_dir())
		.unwrap()
		.map(|entry| {
			let entry = entry.unwrap();
			let (orig_len, orig_module) = {
				let bytes = read(&entry.path()).unwrap();
				(bytes.len(), bytes)
			};
			let (gas_metering_len, gas_module) = {
				let module = gas_metering::inject(
					&orig_module,
					&gas_metering::ConstantCostRules::default(),
					"env",
				)
				.unwrap();
				(module.len(), module)
			};
			let stack_height_len = inject_stack_limiter(&orig_module, 128).unwrap().len();
			let both_len = inject_stack_limiter(&gas_module, 128).unwrap().len();

			let overhead = both_len * 100 / orig_len;
			(
				overhead,
				format!(
					"{:30}: orig = {:4} kb, gas_metering = {} %, stack_limiter = {} %, both = {} %",
					entry.file_name().to_str().unwrap(),
					orig_len / 1024,
					gas_metering_len * 100 / orig_len,
					stack_height_len * 100 / orig_len,
					overhead,
				),
			)
		})
		.collect();
	results.sort_unstable_by(|a, b| b.0.cmp(&a.0));
	for entry in results {
		println!("{}", entry.1);
	}
}

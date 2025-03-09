# fvm-wasm-instrument

This started as a [wasm-instrumet](https://github.com/paritytech/wasm-instrument) with some FVM specific changes but has since been significantly refactored to move away from parity-wasm which reached EOL.

fvm-wasm-instrument is a Rust library containing a collection of WASM module instrumentations and
transformations are mainly useful for wasm based blockchains and smart contracts.

## Provided functionality

This library provides two features:

- Gas metering.
- Stack height limiting.

### Gas Metering

Add gas metering to your platform by injecting the necessary code directly into the wasm module. This allows having a uniform gas metering implementation across different execution engines (interpreters, JIT compilers).

### Stack Height Limiter

Neither the wasm standard nor any sufficiently complex execution engine specifies how many items on the wasm stack are supported before the execution aborts or malfunctions. Even the same execution engine on different operating systems or host architectures could support a different number of stack items and be well within its rights.

This is the kind of indeterminism that can lead to consensus failures when used in a blockchain context.

To address this issue we can inject some code that meters the stack height at runtime and aborts the execution when it reaches a predefined limit. Choosing this limit suffciently small so that it is smaller than what any reasonably parameterized execution engine would support solves the issue: All execution engines would reach the injected limit before hitting any implementation specific limitation.

## License

`fvm-wasm-instrument` is distributed under the terms of both the MIT license and the
Apache License (Version 2.0), at your choice.

See LICENSE-APACHE, and LICENSE-MIT for details.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in `fvm-wasm-instrument` by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

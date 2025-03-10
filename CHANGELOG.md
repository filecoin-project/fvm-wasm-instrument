# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The semantic versioning guarantees cover the interface to the substrate runtime which
includes this pallet as a dependency. This module will also add storage migrations whenever
changes require it. Stability with regard to offchain tooling is explicitly excluded from
this guarantee: For example, adding a new field to an in-storage data structure will require
changes to frontends to properly display it. However, those changes will still be regarded
as a minor version bump.

The interface provided to smart contracts will adhere to semver with one exception: Even
major version bumps will be backwards compatible with regard to already deployed contracts.
In other words: Upgrading this pallet will not break pre-existing contracts.

## [v0.4.0] 2022-12-09

- Update wasmparser/wasmencoder.
- Re-work exported functions for consistency.

## [v0.3.0] 2022-12-09

Reworked for FVM M2.2.

- Dynamic instruction charging.
- Reworked to not use parity-wasm.

## [v0.2.0] 2022-04-14

Reworked for FVM M1.

## [v0.1.1] 2022-01-18

### Fixed

- Stack metering disregarded the activation frame.
[#2](https://github.com/paritytech/wasm-instrument/pull/2)

## [v0.1.0] 2022-01-11

### Changed

- Created from [pwasm-utils](https://github.com/paritytech/wasm-utils) by removing unused code and cleaning up docs.

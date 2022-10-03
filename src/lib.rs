#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
extern crate core;

mod export_globals;
pub mod gas_metering;
mod stack_limiter;
mod utils;

pub use export_globals::export_mutable_globals;
pub use stack_limiter::inject as inject_stack_limiter;

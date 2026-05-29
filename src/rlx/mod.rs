//! RLX-backed NeuroRVQ inference (`rlx::Graph` + `rlx::Session`).
//!
//! Burn-backed types live at the crate root when `--features burn` is enabled.
//! Enable this module with `--features rlx`.

pub mod encoder;
pub mod graph;
pub mod prepare;
pub mod weights;

pub use encoder::{
    FMEncoderResult, ForwardResult, NeuroRVQEncoder, NeuroRVQFoundationModel, ReconstructionResult,
    TokenResult,
};
pub use prepare::{
    build_batch, compute_fft_components, reconstruct_signal, std_norm_4d, RlxInputBatch,
};

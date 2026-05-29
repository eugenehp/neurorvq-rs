//! # neurorvq-rs — NeuroRVQ Biosignal Tokenizer inference in Rust
//!
//! Pure-Rust inference for the NeuroRVQ multi-scale biosignal tokenizer.
//! Two inference engines are available behind Cargo features:
//!
//! | feature | module | runtime |
//! |---------|--------|---------|
//! | `burn`  | crate root (`NeuroRVQEncoder<B>`, …) | [Burn](https://burn.dev) 0.20 |
//! | `rlx`   | [`rlx`] | [RLX](https://docs.rs/rlx) compiler/runtime |
//!
//! NeuroRVQ tokenizes raw EEG/EMG/ECG signals into discrete neural tokens
//! using a multi-scale temporal encoder and Residual Vector Quantization (RVQ).

#[cfg(not(any(feature = "burn", feature = "rlx")))]
compile_error!("enable at least one inference engine: `rlx` (default) and/or `burn`");

/// Configure the global Rayon thread pool (Burn NdArray + RLX CPU).
pub fn init_threads(n: Option<usize>) -> usize {
    let mut builder = rayon::ThreadPoolBuilder::new();
    if let Some(count) = n {
        if count > 0 {
            builder = builder.num_threads(count);
        }
    }
    let _ = builder.build_global();
    rayon::current_num_threads()
}

pub mod channels;
pub mod config;

#[cfg(feature = "burn")]
pub mod data;

#[cfg(feature = "burn")]
pub mod encoder;

#[cfg(feature = "burn")]
pub mod model;

#[cfg(feature = "burn")]
pub mod weights;

#[cfg(feature = "rlx")]
pub mod rlx;

// ── Burn re-exports ───────────────────────────────────────────────────────────

#[cfg(feature = "burn")]
pub use encoder::{
    FMEncoderResult, ForwardResult, NeuroRVQEncoder, NeuroRVQFoundationModel, ReconstructionResult,
    TokenResult,
};

#[cfg(feature = "burn")]
pub use data::{build_batch, build_batch_with_modality, channel_wise_normalize, InputBatch};

// When Burn is off, lift the RLX API to the crate root (default build).
#[cfg(all(feature = "rlx", not(feature = "burn")))]
pub use rlx::{
    build_batch, FMEncoderResult, ForwardResult, NeuroRVQEncoder, NeuroRVQFoundationModel,
    ReconstructionResult, RlxInputBatch, TokenResult,
};

// ── Shared types ──────────────────────────────────────────────────────────────

pub use config::{ConfigOverrides, Modality, NeuroRVQConfig};

pub use channels::{
    channel_index, channel_indices, compute_n_time, create_embedding_ix, create_patches,
    filter_channels, global_channels, global_vocab_size, ECG_CHANNELS, ECG_VOCAB_SIZE,
    EEG_CHANNELS, EEG_VOCAB_SIZE, EMG_CHANNELS, EMG_VOCAB_SIZE,
};

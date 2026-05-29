#![cfg(feature = "burn")]

use crate::config::Modality;
/// Data preparation for NeuroRVQ inference.
///
/// Input format: [B, N_channels, T_time] signal.
use burn::prelude::*;

/// A prepared input batch for the NeuroRVQ tokenizer.
pub struct InputBatch<B: Backend> {
    /// Signal: [1, N_channels, T_time].
    pub signal: Tensor<B, 3>,
    /// Temporal embedding indices: [1, N_channels * n_time].
    pub temporal_ix: Tensor<B, 2, Int>,
    /// Spatial embedding indices: [1, N_channels * n_time].
    pub spatial_ix: Tensor<B, 2, Int>,
    pub n_channels: usize,
    pub n_time_patches: usize,
}

/// Build an InputBatch from raw signal and channel names (defaults to EEG modality).
///
/// signal: [N_channels * T_time] row-major f32
/// channel_names: channel names (must be in the global vocabulary)
/// n_time_patches: number of time patches per channel
/// max_n_patches: maximum patches for temporal alignment
pub fn build_batch<B: Backend>(
    signal: Vec<f32>,
    channel_names: &[&str],
    n_time_patches: usize,
    max_n_patches: usize,
    n_channels: usize,
    n_samples: usize,
    device: &B::Device,
) -> InputBatch<B> {
    build_batch_with_modality(
        signal,
        channel_names,
        n_time_patches,
        max_n_patches,
        n_channels,
        n_samples,
        Modality::EEG,
        device,
    )
}

/// Build an InputBatch with explicit modality.
pub fn build_batch_with_modality<B: Backend>(
    signal: Vec<f32>,
    channel_names: &[&str],
    n_time_patches: usize,
    max_n_patches: usize,
    n_channels: usize,
    n_samples: usize,
    modality: Modality,
    device: &B::Device,
) -> InputBatch<B> {
    let signal_tensor =
        Tensor::<B, 2>::from_data(TensorData::new(signal, vec![n_channels, n_samples]), device)
            .unsqueeze_dim::<3>(0); // [1, N, T]

    let (temp_ix, spat_ix) = crate::channels::create_embedding_ix(
        n_time_patches,
        max_n_patches,
        channel_names,
        modality,
    );

    let n_total = n_channels * n_time_patches;
    let temporal_ix =
        Tensor::<B, 1, Int>::from_data(TensorData::new(temp_ix, vec![n_total]), device)
            .unsqueeze_dim::<2>(0); // [1, N*T]

    let spatial_ix =
        Tensor::<B, 1, Int>::from_data(TensorData::new(spat_ix, vec![n_total]), device)
            .unsqueeze_dim::<2>(0);

    InputBatch {
        signal: signal_tensor,
        temporal_ix,
        spatial_ix,
        n_channels,
        n_time_patches,
    }
}

/// Channel-wise z-score normalization.
pub fn channel_wise_normalize<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let mean = x.clone().mean_dim(2); // [B, C, 1]
    let diff = x.clone() - mean.clone();
    let var = (diff.clone() * diff).mean_dim(2);
    let std = (var + 1e-8).sqrt();
    (x - mean) / std
}

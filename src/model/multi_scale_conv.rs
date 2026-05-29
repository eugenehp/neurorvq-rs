use crate::config::Modality;
use burn::nn::{
    conv::{Conv2d, Conv2dConfig},
    pool::{AvgPool2d, AvgPool2dConfig},
    GroupNorm, GroupNormConfig,
};
/// Multi-Dimensional Temporal Convolution for NeuroRVQ.
///
/// Python: `MultiDimentionalTemporalConv` in NeuroRVQ.py
///
/// Inception-style parallel branches with different kernel sizes capturing
/// different frequency bands. Kernel sizes vary by modality:
///
/// EEG/ECG (fs=200Hz):
///   Branch 1: k=21 (>10 Hz), Branch 2: k=15 (>13 Hz),
///   Branch 3: k=9  (>20 Hz), Branch 4: k=5  (>40 Hz)
///
/// EMG (fs=1000Hz):
///   Branch 1: k=51 (>20 Hz), Branch 2: k=17 (>60 Hz),
///   Branch 3: k=8  (>125 Hz), Branch 4: k=5  (>250 Hz)
///
/// Each branch: Conv2d → GroupNorm → GELU → Pool → Conv2d → GroupNorm → GELU → Pool
use burn::prelude::*;
use burn::tensor::activation::gelu;

/// Kernel configuration for a single convolutional branch (two stages).
#[derive(Debug, Clone, Copy)]
pub struct BranchKernelConfig {
    pub kernel1: usize,
    pub pad1: usize,
    pub pool1_k: usize,
    pub kernel2: usize,
    pub pad2: usize,
    pub pool2_k: usize,
}

/// Get the 4-branch kernel configurations for a given modality.
pub fn kernel_configs(modality: Modality) -> [BranchKernelConfig; 4] {
    match modality {
        Modality::EEG | Modality::ECG => [
            // Branch 1: k=21, pad=10, pool=2; k=9, pad=4, pool=4
            BranchKernelConfig {
                kernel1: 21,
                pad1: 10,
                pool1_k: 2,
                kernel2: 9,
                pad2: 4,
                pool2_k: 4,
            },
            // Branch 2: k=15, pad=7, pool=2; k=7, pad=3, pool=4
            BranchKernelConfig {
                kernel1: 15,
                pad1: 7,
                pool1_k: 2,
                kernel2: 7,
                pad2: 3,
                pool2_k: 4,
            },
            // Branch 3: k=9, pad=4, pool=2; k=5, pad=2, pool=4
            BranchKernelConfig {
                kernel1: 9,
                pad1: 4,
                pool1_k: 2,
                kernel2: 5,
                pad2: 2,
                pool2_k: 4,
            },
            // Branch 4: k=5, pad=2, pool=2; k=3, pad=1, pool=4
            BranchKernelConfig {
                kernel1: 5,
                pad1: 2,
                pool1_k: 2,
                kernel2: 3,
                pad2: 1,
                pool2_k: 4,
            },
        ],
        Modality::EMG => [
            // Branch 1: k=51, pad=25, pool=2; k=25, pad=12, pool=4
            BranchKernelConfig {
                kernel1: 51,
                pad1: 25,
                pool1_k: 2,
                kernel2: 25,
                pad2: 12,
                pool2_k: 4,
            },
            // Branch 2: k=17, pad=8, pool=2; k=9, pad=4, pool=4
            BranchKernelConfig {
                kernel1: 17,
                pad1: 8,
                pool1_k: 2,
                kernel2: 9,
                pad2: 4,
                pool2_k: 4,
            },
            // Branch 3: k=8, pad=4, pool=2; k=4, pad=2, pool=4
            BranchKernelConfig {
                kernel1: 8,
                pad1: 4,
                pool1_k: 2,
                kernel2: 4,
                pad2: 2,
                pool2_k: 4,
            },
            // Branch 4: k=5, pad=2, pool=2; k=3, pad=1, pool=4
            BranchKernelConfig {
                kernel1: 5,
                pad1: 2,
                pool1_k: 2,
                kernel2: 3,
                pad2: 1,
                pool2_k: 4,
            },
        ],
    }
}

/// A single convolutional branch with two conv-norm-pool stages.
#[derive(Module, Debug)]
pub struct ConvBranch<B: Backend> {
    pub conv1: Conv2d<B>,
    pub norm1: GroupNorm<B>,
    pub pool1: AvgPool2d,
    pub conv2: Conv2d<B>,
    pub norm2: GroupNorm<B>,
    pub pool2: AvgPool2d,
}

impl<B: Backend> ConvBranch<B> {
    pub fn new(
        in_chans: usize,
        out_chans: usize,
        cfg: BranchKernelConfig,
        device: &B::Device,
    ) -> Self {
        Self {
            conv1: Conv2dConfig::new([in_chans, out_chans], [1, cfg.kernel1])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(0, cfg.pad1))
                .with_bias(true)
                .init(device),
            norm1: GroupNormConfig::new(4, out_chans)
                .with_epsilon(1e-5)
                .init(device),
            pool1: AvgPool2dConfig::new([1, cfg.pool1_k])
                .with_strides([1, cfg.pool1_k])
                .init(),
            conv2: Conv2dConfig::new([out_chans, out_chans], [1, cfg.kernel2])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(0, cfg.pad2))
                .with_bias(true)
                .init(device),
            norm2: GroupNormConfig::new(4, out_chans)
                .with_epsilon(1e-5)
                .init(device),
            pool2: AvgPool2dConfig::new([1, cfg.pool2_k])
                .with_strides([1, cfg.pool2_k])
                .init(),
        }
    }

    /// x: [B, 1, N*A, T] → [B, out_chans, N*A, T']
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self
            .pool1
            .forward(gelu(self.norm1.forward(self.conv1.forward(x))));
        self.pool2
            .forward(gelu(self.norm2.forward(self.conv2.forward(x))))
    }
}

/// Multi-scale temporal convolution with 4 parallel branches.
#[derive(Module, Debug)]
pub struct MultiScaleTemporalConv<B: Backend> {
    pub branch1: ConvBranch<B>,
    pub branch2: ConvBranch<B>,
    pub branch3: ConvBranch<B>,
    pub branch4: ConvBranch<B>,
}

impl<B: Backend> MultiScaleTemporalConv<B> {
    /// Create with default EEG kernel configuration (backward compatible).
    pub fn new(in_chans: usize, out_chans: usize, device: &B::Device) -> Self {
        Self::new_with_modality(in_chans, out_chans, Modality::EEG, device)
    }

    /// Create with modality-specific kernel configuration.
    pub fn new_with_modality(
        in_chans: usize,
        out_chans: usize,
        modality: Modality,
        device: &B::Device,
    ) -> Self {
        let cfgs = kernel_configs(modality);
        Self {
            branch1: ConvBranch::new(in_chans, out_chans, cfgs[0], device),
            branch2: ConvBranch::new(in_chans, out_chans, cfgs[1], device),
            branch3: ConvBranch::new(in_chans, out_chans, cfgs[2], device),
            branch4: ConvBranch::new(in_chans, out_chans, cfgs[3], device),
        }
    }

    /// x: [B, N, A, T] → (x1, x2, x3, x4) each [B, N*A, T'*C]
    ///
    /// Python rearranges: 'B N A T -> B (N A) T' → unsqueeze(1) → branch → 'B C NA T -> B NA (T C)'
    pub fn forward(
        &self,
        x: Tensor<B, 4>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let [b, n, a, t] = x.dims();
        let na = n * a;

        // Reshape: [B, N, A, T] → [B, N*A, T] → [B, 1, N*A, T]
        let x = x.reshape([b, na, t]).unsqueeze_dim::<4>(1);

        let x1 = self.branch1.forward(x.clone());
        let x2 = self.branch2.forward(x.clone());
        let x3 = self.branch3.forward(x.clone());
        let x4 = self.branch4.forward(x);

        // Rearrange each: [B, C, NA, T'] → [B, NA, T'*C]
        // Python: rearrange(x, 'B C NA T -> B NA (T C)')
        let rearrange = |t: Tensor<B, 4>| -> Tensor<B, 3> {
            let [b2, c, na2, t_prime] = t.dims();
            // swap_dims(1,2) → [B, NA, C, T'] → swap_dims(2,3) → [B, NA, T', C] → reshape
            t.swap_dims(1, 2) // [B, NA, C, T']
                .swap_dims(2, 3) // [B, NA, T', C]
                .reshape([b2, na2, t_prime * c])
        };

        (rearrange(x1), rearrange(x2), rearrange(x3), rearrange(x4))
    }
}

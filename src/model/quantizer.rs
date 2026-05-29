use burn::module::{Param, ParamId};
/// Normalized EMA Vector Quantizer for NeuroRVQ.
///
/// Python: `NormEMAVectorQuantizer` in norm_ema_quantizer.py
///
/// At inference, this simply:
///   1. L2-normalize the input
///   2. Find nearest codebook vector (L2 distance)
///   3. Return quantized output + indices
///
/// The EMA updates are training-only and skipped here.
use burn::prelude::*;

/// Single-level vector quantizer with L2-normalized codebook.
#[derive(Module, Debug)]
pub struct NormVectorQuantizer<B: Backend> {
    /// Codebook weights: [num_tokens, codebook_dim].
    pub weight: Param<Tensor<B, 2>>,
    pub num_tokens: usize,
    pub codebook_dim: usize,
}

impl<B: Backend> NormVectorQuantizer<B> {
    pub fn new(num_tokens: usize, codebook_dim: usize, device: &B::Device) -> Self {
        Self {
            weight: Param::initialized(
                ParamId::new(),
                Tensor::zeros([num_tokens, codebook_dim], device),
            ),
            num_tokens,
            codebook_dim,
        }
    }

    /// L2-normalize along the last dimension.
    fn l2norm(t: Tensor<B, 2>) -> Tensor<B, 2> {
        let norm = t
            .clone()
            .powf_scalar(2.0)
            .sum_dim(1)
            .sqrt()
            .clamp_min(1e-12);
        t / norm
    }

    /// Forward pass (inference only).
    ///
    /// z: [B, C, H, W] → (z_q [B, C, H, W], loss scalar, indices [B*H*W])
    pub fn forward(&self, z: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 1>, Tensor<B, 1, Int>) {
        let [b, c, h, w] = z.dims();
        let device = z.device();

        // Rearrange: [B, C, H, W] → [B, H, W, C] → [B*H*W, C]
        let z_bhwc = z
            .clone()
            .swap_dims(1, 2) // [B, H, C, W]
            .swap_dims(2, 3) // [B, H, W, C]
            .reshape([b * h * w, c]);

        // L2-normalize
        let z_norm = Self::l2norm(z_bhwc.clone());

        // Compute distances: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z@e^T
        // Since both are L2-normalized, ||z||^2 = ||e||^2 = 1
        // d = 2 - 2 * z @ e^T  (but we just need argmin, so -z@e^T suffices)
        let codebook = self.weight.val(); // [num_tokens, codebook_dim]
        let similarity = z_norm.clone().matmul(codebook.transpose()); // [B*H*W, num_tokens]
        let neg_sim = similarity.neg();

        // Argmin → encoding indices
        let encoding_indices = neg_sim.argmin(1).reshape([b * h * w]); // [B*H*W]

        // Gather quantized vectors
        let indices_flat = encoding_indices.clone();
        let z_q_flat = self.gather_embeddings(indices_flat.clone(), &device);

        // Reshape back: [B*H*W, C] → [B, H, W, C] → [B, C, H, W]
        let z_q = z_q_flat
            .reshape([b, h, w, c])
            .swap_dims(2, 3) // [B, H, C, W]
            .swap_dims(1, 2); // [B, C, H, W]

        // At inference, loss = 0
        let loss = Tensor::zeros([1], &device);

        (z_q, loss, encoding_indices)
    }

    /// Look up embeddings by indices.
    fn gather_embeddings(&self, indices: Tensor<B, 1, Int>, device: &B::Device) -> Tensor<B, 2> {
        let n = indices.dims()[0];
        let codebook = self.weight.val(); // [num_tokens, codebook_dim]

        // Convert indices to f32 for gather, then use select
        // We'll use a simple loop-free approach via one-hot matmul
        let one_hot = Tensor::<B, 2>::zeros([n, self.num_tokens], device);
        // Use scatter to build one-hot
        let indices_2d = indices.unsqueeze_dim::<2>(1); // [n, 1]
        let ones = Tensor::<B, 2, Int>::ones([n, 1], device);
        let one_hot = one_hot.scatter(
            1,
            indices_2d,
            ones.float(),
            burn::tensor::IndexingUpdateOp::Add,
        );
        one_hot.matmul(codebook) // [n, codebook_dim]
    }

    /// Encode: z → indices (no gradient).
    pub fn encode(&self, z: Tensor<B, 4>) -> Tensor<B, 1, Int> {
        let (_, _, indices) = self.forward(z);
        indices
    }

    /// Decode: indices → quantized vectors [B*H*W, codebook_dim].
    pub fn decode(&self, indices: Tensor<B, 1, Int>, device: &B::Device) -> Tensor<B, 2> {
        self.gather_embeddings(indices, device)
    }
}

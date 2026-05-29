use crate::model::quantizer::NormVectorQuantizer;
/// Residual Vector Quantization for NeuroRVQ.
///
/// Python: `ResidualVectorQuantization` in RVQ.py
/// Follows Algorithm 1 in https://arxiv.org/pdf/2107.03312.pdf
///
/// At inference:
///   residual = x
///   for each quantizer level:
///     quantized = quantizer(residual)
///     residual = residual - quantized
///     output += quantized
use burn::prelude::*;

/// RVQ with N quantizer levels.
#[derive(Module, Debug)]
pub struct ResidualVQ<B: Backend> {
    pub layers: Vec<NormVectorQuantizer<B>>,
    pub num_quantizers: usize,
}

impl<B: Backend> ResidualVQ<B> {
    pub fn new(
        num_quantizers: usize,
        num_tokens: usize,
        codebook_dim: usize,
        device: &B::Device,
    ) -> Self {
        let layers = (0..num_quantizers)
            .map(|_| NormVectorQuantizer::new(num_tokens, codebook_dim, device))
            .collect();

        Self {
            layers,
            num_quantizers,
        }
    }

    /// Forward pass (inference).
    ///
    /// x: [B, C, H, W]
    /// Returns: (quantized_out [B, C, H, W], indices [num_quantizers, B*H*W], total_loss [1])
    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Vec<Tensor<B, 1, Int>>, Tensor<B, 1>) {
        let device = x.device();
        let mut quantized_out = Tensor::zeros_like(&x);
        let mut residual = x;
        let mut all_indices = Vec::with_capacity(self.num_quantizers);
        let mut total_loss = Tensor::<B, 1>::zeros([1], &device);

        for layer in &self.layers {
            let (quantized, loss, indices) = layer.forward(residual.clone());
            residual = residual - quantized.clone();
            quantized_out = quantized_out + quantized;
            all_indices.push(indices);
            total_loss = total_loss + loss;
        }

        (quantized_out, all_indices, total_loss)
    }

    /// Encode: x → indices [num_quantizers, B*H*W].
    pub fn encode(&self, x: Tensor<B, 4>) -> Vec<Tensor<B, 1, Int>> {
        let mut residual = x;
        let mut all_indices = Vec::with_capacity(self.num_quantizers);

        for layer in &self.layers {
            let (quantized, _, indices) = layer.forward(residual.clone());
            residual = residual - quantized;
            all_indices.push(indices);
        }

        all_indices
    }

    /// Decode: indices → quantized output [B*H*W, codebook_dim].
    pub fn decode(&self, indices: &[Tensor<B, 1, Int>], device: &B::Device) -> Tensor<B, 2> {
        let first = self.layers[0].decode(indices[0].clone(), device);
        let mut out = first;

        for (layer, idx) in self.layers[1..].iter().zip(indices[1..].iter()) {
            out = out + layer.decode(idx.clone(), device);
        }

        out
    }
}

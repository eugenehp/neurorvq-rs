use burn::nn::conv::{Conv2d, Conv2dConfig};
/// Patch Embedding for the decoder side of NeuroRVQ.
///
/// Python: `PatchEmbed` in NeuroRVQ.py
///   Projects each codebook vector to the patch latent space via Conv2d(in_chans, embed_dim, 1x1).
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct PatchEmbed<B: Backend> {
    pub proj: Conv2d<B>,
}

impl<B: Backend> PatchEmbed<B> {
    pub fn new(in_chans: usize, embed_dim: usize, device: &B::Device) -> Self {
        Self {
            proj: Conv2dConfig::new([in_chans, embed_dim], [1, 1])
                .with_stride([1, 1])
                .with_bias(true)
                .init(device),
        }
    }

    /// x: [B, in_chans, H, W] → [B, H*W, embed_dim]
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let out = self.proj.forward(x); // [B, embed_dim, H, W]
        let [b, d, h, w] = out.dims();
        // flatten(2).transpose(1, 2) → [B, H*W, embed_dim]
        out.reshape([b, d, h * w]).swap_dims(1, 2)
    }
}

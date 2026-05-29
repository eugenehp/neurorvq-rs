use burn::nn::{Linear, LinearConfig};
/// MLP / Feed-Forward module for NeuroRVQ transformer blocks.
///
/// Python: `Mlp` class in NeuroRVQ_modules.py
///   fc1(dim → hidden_dim) → GELU → fc2(hidden_dim → dim) → Dropout
use burn::prelude::*;
use burn::tensor::activation::gelu;

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    pub fn new(
        in_features: usize,
        hidden_features: usize,
        out_features: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            fc1: LinearConfig::new(in_features, hidden_features)
                .with_bias(true)
                .init(device),
            fc2: LinearConfig::new(hidden_features, out_features)
                .with_bias(true)
                .init(device),
        }
    }

    /// x: [B, S, dim] → [B, S, out_dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = gelu(self.fc1.forward(x));
        self.fc2.forward(h)
    }
}

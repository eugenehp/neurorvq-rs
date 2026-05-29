use burn::nn::{LayerNorm, LayerNormConfig};
/// LayerNorm wrapper for NeuroRVQ.
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct NeuroLayerNorm<B: Backend> {
    pub inner: LayerNorm<B>,
}

impl<B: Backend> NeuroLayerNorm<B> {
    pub fn new(dim: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            inner: LayerNormConfig::new(dim).with_epsilon(eps).init(device),
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        self.inner.forward(x)
    }
}

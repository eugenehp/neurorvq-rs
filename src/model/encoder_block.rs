use crate::model::attention::Attention;
use crate::model::feedforward::Mlp;
use crate::model::norm::NeuroLayerNorm;
use burn::module::{Param, ParamId};
/// Transformer Block for NeuroRVQ.
///
/// Python: `Block` class in NeuroRVQ_modules.py
///   Pre-norm architecture with optional layer-scale (gamma):
///   x = x + drop_path(gamma_1 * attn(norm1(x)))
///   x = x + drop_path(gamma_2 * mlp(norm2(x)))
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    pub norm1: NeuroLayerNorm<B>,
    pub attn: Attention<B>,
    pub norm2: NeuroLayerNorm<B>,
    pub mlp: Mlp<B>,
    /// Optional layer-scale parameters.
    pub gamma_1: Option<Param<Tensor<B, 1>>>,
    pub gamma_2: Option<Param<Tensor<B, 1>>>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(
        dim: usize,
        num_heads: usize,
        mlp_ratio: f64,
        qkv_bias: bool,
        use_qk_norm: bool,
        init_values: f64,
        norm_eps: f64,
        device: &B::Device,
    ) -> Self {
        let mlp_hidden = (dim as f64 * mlp_ratio) as usize;

        let (gamma_1, gamma_2) = if init_values > 0.0 {
            (
                Some(Param::initialized(
                    ParamId::new(),
                    Tensor::ones([dim], device).mul_scalar(init_values as f32),
                )),
                Some(Param::initialized(
                    ParamId::new(),
                    Tensor::ones([dim], device).mul_scalar(init_values as f32),
                )),
            )
        } else {
            (None, None)
        };

        Self {
            norm1: NeuroLayerNorm::new(dim, norm_eps, device),
            attn: Attention::new(dim, num_heads, qkv_bias, use_qk_norm, norm_eps, device),
            norm2: NeuroLayerNorm::new(dim, norm_eps, device),
            mlp: Mlp::new(dim, mlp_hidden, dim, device),
            gamma_1,
            gamma_2,
        }
    }

    /// x: [B, N, C] → [B, N, C]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let attn_out = self.attn.forward(self.norm1.forward(x.clone()));
        let x = if let Some(ref g1) = self.gamma_1 {
            x + attn_out * g1.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0)
        } else {
            x + attn_out
        };

        let mlp_out = self.mlp.forward(self.norm2.forward(x.clone()));
        if let Some(ref g2) = self.gamma_2 {
            x + mlp_out * g2.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0)
        } else {
            x + mlp_out
        }
    }
}

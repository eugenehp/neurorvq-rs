use crate::model::norm::NeuroLayerNorm;
use burn::module::{Param, ParamId};
use burn::nn::{Linear, LinearConfig};
/// Self-Attention module for NeuroRVQ transformer blocks.
///
/// Python: `Attention` class in NeuroRVQ_modules.py
///   qkv = Linear(dim, 3*dim, bias=False)
///   q_bias, v_bias (optional)
///   q_norm, k_norm (optional LayerNorm on head_dim)
///   attn = softmax(q @ k^T / sqrt(d)) @ v
///   output = proj(attn)
use burn::prelude::*;
use burn::tensor::activation::softmax;

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    /// QKV projection: Linear(dim, 3*dim, bias=False).
    /// Bias is handled separately via q_bias + v_bias.
    pub qkv: Linear<B>,
    /// Output projection.
    pub proj: Linear<B>,
    /// Optional query bias [all_head_dim].
    pub q_bias: Option<Param<Tensor<B, 1>>>,
    /// Optional value bias [all_head_dim].
    pub v_bias: Option<Param<Tensor<B, 1>>>,
    /// Optional query norm (LayerNorm on head_dim).
    pub q_norm: Option<NeuroLayerNorm<B>>,
    /// Optional key norm.
    pub k_norm: Option<NeuroLayerNorm<B>>,
    pub num_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
}

impl<B: Backend> Attention<B> {
    pub fn new(
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        use_qk_norm: bool,
        norm_eps: f64,
        device: &B::Device,
    ) -> Self {
        let head_dim = dim / num_heads;
        let all_head_dim = head_dim * num_heads;

        let qkv = LinearConfig::new(dim, all_head_dim * 3)
            .with_bias(false)
            .init(device);

        let proj = LinearConfig::new(all_head_dim, dim)
            .with_bias(true)
            .init(device);

        let (q_bias, v_bias) = if qkv_bias {
            (
                Some(Param::initialized(
                    ParamId::new(),
                    Tensor::zeros([all_head_dim], device),
                )),
                Some(Param::initialized(
                    ParamId::new(),
                    Tensor::zeros([all_head_dim], device),
                )),
            )
        } else {
            (None, None)
        };

        let (q_norm, k_norm) = if use_qk_norm {
            (
                Some(NeuroLayerNorm::new(head_dim, norm_eps, device)),
                Some(NeuroLayerNorm::new(head_dim, norm_eps, device)),
            )
        } else {
            (None, None)
        };

        Self {
            qkv,
            proj,
            q_bias,
            v_bias,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5) as f32,
            q_norm,
            k_norm,
        }
    }

    /// x: [B, N, C] → [B, N, C]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, _c] = x.dims();
        let h = self.num_heads;
        let dh = self.head_dim;

        // Build QKV with optional bias
        // Python: qkv_bias = cat(q_bias, zeros, v_bias) if q_bias is not None
        //         qkv = F.linear(x, qkv.weight, qkv_bias)
        let qkv = if let (Some(ref qb), Some(ref vb)) = (&self.q_bias, &self.v_bias) {
            let q_bias = qb.val();
            let v_bias = vb.val();
            let zero_bias = Tensor::zeros_like(&v_bias);
            let bias = Tensor::cat(vec![q_bias, zero_bias, v_bias], 0);
            // Manual F.linear: x @ weight^T + bias
            // qkv.weight in burn is [dim, 3*dim] (already transposed for forward)
            // We need to do x @ W + bias manually to inject the fused bias
            let out = self.qkv.forward(x); // This applies W without bias (bias=false)
            out + bias.unsqueeze_dim::<2>(0).expand([b, n, h * dh * 3])
        } else {
            self.qkv.forward(x)
        };

        // Reshape: [B, N, 3*H*D] → [B, N, 3, H, D] → [3, B, H, N, D]
        let qkv = qkv.reshape([b, n, 3, h, dh]);

        let q = qkv.clone().narrow(2, 0, 1).reshape([b, n, h, dh]);
        let k = qkv.clone().narrow(2, 1, 1).reshape([b, n, h, dh]);
        let v = qkv.narrow(2, 2, 1).reshape([b, n, h, dh]);

        // Apply optional Q/K norms
        let q = if let Some(ref norm) = self.q_norm {
            norm.forward(q)
        } else {
            q
        };
        let k = if let Some(ref norm) = self.k_norm {
            norm.forward(k)
        } else {
            k
        };

        // Transpose for attention: [B, H, N, D]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // Scaled dot-product attention
        let attn = softmax(q.matmul(k.transpose()).mul_scalar(self.scale), 3);
        let out = attn.matmul(v); // [B, H, N, D]

        // Reshape back: [B, N, H*D]
        let out = out.swap_dims(1, 2).reshape([b, n, h * dh]);
        self.proj.forward(out)
    }
}

//! NeuroRVQ FM transformer branch expressed as RLX IR.

#![allow(clippy::too_many_arguments)]

use rlx::ir::GraphExt;
use rlx::ops::MaskKind;
use rlx::prelude::*;

#[derive(Clone, Debug)]
pub struct FmBranchSpec {
    pub b: usize,
    /// Sequence length including the prepended CLS token.
    pub s: usize,
    /// Sequence length without CLS (output tokens).
    pub seq_len: usize,
    pub d: usize,
    pub out_dim: usize,
    pub nh: usize,
    pub dh: usize,
    pub depth: usize,
    pub ff: usize,
    pub norm_eps: f32,
    /// e.g. `"blocks"` or `"encoder.blocks"`.
    pub block_prefix: String,
    /// e.g. `""` or `"encoder"`.
    pub head_prefix: String,
    pub branch: usize,
    pub use_qk_norm: bool,
}

impl FmBranchSpec {
    fn head_key(&self, name: &str) -> String {
        if self.head_prefix.is_empty() {
            format!("{name}_{}", self.branch)
        } else {
            format!("{}.{name}_{}", self.head_prefix, self.branch)
        }
    }
}

fn s1(d: usize) -> Shape {
    Shape::new(&[d], DType::F32)
}
fn s2(a: usize, b: usize) -> Shape {
    Shape::new(&[a, b], DType::F32)
}
fn s3(a: usize, b: usize, c: usize) -> Shape {
    Shape::new(&[a, b, c], DType::F32)
}
fn s4(a: usize, b: usize, c: usize, d: usize) -> Shape {
    Shape::new(&[a, b, c, d], DType::F32)
}

fn block_key(prefix: &str, i: usize) -> String {
    if prefix.is_empty() {
        format!("blocks.{i}")
    } else {
        format!("{prefix}.{i}")
    }
}

fn ln(g: &mut Graph, x: NodeId, w: NodeId, b: NodeId, eps: f32) -> NodeId {
    g.ln(x, w, b, eps)
}

fn bias_add(g: &mut Graph, x: NodeId, b: NodeId) -> NodeId {
    g.add(x, b)
}

fn attn_bshd_to_bhsd(g: &mut Graph, t: NodeId) -> NodeId {
    g.transpose_(t, vec![0, 2, 1, 3])
}

fn attn_bhsd_to_bshd(g: &mut Graph, t: NodeId) -> NodeId {
    g.transpose_(t, vec![0, 2, 1, 3])
}

fn ln_on_heads(
    g: &mut Graph,
    x4: NodeId,
    w: NodeId,
    b: NodeId,
    eps: f32,
    batch: usize,
    seq: usize,
    nh: usize,
    dh: usize,
) -> NodeId {
    let flat = g.reshape_(x4, vec![(batch * seq * nh) as i64, dh as i64]);
    let normed = ln(g, flat, w, b, eps);
    g.reshape_(normed, vec![batch as i64, seq as i64, nh as i64, dh as i64])
}

fn neuro_attention(
    g: &mut Graph,
    x: NodeId,
    prefix: &str,
    batch: usize,
    seq: usize,
    d: usize,
    nh: usize,
    dh: usize,
    use_qk_norm: bool,
    eps: f32,
) -> NodeId {
    let h_total = nh * dh;

    let wq = g.param(format!("{prefix}.attn.wq.weight"), s2(d, h_total));
    let wk = g.param(format!("{prefix}.attn.wk.weight"), s2(d, h_total));
    let wv = g.param(format!("{prefix}.attn.wv.weight"), s2(d, h_total));
    let wo = g.param(format!("{prefix}.attn.proj.weight"), s2(h_total, d));
    let wo_b = g.param(format!("{prefix}.attn.proj.bias"), s1(d));

    let qm = g.mm(x, wq);
    let km = g.mm(x, wk);
    let vm = g.mm(x, wv);

    let qb = g.param(format!("{prefix}.attn.q_bias"), s1(h_total));
    let kb = g.param(format!("{prefix}.attn.k_bias"), s1(h_total));
    let vb = g.param(format!("{prefix}.attn.v_bias"), s1(h_total));
    let q = bias_add(g, qm, qb);
    let k = bias_add(g, km, kb);
    let v = bias_add(g, vm, vb);

    let q4 = g.reshape_(q, vec![batch as i64, seq as i64, nh as i64, dh as i64]);
    let k4 = g.reshape_(k, vec![batch as i64, seq as i64, nh as i64, dh as i64]);
    let v4 = g.reshape_(v, vec![batch as i64, seq as i64, nh as i64, dh as i64]);

    let (q4, k4) = if use_qk_norm {
        let qnw = g.param(format!("{prefix}.attn.q_norm.weight"), s1(dh));
        let qnb = g.param(format!("{prefix}.attn.q_norm.bias"), s1(dh));
        let knw = g.param(format!("{prefix}.attn.k_norm.weight"), s1(dh));
        let knb = g.param(format!("{prefix}.attn.k_norm.bias"), s1(dh));
        (
            ln_on_heads(g, q4, qnw, qnb, eps, batch, seq, nh, dh),
            ln_on_heads(g, k4, knw, knb, eps, batch, seq, nh, dh),
        )
    } else {
        (q4, k4)
    };

    let q_bhsd = attn_bshd_to_bhsd(g, q4);
    let k_bhsd = attn_bshd_to_bhsd(g, k4);
    let v_bhsd = attn_bshd_to_bhsd(g, v4);
    let attn = g.attention_kind(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        nh,
        dh,
        MaskKind::None,
        s4(batch, nh, seq, dh),
    );
    let attn_bshd = attn_bhsd_to_bshd(g, attn);
    let attn_3 = g.reshape_(attn_bshd, vec![batch as i64, seq as i64, h_total as i64]);
    let om = g.mm(attn_3, wo);
    bias_add(g, om, wo_b)
}

fn transformer_block(g: &mut Graph, x: NodeId, prefix: &str, spec: &FmBranchSpec) -> NodeId {
    let b = spec.b;
    let s = spec.s;
    let d = spec.d;
    let nh = spec.nh;
    let dh = spec.dh;
    let eps = spec.norm_eps;

    let n1w = g.param(format!("{prefix}.norm1.weight"), s1(d));
    let n1b = g.param(format!("{prefix}.norm1.bias"), s1(d));
    let normed = ln(g, x, n1w, n1b, eps);
    let attn = neuro_attention(g, normed, prefix, b, s, d, nh, dh, spec.use_qk_norm, eps);
    let g1 = g.param(format!("{prefix}.gamma_1"), s1(d));
    let g1_bc = g.reshape_(g1, vec![1, 1, d as i64]);
    let attn_scaled = g.mul(attn, g1_bc);
    let x = g.add(x, attn_scaled);

    let n2w = g.param(format!("{prefix}.norm2.weight"), s1(d));
    let n2b = g.param(format!("{prefix}.norm2.bias"), s1(d));
    let normed = ln(g, x, n2w, n2b, eps);

    let fc1w = g.param(format!("{prefix}.mlp.fc1.weight"), s2(d, spec.ff));
    let fc1b = g.param(format!("{prefix}.mlp.fc1.bias"), s1(spec.ff));
    let fc2w = g.param(format!("{prefix}.mlp.fc2.weight"), s2(spec.ff, d));
    let fc2b = g.param(format!("{prefix}.mlp.fc2.bias"), s1(d));

    let m1 = g.mm(normed, fc1w);
    let m1b = bias_add(g, m1, fc1b);
    let h = g.gelu(m1b);
    let m2 = g.mm(h, fc2w);
    let m2b = bias_add(g, m2, fc2b);
    let g2 = g.param(format!("{prefix}.gamma_2"), s1(d));
    let g2_bc = g.reshape_(g2, vec![1, 1, d as i64]);
    let mlp_scaled = g.mul(m2b, g2_bc);
    g.add(x, mlp_scaled)
}

/// Build the FM branch graph: transformer blocks → drop CLS → fc_norm → head.
pub fn build_fm_branch_graph(spec: &FmBranchSpec) -> Graph {
    let mut g = Graph::new("neurorvq_fm_branch");
    let x = g.input("x", s3(spec.b, spec.s, spec.d));

    let mut h = x;
    for i in 0..spec.depth {
        let bp = block_key(&spec.block_prefix, i);
        h = transformer_block(&mut g, h, &bp, spec);
    }

    let rest = g.narrow_(h, 1, 1, spec.seq_len);

    let fc_norm_w = g.param(format!("{}.weight", spec.head_key("fc_norm")), s1(spec.d));
    let fc_norm_b = g.param(format!("{}.bias", spec.head_key("fc_norm")), s1(spec.d));
    let normed = ln(&mut g, rest, fc_norm_w, fc_norm_b, spec.norm_eps);

    let head_w = g.param(
        format!("{}.weight", spec.head_key("head")),
        s2(spec.d, spec.out_dim),
    );
    let head_b = g.param(format!("{}.bias", spec.head_key("head")), s1(spec.out_dim));
    let hm = g.mm(normed, head_w);
    let out = bias_add(&mut g, hm, head_b);

    g.set_outputs(vec![out]);
    g
}

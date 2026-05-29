//! Safetensors → flat parameter map for RLX graphs.

use std::collections::HashMap;

use half::bf16;
use safetensors::SafeTensors;

use crate::config::NeuroRVQConfig;

#[derive(Clone, Debug)]
pub struct ParamBuf {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

pub type ParamMap = HashMap<String, ParamBuf>;

pub fn load_safetensors(path: &str) -> anyhow::Result<HashMap<String, ParamBuf>> {
    let bytes = std::fs::read(path)?;
    let st = SafeTensors::deserialize(&bytes)?;
    let mut out = HashMap::with_capacity(st.len());
    for (key, view) in st.tensors() {
        let shape: Vec<usize> = view.shape().to_vec();
        let data = match view.dtype() {
            safetensors::Dtype::BF16 => view
                .data()
                .chunks_exact(2)
                .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            safetensors::Dtype::F16 => view
                .data()
                .chunks_exact(2)
                .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            safetensors::Dtype::F32 => view
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            other => anyhow::bail!("unsupported dtype {:?} for key {}", other, key),
        };
        out.insert(key.to_string(), ParamBuf { data, shape });
    }
    Ok(out)
}

fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

fn take_linear_w(raw: &mut HashMap<String, ParamBuf>, key: &str) -> anyhow::Result<ParamBuf> {
    let p = raw
        .remove(key)
        .ok_or_else(|| anyhow::anyhow!("missing weight key: {key}"))?;
    anyhow::ensure!(
        p.shape.len() == 2,
        "Linear weight {key} must be 2-D, got {:?}",
        p.shape
    );
    let (out_d, in_d) = (p.shape[0], p.shape[1]);
    let data = transpose(&p.data, out_d, in_d);
    Ok(ParamBuf {
        data,
        shape: vec![in_d, out_d],
    })
}

fn take(raw: &mut HashMap<String, ParamBuf>, key: &str) -> anyhow::Result<ParamBuf> {
    raw.remove(key)
        .ok_or_else(|| anyhow::anyhow!("missing weight key: {key}"))
}

fn try_take(raw: &mut HashMap<String, ParamBuf>, key: &str) -> Option<ParamBuf> {
    raw.remove(key)
}

/// Split fused `qkv.weight` `[dim, 3*dim]` (RLX layout) → Q/K/V `[dim, dim]`.
fn split_qkv(p: ParamBuf) -> anyhow::Result<(ParamBuf, ParamBuf, ParamBuf)> {
    anyhow::ensure!(p.shape.len() == 2, "qkv must be 2-D, got {:?}", p.shape);
    let dim = p.shape[0];
    anyhow::ensure!(
        p.shape[1] == 3 * dim,
        "qkv shape mismatch: {:?}, expected [{dim}, {}]",
        p.shape,
        3 * dim
    );
    let mut wq = vec![0f32; dim * dim];
    let mut wk = vec![0f32; dim * dim];
    let mut wv = vec![0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let src = i * 3 * dim + j;
            wq[i * dim + j] = p.data[src];
            wk[i * dim + j] = p.data[src + dim];
            wv[i * dim + j] = p.data[src + 2 * dim];
        }
    }
    Ok((
        ParamBuf {
            data: wq,
            shape: vec![dim, dim],
        },
        ParamBuf {
            data: wk,
            shape: vec![dim, dim],
        },
        ParamBuf {
            data: wv,
            shape: vec![dim, dim],
        },
    ))
}

fn insert_fused_qkv_bias(
    p: &mut ParamMap,
    prefix: &str,
    raw: &mut HashMap<String, ParamBuf>,
    dim: usize,
) -> anyhow::Result<()> {
    let h = dim;
    if let Some(qb) = try_take(raw, &format!("{prefix}.attn.q_bias")) {
        p.insert(format!("{prefix}.attn.q_bias"), qb);
        p.insert(
            format!("{prefix}.attn.k_bias"),
            ParamBuf {
                data: vec![0.0; h],
                shape: vec![h],
            },
        );
        if let Some(vb) = try_take(raw, &format!("{prefix}.attn.v_bias")) {
            p.insert(format!("{prefix}.attn.v_bias"), vb);
        } else {
            p.insert(
                format!("{prefix}.attn.v_bias"),
                ParamBuf {
                    data: vec![0.0; h],
                    shape: vec![h],
                },
            );
        }
    }
    Ok(())
}

fn insert_transformer_block(
    p: &mut ParamMap,
    raw: &mut HashMap<String, ParamBuf>,
    prefix: &str,
    dim: usize,
    _ff: usize,
) -> anyhow::Result<()> {
    p.insert(
        format!("{prefix}.norm1.weight"),
        take(raw, &format!("{prefix}.norm1.weight"))?,
    );
    p.insert(
        format!("{prefix}.norm1.bias"),
        take(raw, &format!("{prefix}.norm1.bias"))?,
    );

    let qkv = take_linear_w(raw, &format!("{prefix}.attn.qkv.weight"))?;
    let qkv_dim = qkv.shape[0];
    let (wq, wk, wv) = split_qkv(qkv)?;
    p.insert(format!("{prefix}.attn.wq.weight"), wq);
    p.insert(format!("{prefix}.attn.wk.weight"), wk);
    p.insert(format!("{prefix}.attn.wv.weight"), wv);
    insert_fused_qkv_bias(p, prefix, raw, qkv_dim)?;

    if let (Some(w), Some(b)) = (
        try_take(raw, &format!("{prefix}.attn.q_norm.weight")),
        try_take(raw, &format!("{prefix}.attn.q_norm.bias")),
    ) {
        p.insert(format!("{prefix}.attn.q_norm.weight"), w);
        p.insert(format!("{prefix}.attn.q_norm.bias"), b);
    }
    if let (Some(w), Some(b)) = (
        try_take(raw, &format!("{prefix}.attn.k_norm.weight")),
        try_take(raw, &format!("{prefix}.attn.k_norm.bias")),
    ) {
        p.insert(format!("{prefix}.attn.k_norm.weight"), w);
        p.insert(format!("{prefix}.attn.k_norm.bias"), b);
    }

    p.insert(
        format!("{prefix}.attn.proj.weight"),
        take_linear_w(raw, &format!("{prefix}.attn.proj.weight"))?,
    );
    p.insert(
        format!("{prefix}.attn.proj.bias"),
        take(raw, &format!("{prefix}.attn.proj.bias"))?,
    );

    p.insert(
        format!("{prefix}.norm2.weight"),
        take(raw, &format!("{prefix}.norm2.weight"))?,
    );
    p.insert(
        format!("{prefix}.norm2.bias"),
        take(raw, &format!("{prefix}.norm2.bias"))?,
    );

    p.insert(
        format!("{prefix}.mlp.fc1.weight"),
        take_linear_w(raw, &format!("{prefix}.mlp.fc1.weight"))?,
    );
    p.insert(
        format!("{prefix}.mlp.fc1.bias"),
        take(raw, &format!("{prefix}.mlp.fc1.bias"))?,
    );
    p.insert(
        format!("{prefix}.mlp.fc2.weight"),
        take_linear_w(raw, &format!("{prefix}.mlp.fc2.weight"))?,
    );
    p.insert(
        format!("{prefix}.mlp.fc2.bias"),
        take(raw, &format!("{prefix}.mlp.fc2.bias"))?,
    );

    let ones = vec![1.0f32; dim];
    p.insert(
        format!("{prefix}.gamma_1"),
        try_take(raw, &format!("{prefix}.gamma_1")).unwrap_or(ParamBuf {
            data: ones.clone(),
            shape: vec![dim],
        }),
    );
    p.insert(
        format!("{prefix}.gamma_2"),
        try_take(raw, &format!("{prefix}.gamma_2")).unwrap_or(ParamBuf {
            data: ones,
            shape: vec![dim],
        }),
    );

    Ok(())
}

/// Build transformer + head params for one FM branch graph.
pub fn build_fm_branch_params(
    raw: &mut HashMap<String, ParamBuf>,
    cfg: &NeuroRVQConfig,
    block_prefix: &str,
    head_prefix: &str,
    branch: usize,
    depth: usize,
    embed_dim: usize,
) -> anyhow::Result<ParamMap> {
    let ff = (embed_dim as f64 * cfg.fm_mlp_ratio()) as usize;
    let mut p = ParamMap::new();

    for i in 0..depth {
        let bp = if block_prefix.is_empty() {
            format!("blocks.{i}")
        } else {
            format!("{block_prefix}.{i}")
        };
        insert_transformer_block(&mut p, raw, &bp, embed_dim, ff)?;
    }

    let hp = if head_prefix.is_empty() {
        String::new()
    } else {
        format!("{head_prefix}.")
    };
    p.insert(
        format!("{hp}fc_norm_{branch}.weight"),
        take(raw, &format!("{hp}fc_norm_{branch}.weight"))?,
    );
    p.insert(
        format!("{hp}fc_norm_{branch}.bias"),
        take(raw, &format!("{hp}fc_norm_{branch}.bias"))?,
    );
    p.insert(
        format!("{hp}head_{branch}.weight"),
        take_linear_w(raw, &format!("{hp}head_{branch}.weight"))?,
    );
    p.insert(
        format!("{hp}head_{branch}.bias"),
        take(raw, &format!("{hp}head_{branch}.bias"))?,
    );

    Ok(p)
}

/// Build all CPU-side prepare params (encoder conv, decoder patch embed, RVQ, heads).
pub fn build_all_prepare_params(raw: &mut HashMap<String, ParamBuf>) -> anyhow::Result<ParamMap> {
    build_prepare_params(raw, "encoder.patch_embed")
}

/// Positional / conv / RVQ params consumed on the CPU prepare path.
pub fn build_prepare_params(
    raw: &mut HashMap<String, ParamBuf>,
    conv_prefix: &str,
) -> anyhow::Result<ParamMap> {
    let mut p = ParamMap::new();

    for n in 1..=4 {
        let keys = [
            (
                format!("conv1_{n}.weight"),
                format!("{conv_prefix}.conv1_{n}.weight"),
            ),
            (
                format!("conv1_{n}.bias"),
                format!("{conv_prefix}.conv1_{n}.bias"),
            ),
            (
                format!("norm1_{n}.weight"),
                format!("{conv_prefix}.norm1_{n}.weight"),
            ),
            (
                format!("norm1_{n}.bias"),
                format!("{conv_prefix}.norm1_{n}.bias"),
            ),
            (
                format!("conv2_{n}.weight"),
                format!("{conv_prefix}.conv2_{n}.weight"),
            ),
            (
                format!("conv2_{n}.bias"),
                format!("{conv_prefix}.conv2_{n}.bias"),
            ),
            (
                format!("norm2_{n}.weight"),
                format!("{conv_prefix}.norm2_{n}.weight"),
            ),
            (
                format!("norm2_{n}.bias"),
                format!("{conv_prefix}.norm2_{n}.bias"),
            ),
        ];
        for (dst, src) in keys {
            if let Ok(w) = take(raw, &src) {
                p.insert(dst, w);
            }
        }
    }

    for prefix in ["", "encoder.", "decoder."] {
        let cls = format!("{prefix}cls_token");
        let pos = format!("{prefix}pos_embed");
        let tim = format!("{prefix}time_embed");
        if let Some(t) = try_take(raw, &cls) {
            p.insert(format!("{prefix}cls_token"), t);
        }
        if let Some(t) = try_take(raw, &pos) {
            p.insert(format!("{prefix}pos_embed"), t);
        }
        if let Some(t) = try_take(raw, &tim) {
            p.insert(format!("{prefix}time_embed"), t);
        }
    }

    for n in 1..=4 {
        for l in 0..16 {
            let key = format!("quantize_{n}.layers.{l}.embedding.weight");
            if let Some(w) = try_take(raw, &key) {
                p.insert(key, w);
            }
        }
        for suffix in [".0.weight", ".0.bias", ".2.weight", ".2.bias"] {
            let key = format!("encode_task_layer_{n}{suffix}");
            if let Some(w) = try_take(raw, &key) {
                if suffix.ends_with(".weight") && w.shape.len() == 2 {
                    let (r, c) = (w.shape[0], w.shape[1]);
                    let data = transpose(&w.data, r, c);
                    p.insert(
                        key,
                        ParamBuf {
                            data,
                            shape: vec![c, r],
                        },
                    );
                } else {
                    p.insert(key, w);
                }
            }
        }
    }

    for head in ["amplitude", "angle_sin", "angle_cos"] {
        for suffix in [".0.weight", ".0.bias", ".2.weight", ".2.bias"] {
            let key = format!("decode_task_layer_{head}{suffix}");
            if let Some(w) = try_take(raw, &key) {
                if suffix.ends_with(".weight") && w.shape.len() == 2 {
                    let (r, c) = (w.shape[0], w.shape[1]);
                    let data = transpose(&w.data, r, c);
                    p.insert(
                        key,
                        ParamBuf {
                            data,
                            shape: vec![c, r],
                        },
                    );
                } else {
                    p.insert(key, w);
                }
            }
        }
    }

    for n in 1..=4 {
        if let Some(w) = try_take(raw, &format!("decoder.patch_embed_{n}.proj.weight")) {
            p.insert(format!("decoder.patch_embed_{n}.weight"), w);
        }
        if let Some(b) = try_take(raw, &format!("decoder.patch_embed_{n}.proj.bias")) {
            p.insert(format!("decoder.patch_embed_{n}.bias"), b);
        }
    }

    Ok(p)
}

pub fn apply_params(compiled: &mut rlx::CompiledGraph, params: &ParamMap) {
    for (name, buf) in params {
        compiled.set_param(name, &buf.data);
    }
}

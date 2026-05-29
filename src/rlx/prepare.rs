//! CPU-side token preparation: multi-scale conv, embeddings, RVQ.

#![allow(clippy::too_many_arguments)]

use super::weights::{ParamBuf, ParamMap};
use crate::config::Modality;
use ndarray::ArrayView2;

#[derive(Debug, Clone, Copy)]
struct BranchKernelConfig {
    kernel1: usize,
    pad1: usize,
    pool1_k: usize,
    kernel2: usize,
    pad2: usize,
    pool2_k: usize,
}

fn kernel_configs(modality: Modality) -> [BranchKernelConfig; 4] {
    match modality {
        Modality::EEG | Modality::ECG => [
            BranchKernelConfig {
                kernel1: 21,
                pad1: 10,
                pool1_k: 2,
                kernel2: 9,
                pad2: 4,
                pool2_k: 4,
            },
            BranchKernelConfig {
                kernel1: 15,
                pad1: 7,
                pool1_k: 2,
                kernel2: 7,
                pad2: 3,
                pool2_k: 4,
            },
            BranchKernelConfig {
                kernel1: 9,
                pad1: 4,
                pool1_k: 2,
                kernel2: 5,
                pad2: 2,
                pool2_k: 4,
            },
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
            BranchKernelConfig {
                kernel1: 51,
                pad1: 25,
                pool1_k: 2,
                kernel2: 25,
                pad2: 12,
                pool2_k: 4,
            },
            BranchKernelConfig {
                kernel1: 17,
                pad1: 8,
                pool1_k: 2,
                kernel2: 9,
                pad2: 4,
                pool2_k: 4,
            },
            BranchKernelConfig {
                kernel1: 8,
                pad1: 4,
                pool1_k: 2,
                kernel2: 4,
                pad2: 2,
                pool2_k: 4,
            },
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

/// Prepared CPU input batch (no Burn tensors).
#[derive(Clone, Debug)]
pub struct RlxInputBatch {
    /// Signal `[B, N, T]` row-major.
    pub signal: Vec<f32>,
    pub temporal_ix: Vec<i64>,
    pub spatial_ix: Vec<i64>,
    pub n_channels: usize,
    pub n_time_patches: usize,
}

pub fn build_batch(
    signal: Vec<f32>,
    channel_names: &[&str],
    n_time_patches: usize,
    max_n_patches: usize,
    n_channels: usize,
    _n_samples: usize,
    modality: Modality,
) -> RlxInputBatch {
    let (temp_ix, spat_ix) = crate::channels::create_embedding_ix(
        n_time_patches,
        max_n_patches,
        channel_names,
        modality,
    );
    RlxInputBatch {
        signal,
        temporal_ix: temp_ix,
        spatial_ix: spat_ix,
        n_channels,
        n_time_patches,
    }
}

fn gelu(x: f32) -> f32 {
    let scaled = (x as f64) * std::f64::consts::FRAC_1_SQRT_2;
    0.5 * x * (1.0 + libm::erf(scaled) as f32)
}

fn tanh_f(x: f32) -> f32 {
    libm::tanh(x as f64) as f32
}

fn linear(x: &[f32], w: &[f32], b: &[f32], in_d: usize, out_d: usize) -> Vec<f32> {
    let n = x.len() / in_d;
    let mut y = vec![0f32; n * out_d];
    for i in 0..n {
        for o in 0..out_d {
            let mut acc = b[o];
            for j in 0..in_d {
                acc += x[i * in_d + j] * w[j * out_d + o];
            }
            y[i * out_d + o] = acc;
        }
    }
    y
}

fn conv2d_nchw(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    n: usize,
    c_in: usize,
    h: usize,
    w_in: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    pad_h: usize,
    pad_w: usize,
) -> (usize, usize, Vec<f32>) {
    let h_out = h + 2 * pad_h - kh + 1;
    let w_out = w_in + 2 * pad_w - kw + 1;
    let mut y = vec![0f32; n * c_out * h_out * w_out];
    for ni in 0..n {
        for co in 0..c_out {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let mut acc = b[co];
                    for ci in 0..c_in {
                        for kh_i in 0..kh {
                            for kw_i in 0..kw {
                                let hi = ho + kh_i;
                                let wi = wo + kw_i;
                                if hi < pad_h || wi < pad_w || hi >= h + pad_h || wi >= w_in + pad_w
                                {
                                    continue;
                                }
                                let src_h = hi - pad_h;
                                let src_w = wi - pad_w;
                                let x_idx =
                                    ni * c_in * h * w_in + ci * h * w_in + src_h * w_in + src_w;
                                let w_idx = co * c_in * kh * kw + ci * kh * kw + kh_i * kw + kw_i;
                                acc = f32::mul_add(x[x_idx], w[w_idx], acc);
                            }
                        }
                    }
                    let y_idx = ni * c_out * h_out * w_out + co * h_out * w_out + ho * w_out + wo;
                    y[y_idx] = acc;
                }
            }
        }
    }
    (h_out, w_out, y)
}

fn avg_pool2d_nchw(
    x: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
) -> (usize, usize, Vec<f32>) {
    let h_out = h / kh;
    let w_out = w / kw;
    let mut y = vec![0f32; n * c * h_out * w_out];
    let area = (kh * kw) as f32;
    for ni in 0..n {
        for ci in 0..c {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let mut acc = 0.0f32;
                    for kh_i in 0..kh {
                        for kw_i in 0..kw {
                            let hi = ho * kh + kh_i;
                            let wi = wo * kw + kw_i;
                            let x_idx = ni * c * h * w + ci * h * w + hi * w + wi;
                            acc += x[x_idx];
                        }
                    }
                    let y_idx = ni * c * h_out * w_out + ci * h_out * w_out + ho * w_out + wo;
                    y[y_idx] = acc / area;
                }
            }
        }
    }
    (h_out, w_out, y)
}

fn group_norm_nchw(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    groups: usize,
    eps: f32,
) -> Vec<f32> {
    let mut y = x.to_vec();
    let gc = c / groups;
    for ni in 0..n {
        for g in 0..groups {
            let c0 = g * gc;
            let mut mean = 0.0f64;
            let mut count = 0usize;
            for ci in c0..c0 + gc {
                for hi in 0..h {
                    for wi in 0..w {
                        let idx = ni * c * h * w + ci * h * w + hi * w + wi;
                        mean += y[idx] as f64;
                        count += 1;
                    }
                }
            }
            mean /= count as f64;
            let mut var = 0.0f64;
            for ci in c0..c0 + gc {
                for hi in 0..h {
                    for wi in 0..w {
                        let idx = ni * c * h * w + ci * h * w + hi * w + wi;
                        let d = y[idx] as f64 - mean;
                        var += d * d;
                    }
                }
            }
            var /= count as f64;
            let inv = 1.0 / (var + eps as f64).sqrt();
            for ci in c0..c0 + gc {
                for hi in 0..h {
                    for wi in 0..w {
                        let idx = ni * c * h * w + ci * h * w + hi * w + wi;
                        let normed = ((y[idx] as f64 - mean) * inv) as f32;
                        y[idx] = normed * gamma[ci] + beta[ci];
                    }
                }
            }
        }
    }
    y
}

fn run_conv_branch(
    x: &[f32],
    params: &ParamMap,
    branch: usize,
    cfg: BranchKernelConfig,
    n: usize,
    na: usize,
    t: usize,
) -> Vec<f32> {
    let c_in = 1usize;
    let w1 = &params[&format!("conv1_{branch}.weight")];
    let b1 = &params[&format!("conv1_{branch}.bias")];
    let c_out = w1.shape[0];

    let (h1, w1_out, mut y) = conv2d_nchw(
        x,
        &w1.data,
        &b1.data,
        n,
        c_in,
        na,
        t,
        c_out,
        1,
        cfg.kernel1,
        0,
        cfg.pad1,
    );
    let gn1w = &params[&format!("norm1_{branch}.weight")];
    let gn1b = &params[&format!("norm1_{branch}.bias")];
    y = group_norm_nchw(&y, &gn1w.data, &gn1b.data, n, c_out, h1, w1_out, 4, 1e-5);
    for v in y.iter_mut() {
        *v = gelu(*v);
    }
    let (h1, w1_out, y) = avg_pool2d_nchw(&y, n, c_out, h1, w1_out, 1, cfg.pool1_k);

    let w2 = &params[&format!("conv2_{branch}.weight")];
    let b2 = &params[&format!("conv2_{branch}.bias")];
    let (h2, w2_out, mut y) = conv2d_nchw(
        &y,
        &w2.data,
        &b2.data,
        n,
        c_out,
        h1,
        w1_out,
        c_out,
        1,
        cfg.kernel2,
        0,
        cfg.pad2,
    );
    let gn2w = &params[&format!("norm2_{branch}.weight")];
    let gn2b = &params[&format!("norm2_{branch}.bias")];
    y = group_norm_nchw(&y, &gn2w.data, &gn2b.data, n, c_out, h2, w2_out, 4, 1e-5);
    for v in y.iter_mut() {
        *v = gelu(*v);
    }
    let (_, w2_out, y) = avg_pool2d_nchw(&y, n, c_out, h2, w2_out, 1, cfg.pool2_k);

    // [B, C, NA, T'] → [B, NA, T'*C]
    let t_prime = w2_out;
    let mut out = vec![0f32; n * na * t_prime * c_out];
    for ni in 0..n {
        for na_i in 0..na {
            for tp in 0..t_prime {
                for co in 0..c_out {
                    let src = ni * c_out * na * t_prime + co * na * t_prime + na_i * t_prime + tp;
                    let dst = ni * na * t_prime * c_out + na_i * t_prime * c_out + tp * c_out + co;
                    out[dst] = y[src];
                }
            }
        }
    }
    out
}

/// Multi-scale conv on CPU: `[B,N,A,T]` → 4 branch tensors `[B, seq, feat]`.
pub fn multi_scale_conv(
    signal: &[f32],
    params: &ParamMap,
    modality: Modality,
    b: usize,
    n: usize,
    a: usize,
    t: usize,
) -> [Vec<f32>; 4] {
    let na = n * a;
    let mut x = vec![0f32; b * na * t];
    for bi in 0..b {
        for ni in 0..n {
            for ai in 0..a {
                for ti in 0..t {
                    let src = bi * n * a * t + ni * a * t + ai * t + ti;
                    let dst = bi * na * t + (ni * a + ai) * t + ti;
                    x[dst] = signal[src];
                }
            }
        }
    }
    let x_nchw = {
        let mut v = vec![0f32; b * na * t];
        for bi in 0..b {
            for na_i in 0..na {
                for ti in 0..t {
                    let src = bi * na * t + na_i * t + ti;
                    v[bi * na * t + na_i * t + ti] = x[src];
                }
            }
        }
        v
    };
    let x4 = {
        let mut v = vec![0f32; b * na * t];
        for bi in 0..b {
            for na_i in 0..na {
                for ti in 0..t {
                    v[bi * 1 * na * t + na_i * t + ti] = x_nchw[bi * na * t + na_i * t + ti];
                }
            }
        }
        v
    };

    let cfgs = kernel_configs(modality);
    [
        run_conv_branch(&x4, params, 1, cfgs[0], b, na, t),
        run_conv_branch(&x4, params, 2, cfgs[1], b, na, t),
        run_conv_branch(&x4, params, 3, cfgs[2], b, na, t),
        run_conv_branch(&x4, params, 4, cfgs[3], b, na, t),
    ]
}

pub fn gather_embeddings_2d(table: &ParamBuf, indices: &[i64], b: usize, n: usize) -> Vec<f32> {
    let d = table.shape[1];
    let mut out = vec![0f32; b * n * d];
    for bi in 0..b {
        for j in 0..n {
            let idx = indices[bi * n + j] as usize;
            for k in 0..d {
                out[bi * n * d + j * d + k] = table.data[idx * d + k];
            }
        }
    }
    out
}

/// Add spatial/temporal embeddings and prepend CLS for one branch input.
pub fn prepare_branch_tokens(
    branch_x: &[f32],
    params: &ParamMap,
    embed_prefix: &str,
    temporal_ix: &[i64],
    spatial_ix: &[i64],
    b: usize,
    seq_len: usize,
    embed_dim: usize,
) -> Vec<f32> {
    let cls_key = format!("{embed_prefix}cls_token");
    let pos_key = format!("{embed_prefix}pos_embed");
    let time_key = format!("{embed_prefix}time_embed");

    let cls = &params[&cls_key];
    let pos = &params[&pos_key];
    let time = &params[&time_key];

    let branch_dim = branch_x.len() / (b * seq_len);
    debug_assert_eq!(
        branch_dim, embed_dim,
        "branch feature dim {branch_dim} must match embed_dim {embed_dim}"
    );
    let mut x = vec![0f32; b * (1 + seq_len) * embed_dim];

    for bi in 0..b {
        // CLS
        for d in 0..embed_dim {
            x[bi * (1 + seq_len) * embed_dim + d] = cls.data[d];
        }
        // Branch features
        for s in 0..seq_len {
            for d in 0..embed_dim {
                x[bi * (1 + seq_len) * embed_dim + (1 + s) * embed_dim + d] =
                    branch_x[bi * seq_len * branch_dim + s * branch_dim + d];
            }
        }
    }

    // Spatial: pad index 0 for CLS
    let mut spat_ix = vec![0i64; b * (1 + seq_len)];
    for bi in 0..b {
        spat_ix[bi * (1 + seq_len)] = 0;
        for s in 0..seq_len {
            spat_ix[bi * (1 + seq_len) + 1 + s] = spatial_ix[bi * seq_len + s];
        }
    }
    let spatial_emb = gather_embeddings_2d(pos, &spat_ix, b, 1 + seq_len);
    for i in 0..x.len() {
        x[i] += spatial_emb[i];
    }

    let temporal_emb = gather_embeddings_2d(time, temporal_ix, b, seq_len);
    for bi in 0..b {
        for s in 0..seq_len {
            for d in 0..embed_dim {
                x[bi * (1 + seq_len) * embed_dim + (1 + s) * embed_dim + d] +=
                    temporal_emb[bi * seq_len * embed_dim + s * embed_dim + d];
            }
        }
    }

    x
}

pub fn encode_head(
    branch_out: &[f32],
    params: &ParamMap,
    branch: usize,
    code_dim: usize,
) -> Vec<f32> {
    let w1 = &params[&format!("encode_task_layer_{branch}.0.weight")];
    let b1 = &params[&format!("encode_task_layer_{branch}.0.bias")];
    let in_d = w1.shape[0];
    let hid = w1.shape[1];
    let mut h = linear(branch_out, &w1.data, &b1.data, in_d, hid);
    for v in h.iter_mut() {
        *v = tanh_f(*v);
    }
    let w2 = &params[&format!("encode_task_layer_{branch}.2.weight")];
    let b2 = &params[&format!("encode_task_layer_{branch}.2.bias")];
    linear(&h, &w2.data, &b2.data, w2.shape[0], code_dim)
}

fn l2norm_rows(x: &mut [f32], d: usize) {
    let n = x.len() / d;
    for i in 0..n {
        let slice = &mut x[i * d..(i + 1) * d];
        let norm: f32 = slice.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        for v in slice.iter_mut() {
            *v /= norm;
        }
    }
}

/// RVQ encode one branch → indices per level.
pub fn rvq_encode(
    x: &[f32],
    params: &ParamMap,
    branch: usize,
    code_dim: usize,
    n_levels: usize,
) -> Vec<Vec<i64>> {
    let n = x.len() / code_dim;
    let mut residual = x.to_vec();
    let mut all_indices = Vec::with_capacity(n_levels);

    for l in 0..n_levels {
        let key = format!("quantize_{branch}.layers.{l}.embedding.weight");
        let codebook = &params[&key];
        let mut normed = residual.clone();
        l2norm_rows(&mut normed, code_dim);

        let mut indices = vec![0i64; n];
        for i in 0..n {
            let z = &normed[i * code_dim..(i + 1) * code_dim];
            let mut best = 0usize;
            let mut best_sim = f32::NEG_INFINITY;
            for (tok, cb) in codebook.data.chunks(code_dim).enumerate() {
                let sim: f32 = z.iter().zip(cb.iter()).map(|(a, b)| a * b).sum();
                if sim > best_sim {
                    best_sim = sim;
                    best = tok;
                }
            }
            indices[i] = best as i64;
        }

        for i in 0..n {
            let tok = indices[i] as usize;
            for d in 0..code_dim {
                residual[i * code_dim + d] -= codebook.data[tok * code_dim + d];
            }
        }
        all_indices.push(indices);
    }
    all_indices
}

pub fn num_quantizers(modality: Modality) -> usize {
    match modality {
        Modality::EEG | Modality::ECG => 8,
        Modality::EMG => 16,
    }
}

/// Reshape `[B, seq, C]` → `[B, C, n, w]` (NCHW) for RVQ / patch embed.
pub fn seq_to_nchw(x: &[f32], b: usize, seq: usize, c: usize, n: usize) -> Vec<f32> {
    let w = seq / n;
    let mut out = vec![0f32; b * c * n * w];
    for bi in 0..b {
        for ni in 0..n {
            for wi in 0..w {
                for ci in 0..c {
                    let src = bi * seq * c + (ni * w + wi) * c + ci;
                    let dst = bi * c * n * w + ci * n * w + ni * w + wi;
                    out[dst] = x[src];
                }
            }
        }
    }
    out
}

/// Reshape `[B, C, n, w]` → `[B, seq, C]`.
pub fn nchw_to_seq(x: &[f32], b: usize, c: usize, n: usize, w: usize) -> Vec<f32> {
    let seq = n * w;
    let mut out = vec![0f32; b * seq * c];
    for bi in 0..b {
        for ni in 0..n {
            for wi in 0..w {
                for ci in 0..c {
                    let src = bi * c * n * w + ci * n * w + ni * w + wi;
                    let dst = bi * seq * c + (ni * w + wi) * c + ci;
                    out[dst] = x[src];
                }
            }
        }
    }
    out
}

/// RVQ forward: residual quantize and return summed codebook vectors `[n, code_dim]`.
pub fn rvq_forward(
    x: &[f32],
    params: &ParamMap,
    branch: usize,
    code_dim: usize,
    n_levels: usize,
) -> Vec<f32> {
    let n = x.len() / code_dim;
    let mut residual = x.to_vec();
    let mut quantized_out = vec![0f32; x.len()];

    for l in 0..n_levels {
        let key = format!("quantize_{branch}.layers.{l}.embedding.weight");
        let codebook = &params[&key];
        let mut normed = residual.clone();
        l2norm_rows(&mut normed, code_dim);

        for i in 0..n {
            let z = &normed[i * code_dim..(i + 1) * code_dim];
            let mut best = 0usize;
            let mut best_sim = f32::NEG_INFINITY;
            for (tok, cb) in codebook.data.chunks(code_dim).enumerate() {
                let sim: f32 = z.iter().zip(cb.iter()).map(|(a, b)| a * b).sum();
                if sim > best_sim {
                    best_sim = sim;
                    best = tok;
                }
            }
            for d in 0..code_dim {
                let v = codebook.data[best * code_dim + d];
                quantized_out[i * code_dim + d] += v;
                residual[i * code_dim + d] -= v;
            }
        }
    }
    quantized_out
}

/// Decoder 1×1 conv patch embed: `[B, in, H, W]` → `[B, seq, embed_dim]`.
pub fn patch_embed_branch(
    x_nchw: &[f32],
    params: &ParamMap,
    branch: usize,
    in_ch: usize,
    h: usize,
    w: usize,
    embed_dim: usize,
) -> Vec<f32> {
    let b = 1usize;
    let n = b;
    let weight = &params[&format!("decoder.patch_embed_{branch}.weight")];
    let bias = &params[&format!("decoder.patch_embed_{branch}.bias")];
    let c_out = embed_dim;

    let (_, _, y) = conv2d_nchw(
        x_nchw,
        &weight.data,
        &bias.data,
        n,
        in_ch,
        h,
        w,
        c_out,
        1,
        1,
        0,
        0,
    );

    // [B, embed_dim, H, W] → [B, H*W, embed_dim]
    let seq = h * w;
    let mut out = vec![0f32; b * seq * embed_dim];
    for bi in 0..b {
        for si in 0..seq {
            let hi = si / w;
            let wi = si % w;
            for ei in 0..embed_dim {
                let src = bi * c_out * h * w + ei * h * w + hi * w + wi;
                let dst = bi * seq * embed_dim + si * embed_dim + ei;
                out[dst] = y[src];
            }
        }
    }
    out
}

/// Decode heads on concatenated branch features `[B, seq, 4*embed_dim]`.
pub fn decode_heads(
    concat: &[f32],
    params: &ParamMap,
    embed_dim: usize,
    decoder_out_dim: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let dec_in = 4 * embed_dim;
    let amp = {
        let w1 = &params["decode_task_layer_amplitude.0.weight"];
        let b1 = &params["decode_task_layer_amplitude.0.bias"];
        let mut h = linear(concat, &w1.data, &b1.data, dec_in, w1.shape[1]);
        for v in h.iter_mut() {
            *v = gelu(*v);
        }
        let w2 = &params["decode_task_layer_amplitude.2.weight"];
        let b2 = &params["decode_task_layer_amplitude.2.bias"];
        linear(&h, &w2.data, &b2.data, w2.shape[0], decoder_out_dim)
    };

    let sin = {
        let w1 = &params["decode_task_layer_angle_sin.0.weight"];
        let b1 = &params["decode_task_layer_angle_sin.0.bias"];
        let mut h = linear(concat, &w1.data, &b1.data, dec_in, w1.shape[1]);
        for v in h.iter_mut() {
            *v = tanh_f(*v);
        }
        let w2 = &params["decode_task_layer_angle_sin.2.weight"];
        let b2 = &params["decode_task_layer_angle_sin.2.bias"];
        let mut h2 = linear(&h, &w2.data, &b2.data, w2.shape[0], decoder_out_dim);
        for v in h2.iter_mut() {
            *v = tanh_f(*v);
        }
        h2
    };

    let cos = {
        let w1 = &params["decode_task_layer_angle_cos.0.weight"];
        let b1 = &params["decode_task_layer_angle_cos.0.bias"];
        let mut h = linear(concat, &w1.data, &b1.data, dec_in, w1.shape[1]);
        for v in h.iter_mut() {
            *v = tanh_f(*v);
        }
        let w2 = &params["decode_task_layer_angle_cos.2.weight"];
        let b2 = &params["decode_task_layer_angle_cos.2.bias"];
        let mut h2 = linear(&h, &w2.data, &b2.data, w2.shape[0], decoder_out_dim);
        for v in h2.iter_mut() {
            *v = tanh_f(*v);
        }
        h2
    };

    (amp, sin, cos)
}

fn build_dft_cos_matrix(t: usize) -> Vec<f32> {
    let mut data = vec![0f32; t * t];
    let inv_t = 2.0 * std::f32::consts::PI / t as f32;
    for k in 0..t {
        for j in 0..t {
            data[k * t + j] = (inv_t * k as f32 * j as f32).cos();
        }
    }
    data
}

fn build_dft_sin_matrix(t: usize) -> Vec<f32> {
    let mut data = vec![0f32; t * t];
    let inv_t = 2.0 * std::f32::consts::PI / t as f32;
    for k in 0..t {
        for j in 0..t {
            data[k * t + j] = (inv_t * k as f32 * j as f32).sin();
        }
    }
    data
}

fn transpose_square(m: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            out[j * n + i] = m[i * n + j];
        }
    }
    out
}

fn matmul_2d(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let a_view = ArrayView2::from_shape((m, k), a).expect("matmul a shape");
    let b_view = ArrayView2::from_shape((k, n), b).expect("matmul b shape");
    a_view.dot(&b_view).into_raw_vec_and_offset().0
}

/// FFT decomposition for patched signal `[B, N, A, T]`.
pub fn compute_fft_components(
    x: &[f32],
    b: usize,
    n: usize,
    a: usize,
    t: usize,
) -> (Vec<f32>, f32, f32, Vec<f32>, Vec<f32>) {
    #[cfg(all(feature = "burn", feature = "ndarray"))]
    {
        return fft_burn::compute_fft_components(x, b, n, a, t);
    }
    #[cfg(not(all(feature = "burn", feature = "ndarray")))]
    {
        compute_fft_components_cpu(x, b, n, a, t)
    }
}

fn compute_fft_components_cpu(
    x: &[f32],
    b: usize,
    n: usize,
    a: usize,
    t: usize,
) -> (Vec<f32>, f32, f32, Vec<f32>, Vec<f32>) {
    let dft_cos = build_dft_cos_matrix(t);
    let dft_sin_t = transpose_square(&build_dft_sin_matrix(t), t);
    let flat_len = b * n * a;
    let mut log_amp = vec![0f32; b * n * a * t];
    let mut sin_phase = vec![0f32; b * n * a * t];
    let mut cos_phase = vec![0f32; b * n * a * t];

    let fft_real = matmul_2d(x, &dft_cos, flat_len, t, t);
    let fft_imag = matmul_2d(x, &dft_sin_t, flat_len, t, t);
    for i in 0..flat_len {
        for k in 0..t {
            let re = fft_real[i * t + k];
            let im = fft_imag[i * t + k];
            let amp = (re * re + im * im).sqrt().max(1e-10);
            let off = i * t + k;
            log_amp[off] = amp.ln_1p();
            cos_phase[off] = re / amp;
            sin_phase[off] = -im / amp;
        }
    }

    let (normed, mean, std) = std_norm_with_stats(&log_amp, b, n, a, t);
    (normed, mean, std, sin_phase, cos_phase)
}

fn std_norm_with_stats(x: &[f32], b: usize, n: usize, a: usize, t: usize) -> (Vec<f32>, f32, f32) {
    debug_assert_eq!(x.len(), b * n * a * t);
    if b == 1 {
        let len = n * a * t;
        let mean = x[..len].iter().sum::<f32>() / len as f32;
        let var = x[..len]
            .iter()
            .map(|v| {
                let d = *v - mean;
                d * d
            })
            .sum::<f32>()
            / len as f32;
        let std = (var + 1e-8).sqrt();
        let mut out = vec![0f32; len];
        for (i, v) in x[..len].iter().enumerate() {
            out[i] = (*v - mean) / std;
        }
        return (out, mean, std);
    }

    let plane = n * a * t;
    let mut out = vec![0f32; b * plane];
    let mut mean_acc = 0f32;
    let mut std_acc = 0f32;
    for bi in 0..b {
        let slice = &x[bi * plane..(bi + 1) * plane];
        let mean = slice.iter().sum::<f32>() / plane as f32;
        let var = slice
            .iter()
            .map(|v| {
                let d = *v - mean;
                d * d
            })
            .sum::<f32>()
            / plane as f32;
        let std = (var + 1e-8).sqrt();
        for (i, v) in slice.iter().enumerate() {
            out[bi * plane + i] = (*v - mean) / std;
        }
        mean_acc += mean;
        std_acc += std;
    }
    (out, mean_acc / b as f32, std_acc / b as f32)
}

pub fn std_norm_4d(x: &[f32], b: usize, n: usize, a: usize, t: usize) -> Vec<f32> {
    #[cfg(all(feature = "burn", feature = "ndarray"))]
    {
        return fft_burn::std_norm_4d(x, b, n, a, t);
    }
    #[cfg(not(all(feature = "burn", feature = "ndarray")))]
    {
        let (normed, _, _) = std_norm_with_stats(x, b, n, a, t);
        normed
    }
}

pub fn reconstruct_signal(
    xrec_amp: &[f32],
    xrec_sin: &[f32],
    xrec_cos: &[f32],
    amp_mean: f32,
    amp_std: f32,
    b: usize,
    n: usize,
    a: usize,
    t: usize,
) -> Vec<f32> {
    #[cfg(all(feature = "burn", feature = "ndarray"))]
    {
        return fft_burn::reconstruct_signal(
            xrec_amp, xrec_sin, xrec_cos, amp_mean, amp_std, b, n, a, t,
        );
    }
    #[cfg(not(all(feature = "burn", feature = "ndarray")))]
    {
        reconstruct_signal_cpu(xrec_amp, xrec_sin, xrec_cos, amp_mean, amp_std, b, n, a, t)
    }
}

fn reconstruct_signal_cpu(
    xrec_amp: &[f32],
    xrec_sin: &[f32],
    xrec_cos: &[f32],
    amp_mean: f32,
    amp_std: f32,
    b: usize,
    n: usize,
    a: usize,
    t: usize,
) -> Vec<f32> {
    let seq = n * a;
    let m = b * seq;
    let idft_cos = build_dft_cos_matrix(t);
    let idft_sin_t = transpose_square(&build_dft_sin_matrix(t), t);
    let mut fft_real = vec![0f32; m * t];
    let mut fft_imag = vec![0f32; m * t];

    for bi in 0..b {
        for si in 0..seq {
            for ti in 0..t {
                let idx = bi * seq * t + si * t + ti;
                let row = bi * seq + si;
                let ustd = xrec_amp[idx] * amp_std + amp_mean;
                let amp = ustd.exp() - 1.0;
                fft_real[row * t + ti] = amp * xrec_cos[idx];
                fft_imag[row * t + ti] = amp * xrec_sin[idx];
            }
        }
    }

    let real_out = matmul_2d(&fft_real, &idft_cos, m, t, t);
    let imag_out = matmul_2d(&fft_imag, &idft_sin_t, m, t, t);
    let mut signal = vec![0f32; m * t];
    for i in 0..m * t {
        signal[i] = (real_out[i] - imag_out[i]) / t as f32;
    }
    signal
}

#[cfg(all(feature = "burn", feature = "ndarray"))]
mod fft_burn {
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::NdArray;
    use burn::prelude::*;
    use burn::tensor::TensorData;

    type B = NdArray<f32>;

    fn dft_cos(t: usize, device: &NdArrayDevice) -> Tensor<B, 2> {
        let mut data = vec![0.0f32; t * t];
        let inv_t = 2.0 * std::f32::consts::PI / t as f32;
        for k in 0..t {
            for j in 0..t {
                data[k * t + j] = (inv_t * k as f32 * j as f32).cos();
            }
        }
        Tensor::<B, 2>::from_data(TensorData::new(data, [t, t]), device)
    }

    fn dft_sin(t: usize, device: &NdArrayDevice) -> Tensor<B, 2> {
        let mut data = vec![0.0f32; t * t];
        let inv_t = 2.0 * std::f32::consts::PI / t as f32;
        for k in 0..t {
            for j in 0..t {
                data[k * t + j] = (inv_t * k as f32 * j as f32).sin();
            }
        }
        Tensor::<B, 2>::from_data(TensorData::new(data, [t, t]), device)
    }

    pub fn compute_fft_components(
        x: &[f32],
        b: usize,
        n: usize,
        a: usize,
        t: usize,
    ) -> (Vec<f32>, f32, f32, Vec<f32>, Vec<f32>) {
        let device = NdArrayDevice::Cpu;
        let dft_cos = dft_cos(t, &device);
        let dft_sin = dft_sin(t, &device);
        let x_t = Tensor::<B, 4>::from_data(TensorData::new(x.to_vec(), [b, n, a, t]), &device);
        let x_flat = x_t.reshape([b * n * a, t]);
        let fft_real = x_flat.clone().matmul(dft_cos.transpose());
        let fft_imag = x_flat.matmul(dft_sin.transpose());
        let amp = (fft_real.clone().powf_scalar(2.0) + fft_imag.clone().powf_scalar(2.0))
            .sqrt()
            .clamp_min(1e-10);
        let log_amp = amp.clone().log1p().reshape([b, n, a, t]);
        let cos_phase = fft_real / amp.clone();
        let sin_phase = fft_imag.neg() / amp;

        let mean = log_amp.clone().mean_dim(1).mean_dim(2).mean_dim(3);
        let diff = log_amp.clone() - mean.clone();
        let var = (diff.clone() * diff).mean_dim(1).mean_dim(2).mean_dim(3);
        let std = (var + 1e-8).sqrt();
        let normed = (log_amp - mean.clone()) / std.clone();

        let mean_v = mean.into_data().to_vec::<f32>().unwrap()[0];
        let std_v = std.into_data().to_vec::<f32>().unwrap()[0];
        (
            normed.into_data().to_vec::<f32>().unwrap(),
            mean_v,
            std_v,
            sin_phase.into_data().to_vec::<f32>().unwrap(),
            cos_phase.into_data().to_vec::<f32>().unwrap(),
        )
    }

    pub fn reconstruct_signal(
        xrec_amp: &[f32],
        xrec_sin: &[f32],
        xrec_cos: &[f32],
        amp_mean: f32,
        amp_std: f32,
        b: usize,
        n: usize,
        a: usize,
        t: usize,
    ) -> Vec<f32> {
        let device = NdArrayDevice::Cpu;
        let seq = n * a;
        let amp =
            Tensor::<B, 3>::from_data(TensorData::new(xrec_amp.to_vec(), [b, seq, t]), &device);
        let sin =
            Tensor::<B, 3>::from_data(TensorData::new(xrec_sin.to_vec(), [b, seq, t]), &device);
        let cos =
            Tensor::<B, 3>::from_data(TensorData::new(xrec_cos.to_vec(), [b, seq, t]), &device);
        let mean_t =
            Tensor::<B, 4>::from_data(TensorData::new(vec![amp_mean], [b, 1, 1, 1]), &device);
        let std_t =
            Tensor::<B, 4>::from_data(TensorData::new(vec![amp_std], [b, 1, 1, 1]), &device);

        let amp_4d = amp.reshape([b, n, a, t]);
        let ustd = amp_4d * std_t + mean_t;
        let ustd = ustd.exp() - 1.0;
        let ustd = ustd.reshape([b, seq, t]);
        let fft_real = ustd.clone() * cos;
        let fft_imag = ustd * sin;

        let idft_cos = dft_cos(t, &device);
        let idft_sin = dft_sin(t, &device);
        let fft_real_flat = fft_real.reshape([b * seq, t]);
        let fft_imag_flat = fft_imag.reshape([b * seq, t]);
        let signal = (fft_real_flat.matmul(idft_cos.transpose())
            - fft_imag_flat.matmul(idft_sin.transpose()))
        .mul_scalar(1.0 / t as f32);
        signal
            .reshape([b, seq, t])
            .into_data()
            .to_vec::<f32>()
            .unwrap()
    }

    pub fn std_norm_4d(x: &[f32], b: usize, n: usize, a: usize, t: usize) -> Vec<f32> {
        let device = NdArrayDevice::Cpu;
        let x_t = Tensor::<B, 4>::from_data(TensorData::new(x.to_vec(), [b, n, a, t]), &device);
        let mean = x_t.clone().mean_dim(1).mean_dim(2).mean_dim(3);
        let diff = x_t.clone() - mean.clone();
        let var = (diff.clone() * diff).mean_dim(1).mean_dim(2).mean_dim(3);
        let std = (var + 1e-8).sqrt();
        let normed = (x_t - mean) / std;
        normed
            .reshape([b, n * a, t])
            .into_data()
            .to_vec::<f32>()
            .unwrap()
    }
}

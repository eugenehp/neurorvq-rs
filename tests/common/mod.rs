//! Shared helpers for integration tests.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const SKIP_NO_WEIGHTS: &str =
    "SKIP: set NEURORVQ_WEIGHTS to a converted tokenizer .safetensors file";

pub fn find_weights() -> Option<PathBuf> {
    if let Ok(w) = std::env::var("NEURORVQ_WEIGHTS") {
        let p = PathBuf::from(w);
        return p.exists().then_some(p);
    }
    None
}

pub fn test_config_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("flags/NeuroRVQ_EEG_v1.yml")
}

pub fn diff_max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

pub fn diff_rmse(a: &[f32], b: &[f32]) -> f64 {
    let sum: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| {
            let d = (*x - *y) as f64;
            d * d
        })
        .sum();
    (sum / a.len() as f64).sqrt()
}

/// Deterministic biosignal for parity tests.
pub fn synthetic_signal(n_ch: usize, n_samples: usize) -> Vec<f32> {
    (0..n_ch * n_samples)
        .map(|i| ((i as f32 * 0.013).sin() + (i as f32 * 0.007).cos()) * 0.01)
        .collect()
}

/// Burn-compatible FFT stats for patched `[B, N, A, T]` (reference for parity debugging).
pub fn burn_fft_stats(x: &[f32], b: usize, n: usize, a: usize, t: usize) -> (f32, f32) {
    let dft_cos = build_dft_cos(t);
    let dft_sin_t = transpose_sq(&build_dft_sin(t), t);
    let flat_len = b * n * a;
    let mut log_amp = vec![0f32; flat_len * t];
    for i in 0..flat_len {
        let row = &x[i * t..(i + 1) * t];
        let fft_real = matmul_row(row, &dft_cos, t);
        let fft_imag = matmul_row(row, &dft_sin_t, t);
        for k in 0..t {
            let re = fft_real[k];
            let im = fft_imag[k];
            let amp = (re * re + im * im).sqrt().max(1e-10);
            log_amp[i * t + k] = amp.ln_1p();
        }
    }
    let len = log_amp.len();
    let mean = log_amp.iter().sum::<f32>() / len as f32;
    let var = log_amp
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f32>()
        / len as f32;
    (mean, (var + 1e-8).sqrt())
}

fn build_dft_cos(t: usize) -> Vec<f32> {
    let mut data = vec![0f32; t * t];
    let inv_t = 2.0 * std::f32::consts::PI / t as f32;
    for k in 0..t {
        for j in 0..t {
            data[k * t + j] = (inv_t * k as f32 * j as f32).cos();
        }
    }
    data
}

fn build_dft_sin(t: usize) -> Vec<f32> {
    let mut data = vec![0f32; t * t];
    let inv_t = 2.0 * std::f32::consts::PI / t as f32;
    for k in 0..t {
        for j in 0..t {
            data[k * t + j] = (inv_t * k as f32 * j as f32).sin();
        }
    }
    data
}

fn transpose_sq(m: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            out[j * n + i] = m[i * n + j];
        }
    }
    out
}

fn matmul_row(row: &[f32], m: &[f32], t: usize) -> Vec<f32> {
    let mut out = vec![0f32; t];
    for j in 0..t {
        let mut acc = 0f32;
        for p in 0..t {
            acc += row[p] * m[p * t + j];
        }
        out[j] = acc;
    }
    out
}

pub fn patch_signal(signal: &[f32], n: usize, a: usize, t: usize) -> Vec<f32> {
    let mut patched = vec![0f32; n * a * t];
    for ni in 0..n {
        for ai in 0..a {
            for ti in 0..t {
                let src = ni * a * t + ai * t + ti;
                let dst = ni * a * t + ai * t + ti;
                patched[dst] = signal[src];
            }
        }
    }
    patched
}

/// Burn-compatible `reconstruct_signal` (f32 DFT matmul, matches `tokenizer.rs`).
pub fn burn_reconstruct_signal(
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
    let idft_cos = build_dft_cos(t);
    let idft_sin_t = transpose_sq(&build_dft_sin(t), t);
    let mut signal = vec![0f32; b * seq * t];

    for bi in 0..b {
        for si in 0..seq {
            let mut fft_real = vec![0f32; t];
            let mut fft_imag = vec![0f32; t];
            for ti in 0..t {
                let idx = bi * seq * t + si * t + ti;
                let ustd = xrec_amp[idx] * amp_std + amp_mean;
                let amp = ustd.exp() - 1.0;
                fft_real[ti] = amp * xrec_cos[idx];
                fft_imag[ti] = amp * xrec_sin[idx];
            }
            // Burn: real @ cos^T - imag @ sin^T (cos symmetric; sin uses transpose)
            let real_out = matmul_row(&fft_real, &idft_cos, t);
            let imag_out = matmul_row(&fft_imag, &idft_sin_t, t);
            for ti in 0..t {
                signal[bi * seq * t + si * t + ti] = (real_out[ti] - imag_out[ti]) / t as f32;
            }
        }
    }
    signal
}

#[cfg(feature = "burn")]
pub mod burn_export {
    use std::collections::HashMap;
    use std::path::Path;

    use burn::nn::{GroupNorm, Linear};
    use burn::prelude::*;
    use burn::tensor::Tensor;
    use safetensors::serialize;
    use safetensors::tensor::{Dtype, TensorView};

    use neurorvq_rs::model::encoder_block::TransformerBlock;
    use neurorvq_rs::model::foundation::NeuroRVQFM;
    use neurorvq_rs::model::norm::NeuroLayerNorm;
    use neurorvq_rs::model::patch_embed::PatchEmbed;
    use neurorvq_rs::model::rvq::ResidualVQ;
    use neurorvq_rs::model::tokenizer::NeuroRVQTokenizer;

    fn f32_bytes(data: &[f32]) -> Vec<u8> {
        data.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn insert_f32(
        out: &mut HashMap<String, (Vec<u8>, Vec<usize>, Dtype)>,
        key: &str,
        data: Vec<f32>,
        shape: Vec<usize>,
    ) {
        out.insert(key.to_string(), (f32_bytes(&data), shape, Dtype::F32));
    }

    fn tensor_vec<B: Backend, const D: usize>(t: Tensor<B, D>) -> (Vec<f32>, Vec<usize>) {
        let shape = t.dims().to_vec();
        (t.into_data().to_vec::<f32>().unwrap(), shape)
    }

    fn export_linear_pt<B: Backend>(
        out: &mut HashMap<String, (Vec<u8>, Vec<usize>, Dtype)>,
        key: &str,
        linear: &Linear<B>,
    ) {
        let w = linear.weight.val();
        let [in_d, out_d] = w.dims();
        let wv = w.into_data().to_vec::<f32>().unwrap();
        let mut pt = vec![0f32; in_d * out_d];
        for i in 0..in_d {
            for j in 0..out_d {
                pt[j * in_d + i] = wv[i * out_d + j];
            }
        }
        insert_f32(out, &format!("{key}.weight"), pt, vec![out_d, in_d]);
        if let Some(b) = &linear.bias {
            let (data, shape) = tensor_vec(b.val());
            insert_f32(out, &format!("{key}.bias"), data, shape);
        }
    }

    fn export_layernorm<B: Backend>(
        out: &mut HashMap<String, (Vec<u8>, Vec<usize>, Dtype)>,
        key: &str,
        norm: &NeuroLayerNorm<B>,
    ) {
        let (gw, sh) = tensor_vec(norm.inner.gamma.val());
        insert_f32(out, &format!("{key}.weight"), gw, sh);
        if let Some(ref b) = norm.inner.beta {
            let (gb, sh) = tensor_vec(b.val());
            insert_f32(out, &format!("{key}.bias"), gb, sh);
        }
    }

    fn export_conv2d_pt<B: Backend>(
        out: &mut HashMap<String, (Vec<u8>, Vec<usize>, Dtype)>,
        key: &str,
        w: Tensor<B, 4>,
        b: Option<Tensor<B, 1>>,
    ) {
        let (data, shape) = tensor_vec(w);
        insert_f32(out, &format!("{key}.weight"), data, shape);
        if let Some(bias) = b {
            let (data, shape) = tensor_vec(bias);
            insert_f32(out, &format!("{key}.bias"), data, shape);
        }
    }

    fn export_groupnorm<B: Backend>(
        out: &mut HashMap<String, (Vec<u8>, Vec<usize>, Dtype)>,
        key: &str,
        gn: &GroupNorm<B>,
    ) {
        if let Some(ref g) = gn.gamma {
            let (data, shape) = tensor_vec(g.val());
            insert_f32(out, &format!("{key}.weight"), data, shape);
        }
        if let Some(ref b) = gn.beta {
            let (data, shape) = tensor_vec(b.val());
            insert_f32(out, &format!("{key}.bias"), data, shape);
        }
    }

    fn export_qkv_pt<B: Backend>(
        out: &mut HashMap<String, (Vec<u8>, Vec<usize>, Dtype)>,
        prefix: &str,
        block: &TransformerBlock<B>,
    ) {
        let w = block.attn.qkv.weight.val();
        let [in_d, out_d] = w.dims();
        let wv = w.into_data().to_vec::<f32>().unwrap();
        let mut pt = vec![0f32; out_d * in_d];
        for i in 0..out_d {
            for j in 0..in_d {
                pt[i * in_d + j] = wv[j * out_d + i];
            }
        }
        insert_f32(
            out,
            &format!("{prefix}.attn.qkv.weight"),
            pt,
            vec![out_d, in_d],
        );
        if let Some(ref qb) = block.attn.q_bias {
            let (data, shape) = tensor_vec(qb.val());
            insert_f32(out, &format!("{prefix}.attn.q_bias"), data, shape);
        }
        if let Some(ref vb) = block.attn.v_bias {
            let (data, shape) = tensor_vec(vb.val());
            insert_f32(out, &format!("{prefix}.attn.v_bias"), data, shape);
        }
        export_linear_pt(out, &format!("{prefix}.attn.proj"), &block.attn.proj);
        if let Some(ref qn) = block.attn.q_norm {
            export_layernorm(out, &format!("{prefix}.attn.q_norm"), qn);
        }
        if let Some(ref kn) = block.attn.k_norm {
            export_layernorm(out, &format!("{prefix}.attn.k_norm"), kn);
        }
    }

    fn export_block<B: Backend>(
        out: &mut HashMap<String, (Vec<u8>, Vec<usize>, Dtype)>,
        prefix: &str,
        block: &TransformerBlock<B>,
    ) {
        export_layernorm(out, &format!("{prefix}.norm1"), &block.norm1);
        export_qkv_pt(out, prefix, block);
        export_layernorm(out, &format!("{prefix}.norm2"), &block.norm2);
        export_linear_pt(out, &format!("{prefix}.mlp.fc1"), &block.mlp.fc1);
        export_linear_pt(out, &format!("{prefix}.mlp.fc2"), &block.mlp.fc2);
    }

    fn export_fm<B: Backend>(
        out: &mut HashMap<String, (Vec<u8>, Vec<usize>, Dtype)>,
        prefix: &str,
        fm: &NeuroRVQFM<B>,
    ) {
        let (cls, sh) = tensor_vec(fm.cls_token.val());
        insert_f32(out, &format!("{prefix}cls_token"), cls, sh);
        let (pos, sh) = tensor_vec(fm.pos_embed.val());
        insert_f32(out, &format!("{prefix}pos_embed"), pos, sh);
        let (tim, sh) = tensor_vec(fm.time_embed.val());
        insert_f32(out, &format!("{prefix}time_embed"), tim, sh);

        for (i, block) in fm.blocks.iter().enumerate() {
            export_block(out, &format!("{prefix}blocks.{i}"), block);
        }
        for i in 1..=4 {
            export_layernorm(
                out,
                &format!("{prefix}fc_norm_{i}"),
                match i {
                    1 => &fm.fc_norm_1,
                    2 => &fm.fc_norm_2,
                    3 => &fm.fc_norm_3,
                    _ => &fm.fc_norm_4,
                },
            );
            export_linear_pt(
                out,
                &format!("{prefix}head_{i}"),
                match i {
                    1 => &fm.head_1,
                    2 => &fm.head_2,
                    3 => &fm.head_3,
                    _ => &fm.head_4,
                },
            );
        }
    }

    fn export_patch_embed<B: Backend>(
        out: &mut HashMap<String, (Vec<u8>, Vec<usize>, Dtype)>,
        key: &str,
        pe: &PatchEmbed<B>,
    ) {
        export_conv2d_pt(
            out,
            key,
            pe.proj.weight.val(),
            pe.proj.bias.as_ref().map(|b| b.val()),
        );
    }

    fn export_rvq<B: Backend>(
        out: &mut HashMap<String, (Vec<u8>, Vec<usize>, Dtype)>,
        prefix: &str,
        rvq: &ResidualVQ<B>,
    ) {
        for (i, layer) in rvq.layers.iter().enumerate() {
            let (data, shape) = tensor_vec(layer.weight.val());
            insert_f32(
                out,
                &format!("{prefix}.layers.{i}.embedding.weight"),
                data,
                shape,
            );
        }
    }

    pub fn export_tokenizer<B: Backend>(
        model: &NeuroRVQTokenizer<B>,
        path: &Path,
    ) -> anyhow::Result<()> {
        let mut raw: HashMap<String, (Vec<u8>, Vec<usize>, Dtype)> = HashMap::new();

        if let Some(ref conv) = model.encoder.multi_scale_conv {
            for (branch, i) in [
                (&conv.branch1, 1),
                (&conv.branch2, 2),
                (&conv.branch3, 3),
                (&conv.branch4, 4),
            ] {
                let pfx = "encoder.patch_embed";
                export_conv2d_pt(
                    &mut raw,
                    &format!("{pfx}.conv1_{i}"),
                    branch.conv1.weight.val(),
                    branch.conv1.bias.as_ref().map(|b| b.val()),
                );
                export_groupnorm(&mut raw, &format!("{pfx}.norm1_{i}"), &branch.norm1);
                export_conv2d_pt(
                    &mut raw,
                    &format!("{pfx}.conv2_{i}"),
                    branch.conv2.weight.val(),
                    branch.conv2.bias.as_ref().map(|b| b.val()),
                );
                export_groupnorm(&mut raw, &format!("{pfx}.norm2_{i}"), &branch.norm2);
            }
        }
        export_fm(&mut raw, "encoder.", &model.encoder);

        if let Some(ref pe1) = model.decoder.patch_embed_1 {
            export_patch_embed(&mut raw, "decoder.patch_embed_1.proj", pe1);
        }
        if let Some(ref pe2) = model.decoder.patch_embed_2 {
            export_patch_embed(&mut raw, "decoder.patch_embed_2.proj", pe2);
        }
        if let Some(ref pe3) = model.decoder.patch_embed_3 {
            export_patch_embed(&mut raw, "decoder.patch_embed_3.proj", pe3);
        }
        if let Some(ref pe4) = model.decoder.patch_embed_4 {
            export_patch_embed(&mut raw, "decoder.patch_embed_4.proj", pe4);
        }
        export_fm(&mut raw, "decoder.", &model.decoder);

        export_linear_pt(&mut raw, "encode_task_layer_1.0", &model.encode_head_1_fc1);
        export_linear_pt(&mut raw, "encode_task_layer_1.2", &model.encode_head_1_fc2);
        export_linear_pt(&mut raw, "encode_task_layer_2.0", &model.encode_head_2_fc1);
        export_linear_pt(&mut raw, "encode_task_layer_2.2", &model.encode_head_2_fc2);
        export_linear_pt(&mut raw, "encode_task_layer_3.0", &model.encode_head_3_fc1);
        export_linear_pt(&mut raw, "encode_task_layer_3.2", &model.encode_head_3_fc2);
        export_linear_pt(&mut raw, "encode_task_layer_4.0", &model.encode_head_4_fc1);
        export_linear_pt(&mut raw, "encode_task_layer_4.2", &model.encode_head_4_fc2);

        export_linear_pt(
            &mut raw,
            "decode_task_layer_amplitude.0",
            &model.decode_amp_fc1,
        );
        export_linear_pt(
            &mut raw,
            "decode_task_layer_amplitude.2",
            &model.decode_amp_fc2,
        );
        export_linear_pt(
            &mut raw,
            "decode_task_layer_angle_sin.0",
            &model.decode_sin_fc1,
        );
        export_linear_pt(
            &mut raw,
            "decode_task_layer_angle_sin.2",
            &model.decode_sin_fc2,
        );
        export_linear_pt(
            &mut raw,
            "decode_task_layer_angle_cos.0",
            &model.decode_cos_fc1,
        );
        export_linear_pt(
            &mut raw,
            "decode_task_layer_angle_cos.2",
            &model.decode_cos_fc2,
        );

        export_rvq(&mut raw, "quantize_1", &model.quantize_1);
        export_rvq(&mut raw, "quantize_2", &model.quantize_2);
        export_rvq(&mut raw, "quantize_3", &model.quantize_3);
        export_rvq(&mut raw, "quantize_4", &model.quantize_4);

        let views: HashMap<String, TensorView> = raw
            .iter()
            .map(|(k, (bytes, shape, dtype))| {
                (
                    k.clone(),
                    TensorView::new(*dtype, shape.clone(), bytes).expect("tensor view"),
                )
            })
            .collect();
        let bytes = serialize(views, None)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Export standalone FM weights (unprefixed keys) from a tokenizer encoder.
    pub fn export_foundation_model<B: Backend>(
        model: &NeuroRVQTokenizer<B>,
        path: &Path,
    ) -> anyhow::Result<()> {
        let mut raw: HashMap<String, (Vec<u8>, Vec<usize>, Dtype)> = HashMap::new();

        if let Some(ref conv) = model.encoder.multi_scale_conv {
            for (branch, i) in [
                (&conv.branch1, 1),
                (&conv.branch2, 2),
                (&conv.branch3, 3),
                (&conv.branch4, 4),
            ] {
                let pfx = "patch_embed";
                export_conv2d_pt(
                    &mut raw,
                    &format!("{pfx}.conv1_{i}"),
                    branch.conv1.weight.val(),
                    branch.conv1.bias.as_ref().map(|b| b.val()),
                );
                export_groupnorm(&mut raw, &format!("{pfx}.norm1_{i}"), &branch.norm1);
                export_conv2d_pt(
                    &mut raw,
                    &format!("{pfx}.conv2_{i}"),
                    branch.conv2.weight.val(),
                    branch.conv2.bias.as_ref().map(|b| b.val()),
                );
                export_groupnorm(&mut raw, &format!("{pfx}.norm2_{i}"), &branch.norm2);
            }
        }
        export_fm(&mut raw, "", &model.encoder);

        let views: HashMap<String, TensorView> = raw
            .iter()
            .map(|(k, (bytes, shape, dtype))| {
                (
                    k.clone(),
                    TensorView::new(*dtype, shape.clone(), bytes).expect("tensor view"),
                )
            })
            .collect();
        let bytes = serialize(views, None)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }
}

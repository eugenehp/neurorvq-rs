#![cfg(feature = "burn")]

use burn::nn::Linear;
use burn::prelude::*;
/// Load pretrained NeuroRVQ weights from PyTorch .pt files converted to safetensors.
///
/// NeuroRVQ ships weights as .pt (pickle) files. To use in Rust, first convert
/// to safetensors format using the provided Python script, then load here.
///
/// Alternatively, this module can load raw f32 tensors from a safetensors file
/// with PyTorch-style state_dict keys.
use std::collections::HashMap;

use crate::model::norm::NeuroLayerNorm;
use crate::model::quantizer::NormVectorQuantizer;
use crate::model::tokenizer::NeuroRVQTokenizer;

// ── WeightMap ─────────────────────────────────────────────────────────────────

pub struct WeightMap {
    pub tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl WeightMap {
    /// Load all tensors from a safetensors file.
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        let st = safetensors::SafeTensors::deserialize(&bytes)?;
        let mut tensors = HashMap::with_capacity(st.len());

        for (key, view) in st.tensors() {
            let shape: Vec<usize> = view.shape().to_vec();
            let data = view.data();

            let f32s: Vec<f32> = match view.dtype() {
                safetensors::Dtype::BF16 => data
                    .chunks_exact(2)
                    .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect(),
                safetensors::Dtype::F16 => data
                    .chunks_exact(2)
                    .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect(),
                safetensors::Dtype::F32 => data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect(),
                other => anyhow::bail!("unsupported dtype {:?} for key {key}", other),
            };

            tensors.insert(key.to_string(), (f32s, shape));
        }

        Ok(Self { tensors })
    }

    /// Take a tensor by key, removing it from the map.
    pub fn take<B: Backend, const N: usize>(
        &mut self,
        key: &str,
        device: &B::Device,
    ) -> anyhow::Result<Tensor<B, N>> {
        let (data, shape) = self
            .tensors
            .remove(key)
            .ok_or_else(|| anyhow::anyhow!("weight key not found: {key}"))?;
        if shape.len() != N {
            anyhow::bail!("rank mismatch for {key}: expected {N}, got {}", shape.len());
        }
        Ok(Tensor::<B, N>::from_data(
            TensorData::new(data, shape),
            device,
        ))
    }

    pub fn has(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    pub fn print_keys(&self) {
        let mut keys: Vec<&str> = self.tensors.keys().map(String::as_str).collect();
        keys.sort();
        for k in keys {
            let (_, s) = &self.tensors[k];
            println!("  {k:80}  {s:?}");
        }
    }
}

// ── Weight assignment helpers ─────────────────────────────────────────────────

fn set_linear_wb<B: Backend>(linear: &mut Linear<B>, w: Tensor<B, 2>, b: Option<Tensor<B, 1>>) {
    // PyTorch Linear stores [out, in]; burn stores [in, out] (transposed in forward)
    linear.weight = linear.weight.clone().map(|_| w.transpose());
    if let (Some(ref bias), Some(b_val)) = (&linear.bias, b) {
        linear.bias = Some(bias.clone().map(|_| b_val));
    }
}

fn set_linear_no_bias<B: Backend>(linear: &mut Linear<B>, w: Tensor<B, 2>) {
    linear.weight = linear.weight.clone().map(|_| w.transpose());
}

fn set_layernorm<B: Backend>(norm: &mut NeuroLayerNorm<B>, w: Tensor<B, 1>, b: Tensor<B, 1>) {
    norm.inner.gamma = norm.inner.gamma.clone().map(|_| w);
    if let Some(ref beta) = norm.inner.beta {
        norm.inner.beta = Some(beta.clone().map(|_| b));
    }
}

fn set_groupnorm<B: Backend>(gn: &mut burn::nn::GroupNorm<B>, w: Tensor<B, 1>, b: Tensor<B, 1>) {
    if let Some(ref gamma) = gn.gamma {
        gn.gamma = Some(gamma.clone().map(|_| w));
    }
    if let Some(ref beta) = gn.beta {
        gn.beta = Some(beta.clone().map(|_| b));
    }
}

fn set_conv2d_wb<B: Backend>(
    conv: &mut burn::nn::conv::Conv2d<B>,
    w: Tensor<B, 4>,
    b: Tensor<B, 1>,
) {
    conv.weight = conv.weight.clone().map(|_| w);
    if let Some(ref bias) = conv.bias {
        conv.bias = Some(bias.clone().map(|_| b));
    }
}

fn set_quantizer_weight<B: Backend>(q: &mut NormVectorQuantizer<B>, w: Tensor<B, 2>) {
    q.weight = q.weight.clone().map(|_| w);
}

// ── Tokenizer loader ──────────────────────────────────────────────────────────

/// Load tokenizer weights from a safetensors file.
///
/// Keys match PyTorch state_dict naming.
pub fn load_tokenizer<B: Backend>(
    wm: &mut WeightMap,
    model: &mut NeuroRVQTokenizer<B>,
    device: &B::Device,
) -> anyhow::Result<()> {
    // ── Encoder: multi-scale conv branches ────────────────────────────────
    load_multi_scale_conv(wm, model, device)?;

    // ── Encoder: transformer blocks ──────────────────────────────────────
    load_transformer_blocks(wm, "encoder", &mut model.encoder, device)?;

    // ── Encoder: positional embeddings ───────────────────────────────────
    load_pos_embeds(wm, "encoder", &mut model.encoder, device)?;

    // ── Encoder: output heads ────────────────────────────────────────────
    load_encoder_heads(wm, "encoder", &mut model.encoder, device)?;

    // ── Decoder: transformer blocks ──────────────────────────────────────
    load_transformer_blocks(wm, "decoder", &mut model.decoder, device)?;

    // ── Decoder: positional embeddings ───────────────────────────────────
    load_pos_embeds(wm, "decoder", &mut model.decoder, device)?;

    // ── Decoder: patch embeds ────────────────────────────────────────────
    load_decoder_patch_embeds(wm, model, device)?;

    // ── Decoder: output heads ────────────────────────────────────────────
    load_encoder_heads(wm, "decoder", &mut model.decoder, device)?;

    // ── Encode task layers ───────────────────────────────────────────────
    load_encode_task_layers(wm, model, device)?;

    // ── Decode task layers ───────────────────────────────────────────────
    load_decode_task_layers(wm, model, device)?;

    // ── RVQ codebooks ────────────────────────────────────────────────────
    load_rvq_codebooks(wm, model, device)?;

    Ok(())
}

fn load_multi_scale_conv<B: Backend>(
    wm: &mut WeightMap,
    model: &mut NeuroRVQTokenizer<B>,
    device: &B::Device,
) -> anyhow::Result<()> {
    let conv = model
        .encoder
        .multi_scale_conv
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("encoder has no multi_scale_conv"))?;

    // Each branch has conv1, norm1, conv2, norm2
    let branches = [
        (
            "encoder.patch_embed.conv1_1",
            "encoder.patch_embed.norm1_1",
            "encoder.patch_embed.conv2_1",
            "encoder.patch_embed.norm2_1",
            &mut conv.branch1,
        ),
        (
            "encoder.patch_embed.conv1_2",
            "encoder.patch_embed.norm1_2",
            "encoder.patch_embed.conv2_2",
            "encoder.patch_embed.norm2_2",
            &mut conv.branch2,
        ),
        (
            "encoder.patch_embed.conv1_3",
            "encoder.patch_embed.norm1_3",
            "encoder.patch_embed.conv2_3",
            "encoder.patch_embed.norm2_3",
            &mut conv.branch3,
        ),
        (
            "encoder.patch_embed.conv1_4",
            "encoder.patch_embed.norm1_4",
            "encoder.patch_embed.conv2_4",
            "encoder.patch_embed.norm2_4",
            &mut conv.branch4,
        ),
    ];

    for (c1_pfx, n1_pfx, c2_pfx, n2_pfx, branch) in branches {
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 4>(&format!("{c1_pfx}.weight"), device),
            wm.take::<B, 1>(&format!("{c1_pfx}.bias"), device),
        ) {
            set_conv2d_wb(&mut branch.conv1, w, b);
        }
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 1>(&format!("{n1_pfx}.weight"), device),
            wm.take::<B, 1>(&format!("{n1_pfx}.bias"), device),
        ) {
            set_groupnorm(&mut branch.norm1, w, b);
        }
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 4>(&format!("{c2_pfx}.weight"), device),
            wm.take::<B, 1>(&format!("{c2_pfx}.bias"), device),
        ) {
            set_conv2d_wb(&mut branch.conv2, w, b);
        }
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 1>(&format!("{n2_pfx}.weight"), device),
            wm.take::<B, 1>(&format!("{n2_pfx}.bias"), device),
        ) {
            set_groupnorm(&mut branch.norm2, w, b);
        }
    }
    Ok(())
}

fn load_transformer_blocks<B: Backend>(
    wm: &mut WeightMap,
    prefix: &str,
    fm: &mut crate::model::foundation::NeuroRVQFM<B>,
    device: &B::Device,
) -> anyhow::Result<()> {
    for (i, block) in fm.blocks.iter_mut().enumerate() {
        let p = if prefix.is_empty() {
            format!("blocks.{i}")
        } else {
            format!("{prefix}.blocks.{i}")
        };

        // norm1
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 1>(&format!("{p}.norm1.weight"), device),
            wm.take::<B, 1>(&format!("{p}.norm1.bias"), device),
        ) {
            set_layernorm(&mut block.norm1, w, b);
        }

        // attention
        if let Ok(w) = wm.take::<B, 2>(&format!("{p}.attn.qkv.weight"), device) {
            set_linear_no_bias(&mut block.attn.qkv, w);
        }
        if let Ok(b) = wm.take::<B, 1>(&format!("{p}.attn.q_bias"), device) {
            if let Some(ref qb) = block.attn.q_bias {
                block.attn.q_bias = Some(qb.clone().map(|_| b));
            }
        }
        if let Ok(b) = wm.take::<B, 1>(&format!("{p}.attn.v_bias"), device) {
            if let Some(ref vb) = block.attn.v_bias {
                block.attn.v_bias = Some(vb.clone().map(|_| b));
            }
        }
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 2>(&format!("{p}.attn.proj.weight"), device),
            wm.take::<B, 1>(&format!("{p}.attn.proj.bias"), device),
        ) {
            set_linear_wb(&mut block.attn.proj, w, Some(b));
        }

        // q_norm, k_norm
        if let Some(ref mut qn) = block.attn.q_norm {
            if let (Ok(w), Ok(b)) = (
                wm.take::<B, 1>(&format!("{p}.attn.q_norm.weight"), device),
                wm.take::<B, 1>(&format!("{p}.attn.q_norm.bias"), device),
            ) {
                set_layernorm(qn, w, b);
            }
        }
        if let Some(ref mut kn) = block.attn.k_norm {
            if let (Ok(w), Ok(b)) = (
                wm.take::<B, 1>(&format!("{p}.attn.k_norm.weight"), device),
                wm.take::<B, 1>(&format!("{p}.attn.k_norm.bias"), device),
            ) {
                set_layernorm(kn, w, b);
            }
        }

        // norm2
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 1>(&format!("{p}.norm2.weight"), device),
            wm.take::<B, 1>(&format!("{p}.norm2.bias"), device),
        ) {
            set_layernorm(&mut block.norm2, w, b);
        }

        // mlp
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 2>(&format!("{p}.mlp.fc1.weight"), device),
            wm.take::<B, 1>(&format!("{p}.mlp.fc1.bias"), device),
        ) {
            set_linear_wb(&mut block.mlp.fc1, w, Some(b));
        }
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 2>(&format!("{p}.mlp.fc2.weight"), device),
            wm.take::<B, 1>(&format!("{p}.mlp.fc2.bias"), device),
        ) {
            set_linear_wb(&mut block.mlp.fc2, w, Some(b));
        }

        // gamma_1, gamma_2
        if let Ok(g) = wm.take::<B, 1>(&format!("{p}.gamma_1"), device) {
            if let Some(ref g1) = block.gamma_1 {
                block.gamma_1 = Some(g1.clone().map(|_| g));
            }
        }
        if let Ok(g) = wm.take::<B, 1>(&format!("{p}.gamma_2"), device) {
            if let Some(ref g2) = block.gamma_2 {
                block.gamma_2 = Some(g2.clone().map(|_| g));
            }
        }
    }
    Ok(())
}

fn load_pos_embeds<B: Backend>(
    wm: &mut WeightMap,
    prefix: &str,
    fm: &mut crate::model::foundation::NeuroRVQFM<B>,
    device: &B::Device,
) -> anyhow::Result<()> {
    if let Ok(t) = wm.take::<B, 3>(&format!("{prefix}.cls_token"), device) {
        fm.cls_token = fm.cls_token.clone().map(|_| t);
    }
    if let Ok(t) = wm.take::<B, 2>(&format!("{prefix}.pos_embed"), device) {
        fm.pos_embed = fm.pos_embed.clone().map(|_| t);
    }
    if let Ok(t) = wm.take::<B, 2>(&format!("{prefix}.time_embed"), device) {
        fm.time_embed = fm.time_embed.clone().map(|_| t);
    }
    Ok(())
}

fn load_encoder_heads<B: Backend>(
    wm: &mut WeightMap,
    prefix: &str,
    fm: &mut crate::model::foundation::NeuroRVQFM<B>,
    device: &B::Device,
) -> anyhow::Result<()> {
    for (i, (norm, head)) in [
        (&mut fm.fc_norm_1, &mut fm.head_1),
        (&mut fm.fc_norm_2, &mut fm.head_2),
        (&mut fm.fc_norm_3, &mut fm.head_3),
        (&mut fm.fc_norm_4, &mut fm.head_4),
    ]
    .into_iter()
    .enumerate()
    {
        let n = i + 1;
        let fc_norm_w = if prefix.is_empty() {
            format!("fc_norm_{n}.weight")
        } else {
            format!("{prefix}.fc_norm_{n}.weight")
        };
        let fc_norm_b = if prefix.is_empty() {
            format!("fc_norm_{n}.bias")
        } else {
            format!("{prefix}.fc_norm_{n}.bias")
        };
        let head_w = if prefix.is_empty() {
            format!("head_{n}.weight")
        } else {
            format!("{prefix}.head_{n}.weight")
        };
        let head_b = if prefix.is_empty() {
            format!("head_{n}.bias")
        } else {
            format!("{prefix}.head_{n}.bias")
        };
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 1>(&fc_norm_w, device),
            wm.take::<B, 1>(&fc_norm_b, device),
        ) {
            set_layernorm(norm, w, b);
        }
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 2>(&head_w, device),
            wm.take::<B, 1>(&head_b, device),
        ) {
            set_linear_wb(head, w, Some(b));
        }
    }
    Ok(())
}

fn load_decoder_patch_embeds<B: Backend>(
    wm: &mut WeightMap,
    model: &mut NeuroRVQTokenizer<B>,
    device: &B::Device,
) -> anyhow::Result<()> {
    for (i, pe_opt) in [
        &mut model.decoder.patch_embed_1,
        &mut model.decoder.patch_embed_2,
        &mut model.decoder.patch_embed_3,
        &mut model.decoder.patch_embed_4,
    ]
    .into_iter()
    .enumerate()
    {
        let n = i + 1;
        if let Some(ref mut pe) = pe_opt {
            if let (Ok(w), Ok(b)) = (
                wm.take::<B, 4>(&format!("decoder.patch_embed_{n}.proj.weight"), device),
                wm.take::<B, 1>(&format!("decoder.patch_embed_{n}.proj.bias"), device),
            ) {
                set_conv2d_wb(&mut pe.proj, w, b);
            }
        }
    }
    Ok(())
}

fn load_encode_task_layers<B: Backend>(
    wm: &mut WeightMap,
    model: &mut NeuroRVQTokenizer<B>,
    device: &B::Device,
) -> anyhow::Result<()> {
    let heads = [
        (
            &mut model.encode_head_1_fc1,
            &mut model.encode_head_1_fc2,
            1,
        ),
        (
            &mut model.encode_head_2_fc1,
            &mut model.encode_head_2_fc2,
            2,
        ),
        (
            &mut model.encode_head_3_fc1,
            &mut model.encode_head_3_fc2,
            3,
        ),
        (
            &mut model.encode_head_4_fc1,
            &mut model.encode_head_4_fc2,
            4,
        ),
    ];

    for (fc1, fc2, n) in heads {
        // Python: encode_task_layer_{n}.0 = Linear, .2 = Linear (1 = Tanh, no params)
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 2>(&format!("encode_task_layer_{n}.0.weight"), device),
            wm.take::<B, 1>(&format!("encode_task_layer_{n}.0.bias"), device),
        ) {
            set_linear_wb(fc1, w, Some(b));
        }
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 2>(&format!("encode_task_layer_{n}.2.weight"), device),
            wm.take::<B, 1>(&format!("encode_task_layer_{n}.2.bias"), device),
        ) {
            set_linear_wb(fc2, w, Some(b));
        }
    }
    Ok(())
}

fn load_decode_task_layers<B: Backend>(
    wm: &mut WeightMap,
    model: &mut NeuroRVQTokenizer<B>,
    device: &B::Device,
) -> anyhow::Result<()> {
    // Amplitude: decode_task_layer_amplitude.0 = Linear, .2 = Linear (1 = GELU)
    if let (Ok(w), Ok(b)) = (
        wm.take::<B, 2>("decode_task_layer_amplitude.0.weight", device),
        wm.take::<B, 1>("decode_task_layer_amplitude.0.bias", device),
    ) {
        set_linear_wb(&mut model.decode_amp_fc1, w, Some(b));
    }
    if let (Ok(w), Ok(b)) = (
        wm.take::<B, 2>("decode_task_layer_amplitude.2.weight", device),
        wm.take::<B, 1>("decode_task_layer_amplitude.2.bias", device),
    ) {
        set_linear_wb(&mut model.decode_amp_fc2, w, Some(b));
    }

    // Sin: decode_task_layer_angle_sin.0, .2 (with Tanh in between and at end)
    if let (Ok(w), Ok(b)) = (
        wm.take::<B, 2>("decode_task_layer_angle_sin.0.weight", device),
        wm.take::<B, 1>("decode_task_layer_angle_sin.0.bias", device),
    ) {
        set_linear_wb(&mut model.decode_sin_fc1, w, Some(b));
    }
    if let (Ok(w), Ok(b)) = (
        wm.take::<B, 2>("decode_task_layer_angle_sin.2.weight", device),
        wm.take::<B, 1>("decode_task_layer_angle_sin.2.bias", device),
    ) {
        set_linear_wb(&mut model.decode_sin_fc2, w, Some(b));
    }

    // Cos: decode_task_layer_angle_cos.0, .2
    if let (Ok(w), Ok(b)) = (
        wm.take::<B, 2>("decode_task_layer_angle_cos.0.weight", device),
        wm.take::<B, 1>("decode_task_layer_angle_cos.0.bias", device),
    ) {
        set_linear_wb(&mut model.decode_cos_fc1, w, Some(b));
    }
    if let (Ok(w), Ok(b)) = (
        wm.take::<B, 2>("decode_task_layer_angle_cos.2.weight", device),
        wm.take::<B, 1>("decode_task_layer_angle_cos.2.bias", device),
    ) {
        set_linear_wb(&mut model.decode_cos_fc2, w, Some(b));
    }
    Ok(())
}

fn load_rvq_codebooks<B: Backend>(
    wm: &mut WeightMap,
    model: &mut NeuroRVQTokenizer<B>,
    device: &B::Device,
) -> anyhow::Result<()> {
    let rvqs = [
        &mut model.quantize_1,
        &mut model.quantize_2,
        &mut model.quantize_3,
        &mut model.quantize_4,
    ];

    for (q_idx, rvq) in rvqs.into_iter().enumerate() {
        let n = q_idx + 1;
        for (l_idx, layer) in rvq.layers.iter_mut().enumerate() {
            // Python key: quantize_{n}.layers.{l_idx}.embedding.weight
            let key = format!("quantize_{n}.layers.{l_idx}.embedding.weight");
            if let Ok(w) = wm.take::<B, 2>(&key, device) {
                set_quantizer_weight(layer, w);
            }
        }
    }
    Ok(())
}

// ── Standalone Foundation Model loader ────────────────────────────────────────

/// Load standalone foundation model weights from a safetensors file.
///
/// Python: `foundation_model.load_state_dict(torch.load(model_path), strict=False)`
///
/// The FM weights use unprefixed keys (e.g. `patch_embed.conv1_1.weight`),
/// not `encoder.` prefixed like in the tokenizer.
pub fn load_foundation_model<B: Backend>(
    wm: &mut WeightMap,
    model: &mut crate::model::foundation::NeuroRVQFM<B>,
    device: &B::Device,
) -> anyhow::Result<()> {
    // Multi-scale conv (encoder mode)
    load_fm_multi_scale_conv(wm, model, device)?;

    // Transformer blocks (unprefixed keys: blocks.{i}.*)
    load_transformer_blocks(wm, "", model, device)?;

    // Positional embeddings
    load_pos_embeds_fm(wm, model, device)?;

    // Output heads (fc_norm_{n}, head_{n})
    load_encoder_heads(wm, "", model, device)?;

    Ok(())
}

fn load_fm_multi_scale_conv<B: Backend>(
    wm: &mut WeightMap,
    model: &mut crate::model::foundation::NeuroRVQFM<B>,
    device: &B::Device,
) -> anyhow::Result<()> {
    let conv = match model.multi_scale_conv.as_mut() {
        Some(c) => c,
        None => return Ok(()), // decoder mode, no conv
    };

    let branches = [
        (
            "patch_embed.conv1_1",
            "patch_embed.norm1_1",
            "patch_embed.conv2_1",
            "patch_embed.norm2_1",
            &mut conv.branch1,
        ),
        (
            "patch_embed.conv1_2",
            "patch_embed.norm1_2",
            "patch_embed.conv2_2",
            "patch_embed.norm2_2",
            &mut conv.branch2,
        ),
        (
            "patch_embed.conv1_3",
            "patch_embed.norm1_3",
            "patch_embed.conv2_3",
            "patch_embed.norm2_3",
            &mut conv.branch3,
        ),
        (
            "patch_embed.conv1_4",
            "patch_embed.norm1_4",
            "patch_embed.conv2_4",
            "patch_embed.norm2_4",
            &mut conv.branch4,
        ),
    ];

    for (c1_pfx, n1_pfx, c2_pfx, n2_pfx, branch) in branches {
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 4>(&format!("{c1_pfx}.weight"), device),
            wm.take::<B, 1>(&format!("{c1_pfx}.bias"), device),
        ) {
            set_conv2d_wb(&mut branch.conv1, w, b);
        }
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 1>(&format!("{n1_pfx}.weight"), device),
            wm.take::<B, 1>(&format!("{n1_pfx}.bias"), device),
        ) {
            set_groupnorm(&mut branch.norm1, w, b);
        }
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 4>(&format!("{c2_pfx}.weight"), device),
            wm.take::<B, 1>(&format!("{c2_pfx}.bias"), device),
        ) {
            set_conv2d_wb(&mut branch.conv2, w, b);
        }
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 1>(&format!("{n2_pfx}.weight"), device),
            wm.take::<B, 1>(&format!("{n2_pfx}.bias"), device),
        ) {
            set_groupnorm(&mut branch.norm2, w, b);
        }
    }
    Ok(())
}

fn load_pos_embeds_fm<B: Backend>(
    wm: &mut WeightMap,
    fm: &mut crate::model::foundation::NeuroRVQFM<B>,
    device: &B::Device,
) -> anyhow::Result<()> {
    // Try unprefixed keys first, then prefixed
    if let Ok(t) = wm.take::<B, 3>("cls_token", device) {
        fm.cls_token = fm.cls_token.clone().map(|_| t);
    }
    if let Ok(t) = wm.take::<B, 2>("pos_embed", device) {
        fm.pos_embed = fm.pos_embed.clone().map(|_| t);
    }
    if let Ok(t) = wm.take::<B, 2>("time_embed", device) {
        fm.time_embed = fm.time_embed.clone().map(|_| t);
    }
    Ok(())
}

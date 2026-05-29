#![cfg(feature = "burn")]

use anyhow::Context;
use burn::prelude::*;
/// High-level APIs for NeuroRVQ inference:
///   - NeuroRVQEncoder: tokenizer pipeline (encode → quantize → decode)
///   - NeuroRVQFoundationModel: standalone foundation model encoder
use std::{path::Path, time::Instant};

use crate::{
    channels,
    config::{ConfigOverrides, Modality, NeuroRVQConfig},
    data::InputBatch,
    model::foundation::NeuroRVQFM,
    model::tokenizer::NeuroRVQTokenizer,
    weights::{load_foundation_model, load_tokenizer, WeightMap},
};

// ── Token Result ──────────────────────────────────────────────────────────────

/// Result of tokenization: per-branch token indices.
pub struct TokenResult {
    /// Token indices per branch (4 branches × 8 RVQ levels).
    /// Each inner Vec has 8 entries (one per RVQ level), each a Vec<i64> of indices.
    pub branch_tokens: Vec<Vec<Vec<i64>>>,
    /// Number of channels.
    pub n_channels: usize,
    /// Number of time patches per channel.
    pub n_time_patches: usize,
}

// ── Reconstruction Result ─────────────────────────────────────────────────────

/// Full signal reconstruction result.
pub struct ReconstructionResult {
    /// Reconstructed amplitude: [seq_len, decoder_out_dim].
    pub amplitude: Vec<f32>,
    /// Reconstructed sin(phase): [seq_len, decoder_out_dim].
    pub sin_phase: Vec<f32>,
    /// Reconstructed cos(phase): [seq_len, decoder_out_dim].
    pub cos_phase: Vec<f32>,
    pub shape: Vec<usize>,
}

// ── Full Forward Result ───────────────────────────────────────────────────────

/// Result from the full forward pass (standardized original + reconstructed signals).
pub struct ForwardResult {
    /// Standardized original signal, flattened [B * N*A * T].
    pub original_std: Vec<f32>,
    /// Standardized reconstructed signal, flattened [B * N*A * T].
    pub reconstructed_std: Vec<f32>,
    /// Shape: [B, N*A, T].
    pub shape: Vec<usize>,
}

// ── Foundation Model Result ───────────────────────────────────────────────────

/// Result from standalone foundation model encoder.
pub struct FMEncoderResult {
    /// 4 branch outputs, each flattened [B * seq_len * embed_dim].
    pub branch_features: Vec<Vec<f32>>,
    /// Shape of each branch output: [B, seq_len, embed_dim].
    pub shape: Vec<usize>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// NeuroRVQ Tokenizer Encoder
// ═══════════════════════════════════════════════════════════════════════════════

/// High-level NeuroRVQ tokenizer encoder.
pub struct NeuroRVQEncoder<B: Backend> {
    model: NeuroRVQTokenizer<B>,
    pub config: NeuroRVQConfig,
    pub modality: Modality,
    device: B::Device,
}

impl<B: Backend> NeuroRVQEncoder<B> {
    /// Load model from config YAML and safetensors weights.
    ///
    /// Modality is auto-detected from the YAML filename (e.g. `NeuroRVQ_EEG_v1.yml`).
    /// Falls back to EEG if undetectable. Use [`load_with_modality`] to override.
    pub fn load(
        config_path: &Path,
        weights_path: &Path,
        device: B::Device,
    ) -> anyhow::Result<(Self, f64)> {
        let config =
            NeuroRVQConfig::from_yaml(config_path.to_str().context("config path not UTF-8")?)?;
        let modality = config.resolve_modality();
        Self::load_with_modality(config_path, weights_path, modality, device)
    }

    /// Load model with explicit modality override.
    pub fn load_with_modality(
        config_path: &Path,
        weights_path: &Path,
        modality: Modality,
        device: B::Device,
    ) -> anyhow::Result<(Self, f64)> {
        Self::load_full(config_path, weights_path, modality, None, device)
    }

    /// Load model with modality and optional config overrides.
    pub fn load_full(
        config_path: &Path,
        weights_path: &Path,
        modality: Modality,
        overrides: Option<&ConfigOverrides>,
        device: B::Device,
    ) -> anyhow::Result<(Self, f64)> {
        let mut config = NeuroRVQConfig::from_yaml_with_modality(
            config_path.to_str().context("config path not UTF-8")?,
            modality,
        )?;
        if let Some(ovr) = overrides {
            config.apply_overrides(ovr);
        }

        // Set n_global_electrodes from modality
        config.n_global_electrodes = channels::global_vocab_size(modality);

        let t = Instant::now();

        let mut model = NeuroRVQTokenizer::new_with_modality(
            config.n_patches,
            config.patch_size,
            config.embed_dim,
            config.code_dim,
            config.n_code,
            config.decoder_out_dim,
            config.out_chans_encoder,
            config.depth_encoder,
            config.depth_decoder,
            config.num_heads_tokenizer,
            config.mlp_ratio_tokenizer,
            config.qkv_bias_tokenizer,
            config.init_values_tokenizer,
            config.init_scale_tokenizer,
            config.n_global_electrodes,
            modality,
            &device,
        );

        let mut wm =
            WeightMap::from_file(weights_path.to_str().context("weights path not UTF-8")?)?;
        eprintln!("Loading {} weight tensors...", wm.tensors.len());
        load_tokenizer(&mut wm, &mut model, &device)?;

        let ms = t.elapsed().as_secs_f64() * 1000.0;

        Ok((
            Self {
                model,
                config,
                modality,
                device,
            },
            ms,
        ))
    }

    pub fn describe(&self) -> String {
        let c = &self.config;
        format!(
            "NeuroRVQ-{} embed_dim={} patch={} n_patches={} code_dim={} n_code={} enc_depth={} dec_depth={}",
            self.modality, c.embed_dim, c.patch_size, c.n_patches, c.code_dim, c.n_code,
            c.depth_encoder, c.depth_decoder,
        )
    }

    /// Run tokenization on a prepared batch.
    /// Returns token indices for all 4 branches × 8 RVQ levels.
    pub fn tokenize(&self, batch: &InputBatch<B>) -> anyhow::Result<TokenResult> {
        let signal = batch.signal.clone();
        let [b, n, t] = signal.dims();
        let a = t / self.config.patch_size;

        let x = signal.reshape([b, n, a, self.config.patch_size]);

        let enc_out = self
            .model
            .get_tokens(x, batch.temporal_ix.clone(), batch.spatial_ix.clone());

        let mut branch_tokens = Vec::with_capacity(4);
        for branch_indices in &enc_out.indices {
            let mut levels = Vec::with_capacity(branch_indices.len());
            for level_indices in branch_indices {
                let indices_vec = level_indices
                    .clone()
                    .into_data()
                    .to_vec::<i64>()
                    .map_err(|e| anyhow::anyhow!("indices→vec: {e:?}"))?;
                levels.push(indices_vec);
            }
            branch_tokens.push(levels);
        }

        Ok(TokenResult {
            branch_tokens,
            n_channels: batch.n_channels,
            n_time_patches: batch.n_time_patches,
        })
    }

    /// Run full encode → quantize → decode pipeline (returns FFT components).
    pub fn reconstruct(&self, batch: &InputBatch<B>) -> anyhow::Result<ReconstructionResult> {
        let signal = batch.signal.clone();
        let [b, n, t] = signal.dims();
        let a = t / self.config.patch_size;

        let x = signal.reshape([b, n, a, self.config.patch_size]);

        let enc_out = self
            .model
            .encode(x, batch.temporal_ix.clone(), batch.spatial_ix.clone());

        let (amp, sin, cos) = self.model.decode(
            &enc_out.quantized,
            batch.temporal_ix.clone(),
            batch.spatial_ix.clone(),
        );

        let shape = amp.dims().to_vec();
        let amp_vec = amp
            .squeeze::<2>()
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("amp→vec: {e:?}"))?;
        let sin_vec = sin
            .squeeze::<2>()
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("sin→vec: {e:?}"))?;
        let cos_vec = cos
            .squeeze::<2>()
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("cos→vec: {e:?}"))?;

        Ok(ReconstructionResult {
            amplitude: amp_vec,
            sin_phase: sin_vec,
            cos_phase: cos_vec,
            shape: shape[1..].to_vec(),
        })
    }

    /// Run the full forward pass: encode → decode → iFFT → standardize.
    ///
    /// Returns standardized original and reconstructed signals (matching Python's
    /// `NeuroRVQTokenizer.forward()`).
    pub fn forward(&self, batch: &InputBatch<B>) -> anyhow::Result<ForwardResult> {
        let fwd = self.model.forward(
            batch.signal.clone(),
            batch.temporal_ix.clone(),
            batch.spatial_ix.clone(),
        );

        let shape = fwd.original_std.dims().to_vec();
        let orig = fwd
            .original_std
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("orig→vec: {e:?}"))?;
        let recon = fwd
            .reconstructed_std
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("recon→vec: {e:?}"))?;

        Ok(ForwardResult {
            original_std: orig,
            reconstructed_std: recon,
            shape,
        })
    }

    pub fn device(&self) -> &B::Device {
        &self.device
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Standalone NeuroRVQ Foundation Model
// ═══════════════════════════════════════════════════════════════════════════════

/// Standalone NeuroRVQ Foundation Model encoder.
///
/// Python: `load_neurorqv_fm()` in `NeuroRVQ_*_FM_example.py`
///
/// Uses the second-stage config params from the YAML (e.g. `embed_dim_second_stage`).
/// The FM is always used as an encoder (`use_as_encoder=True`).
pub struct NeuroRVQFoundationModel<B: Backend> {
    model: NeuroRVQFM<B>,
    pub config: NeuroRVQConfig,
    pub modality: Modality,
    device: B::Device,
}

impl<B: Backend> NeuroRVQFoundationModel<B> {
    /// Load foundation model from config YAML and safetensors weights.
    ///
    /// Modality is auto-detected from the YAML filename if not supplied.
    /// Use the explicit overload to force a specific modality.
    pub fn load(
        config_path: &Path,
        weights_path: &Path,
        modality: Modality,
        device: B::Device,
    ) -> anyhow::Result<(Self, f64)> {
        Self::load_full(config_path, weights_path, modality, None, device)
    }

    pub fn load_full(
        config_path: &Path,
        weights_path: &Path,
        modality: Modality,
        overrides: Option<&ConfigOverrides>,
        device: B::Device,
    ) -> anyhow::Result<(Self, f64)> {
        let mut config = NeuroRVQConfig::from_yaml_with_modality(
            config_path.to_str().context("config path not UTF-8")?,
            modality,
        )?;
        if let Some(ovr) = overrides {
            config.apply_overrides(ovr);
        }

        // Set n_global_electrodes from modality
        config.n_global_electrodes = channels::global_vocab_size(modality);

        let t = Instant::now();

        let mut model = NeuroRVQFM::new_with_modality(
            config.n_patches,
            config.patch_size,
            config.fm_in_chans(),
            config.fm_out_chans(),
            0, // num_classes = 0 for encoder
            config.fm_embed_dim(),
            config.fm_depth(),
            config.fm_num_heads(),
            config.fm_mlp_ratio(),
            config.fm_qkv_bias(),
            config.fm_init_values(),
            config.n_global_electrodes,
            true, // use_as_encoder
            modality,
            &device,
        );

        let mut wm =
            WeightMap::from_file(weights_path.to_str().context("weights path not UTF-8")?)?;
        eprintln!("Loading {} weight tensors for FM...", wm.tensors.len());
        load_foundation_model(&mut wm, &mut model, &device)?;

        let ms = t.elapsed().as_secs_f64() * 1000.0;

        Ok((
            Self {
                model,
                config,
                modality,
                device,
            },
            ms,
        ))
    }

    pub fn describe(&self) -> String {
        let c = &self.config;
        format!(
            "NeuroRVQ-{}-FM  embed_dim={}  depth={}  n_patches={}  patch_size={}",
            self.modality,
            c.fm_embed_dim(),
            c.fm_depth(),
            c.n_patches,
            c.patch_size,
        )
    }

    /// Run the encoder forward pass.
    ///
    /// batch: prepared InputBatch
    /// Returns: 4 branch feature tensors (one per multi-scale branch).
    pub fn encode(&self, batch: &InputBatch<B>) -> anyhow::Result<FMEncoderResult> {
        let signal = batch.signal.clone();
        let [b, n, t] = signal.dims();
        let a = t / self.config.patch_size;

        let x = signal.reshape([b, n, a, self.config.patch_size]);

        let (o1, o2, o3, o4) =
            self.model
                .forward_encoder(x, batch.temporal_ix.clone(), batch.spatial_ix.clone());

        let shape = o1.dims().to_vec();
        let mut branch_features = Vec::with_capacity(4);
        for out in [o1, o2, o3, o4] {
            let v = out
                .into_data()
                .to_vec::<f32>()
                .map_err(|e| anyhow::anyhow!("feature→vec: {e:?}"))?;
            branch_features.push(v);
        }

        Ok(FMEncoderResult {
            branch_features,
            shape,
        })
    }

    /// Concatenate all 4 branch features and return mean-pooled representation.
    ///
    /// Python (fine-tuning path):
    ///   x = concat([x1,x2,x3,x4], dim=-1)  # [B, seq, 4*embed_dim]
    ///   return head(fc_norm(x.mean(1)))
    pub fn encode_pooled(&self, batch: &InputBatch<B>) -> anyhow::Result<Vec<f32>> {
        let signal = batch.signal.clone();
        let [b, n, t] = signal.dims();
        let a = t / self.config.patch_size;

        let x = signal.reshape([b, n, a, self.config.patch_size]);

        let (o1, o2, o3, o4) =
            self.model
                .forward_encoder(x, batch.temporal_ix.clone(), batch.spatial_ix.clone());

        // Concat: [B, seq, 4*embed_dim]
        let concat = Tensor::cat(vec![o1, o2, o3, o4], 2);
        // Mean pool: [B, 4*embed_dim]
        let pooled = concat.mean_dim(1);

        let v = pooled
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("pooled→vec: {e:?}"))?;
        Ok(v)
    }

    pub fn device(&self) -> &B::Device {
        &self.device
    }
}

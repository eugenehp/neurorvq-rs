//! RLX-backed NeuroRVQ encoder APIs.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Context;

use super::graph::{build_fm_branch_graph, FmBranchSpec};
use super::prepare::{
    compute_fft_components, decode_heads, encode_head, multi_scale_conv, num_quantizers,
    patch_embed_branch, prepare_branch_tokens, reconstruct_signal, rvq_encode, rvq_forward,
    seq_to_nchw, std_norm_4d, RlxInputBatch,
};
use super::weights::{
    apply_params, build_all_prepare_params, build_fm_branch_params, build_prepare_params,
    load_safetensors, ParamMap,
};
use crate::channels;
use crate::config::{ConfigOverrides, Modality, NeuroRVQConfig};

// ── Result types ──────────────────────────────────────────────────────────────

pub struct TokenResult {
    pub branch_tokens: Vec<Vec<Vec<i64>>>,
    pub n_channels: usize,
    pub n_time_patches: usize,
}

pub struct ReconstructionResult {
    pub amplitude: Vec<f32>,
    pub sin_phase: Vec<f32>,
    pub cos_phase: Vec<f32>,
    pub shape: Vec<usize>,
}

pub struct ForwardResult {
    pub original_std: Vec<f32>,
    pub reconstructed_std: Vec<f32>,
    pub shape: Vec<usize>,
}

pub struct FMEncoderResult {
    pub branch_features: Vec<Vec<f32>>,
    pub shape: Vec<usize>,
}

// ── Foundation Model ──────────────────────────────────────────────────────────

pub struct NeuroRVQFoundationModel {
    pub config: NeuroRVQConfig,
    pub modality: Modality,
    pub device: rlx::Device,

    branch_params: [ParamMap; 4],
    prepare_params: ParamMap,
    session: rlx::Session,
    cache: HashMap<u64, [rlx::CompiledGraph; 4]>,
}

impl NeuroRVQFoundationModel {
    pub fn load(
        config_path: &Path,
        weights_path: &Path,
        modality: Modality,
        device: rlx::Device,
    ) -> anyhow::Result<(Self, f64)> {
        Self::load_full(config_path, weights_path, modality, None, device)
    }

    pub fn load_full(
        config_path: &Path,
        weights_path: &Path,
        modality: Modality,
        overrides: Option<&crate::config::ConfigOverrides>,
        device: rlx::Device,
    ) -> anyhow::Result<(Self, f64)> {
        let mut config = NeuroRVQConfig::from_yaml_with_modality(
            config_path.to_str().context("config path not UTF-8")?,
            modality,
        )?;
        if let Some(ovr) = overrides {
            config.apply_overrides(ovr);
        }
        config.n_global_electrodes = channels::global_vocab_size(modality);

        let t = std::time::Instant::now();
        let raw = load_safetensors(weights_path.to_str().context("weights path not UTF-8")?)?;
        let mut raw_prep = raw.clone();
        let prepare_params = build_prepare_params(&mut raw_prep, "patch_embed")?;

        let embed_dim = config.fm_embed_dim();
        let depth = config.fm_depth();
        let branch_params = [
            build_fm_branch_params(&mut raw.clone(), &config, "blocks", "", 1, depth, embed_dim)?,
            build_fm_branch_params(&mut raw.clone(), &config, "blocks", "", 2, depth, embed_dim)?,
            build_fm_branch_params(&mut raw.clone(), &config, "blocks", "", 3, depth, embed_dim)?,
            build_fm_branch_params(&mut raw.clone(), &config, "blocks", "", 4, depth, embed_dim)?,
        ];

        let session = rlx::Session::new(device);
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        Ok((
            Self {
                config,
                modality,
                device,
                branch_params,
                prepare_params,
                session,
                cache: HashMap::new(),
            },
            ms,
        ))
    }

    pub fn describe(&self) -> String {
        let c = &self.config;
        format!(
            "NeuroRVQ-{}-FM (RLX, dev={:?})  embed_dim={}  depth={}  n_patches={}  patch_size={}",
            self.modality,
            self.device,
            c.fm_embed_dim(),
            c.fm_depth(),
            c.n_patches,
            c.patch_size,
        )
    }

    fn branch_spec(&self, b: usize, seq_len: usize, branch: usize) -> FmBranchSpec {
        let c = &self.config;
        let d = c.fm_embed_dim();
        let nh = c.fm_num_heads();
        FmBranchSpec {
            b,
            s: 1 + seq_len,
            seq_len,
            d,
            out_dim: d,
            nh,
            dh: d / nh,
            depth: c.fm_depth(),
            ff: (d as f64 * c.fm_mlp_ratio()) as usize,
            norm_eps: 1e-6,
            block_prefix: "blocks".into(),
            head_prefix: String::new(),
            branch,
            use_qk_norm: true,
        }
    }

    fn cache_key(&self, b: usize, seq_len: usize) -> u64 {
        (b as u64) << 32 | (seq_len as u64)
    }

    fn compiled_branches(&mut self, b: usize, seq_len: usize) -> &mut [rlx::CompiledGraph; 4] {
        let key = self.cache_key(b, seq_len);
        if !self.cache.contains_key(&key) {
            let mut graphs = [
                self.session
                    .compile(build_fm_branch_graph(&self.branch_spec(b, seq_len, 1))),
                self.session
                    .compile(build_fm_branch_graph(&self.branch_spec(b, seq_len, 2))),
                self.session
                    .compile(build_fm_branch_graph(&self.branch_spec(b, seq_len, 3))),
                self.session
                    .compile(build_fm_branch_graph(&self.branch_spec(b, seq_len, 4))),
            ];
            for (i, g) in graphs.iter_mut().enumerate() {
                apply_params(g, &self.branch_params[i]);
            }
            self.cache.insert(key, graphs);
        }
        self.cache.get_mut(&key).expect("just inserted")
    }

    pub fn encode(&mut self, batch: &RlxInputBatch) -> anyhow::Result<FMEncoderResult> {
        let b = 1usize;
        let n = batch.n_channels;
        let a = batch.n_time_patches;
        let t = self.config.patch_size;
        let seq_len = n * a;
        let embed_dim = self.config.fm_embed_dim();

        let branches = multi_scale_conv(
            &batch.signal,
            &self.prepare_params,
            self.modality,
            b,
            n,
            a,
            t,
        );

        let mut branch_features = Vec::with_capacity(4);
        for (i, branch_x) in branches.into_iter().enumerate() {
            let tokens = prepare_branch_tokens(
                &branch_x,
                &self.prepare_params,
                "",
                &batch.temporal_ix,
                &batch.spatial_ix,
                b,
                seq_len,
                embed_dim,
            );
            let compiled = self.compiled_branches(b, seq_len);
            let outs = compiled[i].run(&[("x", &tokens)]);
            branch_features.push(
                outs.into_iter()
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("branch {} produced no output", i + 1))?,
            );
        }

        Ok(FMEncoderResult {
            branch_features,
            shape: vec![b, seq_len, embed_dim],
        })
    }
}

// ── Tokenizer ─────────────────────────────────────────────────────────────────

pub struct NeuroRVQEncoder {
    pub config: NeuroRVQConfig,
    pub modality: Modality,
    pub device: rlx::Device,

    enc_branch_params: [ParamMap; 4],
    dec_branch_params: [ParamMap; 4],
    prepare_params: ParamMap,
    session: rlx::Session,
    enc_cache: HashMap<u64, [rlx::CompiledGraph; 4]>,
    dec_cache: HashMap<u64, [rlx::CompiledGraph; 4]>,
}

impl NeuroRVQEncoder {
    pub fn load(
        config_path: &Path,
        weights_path: &Path,
        device: rlx::Device,
    ) -> anyhow::Result<(Self, f64)> {
        let config =
            NeuroRVQConfig::from_yaml(config_path.to_str().context("config path not UTF-8")?)?;
        let modality = config.resolve_modality();
        Self::load_with_modality(config_path, weights_path, modality, device)
    }

    pub fn load_with_modality(
        config_path: &Path,
        weights_path: &Path,
        modality: Modality,
        device: rlx::Device,
    ) -> anyhow::Result<(Self, f64)> {
        Self::load_full(config_path, weights_path, modality, None, device)
    }

    pub fn load_full(
        config_path: &Path,
        weights_path: &Path,
        modality: Modality,
        overrides: Option<&ConfigOverrides>,
        device: rlx::Device,
    ) -> anyhow::Result<(Self, f64)> {
        let mut config = NeuroRVQConfig::from_yaml_with_modality(
            config_path.to_str().context("config path not UTF-8")?,
            modality,
        )?;
        if let Some(ovr) = overrides {
            config.apply_overrides(ovr);
        }
        config.n_global_electrodes = channels::global_vocab_size(modality);

        let t = std::time::Instant::now();
        let raw = load_safetensors(weights_path.to_str().context("weights path not UTF-8")?)?;
        let mut raw_prep = raw.clone();
        let prepare_params = build_all_prepare_params(&mut raw_prep)?;

        let embed_dim = config.embed_dim;
        let depth_enc = config.depth_encoder;
        let depth_dec = config.depth_decoder;
        let enc_branch_params = [
            build_fm_branch_params(
                &mut raw.clone(),
                &config,
                "encoder.blocks",
                "encoder",
                1,
                depth_enc,
                embed_dim,
            )?,
            build_fm_branch_params(
                &mut raw.clone(),
                &config,
                "encoder.blocks",
                "encoder",
                2,
                depth_enc,
                embed_dim,
            )?,
            build_fm_branch_params(
                &mut raw.clone(),
                &config,
                "encoder.blocks",
                "encoder",
                3,
                depth_enc,
                embed_dim,
            )?,
            build_fm_branch_params(
                &mut raw.clone(),
                &config,
                "encoder.blocks",
                "encoder",
                4,
                depth_enc,
                embed_dim,
            )?,
        ];
        let dec_branch_params = [
            build_fm_branch_params(
                &mut raw.clone(),
                &config,
                "decoder.blocks",
                "decoder",
                1,
                depth_dec,
                embed_dim,
            )?,
            build_fm_branch_params(
                &mut raw.clone(),
                &config,
                "decoder.blocks",
                "decoder",
                2,
                depth_dec,
                embed_dim,
            )?,
            build_fm_branch_params(
                &mut raw.clone(),
                &config,
                "decoder.blocks",
                "decoder",
                3,
                depth_dec,
                embed_dim,
            )?,
            build_fm_branch_params(
                &mut raw.clone(),
                &config,
                "decoder.blocks",
                "decoder",
                4,
                depth_dec,
                embed_dim,
            )?,
        ];

        let session = rlx::Session::new(device);
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        Ok((
            Self {
                config,
                modality,
                device,
                enc_branch_params,
                dec_branch_params,
                prepare_params,
                session,
                enc_cache: HashMap::new(),
                dec_cache: HashMap::new(),
            },
            ms,
        ))
    }

    pub fn describe(&self) -> String {
        let c = &self.config;
        format!(
            "NeuroRVQ-{} (RLX, dev={:?}) embed_dim={} patch={} n_patches={} code_dim={} enc_depth={}",
            self.modality,
            self.device,
            c.embed_dim,
            c.patch_size,
            c.n_patches,
            c.code_dim,
            c.depth_encoder,
        )
    }

    fn enc_branch_spec(&self, b: usize, seq_len: usize, branch: usize) -> FmBranchSpec {
        let c = &self.config;
        let d = c.embed_dim;
        let nh = c.num_heads_tokenizer;
        FmBranchSpec {
            b,
            s: 1 + seq_len,
            seq_len,
            d,
            out_dim: d,
            nh,
            dh: d / nh,
            depth: c.depth_encoder,
            ff: (d as f64 * c.mlp_ratio_tokenizer) as usize,
            norm_eps: 1e-6,
            block_prefix: "encoder.blocks".into(),
            head_prefix: "encoder".into(),
            branch,
            use_qk_norm: true,
        }
    }

    fn enc_cache_key(&self, b: usize, seq_len: usize) -> u64 {
        (b as u64) << 32 | (seq_len as u64)
    }

    fn compiled_encoder(&mut self, b: usize, seq_len: usize) -> &mut [rlx::CompiledGraph; 4] {
        let key = self.enc_cache_key(b, seq_len);
        if !self.enc_cache.contains_key(&key) {
            let mut graphs = [
                self.session
                    .compile(build_fm_branch_graph(&self.enc_branch_spec(b, seq_len, 1))),
                self.session
                    .compile(build_fm_branch_graph(&self.enc_branch_spec(b, seq_len, 2))),
                self.session
                    .compile(build_fm_branch_graph(&self.enc_branch_spec(b, seq_len, 3))),
                self.session
                    .compile(build_fm_branch_graph(&self.enc_branch_spec(b, seq_len, 4))),
            ];
            for (i, g) in graphs.iter_mut().enumerate() {
                apply_params(g, &self.enc_branch_params[i]);
            }
            self.enc_cache.insert(key, graphs);
        }
        self.enc_cache.get_mut(&key).expect("just inserted")
    }

    fn run_encoder_branches(&mut self, batch: &RlxInputBatch) -> anyhow::Result<[Vec<f32>; 4]> {
        let b = 1usize;
        let n = batch.n_channels;
        let a = batch.n_time_patches;
        let t = self.config.patch_size;
        let seq_len = n * a;
        let embed_dim = self.config.embed_dim;

        let branches = multi_scale_conv(
            &batch.signal,
            &self.prepare_params,
            self.modality,
            b,
            n,
            a,
            t,
        );

        let mut outs = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

        for (i, branch_x) in branches.into_iter().enumerate() {
            let tokens = prepare_branch_tokens(
                &branch_x,
                &self.prepare_params,
                "encoder.",
                &batch.temporal_ix,
                &batch.spatial_ix,
                b,
                seq_len,
                embed_dim,
            );
            let compiled = self.compiled_encoder(b, seq_len);
            let run_out = compiled[i].run(&[("x", &tokens)]);
            outs[i] = run_out
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("encoder branch {} produced no output", i + 1))?;
        }
        Ok(outs)
    }

    fn dec_branch_spec(&self, b: usize, seq_len: usize, branch: usize) -> FmBranchSpec {
        let c = &self.config;
        let d = c.embed_dim;
        let nh = c.num_heads_tokenizer;
        FmBranchSpec {
            b,
            s: 1 + seq_len,
            seq_len,
            d,
            out_dim: d,
            nh,
            dh: d / nh,
            depth: c.depth_decoder,
            ff: (d as f64 * c.mlp_ratio_tokenizer) as usize,
            norm_eps: 1e-6,
            block_prefix: "decoder.blocks".into(),
            head_prefix: "decoder".into(),
            branch,
            use_qk_norm: true,
        }
    }

    fn compiled_decoder(&mut self, b: usize, seq_len: usize) -> &mut [rlx::CompiledGraph; 4] {
        let key = self.enc_cache_key(b, seq_len);
        if !self.dec_cache.contains_key(&key) {
            let mut graphs = [
                self.session
                    .compile(build_fm_branch_graph(&self.dec_branch_spec(b, seq_len, 1))),
                self.session
                    .compile(build_fm_branch_graph(&self.dec_branch_spec(b, seq_len, 2))),
                self.session
                    .compile(build_fm_branch_graph(&self.dec_branch_spec(b, seq_len, 3))),
                self.session
                    .compile(build_fm_branch_graph(&self.dec_branch_spec(b, seq_len, 4))),
            ];
            for (i, g) in graphs.iter_mut().enumerate() {
                apply_params(g, &self.dec_branch_params[i]);
            }
            self.dec_cache.insert(key, graphs);
        }
        self.dec_cache.get_mut(&key).expect("just inserted")
    }

    fn encode_quantized(&mut self, batch: &RlxInputBatch) -> anyhow::Result<[Vec<f32>; 4]> {
        let branch_outs = self.run_encoder_branches(batch)?;
        let b = 1usize;
        let n = batch.n_channels;
        let a = batch.n_time_patches;
        let seq_len = n * a;
        let code_dim = self.config.code_dim;
        let n_levels = num_quantizers(self.modality);

        let mut quantized = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for (i, out) in branch_outs.into_iter().enumerate() {
            let encoded = encode_head(&out, &self.prepare_params, i + 1, code_dim);
            let q_vec = rvq_forward(&encoded, &self.prepare_params, i + 1, code_dim, n_levels);
            quantized[i] = seq_to_nchw(&q_vec, b, seq_len, code_dim, n);
        }
        Ok(quantized)
    }

    fn run_decoder_branches(
        &mut self,
        batch: &RlxInputBatch,
        quantized_nchw: &[Vec<f32>; 4],
    ) -> anyhow::Result<Vec<f32>> {
        let b = 1usize;
        let n = batch.n_channels;
        let a = batch.n_time_patches;
        let seq_len = n * a;
        let embed_dim = self.config.embed_dim;
        let code_dim = self.config.code_dim;
        let w = seq_len / n;

        let mut branch_outs = Vec::with_capacity(4);
        for (i, q_nchw) in quantized_nchw.iter().enumerate() {
            let pe_out = patch_embed_branch(
                q_nchw,
                &self.prepare_params,
                i + 1,
                code_dim,
                n,
                w,
                embed_dim,
            );
            let tokens = prepare_branch_tokens(
                &pe_out,
                &self.prepare_params,
                "decoder.",
                &batch.temporal_ix,
                &batch.spatial_ix,
                b,
                seq_len,
                embed_dim,
            );
            let compiled = self.compiled_decoder(b, seq_len);
            let run_out = compiled[i].run(&[("x", &tokens)]);
            branch_outs.push(
                run_out.into_iter().next().ok_or_else(|| {
                    anyhow::anyhow!("decoder branch {} produced no output", i + 1)
                })?,
            );
        }

        let mut concat = vec![0f32; b * seq_len * 4 * embed_dim];
        for bi in 0..b {
            for s in 0..seq_len {
                for (br, out) in branch_outs.iter().enumerate() {
                    for d in 0..embed_dim {
                        let src = bi * seq_len * embed_dim + s * embed_dim + d;
                        let dst =
                            bi * seq_len * 4 * embed_dim + s * 4 * embed_dim + br * embed_dim + d;
                        concat[dst] = out[src];
                    }
                }
            }
        }
        Ok(concat)
    }

    pub fn tokenize(&mut self, batch: &RlxInputBatch) -> anyhow::Result<TokenResult> {
        let branch_outs = self.run_encoder_branches(batch)?;
        let code_dim = self.config.code_dim;
        let n_levels = num_quantizers(self.modality);

        let mut branch_tokens = Vec::with_capacity(4);
        for (i, out) in branch_outs.into_iter().enumerate() {
            let encoded = encode_head(&out, &self.prepare_params, i + 1, code_dim);
            let indices = rvq_encode(&encoded, &self.prepare_params, i + 1, code_dim, n_levels);
            branch_tokens.push(indices);
        }

        Ok(TokenResult {
            branch_tokens,
            n_channels: batch.n_channels,
            n_time_patches: batch.n_time_patches,
        })
    }

    pub fn reconstruct(&mut self, batch: &RlxInputBatch) -> anyhow::Result<ReconstructionResult> {
        let n = batch.n_channels;
        let a = batch.n_time_patches;
        let seq_len = n * a;
        let embed_dim = self.config.embed_dim;
        let dec_out_dim = self.config.decoder_out_dim;

        let quantized = self.encode_quantized(batch)?;
        let concat = self.run_decoder_branches(batch, &quantized)?;
        let (amp, sin, cos) = decode_heads(&concat, &self.prepare_params, embed_dim, dec_out_dim);

        Ok(ReconstructionResult {
            amplitude: amp,
            sin_phase: sin,
            cos_phase: cos,
            shape: vec![seq_len, dec_out_dim],
        })
    }

    pub fn forward(&mut self, batch: &RlxInputBatch) -> anyhow::Result<ForwardResult> {
        let b = 1usize;
        let n = batch.n_channels;
        let a = batch.n_time_patches;
        let t = self.config.patch_size;
        let embed_dim = self.config.embed_dim;
        let dec_out_dim = self.config.decoder_out_dim;
        let n_samples = n * a * t;

        // Reshape signal [B, N, T_total] → patched [B, N, A, T_patch]
        let mut patched = vec![0f32; b * n * a * t];
        for bi in 0..b {
            for ni in 0..n {
                for ai in 0..a {
                    for ti in 0..t {
                        let src = bi * n_samples + ni * a * t + ai * t + ti;
                        let dst = bi * n * a * t + ni * a * t + ai * t + ti;
                        patched[dst] = batch.signal[src];
                    }
                }
            }
        }

        let (_log_amp, amp_mean, amp_std, _sin, _cos) =
            compute_fft_components(&patched, b, n, a, t);

        let quantized = self.encode_quantized(batch)?;
        let concat = self.run_decoder_branches(batch, &quantized)?;
        let (xrec_amp, xrec_sin, xrec_cos) =
            decode_heads(&concat, &self.prepare_params, embed_dim, dec_out_dim);

        let xrec_signal = reconstruct_signal(
            &xrec_amp, &xrec_sin, &xrec_cos, amp_mean, amp_std, b, n, a, t,
        );

        let original_std = std_norm_4d(&patched, b, n, a, t);
        let reconstructed_std = std_norm_4d(&xrec_signal, b, n, a, t);

        Ok(ForwardResult {
            original_std,
            reconstructed_std,
            shape: vec![b, n * a, t],
        })
    }
}

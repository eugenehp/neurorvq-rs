//! Parity: RLX `NeuroRVQEncoder` vs Burn on identical weights and inputs.
//!
//! ```text
//! cargo test --release --no-default-features \
//!     --features burn,rlx,ndarray,rlx-cpu \
//!     --test parity_rlx_vs_burn -- --nocapture
//! ```

#![cfg(all(feature = "burn", feature = "rlx"))]

mod common;

use std::path::PathBuf;

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
use common::{burn_export, diff_max_abs, diff_rmse, synthetic_signal, test_config_path};
use neurorvq_rs::config::{ConfigOverrides, Modality, NeuroRVQConfig};
use neurorvq_rs::data::build_batch_with_modality;
use neurorvq_rs::model::tokenizer::NeuroRVQTokenizer;
use neurorvq_rs::rlx::NeuroRVQEncoder as RlxEncoder;
use rlx::Device as RlxDevice;

type B = NdArray<f32>;

fn tiny_overrides() -> ConfigOverrides {
    ConfigOverrides {
        patch_size: None,
        n_patches: Some(32),
        embed_dim: None,
        code_dim: None,
        n_code: Some(64),
        decoder_out_dim: None,
        out_chans_encoder: None,
        depth_encoder: Some(2),
        depth_decoder: Some(2),
        depth_second_stage: Some(2),
        num_heads_tokenizer: None,
        mlp_ratio_tokenizer: None,
        qkv_bias_tokenizer: None,
        init_values_tokenizer: None,
        init_values_second_stage: Some(0.0),
        init_scale_tokenizer: None,
        n_global_electrodes: Some(16),
    }
}

fn prepare_tiny_weights() -> PathBuf {
    let dir = std::env::temp_dir().join("neurorvq_parity");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("tiny_tokenizer_v2.safetensors");
    let fm_path = dir.join("tiny_fm.safetensors");
    if path.exists() && fm_path.exists() {
        return path;
    }

    let dev = NdArrayDevice::Cpu;
    let mut cfg = NeuroRVQConfig::from_yaml_with_modality(
        test_config_path().to_str().unwrap(),
        Modality::EEG,
    )
    .unwrap();
    cfg.apply_overrides(&tiny_overrides());
    cfg.n_global_electrodes = 16;

    let model = NeuroRVQTokenizer::<B>::new_with_modality(
        cfg.n_patches,
        cfg.patch_size,
        cfg.embed_dim,
        cfg.code_dim,
        cfg.n_code,
        cfg.decoder_out_dim,
        cfg.out_chans_encoder,
        cfg.depth_encoder,
        cfg.depth_decoder,
        cfg.num_heads_tokenizer,
        cfg.mlp_ratio_tokenizer,
        cfg.qkv_bias_tokenizer,
        cfg.init_values_tokenizer,
        cfg.init_scale_tokenizer,
        cfg.n_global_electrodes,
        Modality::EEG,
        &dev,
    );
    burn_export::export_tokenizer(&model, &path).expect("export tokenizer");
    burn_export::export_foundation_model(&model, &fm_path).expect("export FM");
    path
}

fn fm_weights_path() -> PathBuf {
    if let Some(w) = common::find_weights() {
        return w;
    }
    prepare_tiny_weights();
    std::env::temp_dir()
        .join("neurorvq_parity")
        .join("tiny_fm.safetensors")
}

fn weights_path() -> PathBuf {
    common::find_weights().unwrap_or_else(prepare_tiny_weights)
}

fn make_batch_burn(cfg: &NeuroRVQConfig, dev: &NdArrayDevice) -> neurorvq_rs::data::InputBatch<B> {
    let ch = &neurorvq_rs::EEG_CHANNELS[..4];
    let n_channels = ch.len();
    let n_time = neurorvq_rs::compute_n_time(cfg.n_patches, n_channels);
    let n_samples = n_time * cfg.patch_size;
    let signal = synthetic_signal(n_channels, n_samples);
    build_batch_with_modality(
        signal,
        ch,
        n_time,
        cfg.n_patches,
        n_channels,
        n_samples,
        Modality::EEG,
        dev,
    )
}

fn make_batch_rlx(cfg: &NeuroRVQConfig) -> neurorvq_rs::rlx::RlxInputBatch {
    let ch = &neurorvq_rs::EEG_CHANNELS[..4];
    let n_channels = ch.len();
    let n_time = neurorvq_rs::compute_n_time(cfg.n_patches, n_channels);
    let n_samples = n_time * cfg.patch_size;
    let signal = synthetic_signal(n_channels, n_samples);
    neurorvq_rs::rlx::build_batch(
        signal,
        ch,
        n_time,
        cfg.n_patches,
        n_channels,
        n_samples,
        Modality::EEG,
    )
}

fn load_cfg() -> NeuroRVQConfig {
    let mut cfg = NeuroRVQConfig::from_yaml_with_modality(
        test_config_path().to_str().unwrap(),
        Modality::EEG,
    )
    .unwrap();
    if common::find_weights().is_none() {
        cfg.apply_overrides(&tiny_overrides());
    }
    cfg.n_global_electrodes = 16;
    cfg
}

#[test]
fn rlx_tokenize_matches_burn() {
    let weights = weights_path();
    let cfg = load_cfg();
    let dev = NdArrayDevice::Cpu;

    let (burn_model, _) = neurorvq_rs::NeuroRVQEncoder::<B>::load_full(
        &test_config_path(),
        &weights,
        Modality::EEG,
        Some(&tiny_overrides()),
        dev.clone(),
    )
    .unwrap_or_else(|_| {
        neurorvq_rs::NeuroRVQEncoder::<B>::load_full(
            &test_config_path(),
            &weights,
            Modality::EEG,
            None,
            dev.clone(),
        )
        .unwrap()
    });

    let (mut rlx_model, _) = RlxEncoder::load_full(
        &test_config_path(),
        &weights,
        Modality::EEG,
        Some(&tiny_overrides()),
        RlxDevice::Cpu,
    )
    .unwrap();

    let burn_batch = make_batch_burn(&cfg, &dev);
    let rlx_batch = make_batch_rlx(&cfg);

    let burn_tok = burn_model.tokenize(&burn_batch).unwrap();
    let rlx_tok = rlx_model.tokenize(&rlx_batch).unwrap();

    assert_eq!(burn_tok.branch_tokens.len(), rlx_tok.branch_tokens.len());
    for (br, (b_levels, r_levels)) in burn_tok
        .branch_tokens
        .iter()
        .zip(rlx_tok.branch_tokens.iter())
        .enumerate()
    {
        assert_eq!(b_levels.len(), r_levels.len(), "branch {br} level count");
        for (lvl, (b_idx, r_idx)) in b_levels.iter().zip(r_levels.iter()).enumerate() {
            assert_eq!(b_idx, r_idx, "branch {br} level {lvl} token mismatch");
        }
    }
    eprintln!("→ tokenize parity: exact token match");
}

#[test]
fn rlx_reconstruct_matches_burn() {
    let weights = weights_path();
    let cfg = load_cfg();
    let dev = NdArrayDevice::Cpu;

    let overrides = if common::find_weights().is_none() {
        Some(tiny_overrides())
    } else {
        None
    };

    let (burn_model, _) = neurorvq_rs::NeuroRVQEncoder::<B>::load_full(
        &test_config_path(),
        &weights,
        Modality::EEG,
        overrides.as_ref(),
        dev.clone(),
    )
    .unwrap();

    let (mut rlx_model, _) = RlxEncoder::load_full(
        &test_config_path(),
        &weights,
        Modality::EEG,
        overrides.as_ref(),
        RlxDevice::Cpu,
    )
    .unwrap();

    let burn_batch = make_batch_burn(&cfg, &dev);
    let rlx_batch = make_batch_rlx(&cfg);

    let burn_out = burn_model.reconstruct(&burn_batch).unwrap();
    let rlx_out = rlx_model.reconstruct(&rlx_batch).unwrap();

    for (name, b, r) in [
        ("amp", &burn_out.amplitude, &rlx_out.amplitude),
        ("sin", &burn_out.sin_phase, &rlx_out.sin_phase),
        ("cos", &burn_out.cos_phase, &rlx_out.cos_phase),
    ] {
        assert_eq!(b.len(), r.len(), "{name} length");
        let max_abs = diff_max_abs(b, r);
        let rmse = diff_rmse(b, r);
        eprintln!("→ reconstruct {name}: max_abs={max_abs:.8}  rmse={rmse:.8}");
        assert!(max_abs < 1e-5, "{name} parity failed: max_abs={max_abs:.8}");
    }
}

#[test]
fn rlx_forward_matches_burn() {
    let weights = weights_path();
    let cfg = load_cfg();
    let dev = NdArrayDevice::Cpu;

    let overrides = if common::find_weights().is_none() {
        Some(tiny_overrides())
    } else {
        None
    };

    let (burn_model, _) = neurorvq_rs::NeuroRVQEncoder::<B>::load_full(
        &test_config_path(),
        &weights,
        Modality::EEG,
        overrides.as_ref(),
        dev.clone(),
    )
    .unwrap();

    let (mut rlx_model, _) = RlxEncoder::load_full(
        &test_config_path(),
        &weights,
        Modality::EEG,
        overrides.as_ref(),
        RlxDevice::Cpu,
    )
    .unwrap();

    let burn_batch = make_batch_burn(&cfg, &dev);
    let rlx_batch = make_batch_rlx(&cfg);

    let burn_out = burn_model.forward(&burn_batch).unwrap();
    let rlx_out = rlx_model.forward(&rlx_batch).unwrap();

    assert_eq!(burn_out.shape, rlx_out.shape);
    for (name, b, r) in [
        ("orig", &burn_out.original_std, &rlx_out.original_std),
        (
            "recon",
            &burn_out.reconstructed_std,
            &rlx_out.reconstructed_std,
        ),
    ] {
        let max_abs = diff_max_abs(b, r);
        let rmse = diff_rmse(b, r);
        eprintln!("→ forward {name}: max_abs={max_abs:.8}  rmse={rmse:.8}");
        assert!(max_abs < 1e-5, "{name} parity failed: max_abs={max_abs:.8}");
    }
}

#[test]
fn rlx_fm_encode_matches_burn() {
    let weights = fm_weights_path();
    let cfg = load_cfg();
    let dev = NdArrayDevice::Cpu;
    let overrides = if common::find_weights().is_none() {
        Some(tiny_overrides())
    } else {
        None
    };

    let (burn_fm, _) = neurorvq_rs::NeuroRVQFoundationModel::<B>::load_full(
        &test_config_path(),
        &weights,
        Modality::EEG,
        overrides.as_ref(),
        dev.clone(),
    )
    .unwrap();

    let (mut rlx_fm, _) = neurorvq_rs::rlx::NeuroRVQFoundationModel::load_full(
        &test_config_path(),
        &weights,
        Modality::EEG,
        overrides.as_ref(),
        RlxDevice::Cpu,
    )
    .unwrap();

    let burn_batch = make_batch_burn(&cfg, &dev);
    let rlx_batch = make_batch_rlx(&cfg);

    let burn_out = burn_fm.encode(&burn_batch).unwrap();
    let rlx_out = rlx_fm.encode(&rlx_batch).unwrap();

    assert_eq!(
        burn_out.branch_features.len(),
        rlx_out.branch_features.len()
    );
    for (i, (b, r)) in burn_out
        .branch_features
        .iter()
        .zip(rlx_out.branch_features.iter())
        .enumerate()
    {
        assert_eq!(b.len(), r.len(), "FM branch {i} length");
        let max_abs = diff_max_abs(b, r);
        eprintln!("→ FM branch {i}: max_abs={max_abs:.8}");
        assert!(max_abs < 1e-5, "FM branch {i} failed: max_abs={max_abs:.8}");
    }
}

//! NeuroRVQ inference — thin CLI over [`neurorvq_rs::rlx`].

use std::path::Path;
use std::time::Instant;

use clap::{Parser, ValueEnum};

use neurorvq_rs::rlx::{build_batch, NeuroRVQEncoder, NeuroRVQFoundationModel, RlxInputBatch};
use neurorvq_rs::{channels, init_threads, ConfigOverrides, Modality, NeuroRVQConfig};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DeviceArg {
    Cpu,
    Metal,
    Mlx,
    Gpu,
    Cuda,
    Rocm,
    Tpu,
}

impl DeviceArg {
    fn into_rlx(self) -> rlx::Device {
        match self {
            Self::Cpu => rlx::Device::Cpu,
            Self::Metal => rlx::Device::Metal,
            Self::Mlx => rlx::Device::Mlx,
            Self::Gpu => rlx::Device::Gpu,
            Self::Cuda => rlx::Device::Cuda,
            Self::Rocm => rlx::Device::Rocm,
            Self::Tpu => rlx::Device::Tpu,
        }
    }
}

#[derive(Parser, Debug)]
#[command(about = "NeuroRVQ biosignal tokenizer inference (RLX runtime)")]
struct Args {
    #[arg(long, default_value = "cpu")]
    device: DeviceArg,

    #[arg(long)]
    weights: String,

    #[arg(long)]
    config: String,

    #[arg(long)]
    modality: Option<String>,

    #[arg(long, default_value = "tokenize")]
    mode: String,

    #[arg(long, short = 'v')]
    verbose: bool,

    #[arg(long, env = "RAYON_NUM_THREADS")]
    threads: Option<usize>,

    #[arg(long)]
    patch_size: Option<usize>,

    #[arg(long)]
    n_patches: Option<usize>,

    #[arg(long)]
    embed_dim: Option<usize>,

    #[arg(long)]
    code_dim: Option<usize>,

    #[arg(long)]
    n_code: Option<usize>,

    #[arg(long)]
    decoder_out_dim: Option<usize>,

    #[arg(long)]
    out_chans_encoder: Option<usize>,

    #[arg(long)]
    depth_encoder: Option<usize>,

    #[arg(long)]
    depth_decoder: Option<usize>,

    #[arg(long)]
    num_heads: Option<usize>,

    #[arg(long)]
    mlp_ratio: Option<f64>,

    #[arg(long)]
    init_values: Option<f64>,

    #[arg(long)]
    init_scale: Option<f64>,

    #[arg(long)]
    qkv_bias: Option<bool>,

    #[arg(long)]
    n_global_electrodes: Option<usize>,
}

impl Args {
    fn overrides(&self) -> ConfigOverrides {
        ConfigOverrides {
            patch_size: self.patch_size,
            n_patches: self.n_patches,
            embed_dim: self.embed_dim,
            code_dim: self.code_dim,
            n_code: self.n_code,
            decoder_out_dim: self.decoder_out_dim,
            out_chans_encoder: self.out_chans_encoder,
            depth_encoder: self.depth_encoder,
            depth_decoder: self.depth_decoder,
            depth_second_stage: None,
            num_heads_tokenizer: self.num_heads,
            mlp_ratio_tokenizer: self.mlp_ratio,
            qkv_bias_tokenizer: self.qkv_bias,
            init_values_tokenizer: self.init_values,
            init_values_second_stage: None,
            init_scale_tokenizer: self.init_scale,
            n_global_electrodes: self.n_global_electrodes,
        }
    }

    fn has_overrides(&self) -> bool {
        let o = self.overrides();
        o.patch_size.is_some()
            || o.n_patches.is_some()
            || o.embed_dim.is_some()
            || o.code_dim.is_some()
            || o.n_code.is_some()
            || o.decoder_out_dim.is_some()
            || o.out_chans_encoder.is_some()
            || o.depth_encoder.is_some()
            || o.depth_decoder.is_some()
            || o.num_heads_tokenizer.is_some()
            || o.mlp_ratio_tokenizer.is_some()
            || o.qkv_bias_tokenizer.is_some()
            || o.init_values_tokenizer.is_some()
            || o.init_scale_tokenizer.is_some()
            || o.n_global_electrodes.is_some()
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let n_threads = init_threads(args.threads);
    let device = args.device.into_rlx();
    let t0 = Instant::now();

    let modality: Modality = match &args.modality {
        Some(m) => m.parse()?,
        None => {
            let cfg = NeuroRVQConfig::from_yaml(&args.config)?;
            let m = cfg.resolve_modality();
            eprintln!("Auto-detected modality: {m} (from config filename)");
            m
        }
    };

    eprintln!("Device   : {:?}  ({n_threads} threads)", device);
    eprintln!("Modality : {modality}");

    match args.mode.as_str() {
        "fm" => run_fm(&args, modality, device, t0),
        _ => run_tokenizer(&args, modality, device, t0),
    }
}

fn run_tokenizer(
    args: &Args,
    modality: Modality,
    device: rlx::Device,
    t0: Instant,
) -> anyhow::Result<()> {
    let overrides = args.overrides();
    let ovr = if args.has_overrides() {
        Some(&overrides)
    } else {
        None
    };

    let (mut model, ms_weights) = NeuroRVQEncoder::load_full(
        Path::new(&args.config),
        Path::new(&args.weights),
        modality,
        ovr,
        device,
    )?;

    eprintln!("Model    : {}  ({ms_weights:.0} ms)", model.describe());

    let batch = make_dummy_batch(&model.config, modality);

    let t_inf = Instant::now();
    match args.mode.as_str() {
        "tokenize" => {
            let result = model.tokenize(&batch)?;
            let ms_infer = t_inf.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "Tokens   : {} branches × {} RVQ levels  ({ms_infer:.1} ms)",
                result.branch_tokens.len(),
                result.branch_tokens[0].len(),
            );
            if args.verbose {
                for (br, tokens) in result.branch_tokens.iter().enumerate() {
                    for (lvl, indices) in tokens.iter().enumerate() {
                        eprintln!(
                            "  Branch {br} Level {lvl}: {} indices, first 5: {:?}",
                            indices.len(),
                            &indices[..5.min(indices.len())]
                        );
                    }
                }
            }
        }
        "reconstruct" => {
            let result = model.reconstruct(&batch)?;
            let ms_infer = t_inf.elapsed().as_secs_f64() * 1000.0;
            eprintln!("Output   : shape={:?}  ({ms_infer:.1} ms)", result.shape);
        }
        "forward" => {
            let result = model.forward(&batch)?;
            let ms_infer = t_inf.elapsed().as_secs_f64() * 1000.0;
            eprintln!("Forward  : shape={:?}  ({ms_infer:.1} ms)", result.shape);
        }
        other => {
            anyhow::bail!(
                "Unknown mode: {other}. Use 'tokenize', 'reconstruct', 'forward', or 'fm'."
            );
        }
    }

    print_timing(ms_weights, t0);
    Ok(())
}

fn run_fm(args: &Args, modality: Modality, device: rlx::Device, t0: Instant) -> anyhow::Result<()> {
    let (mut fm, ms_weights) = NeuroRVQFoundationModel::load(
        Path::new(&args.config),
        Path::new(&args.weights),
        modality,
        device,
    )?;

    eprintln!("FM       : {}  ({ms_weights:.0} ms)", fm.describe());

    let batch = make_dummy_batch(&fm.config, modality);

    let t_inf = Instant::now();
    let result = fm.encode(&batch)?;
    let ms_infer = t_inf.elapsed().as_secs_f64() * 1000.0;

    eprintln!(
        "Features : {} branches × shape={:?}  ({ms_infer:.1} ms)",
        result.branch_features.len(),
        result.shape
    );

    if args.verbose {
        for (i, feats) in result.branch_features.iter().enumerate() {
            let mean: f64 = feats.iter().map(|&v| v as f64).sum::<f64>() / feats.len() as f64;
            eprintln!("  Branch {i}: len={} mean={mean:+.6}", feats.len());
        }
    }

    print_timing(ms_weights, t0);
    Ok(())
}

fn make_dummy_batch(config: &NeuroRVQConfig, modality: Modality) -> RlxInputBatch {
    let ch = channels::global_channels(modality);
    let n_channels = ch.len().min(16);
    let channel_names: Vec<&str> = ch[..n_channels].to_vec();
    let patch_size = config.patch_size;
    let n_time = channels::compute_n_time(config.n_patches, n_channels);
    let n_samples = n_time * patch_size;

    let signal = vec![0.0f32; n_channels * n_samples];
    build_batch(
        signal,
        &channel_names,
        n_time,
        config.n_patches,
        n_channels,
        n_samples,
        modality,
    )
}

fn print_timing(ms_weights: f64, t0: Instant) {
    let ms_total = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!("── Timing ───────────────────────────────────────────────────────");
    eprintln!("  Weights  : {ms_weights:.0} ms");
    eprintln!("  Total    : {ms_total:.0} ms");
}

#![cfg(all(feature = "burn", feature = "ndarray"))]

use std::io::Write;
/// Benchmark suite for neurorvq-rs.
///
/// Measures model construction time and inference latency across all three
/// modalities (EEG, ECG, EMG) and varying input sizes. Outputs CSV results
/// and generates SVG charts.
///
/// Run:
///   cargo run --release --bin bench
use std::panic;
use std::time::Instant;

use neurorvq_rs::channels;
use neurorvq_rs::config::{Modality, NeuroRVQConfig};
use neurorvq_rs::data;
use neurorvq_rs::model::tokenizer::NeuroRVQTokenizer;

#[cfg(feature = "ndarray")]
type B = burn::backend::NdArray;
#[cfg(feature = "ndarray")]
fn device() -> burn::backend::ndarray::NdArrayDevice {
    burn::backend::ndarray::NdArrayDevice::Cpu
}

#[cfg(all(feature = "wgpu", not(feature = "ndarray")))]
type B = burn::backend::Wgpu;
#[cfg(all(feature = "wgpu", not(feature = "ndarray")))]
fn device() -> burn::backend::wgpu::WgpuDevice {
    burn::backend::wgpu::WgpuDevice::DefaultDevice
}

// ── Benchmark Parameters ──────────────────────────────────────────────────────

const WARMUP: usize = 2;
const ITERS: usize = 10;

struct BenchCase {
    label: String,
    modality: Modality,
    n_channels: usize,
    n_time: usize,
}

fn bench_cases() -> Vec<BenchCase> {
    let mut cases = Vec::new();

    for &nch in &[4, 8, 16, 32, 64] {
        let n_time = (256 / nch).max(1);
        cases.push(BenchCase {
            label: format!("EEG {}ch x{}t", nch, n_time),
            modality: Modality::EEG,
            n_channels: nch,
            n_time,
        });
    }

    for &nch in &[4, 8, 12, 15] {
        let n_time = (600usize / nch).max(1);
        cases.push(BenchCase {
            label: format!("ECG {}ch x{}t", nch, n_time),
            modality: Modality::ECG,
            n_channels: nch,
            n_time,
        });
    }

    for &nch in &[4, 8, 16] {
        let n_time = (256 / nch).max(1);
        cases.push(BenchCase {
            label: format!("EMG {}ch x{}t", nch, n_time),
            modality: Modality::EMG,
            n_channels: nch,
            n_time,
        });
    }

    cases
}

// ── Benchmark Runner ──────────────────────────────────────────────────────────

struct BenchResult {
    label: String,
    modality: String,
    n_channels: usize,
    n_time: usize,
    n_patches: usize,
    patch_size: usize,
    construct_ms: f64,
    encode_mean_ms: f64,
    encode_std_ms: f64,
    tokenize_mean_ms: f64,
    tokenize_std_ms: f64,
}

fn load_config(modality: Modality) -> NeuroRVQConfig {
    let path = match modality {
        Modality::EEG => "flags/NeuroRVQ_EEG_v1.yml",
        Modality::ECG => "flags/NeuroRVQ_ECG_v1.yml",
        Modality::EMG => "flags/NeuroRVQ_EMG_v1.yml",
    };
    let mut cfg = NeuroRVQConfig::from_yaml(path).unwrap();
    cfg.n_global_electrodes = channels::global_vocab_size(modality);
    cfg
}

fn run_bench(case: &BenchCase) -> BenchResult {
    let dev = device();
    let cfg = load_config(case.modality);

    let ch = channels::global_channels(case.modality);
    let n_channels = case.n_channels.min(ch.len());
    let channel_names: Vec<&str> = ch[..n_channels].to_vec();
    let n_time = case.n_time;
    let n_patches = cfg.n_patches;
    let patch_size = cfg.patch_size;
    let n_samples = n_time * patch_size;

    // Measure model construction
    let t0 = Instant::now();
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
        case.modality,
        &dev,
    );
    let construct_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Build input batch
    let signal = vec![0.1f32; n_channels * n_samples];
    let batch = data::build_batch_with_modality(
        signal,
        &channel_names,
        n_time,
        n_patches,
        n_channels,
        n_samples,
        case.modality,
        &dev,
    );

    // --- Benchmark: encode (encoder + RVQ quantize) ---
    let mut encode_times = Vec::with_capacity(WARMUP + ITERS);
    for i in 0..(WARMUP + ITERS) {
        let sig = batch.signal.clone();
        let [b, n, t] = sig.dims();
        let a = t / patch_size;
        let x = sig.reshape([b, n, a, patch_size]);

        let t0 = Instant::now();
        let _enc = model.encode(x, batch.temporal_ix.clone(), batch.spatial_ix.clone());
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if i >= WARMUP {
            encode_times.push(ms);
        }
    }

    // --- Benchmark: full tokenize (encode + indices extraction) ---
    let mut tok_times = Vec::with_capacity(WARMUP + ITERS);
    for i in 0..(WARMUP + ITERS) {
        let sig = batch.signal.clone();
        let [b, n, t] = sig.dims();
        let a = t / patch_size;
        let x = sig.reshape([b, n, a, patch_size]);

        let t0 = Instant::now();
        let enc = model.get_tokens(x, batch.temporal_ix.clone(), batch.spatial_ix.clone());
        // Force materialization of indices
        for branch_idx in &enc.indices {
            for level_idx in branch_idx {
                let _ = level_idx.clone().into_data();
            }
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if i >= WARMUP {
            tok_times.push(ms);
        }
    }

    let (enc_mean, enc_std) = mean_std(&encode_times);
    let (tok_mean, tok_std) = mean_std(&tok_times);

    BenchResult {
        label: case.label.clone(),
        modality: case.modality.to_string(),
        n_channels,
        n_time,
        n_patches,
        patch_size,
        construct_ms,
        encode_mean_ms: enc_mean,
        encode_std_ms: enc_std,
        tokenize_mean_ms: tok_mean,
        tokenize_std_ms: tok_std,
    }
}

fn mean_std(vals: &[f64]) -> (f64, f64) {
    let n = vals.len() as f64;
    let mean = vals.iter().sum::<f64>() / n;
    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

// ── SVG Chart Generation ──────────────────────────────────────────────────────

fn bar_chart_svg(
    title: &str,
    labels: &[String],
    values: &[f64],
    errors: &[f64],
    colors: &[String],
    y_label: &str,
) -> String {
    let n = values.len();
    let margin_l = 80.0_f64;
    let margin_r = 20.0;
    let margin_t = 50.0;
    let margin_b = 120.0;
    let bar_w = 36.0;
    let gap = 12.0;
    let chart_w = n as f64 * (bar_w + gap) + gap;
    let w = margin_l + chart_w + margin_r;
    let chart_h = 260.0;
    let h = margin_t + chart_h + margin_b;

    let max_val = values
        .iter()
        .zip(errors.iter())
        .map(|(v, e)| v + e)
        .fold(0.0_f64, f64::max)
        * 1.15;
    let max_val = if max_val < 1e-9 { 1.0 } else { max_val };

    let mut s = String::new();
    s += &format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" \
         font-family=\"system-ui,-apple-system,sans-serif\" font-size=\"12\">\n",
        w, h
    );
    s += &format!("<rect width=\"{}\" height=\"{}\" fill=\"white\"/>\n", w, h);
    s += &format!(
        "<text x=\"{}\" y=\"28\" text-anchor=\"middle\" font-size=\"15\" \
         font-weight=\"600\">{}</text>\n",
        w / 2.0,
        title
    );

    // Y axis gridlines + labels
    let n_ticks = 5;
    for i in 0..=n_ticks {
        let frac = i as f64 / n_ticks as f64;
        let y = margin_t + chart_h * (1.0 - frac);
        let val = max_val * frac;
        s += &format!(
            "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#e0e0e0\" stroke-width=\"1\"/>\n",
            margin_l, y, margin_l + chart_w, y
        );
        s += &format!(
            "<text x=\"{}\" y=\"{}\" text-anchor=\"end\" fill=\"#666\" font-size=\"10\">{:.1}</text>\n",
            margin_l - 6.0,
            y + 4.0,
            val
        );
    }

    // Y axis label
    let ym = margin_t + chart_h / 2.0;
    s += &format!(
        "<text x=\"14\" y=\"{}\" text-anchor=\"middle\" \
         transform=\"rotate(-90,14,{})\" fill=\"#333\" font-size=\"11\">{}</text>\n",
        ym, ym, y_label
    );

    // Bars
    for i in 0..n {
        let x = margin_l + gap + i as f64 * (bar_w + gap);
        let bar_h = (values[i] / max_val) * chart_h;
        let y = margin_t + chart_h - bar_h;
        let color = &colors[i % colors.len()];

        s += &format!(
            "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{}\" rx=\"3\"/>\n",
            x, y, bar_w, bar_h, color
        );

        // Error bar
        if errors[i] > 0.0 {
            let err_h = (errors[i] / max_val) * chart_h;
            let cx = x + bar_w / 2.0;
            s += &format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#333\" stroke-width=\"1.5\"/>\n",
                cx, y - err_h, cx, y + err_h
            );
            s += &format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#333\" stroke-width=\"1.5\"/>\n",
                cx - 4.0, y - err_h, cx + 4.0, y - err_h
            );
            s += &format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#333\" stroke-width=\"1.5\"/>\n",
                cx - 4.0, y + err_h, cx + 4.0, y + err_h
            );
        }

        // Value on top
        s += &format!(
            "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" fill=\"#333\" font-size=\"9\">{:.1}</text>\n",
            x + bar_w / 2.0,
            y - 6.0,
            values[i]
        );

        // X label (rotated)
        let lx = x + bar_w / 2.0;
        let ly = margin_t + chart_h + 10.0;
        s += &format!(
            "<text x=\"{}\" y=\"{}\" text-anchor=\"start\" \
             transform=\"rotate(45,{},{})\" fill=\"#333\" font-size=\"10\">{}</text>\n",
            lx, ly, lx, ly, labels[i]
        );
    }

    s += "</svg>\n";
    s
}

fn color_for_modality(m: &str) -> String {
    match m {
        "EEG" => "#4285f4".to_string(),
        "ECG" => "#ea4335".to_string(),
        "EMG" => "#34a853".to_string(),
        _ => "#999999".to_string(),
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    println!("neurorvq-rs benchmark suite");
    println!("==========================\n");

    let backend_name;
    #[cfg(feature = "blas-accelerate")]
    {
        backend_name = "NdArray + Apple Accelerate";
    }
    #[cfg(all(feature = "ndarray", not(feature = "blas-accelerate")))]
    {
        backend_name = "NdArray + Rayon";
    }
    #[cfg(all(feature = "wgpu", not(feature = "ndarray")))]
    {
        backend_name = "wgpu";
    }

    println!("Backend: {backend_name}");
    println!("Warmup: {WARMUP} iters, Measurement: {ITERS} iters\n");

    let cases = bench_cases();
    let mut results = Vec::new();

    for (i, case) in cases.iter().enumerate() {
        print!("[{}/{}] {} ... ", i + 1, cases.len(), case.label);
        std::io::stdout().flush().unwrap();

        // Suppress panic output for skipped cases
        let prev_hook = panic::take_hook();
        panic::set_hook(Box::new(|_| {}));
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| run_bench(case)));
        panic::set_hook(prev_hook);

        match result {
            Ok(r) => {
                println!(
                    "construct={:.0}ms  encode={:.1}+/-{:.1}ms  tokenize={:.1}+/-{:.1}ms",
                    r.construct_ms,
                    r.encode_mean_ms,
                    r.encode_std_ms,
                    r.tokenize_mean_ms,
                    r.tokenize_std_ms,
                );
                results.push(r);
            }
            Err(_) => {
                println!("SKIPPED (backend limitation for this config)");
            }
        }
    }

    std::fs::create_dir_all("figures").unwrap();

    // ── CSV ──────────────────────────────────────────────────────────────
    let csv_suffix: &str;
    #[cfg(feature = "blas-accelerate")]
    {
        csv_suffix = "_accelerate";
    }
    #[cfg(all(feature = "ndarray", not(feature = "blas-accelerate")))]
    {
        csv_suffix = "_ndarray";
    }
    #[cfg(all(feature = "wgpu", not(feature = "ndarray")))]
    {
        csv_suffix = "_wgpu";
    }
    let csv_path = format!("figures/benchmark_results{csv_suffix}.csv");
    // Also write to default path for backward compat
    let csv_path_default = "figures/benchmark_results.csv";
    {
        let mut f = std::fs::File::create(&csv_path).unwrap();
        writeln!(f, "label,modality,n_channels,n_time,n_patches,patch_size,construct_ms,encode_mean_ms,encode_std_ms,tokenize_mean_ms,tokenize_std_ms").unwrap();
        for r in &results {
            writeln!(
                f,
                "{},{},{},{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2}",
                r.label,
                r.modality,
                r.n_channels,
                r.n_time,
                r.n_patches,
                r.patch_size,
                r.construct_ms,
                r.encode_mean_ms,
                r.encode_std_ms,
                r.tokenize_mean_ms,
                r.tokenize_std_ms,
            )
            .unwrap();
        }
    }
    // Copy to default path too
    let _ = std::fs::copy(&csv_path, csv_path_default);
    println!("\nCSV: {csv_path}");

    // ── Chart 1: Tokenize latency (all) ──────────────────────────────────
    {
        let labels: Vec<String> = results.iter().map(|r| r.label.clone()).collect();
        let values: Vec<f64> = results.iter().map(|r| r.tokenize_mean_ms).collect();
        let errors: Vec<f64> = results.iter().map(|r| r.tokenize_std_ms).collect();
        let colors: Vec<String> = results
            .iter()
            .map(|r| color_for_modality(&r.modality))
            .collect();
        let svg = bar_chart_svg(
            "Tokenize Latency (all configurations)",
            &labels,
            &values,
            &errors,
            &colors,
            "Latency (ms)",
        );
        std::fs::write("figures/tokenize_latency.svg", &svg).unwrap();
        println!("Chart: figures/tokenize_latency.svg");
    }

    // ── Chart 2: Encode latency (all) ────────────────────────────────────
    {
        let labels: Vec<String> = results.iter().map(|r| r.label.clone()).collect();
        let values: Vec<f64> = results.iter().map(|r| r.encode_mean_ms).collect();
        let errors: Vec<f64> = results.iter().map(|r| r.encode_std_ms).collect();
        let colors: Vec<String> = results
            .iter()
            .map(|r| color_for_modality(&r.modality))
            .collect();
        let svg = bar_chart_svg(
            "Encode Latency (encoder + RVQ quantize)",
            &labels,
            &values,
            &errors,
            &colors,
            "Latency (ms)",
        );
        std::fs::write("figures/encode_latency.svg", &svg).unwrap();
        println!("Chart: figures/encode_latency.svg");
    }

    // ── Chart 3: Construction time per modality ──────────────────────────
    {
        let mods = ["EEG", "ECG", "EMG"];
        let mut labels = Vec::new();
        let mut values = Vec::new();
        let colors: Vec<String> = mods.iter().map(|m| color_for_modality(m)).collect();
        for m in &mods {
            let mr: Vec<&BenchResult> = results.iter().filter(|r| r.modality == *m).collect();
            if mr.is_empty() {
                continue;
            }
            labels.push(m.to_string());
            values.push(mr.iter().map(|r| r.construct_ms).sum::<f64>() / mr.len() as f64);
        }
        let errors = vec![0.0; values.len()];
        let svg = bar_chart_svg(
            "Model Construction Time (avg per modality)",
            &labels,
            &values,
            &errors,
            &colors,
            "Time (ms)",
        );
        std::fs::write("figures/construction_time.svg", &svg).unwrap();
        println!("Chart: figures/construction_time.svg");
    }

    // ── Chart 4: EEG scaling ─────────────────────────────────────────────
    {
        let eeg: Vec<&BenchResult> = results.iter().filter(|r| r.modality == "EEG").collect();
        let labels: Vec<String> = eeg.iter().map(|r| format!("{}ch", r.n_channels)).collect();
        let values: Vec<f64> = eeg.iter().map(|r| r.tokenize_mean_ms).collect();
        let errors: Vec<f64> = eeg.iter().map(|r| r.tokenize_std_ms).collect();
        let colors = vec!["#4285f4".to_string()];
        let svg = bar_chart_svg(
            "EEG Tokenize Latency vs Channel Count",
            &labels,
            &values,
            &errors,
            &colors,
            "Latency (ms)",
        );
        std::fs::write("figures/eeg_scaling.svg", &svg).unwrap();
        println!("Chart: figures/eeg_scaling.svg");
    }

    // ── Markdown summary ─────────────────────────────────────────────────
    let md_path = "figures/benchmark_summary.md";
    {
        let mut f = std::fs::File::create(md_path).unwrap();
        writeln!(f, "# neurorvq-rs Benchmark Results\n").unwrap();
        writeln!(f, "**Platform:** Apple M4 Pro, 64 GB RAM, macOS (arm64)  ").unwrap();
        writeln!(f, "**Backend:** {}  ", backend_name).unwrap();
        writeln!(f, "**Iterations:** {} (after {} warmup)\n", ITERS, WARMUP).unwrap();
        writeln!(f, "| Configuration | Modality | Channels | Patches | Construct (ms) | Encode (ms) | Tokenize (ms) |").unwrap();
        writeln!(f, "|---|---|---:|---:|---:|---:|---:|").unwrap();
        for r in &results {
            writeln!(
                f,
                "| {} | {} | {} | {} | {:.0} | {:.1} ± {:.1} | {:.1} ± {:.1} |",
                r.label,
                r.modality,
                r.n_channels,
                r.n_time,
                r.construct_ms,
                r.encode_mean_ms,
                r.encode_std_ms,
                r.tokenize_mean_ms,
                r.tokenize_std_ms,
            )
            .unwrap();
        }
        writeln!(f).unwrap();
        writeln!(f, "### Charts\n").unwrap();
        writeln!(f, "![Tokenize Latency](tokenize_latency.svg)\n").unwrap();
        writeln!(f, "![Encode Latency](encode_latency.svg)\n").unwrap();
        writeln!(f, "![Construction Time](construction_time.svg)\n").unwrap();
        writeln!(f, "![EEG Scaling](eeg_scaling.svg)\n").unwrap();
    }
    println!("Summary: {md_path}");
    println!("\nDone!");
}

use crate::config::Modality;
use crate::model::foundation::NeuroRVQFM;
use crate::model::rvq::ResidualVQ;
use burn::nn::{Linear, LinearConfig};
/// NeuroRVQ Tokenizer — the full encoder-quantizer-decoder pipeline.
///
/// Python: `NeuroRVQTokenizer` class in NeuroRVQ.py
///
/// Pipeline:
///   1. Encoder: raw signal → 4 multi-scale feature branches
///   2. Encode heads: project each branch to codebook dim
///   3. RVQ: quantize each branch (8 levels × 4 branches)
///   4. Decoder: quantized → 4 decoded branches → concat
///   5. Reconstruction heads: predict FFT amplitude, sin(phase), cos(phase)
///   6. Inverse FFT → reconstructed signal
use burn::prelude::*;
use burn::tensor::activation::{gelu, tanh};

/// Number of RVQ quantizer levels per modality.
/// Python: hardcoded in NeuroRVQTokenizer.__init__
///   EEG/ECG: num_quantizers = 8
///   EMG:     num_quantizers = 16
pub fn num_quantizers(modality: Modality) -> usize {
    match modality {
        Modality::EEG | Modality::ECG => 8,
        Modality::EMG => 16,
    }
}

/// NeuroRVQ Tokenizer.
#[derive(Module, Debug)]
pub struct NeuroRVQTokenizer<B: Backend> {
    /// Encoder (use_as_encoder = true).
    pub encoder: NeuroRVQFM<B>,
    /// Decoder (use_as_encoder = false).
    pub decoder: NeuroRVQFM<B>,

    /// 4 RVQ codebooks, one per multi-scale branch.
    pub quantize_1: ResidualVQ<B>,
    pub quantize_2: ResidualVQ<B>,
    pub quantize_3: ResidualVQ<B>,
    pub quantize_4: ResidualVQ<B>,

    /// Encoding heads: project encoder output → codebook dim.
    pub encode_head_1_fc1: Linear<B>,
    pub encode_head_1_fc2: Linear<B>,
    pub encode_head_2_fc1: Linear<B>,
    pub encode_head_2_fc2: Linear<B>,
    pub encode_head_3_fc1: Linear<B>,
    pub encode_head_3_fc2: Linear<B>,
    pub encode_head_4_fc1: Linear<B>,
    pub encode_head_4_fc2: Linear<B>,

    /// Decoding heads: predict FFT amplitude, sin(phase), cos(phase).
    pub decode_amp_fc1: Linear<B>,
    pub decode_amp_fc2: Linear<B>,
    pub decode_sin_fc1: Linear<B>,
    pub decode_sin_fc2: Linear<B>,
    pub decode_cos_fc1: Linear<B>,
    pub decode_cos_fc2: Linear<B>,

    pub patch_size: usize,
    pub code_dim: usize,
    pub embed_dim: usize,
    pub decoder_out_dim: usize,
}

impl<B: Backend> NeuroRVQTokenizer<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_patches: usize,
        patch_size: usize,
        embed_dim: usize,
        code_dim: usize,
        n_code: usize,
        decoder_out_dim: usize,
        // Encoder config
        out_chans_encoder: usize,
        depth_encoder: usize,
        // Decoder config
        depth_decoder: usize,
        // Shared config
        num_heads: usize,
        mlp_ratio: f64,
        qkv_bias: bool,
        init_values: f64,
        _init_scale: f64,
        n_global_electrodes: usize,
        device: &B::Device,
    ) -> Self {
        Self::new_with_modality(
            n_patches,
            patch_size,
            embed_dim,
            code_dim,
            n_code,
            decoder_out_dim,
            out_chans_encoder,
            depth_encoder,
            depth_decoder,
            num_heads,
            mlp_ratio,
            qkv_bias,
            init_values,
            _init_scale,
            n_global_electrodes,
            Modality::EEG,
            device,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_modality(
        n_patches: usize,
        patch_size: usize,
        embed_dim: usize,
        code_dim: usize,
        n_code: usize,
        decoder_out_dim: usize,
        out_chans_encoder: usize,
        depth_encoder: usize,
        depth_decoder: usize,
        num_heads: usize,
        mlp_ratio: f64,
        qkv_bias: bool,
        init_values: f64,
        _init_scale: f64,
        n_global_electrodes: usize,
        modality: Modality,
        device: &B::Device,
    ) -> Self {
        // Encoder
        let encoder = NeuroRVQFM::new_with_modality(
            n_patches,
            patch_size,
            1,
            out_chans_encoder,
            0,
            embed_dim,
            depth_encoder,
            num_heads,
            mlp_ratio,
            qkv_bias,
            init_values,
            n_global_electrodes,
            true,
            modality,
            device,
        );

        // Decoder (modality doesn't affect decoder since it uses PatchEmbed, not MultiScaleConv)
        let decoder = NeuroRVQFM::new_with_modality(
            n_patches,
            patch_size,
            code_dim,
            0,
            0,
            embed_dim,
            depth_decoder,
            num_heads,
            mlp_ratio,
            qkv_bias,
            init_values,
            n_global_electrodes,
            false,
            modality,
            device,
        );

        // 4 RVQ codebooks — num_quantizers depends on modality (EEG/ECG=8, EMG=16)
        let nq = num_quantizers(modality);
        let quantize_1 = ResidualVQ::new(nq, n_code, code_dim, device);
        let quantize_2 = ResidualVQ::new(nq, n_code, code_dim, device);
        let quantize_3 = ResidualVQ::new(nq, n_code, code_dim, device);
        let quantize_4 = ResidualVQ::new(nq, n_code, code_dim, device);

        // Encode heads: embed_dim → embed_dim → Tanh → code_dim
        let mk_encode_head = |device: &B::Device| -> (Linear<B>, Linear<B>) {
            (
                LinearConfig::new(embed_dim, embed_dim)
                    .with_bias(true)
                    .init(device),
                LinearConfig::new(embed_dim, code_dim)
                    .with_bias(true)
                    .init(device),
            )
        };
        let (eh1_fc1, eh1_fc2) = mk_encode_head(device);
        let (eh2_fc1, eh2_fc2) = mk_encode_head(device);
        let (eh3_fc1, eh3_fc2) = mk_encode_head(device);
        let (eh4_fc1, eh4_fc2) = mk_encode_head(device);

        // Decode heads
        let dec_in = 4 * embed_dim; // concat of 4 decoder branches
        let decode_amp_fc1 = LinearConfig::new(dec_in, embed_dim)
            .with_bias(true)
            .init(device);
        let decode_amp_fc2 = LinearConfig::new(embed_dim, decoder_out_dim)
            .with_bias(true)
            .init(device);
        let decode_sin_fc1 = LinearConfig::new(dec_in, embed_dim)
            .with_bias(true)
            .init(device);
        let decode_sin_fc2 = LinearConfig::new(embed_dim, decoder_out_dim)
            .with_bias(true)
            .init(device);
        let decode_cos_fc1 = LinearConfig::new(dec_in, embed_dim)
            .with_bias(true)
            .init(device);
        let decode_cos_fc2 = LinearConfig::new(embed_dim, decoder_out_dim)
            .with_bias(true)
            .init(device);

        Self {
            encoder,
            decoder,
            quantize_1,
            quantize_2,
            quantize_3,
            quantize_4,
            encode_head_1_fc1: eh1_fc1,
            encode_head_1_fc2: eh1_fc2,
            encode_head_2_fc1: eh2_fc1,
            encode_head_2_fc2: eh2_fc2,
            encode_head_3_fc1: eh3_fc1,
            encode_head_3_fc2: eh3_fc2,
            encode_head_4_fc1: eh4_fc1,
            encode_head_4_fc2: eh4_fc2,
            decode_amp_fc1,
            decode_amp_fc2,
            decode_sin_fc1,
            decode_sin_fc2,
            decode_cos_fc1,
            decode_cos_fc2,
            patch_size,
            code_dim,
            embed_dim,
            decoder_out_dim,
        }
    }

    /// Encode raw signal to quantized vectors.
    ///
    /// x: [B, N, A, T]
    /// Returns: 4 quantized tensors (one per branch), each [B, code_dim, n_electrodes, n_time_per_electrode]
    pub fn encode(
        &self,
        x: Tensor<B, 4>,
        temporal_ix: Tensor<B, 2, Int>,
        spatial_ix: Tensor<B, 2, Int>,
    ) -> EncodeOutput<B> {
        let [b, n, _a, _t] = x.dims();

        // Encoder forward → 4 branch features
        let (f1, f2, f3, f4) = self.encoder.forward_encoder(x, temporal_ix, spatial_ix);

        // Apply encode task layers: Linear → Tanh → Linear
        let apply_head = |f: Tensor<B, 3>, fc1: &Linear<B>, fc2: &Linear<B>| -> Tensor<B, 3> {
            fc2.forward(tanh(fc1.forward(f)))
        };

        let q1 = apply_head(f1, &self.encode_head_1_fc1, &self.encode_head_1_fc2);
        let q2 = apply_head(f2, &self.encode_head_2_fc1, &self.encode_head_2_fc2);
        let q3 = apply_head(f3, &self.encode_head_3_fc1, &self.encode_head_3_fc2);
        let q4 = apply_head(f4, &self.encode_head_4_fc1, &self.encode_head_4_fc2);

        // Reshape for quantizer: [B, seq, code_dim] → [B, code_dim, n, n_time_per_electrode]
        let seq_len = q1.dims()[1];
        let w = seq_len / n;

        let to_4d = |t: Tensor<B, 3>| -> Tensor<B, 4> {
            // [B, (h w), c] → [B, c, h, w]
            t.swap_dims(1, 2).reshape([b, self.code_dim, n, w])
        };

        let q1_4d = to_4d(q1);
        let q2_4d = to_4d(q2);
        let q3_4d = to_4d(q3);
        let q4_4d = to_4d(q4);

        // RVQ quantize
        let (quant1, idx1, loss1) = self.quantize_1.forward(q1_4d);
        let (quant2, idx2, loss2) = self.quantize_2.forward(q2_4d);
        let (quant3, idx3, loss3) = self.quantize_3.forward(q3_4d);
        let (quant4, idx4, loss4) = self.quantize_4.forward(q4_4d);

        let total_loss = loss1 + loss2 + loss3 + loss4;

        EncodeOutput {
            quantized: [quant1, quant2, quant3, quant4],
            indices: [idx1, idx2, idx3, idx4],
            loss: total_loss,
        }
    }

    /// Decode quantized vectors to FFT components.
    ///
    /// Returns: (amplitude, sin_phase, cos_phase) each [B, seq, decoder_out_dim]
    pub fn decode(
        &self,
        quantized: &[Tensor<B, 4>; 4],
        temporal_ix: Tensor<B, 2, Int>,
        spatial_ix: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        // Decode each branch
        let d1 = self.decoder.forward_decoder(
            quantized[0].clone(),
            temporal_ix.clone(),
            spatial_ix.clone(),
            0,
        );
        let d2 = self.decoder.forward_decoder(
            quantized[1].clone(),
            temporal_ix.clone(),
            spatial_ix.clone(),
            1,
        );
        let d3 = self.decoder.forward_decoder(
            quantized[2].clone(),
            temporal_ix.clone(),
            spatial_ix.clone(),
            2,
        );
        let d4 = self.decoder.forward_decoder(
            quantized[3].clone(),
            temporal_ix.clone(),
            spatial_ix.clone(),
            3,
        );

        // Concatenate along feature dim: [B, seq, 4*embed_dim]
        let dec_features = Tensor::cat(vec![d1, d2, d3, d4], 2);

        // Amplitude head: Linear → GELU → Linear
        let amp = self
            .decode_amp_fc2
            .forward(gelu(self.decode_amp_fc1.forward(dec_features.clone())));
        // Sin phase head: Linear → Tanh → Linear → Tanh
        let sin = tanh(
            self.decode_sin_fc2
                .forward(tanh(self.decode_sin_fc1.forward(dec_features.clone()))),
        );
        // Cos phase head: Linear → Tanh → Linear → Tanh
        let cos = tanh(
            self.decode_cos_fc2
                .forward(tanh(self.decode_cos_fc1.forward(dec_features))),
        );

        (amp, sin, cos)
    }

    /// Full forward pass matching the Python NeuroRVQTokenizer.forward():
    ///   1. Reshape signal into patches
    ///   2. Compute FFT → amplitude + sin/cos phase
    ///   3. Encode + quantize
    ///   4. Decode → predicted amplitude + sin/cos phase
    ///   5. Inverse FFT → reconstructed signal
    ///   6. Standardize both original and reconstructed for comparison
    ///
    /// x: [B, N, T] (raw signal, N channels, T time samples)
    /// Returns: ForwardOutput with standardized original and reconstructed signals
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        temporal_ix: Tensor<B, 2, Int>,
        spatial_ix: Tensor<B, 2, Int>,
    ) -> ForwardOutput<B> {
        let [b, n_total, t_total] = x.dims();
        let a = t_total / self.patch_size;
        let t = self.patch_size;

        // Reshape to patches: [B, N, A, T]
        let x_patched = x.reshape([b, n_total, a, t]);

        // ── FFT decomposition (real-valued DFT via manual computation) ──

        // Compute FFT amplitude and phase using the real DFT approach.
        // Since Burn doesn't have native FFT, we compute DFT manually for small patch_size.
        // Python: x_fft = torch.fft.fft(x, dim=-1)
        //         amplitude = log1p(abs(x_fft))
        //         angle = angle(x_fft)
        //         sin_angle = sin(angle), cos_angle = cos(angle)
        let (_fft_amp_log_normed, amp_mean, amp_std, _fft_sin, _fft_cos) =
            Self::compute_fft_components(x_patched.clone());

        // ── Encode + Quantize ──
        let enc_out = self.encode(x_patched.clone(), temporal_ix.clone(), spatial_ix.clone());

        // ── Decode ──
        let (xrec_amp, xrec_sin, xrec_cos) =
            self.decode(&enc_out.quantized, temporal_ix, spatial_ix);

        // ── Inverse FFT reconstruction ──
        // Python: ustd_xrec = (xrec_amp * amp_std) + amp_mean
        //         ustd_xrec = expm1(ustd_xrec)
        //         xrec_signal = real(inverse_fft_cos_sin(ustd_xrec, xrec_sin, xrec_cos))
        let xrec_signal = Self::reconstruct_signal(
            xrec_amp, xrec_sin, xrec_cos, amp_mean, amp_std, b, n_total, a, t,
        );

        // ── Standardize both signals ──
        let std_x = Self::std_norm_4d(x_patched);
        let xrec_reshaped = xrec_signal.reshape([b, n_total, a, t]);
        let std_xrec = Self::std_norm_4d(xrec_reshaped);

        ForwardOutput {
            original_std: std_x,
            reconstructed_std: std_xrec,
            encode_output: enc_out,
        }
    }

    /// Get codebook token indices for the input.
    ///
    /// x: [B, N, A, T]
    /// Returns: indices for all 4 branches × 8 RVQ levels
    pub fn get_tokens(
        &self,
        x: Tensor<B, 4>,
        temporal_ix: Tensor<B, 2, Int>,
        spatial_ix: Tensor<B, 2, Int>,
    ) -> EncodeOutput<B> {
        self.encode(x, temporal_ix, spatial_ix)
    }

    // ── FFT helper methods ──────────────────────────────────────────────────

    /// Compute FFT components for the input patches.
    ///
    /// x: [B, N, A, T]
    /// Returns: (amplitude_log_normed, amp_mean, amp_std, sin_phase, cos_phase)
    ///   amplitude_log_normed: [B, N, A, T] - standardized log amplitude
    ///   amp_mean: [B, 1, 1, 1]
    ///   amp_std: [B, 1, 1, 1]
    ///   sin_phase, cos_phase: [B, N, A, T]
    fn compute_fft_components(
        x: Tensor<B, 4>,
    ) -> (
        Tensor<B, 4>,
        Tensor<B, 4>,
        Tensor<B, 4>,
        Tensor<B, 4>,
        Tensor<B, 4>,
    ) {
        let [b, n, a, t] = x.dims();
        let device = x.device();

        // Build the DFT matrix: W[k, j] = e^{-2πi·k·j / T}
        // real_part[k,j] = cos(2π·k·j / T)
        // imag_part[k,j] = -sin(2π·k·j / T)
        let dft_cos = Self::build_dft_cos_matrix(t, &device); // [T, T]
        let dft_sin = Self::build_dft_sin_matrix(t, &device); // [T, T]

        // x_flat: [B*N*A, T]
        let x_flat = x.clone().reshape([b * n * a, t]);

        // real = x @ dft_cos^T, imag = x @ dft_sin^T
        // (DFT: X[k] = sum_j x[j] * cos(2π·k·j/T) - i * sum_j x[j] * sin(2π·k·j/T))
        let fft_real = x_flat.clone().matmul(dft_cos.transpose()); // [B*N*A, T]
        let fft_imag = x_flat.matmul(dft_sin.transpose()); // [B*N*A, T] (this is -imag)

        // Amplitude: |X[k]| = sqrt(real^2 + imag^2)
        let amp = (fft_real.clone().powf_scalar(2.0) + fft_imag.clone().powf_scalar(2.0))
            .sqrt()
            .clamp_min(1e-10);

        // Phase components: cos(angle) = real / |X|, sin(angle) = -imag / |X| (note sign)
        let cos_phase = fft_real / amp.clone();
        let sin_phase = fft_imag.neg() / amp.clone(); // negate because dft_sin has -sin

        // Log amplitude
        let log_amp = amp.log1p();

        // Reshape back to [B, N, A, T]
        let log_amp = log_amp.reshape([b, n, a, t]);
        let sin_phase = sin_phase.reshape([b, n, a, t]);
        let cos_phase = cos_phase.reshape([b, n, a, t]);

        // Standardize log amplitude: zero mean, unit variance
        let (log_amp_normed, amp_mean, amp_std) = Self::std_norm_with_stats(log_amp);

        (log_amp_normed, amp_mean, amp_std, sin_phase, cos_phase)
    }

    /// Build DFT cosine matrix: cos(2π·k·j / T) for k,j in [0, T).
    fn build_dft_cos_matrix(t: usize, device: &B::Device) -> Tensor<B, 2> {
        let mut data = vec![0.0f32; t * t];
        let inv_t = 2.0 * std::f32::consts::PI / t as f32;
        for k in 0..t {
            for j in 0..t {
                data[k * t + j] = (inv_t * k as f32 * j as f32).cos();
            }
        }
        Tensor::<B, 2>::from_data(TensorData::new(data, [t, t]), device)
    }

    /// Build DFT sine matrix: sin(2π·k·j / T) for k,j in [0, T).
    fn build_dft_sin_matrix(t: usize, device: &B::Device) -> Tensor<B, 2> {
        let mut data = vec![0.0f32; t * t];
        let inv_t = 2.0 * std::f32::consts::PI / t as f32;
        for k in 0..t {
            for j in 0..t {
                data[k * t + j] = (inv_t * k as f32 * j as f32).sin();
            }
        }
        Tensor::<B, 2>::from_data(TensorData::new(data, [t, t]), device)
    }

    /// Reconstruct time-domain signal from predicted FFT components.
    ///
    /// Python:
    ///   ustd_xrec = (xrec_amp * amp_std) + amp_mean  # un-standardize
    ///   ustd_xrec = expm1(ustd_xrec)                  # reverse log1p
    ///   real = ustd_xrec * cos_phase
    ///   imag = ustd_xrec * sin_phase
    ///   signal = real(ifft(complex(real, imag)))
    fn reconstruct_signal(
        xrec_amp: Tensor<B, 3>, // [B, N*A, T]
        xrec_sin: Tensor<B, 3>, // [B, N*A, T]
        xrec_cos: Tensor<B, 3>, // [B, N*A, T]
        amp_mean: Tensor<B, 4>, // [B, 1, 1, 1]
        amp_std: Tensor<B, 4>,  // [B, 1, 1, 1]
        b: usize,
        n: usize,
        a: usize,
        t: usize,
    ) -> Tensor<B, 3> {
        let device = xrec_amp.device();

        // Un-standardize amplitude: reshape xrec_amp [B, N*A, T] → [B, N, A, T]
        let xrec_amp_4d = xrec_amp.reshape([b, n, a, t]);
        let ustd_amp = xrec_amp_4d * amp_std + amp_mean;

        // Reverse log1p: expm1(x) = exp(x) - 1
        let ustd_amp = ustd_amp.exp() - 1.0;

        // Flatten: [B, N, A, T] → [B, N*A, T]
        let ustd_amp = ustd_amp.reshape([b, n * a, t]);

        // Inverse FFT via DFT matrices
        // Python: inverse_fft_cos_sin(amp, sin_pha, cos_pha)
        //   imag = amp * sin_pha
        //   real = amp * cos_pha
        //   fft_y = complex(real, imag)
        //   y = ifft(fft_y)
        //
        // IFFT: x[j] = (1/T) * sum_k (real[k]*cos(2π·k·j/T) - imag[k]*sin(2π·k·j/T))
        let fft_real = ustd_amp.clone() * xrec_cos; // real part of frequency domain
        let fft_imag = ustd_amp * xrec_sin; // imag part of frequency domain

        // Build IDFT matrix
        let idft_cos = Self::build_dft_cos_matrix(t, &device); // cos is symmetric
        let idft_sin = Self::build_dft_sin_matrix(t, &device); // sin for IDFT uses +sin

        // Flatten for matmul: [B*N*A, T]
        let fft_real_flat = fft_real.reshape([b * n * a, t]);
        let fft_imag_flat = fft_imag.reshape([b * n * a, t]);

        // IFFT: x[j] = (1/T) * (real @ cos^T - imag @ sin^T)
        let signal = (fft_real_flat.matmul(idft_cos.transpose())
            - fft_imag_flat.matmul(idft_sin.transpose()))
        .mul_scalar(1.0 / t as f32);

        signal.reshape([b, n * a, t])
    }

    /// Standardize tensor: zero mean, unit variance across dims (1,2,3).
    /// Python: std_norm(x) → (normed_x, mean, std)
    fn std_norm_with_stats(x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        // mean over dims 1,2,3 → [B, 1, 1, 1]
        let mean = x.clone().mean_dim(1).mean_dim(2).mean_dim(3);
        let diff = x.clone() - mean.clone();
        // var over dims 1,2,3
        let var = (diff.clone() * diff).mean_dim(1).mean_dim(2).mean_dim(3);
        let std = (var + 1e-8).sqrt();
        let normed = (x - mean.clone()) / std.clone();
        (normed, mean, std)
    }

    /// Standardize 4D tensor and return as 3D [B, N*A, T].
    /// Python: std_norm → then view/squeeze to match output shape.
    fn std_norm_4d(x: Tensor<B, 4>) -> Tensor<B, 3> {
        let [b, n, a, t] = x.dims();
        let (normed, _, _) = Self::std_norm_with_stats(x);
        normed.reshape([b, n * a, t])
    }
}

/// Output from the encode step.
pub struct EncodeOutput<B: Backend> {
    /// Quantized vectors, one per branch: [B, code_dim, H, W].
    pub quantized: [Tensor<B, 4>; 4],
    /// Token indices per branch, each is Vec of [B*H*W] per RVQ level.
    pub indices: [Vec<Tensor<B, 1, Int>>; 4],
    /// Total quantization loss.
    pub loss: Tensor<B, 1>,
}

/// Output from the full forward pass.
pub struct ForwardOutput<B: Backend> {
    /// Standardized original signal: [B, N*A, T].
    pub original_std: Tensor<B, 3>,
    /// Standardized reconstructed signal: [B, N*A, T].
    pub reconstructed_std: Tensor<B, 3>,
    /// Encode output (quantized vectors + indices + loss).
    pub encode_output: EncodeOutput<B>,
}

/// Model configuration for NeuroRVQ, parsed from YAML config files.
use serde::Deserialize;

/// Top-level config matching the NeuroRVQ YAML flags files.
///
/// Loads directly from the upstream flag files shipped with the Python repo:
///   - `flags/NeuroRVQ_EEG_v1.yml`
///   - `flags/NeuroRVQ_ECG_v1.yml`
///   - `flags/NeuroRVQ_EMG_v1.yml`
///
/// Unknown keys (e.g. fine-tuning hyperparameters) are silently ignored.
#[derive(Debug, Clone, Deserialize)]
pub struct NeuroRVQConfig {
    /// Optional modality field. If absent, can be auto-detected from filename
    /// or set explicitly via [`NeuroRVQConfig::from_yaml_with_modality`].
    #[serde(default)]
    pub modality: Option<String>,

    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_n_patches")]
    pub n_patches: usize,
    #[serde(default = "default_num_classes")]
    pub num_classes: usize,
    #[serde(default = "default_n_code")]
    pub n_code: usize,
    #[serde(default = "default_code_dim")]
    pub code_dim: usize,
    #[serde(default = "default_embed_dim")]
    pub embed_dim: usize,

    // Encoder
    #[serde(default = "default_in_chans_encoder")]
    pub in_chans_encoder: usize,
    #[serde(default = "default_out_chans_encoder")]
    pub out_chans_encoder: usize,
    #[serde(default = "default_depth_encoder")]
    pub depth_encoder: usize,
    #[serde(default = "default_depth_decoder")]
    pub depth_decoder: usize,
    #[serde(default = "default_decoder_out_dim")]
    pub decoder_out_dim: usize,

    // Tokenizer transformer config
    #[serde(default = "default_num_heads")]
    pub num_heads_tokenizer: usize,
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio_tokenizer: f64,
    #[serde(default = "default_true")]
    pub qkv_bias_tokenizer: bool,
    #[serde(default)]
    pub drop_rate_tokenizer: f64,
    #[serde(default)]
    pub attn_drop_rate_tokenizer: f64,
    #[serde(default)]
    pub drop_path_rate_tokenizer: f64,
    #[serde(default)]
    pub init_values_tokenizer: f64,
    #[serde(default = "default_init_scale")]
    pub init_scale_tokenizer: f64,

    // Second-stage (standalone Foundation Model) config
    #[serde(default)]
    pub in_chans_second_stage: Option<usize>,
    #[serde(default)]
    pub out_chans_second_stage: Option<usize>,
    #[serde(default)]
    pub embed_dim_second_stage: Option<usize>,
    #[serde(default)]
    pub depth_second_stage: Option<usize>,
    #[serde(default)]
    pub num_heads_second_stage: Option<usize>,
    #[serde(default)]
    pub mlp_ratio_second_stage: Option<f64>,
    #[serde(default)]
    pub qkv_bias_second_stage: Option<bool>,
    #[serde(default)]
    pub drop_rate_second_stage: Option<f64>,
    #[serde(default)]
    pub attn_drop_rate_second_stage: Option<f64>,
    #[serde(default)]
    pub drop_path_rate_second_stage: Option<f64>,
    #[serde(default)]
    pub init_values_second_stage: Option<f64>,
    #[serde(default)]
    pub init_scale_second_stage: Option<f64>,

    // Global electrode count (set at runtime from modality)
    #[serde(default = "default_n_global_electrodes")]
    pub n_global_electrodes: usize,
}

fn default_patch_size() -> usize {
    200
}
fn default_n_patches() -> usize {
    256
}
fn default_num_classes() -> usize {
    5
}
fn default_n_code() -> usize {
    8192
}
fn default_code_dim() -> usize {
    128
}
fn default_embed_dim() -> usize {
    200
}
fn default_in_chans_encoder() -> usize {
    1
}
fn default_out_chans_encoder() -> usize {
    8
}
fn default_depth_encoder() -> usize {
    12
}
fn default_depth_decoder() -> usize {
    3
}
fn default_decoder_out_dim() -> usize {
    200
}
fn default_num_heads() -> usize {
    10
}
fn default_mlp_ratio() -> f64 {
    4.0
}
fn default_true() -> bool {
    true
}
fn default_init_scale() -> f64 {
    0.001
}
fn default_n_global_electrodes() -> usize {
    103
}

impl Default for NeuroRVQConfig {
    fn default() -> Self {
        Self {
            modality: None,
            patch_size: default_patch_size(),
            n_patches: default_n_patches(),
            num_classes: default_num_classes(),
            n_code: default_n_code(),
            code_dim: default_code_dim(),
            embed_dim: default_embed_dim(),
            in_chans_encoder: default_in_chans_encoder(),
            out_chans_encoder: default_out_chans_encoder(),
            depth_encoder: default_depth_encoder(),
            depth_decoder: default_depth_decoder(),
            decoder_out_dim: default_decoder_out_dim(),
            num_heads_tokenizer: default_num_heads(),
            mlp_ratio_tokenizer: default_mlp_ratio(),
            qkv_bias_tokenizer: default_true(),
            drop_rate_tokenizer: 0.0,
            attn_drop_rate_tokenizer: 0.0,
            drop_path_rate_tokenizer: 0.0,
            init_values_tokenizer: 0.0,
            init_scale_tokenizer: default_init_scale(),
            in_chans_second_stage: None,
            out_chans_second_stage: None,
            embed_dim_second_stage: None,
            depth_second_stage: None,
            num_heads_second_stage: None,
            mlp_ratio_second_stage: None,
            qkv_bias_second_stage: None,
            drop_rate_second_stage: None,
            attn_drop_rate_second_stage: None,
            drop_path_rate_second_stage: None,
            init_values_second_stage: None,
            init_scale_second_stage: None,
            n_global_electrodes: default_n_global_electrodes(),
        }
    }
}

impl NeuroRVQConfig {
    /// Load config from a YAML file.
    ///
    /// Accepts the upstream flag files directly (e.g. `NeuroRVQ_EEG_v1.yml`).
    /// Unknown keys are silently ignored.
    pub fn from_yaml(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut cfg: Self = serde_yaml::from_str(&content)?;
        // Auto-detect modality from filename if not set in YAML
        if cfg.modality.is_none() {
            cfg.modality = detect_modality_from_path(path);
        }
        Ok(cfg)
    }

    /// Load config with an explicit modality override.
    pub fn from_yaml_with_modality(path: &str, modality: Modality) -> anyhow::Result<Self> {
        let mut cfg = Self::from_yaml(path)?;
        cfg.modality = Some(modality.to_string());
        Ok(cfg)
    }

    /// Resolve the modality from config. Falls back to EEG if unset.
    pub fn resolve_modality(&self) -> Modality {
        self.modality
            .as_deref()
            .and_then(|s| s.parse::<Modality>().ok())
            .unwrap_or(Modality::EEG)
    }

    /// Apply CLI overrides on top of YAML-loaded config.
    /// Any `Some(value)` replaces the corresponding field.
    pub fn apply_overrides(&mut self, ovr: &ConfigOverrides) {
        macro_rules! override_field {
            ($field:ident) => {
                if let Some(v) = ovr.$field {
                    self.$field = v;
                }
            };
        }
        override_field!(patch_size);
        override_field!(n_patches);
        override_field!(embed_dim);
        override_field!(code_dim);
        override_field!(n_code);
        override_field!(decoder_out_dim);
        override_field!(out_chans_encoder);
        override_field!(depth_encoder);
        override_field!(depth_decoder);
        if let Some(v) = ovr.depth_second_stage {
            self.depth_second_stage = Some(v);
        }
        override_field!(num_heads_tokenizer);
        override_field!(mlp_ratio_tokenizer);
        override_field!(init_values_tokenizer);
        if let Some(v) = ovr.init_values_second_stage {
            self.init_values_second_stage = Some(v);
        }
        override_field!(init_scale_tokenizer);
        override_field!(n_global_electrodes);
        if let Some(v) = ovr.qkv_bias_tokenizer {
            self.qkv_bias_tokenizer = v;
        }
    }

    /// Get second-stage FM embed_dim, falling back to tokenizer embed_dim.
    pub fn fm_embed_dim(&self) -> usize {
        self.embed_dim_second_stage.unwrap_or(self.embed_dim)
    }

    /// Get second-stage FM depth, falling back to encoder depth.
    pub fn fm_depth(&self) -> usize {
        self.depth_second_stage.unwrap_or(self.depth_encoder)
    }

    /// Get second-stage FM num_heads.
    pub fn fm_num_heads(&self) -> usize {
        self.num_heads_second_stage
            .unwrap_or(self.num_heads_tokenizer)
    }

    /// Get second-stage FM mlp_ratio.
    pub fn fm_mlp_ratio(&self) -> f64 {
        self.mlp_ratio_second_stage
            .unwrap_or(self.mlp_ratio_tokenizer)
    }

    /// Get second-stage FM qkv_bias.
    pub fn fm_qkv_bias(&self) -> bool {
        self.qkv_bias_second_stage
            .unwrap_or(self.qkv_bias_tokenizer)
    }

    /// Get second-stage FM init_values.
    pub fn fm_init_values(&self) -> f64 {
        self.init_values_second_stage
            .unwrap_or(self.init_values_tokenizer)
    }

    /// Get second-stage FM init_scale.
    pub fn fm_init_scale(&self) -> f64 {
        self.init_scale_second_stage
            .unwrap_or(self.init_scale_tokenizer)
    }

    /// Get second-stage FM out_chans.
    pub fn fm_out_chans(&self) -> usize {
        self.out_chans_second_stage
            .unwrap_or(self.out_chans_encoder)
    }

    /// Get second-stage FM in_chans.
    pub fn fm_in_chans(&self) -> usize {
        self.in_chans_second_stage.unwrap_or(self.in_chans_encoder)
    }
}

/// Modality type for the tokenizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modality {
    EEG,
    EMG,
    ECG,
}

impl std::fmt::Display for Modality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Modality::EEG => write!(f, "EEG"),
            Modality::EMG => write!(f, "EMG"),
            Modality::ECG => write!(f, "ECG"),
        }
    }
}

impl std::str::FromStr for Modality {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "EEG" => Ok(Modality::EEG),
            "EMG" => Ok(Modality::EMG),
            "ECG" => Ok(Modality::ECG),
            _ => Err(anyhow::anyhow!(
                "Unknown modality: {s}. Must be EEG, EMG, or ECG."
            )),
        }
    }
}

/// CLI-level overrides for [`NeuroRVQConfig`].
///
/// Every field is `Option` — only `Some` values are applied.
/// Use with [`NeuroRVQConfig::apply_overrides`].
#[derive(Debug, Clone, Default)]
pub struct ConfigOverrides {
    pub patch_size: Option<usize>,
    pub n_patches: Option<usize>,
    pub embed_dim: Option<usize>,
    pub code_dim: Option<usize>,
    pub n_code: Option<usize>,
    pub decoder_out_dim: Option<usize>,
    pub out_chans_encoder: Option<usize>,
    pub depth_encoder: Option<usize>,
    pub depth_decoder: Option<usize>,
    pub depth_second_stage: Option<usize>,
    pub num_heads_tokenizer: Option<usize>,
    pub mlp_ratio_tokenizer: Option<f64>,
    pub qkv_bias_tokenizer: Option<bool>,
    pub init_values_tokenizer: Option<f64>,
    pub init_values_second_stage: Option<f64>,
    pub init_scale_tokenizer: Option<f64>,
    pub n_global_electrodes: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yaml_loading() {
        let cfg = NeuroRVQConfig::from_yaml("flags/NeuroRVQ_EEG_v1.yml").unwrap();
        assert_eq!(cfg.patch_size, 200);
        assert_eq!(cfg.n_patches, 256);
        assert_eq!(cfg.embed_dim, 200);
        assert_eq!(cfg.depth_encoder, 12);
        assert_eq!(cfg.resolve_modality(), Modality::EEG);
    }

    #[test]
    fn test_yaml_ecg() {
        let cfg = NeuroRVQConfig::from_yaml("flags/NeuroRVQ_ECG_v1.yml").unwrap();
        assert_eq!(cfg.patch_size, 40);
        assert_eq!(cfg.n_patches, 600);
        assert_eq!(cfg.embed_dim, 40);
        assert_eq!(cfg.resolve_modality(), Modality::ECG);
    }

    #[test]
    fn test_yaml_emg() {
        let cfg = NeuroRVQConfig::from_yaml("flags/NeuroRVQ_EMG_v1.yml").unwrap();
        assert_eq!(cfg.patch_size, 200);
        assert_eq!(cfg.resolve_modality(), Modality::EMG);
    }

    #[test]
    fn test_overrides() {
        let mut cfg = NeuroRVQConfig::from_yaml("flags/NeuroRVQ_EEG_v1.yml").unwrap();
        assert_eq!(cfg.embed_dim, 200);
        assert_eq!(cfg.depth_encoder, 12);

        cfg.apply_overrides(&ConfigOverrides {
            embed_dim: Some(64),
            depth_encoder: Some(6),
            ..Default::default()
        });

        assert_eq!(cfg.embed_dim, 64);
        assert_eq!(cfg.depth_encoder, 6);
        // Unchanged fields stay the same
        assert_eq!(cfg.patch_size, 200);
        assert_eq!(cfg.n_patches, 256);
    }

    #[test]
    fn test_modality_detection() {
        assert_eq!(
            detect_modality_from_path("flags/NeuroRVQ_EEG_v1.yml"),
            Some("EEG".into())
        );
        assert_eq!(
            detect_modality_from_path("flags/NeuroRVQ_ECG_v1.yml"),
            Some("ECG".into())
        );
        assert_eq!(
            detect_modality_from_path("flags/NeuroRVQ_EMG_v1.yml"),
            Some("EMG".into())
        );
        assert_eq!(detect_modality_from_path("some/random/config.yml"), None);
    }

    #[test]
    fn test_modality_override() {
        let cfg =
            NeuroRVQConfig::from_yaml_with_modality("flags/NeuroRVQ_EEG_v1.yml", Modality::EMG)
                .unwrap();
        assert_eq!(cfg.resolve_modality(), Modality::EMG);
    }

    #[test]
    fn test_second_stage_fallback() {
        let cfg = NeuroRVQConfig::from_yaml("flags/NeuroRVQ_EEG_v1.yml").unwrap();
        // second-stage embed_dim is set in the YAML
        assert_eq!(cfg.fm_embed_dim(), 200);
        // init_values_second_stage is 1e-5 in the YAML
        assert!((cfg.fm_init_values() - 1e-5).abs() < 1e-10);
    }
}

/// Try to detect modality from a YAML filename.
///
/// Matches patterns like `NeuroRVQ_EEG_v1.yml`, `neurorvq_ecg.yaml`, etc.
fn detect_modality_from_path(path: &str) -> Option<String> {
    let upper = path.to_uppercase();
    if upper.contains("_EEG") || upper.contains("-EEG") {
        Some("EEG".to_string())
    } else if upper.contains("_ECG") || upper.contains("-ECG") {
        Some("ECG".to_string())
    } else if upper.contains("_EMG") || upper.contains("-EMG") {
        Some("EMG".to_string())
    } else {
        None
    }
}

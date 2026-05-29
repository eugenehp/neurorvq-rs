/// Channel vocabularies for NeuroRVQ modalities (EEG, ECG, EMG).
///
/// Each modality has a global channel list used during pre-training.
/// Channel names are lowercased for matching.
use crate::config::Modality;

// ── EEG (103 channels) ───────────────────────────────────────────────────────

/// Global channel vocabulary from NeuroRVQ_EEG_v1 (103 channels).
pub const EEG_CHANNELS: &[&str] = &[
    "a1", "a2", "af3", "af4", "af7", "af8", "afz", "c1", "c2", "c3", "c4", "c5", "c6", "ccp1",
    "ccp2", "ccp3", "ccp4", "ccp5", "ccp6", "ccp7", "ccp8", "cfc1", "cfc2", "cfc3", "cfc4", "cfc5",
    "cfc6", "cfc7", "cfc8", "cp1", "cp2", "cp3", "cp4", "cp5", "cp6", "cpz", "cz", "eog", "f1",
    "f10", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "fc1", "fc2", "fc3", "fc4", "fc5",
    "fc6", "fcz", "fp1", "fp2", "fpz", "ft7", "ft8", "fz", "iz", "loc", "o1", "o2", "oz", "p08",
    "p1", "p10", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "po1", "po10", "po2", "po3",
    "po4", "po7", "po8", "po9", "poz", "pz", "roc", "sp1", "sp2", "t1", "t10", "t2", "t3", "t4",
    "t5", "t6", "t7", "t8", "t9", "tp10", "tp7", "tp8", "tp9",
];

pub const EEG_VOCAB_SIZE: usize = 103;

// ── ECG (15 channels) ────────────────────────────────────────────────────────

/// Global channel vocabulary from NeuroRVQ_ECG_v1 (15 channels).
pub const ECG_CHANNELS: &[&str] = &[
    "avf", "avl", "avr", "i", "ii", "iii", "v1", "v2", "v3", "v4", "v5", "v6", "vx", "vy", "vz",
];

pub const ECG_VOCAB_SIZE: usize = 15;

// ── EMG (16 channels) ────────────────────────────────────────────────────────

/// Global channel vocabulary from NeuroRVQ_EMG_v1 (16 channels).
pub const EMG_CHANNELS: &[&str] = &[
    "c1", "c10", "c11", "c12", "c13", "c14", "c15", "c16", "c2", "c3", "c4", "c5", "c6", "c7",
    "c8", "c9",
];

pub const EMG_VOCAB_SIZE: usize = 16;

// ── Modality-aware helpers ───────────────────────────────────────────────────

/// Return the global channel list for a given modality.
pub fn global_channels(modality: Modality) -> &'static [&'static str] {
    match modality {
        Modality::EEG => EEG_CHANNELS,
        Modality::ECG => ECG_CHANNELS,
        Modality::EMG => EMG_CHANNELS,
    }
}

/// Return the number of global electrodes for a given modality.
pub fn global_vocab_size(modality: Modality) -> usize {
    match modality {
        Modality::EEG => EEG_VOCAB_SIZE,
        Modality::ECG => ECG_VOCAB_SIZE,
        Modality::EMG => EMG_VOCAB_SIZE,
    }
}

/// Look up the index of a channel name in a modality's global vocabulary.
/// Returns None if not found.
pub fn channel_index(name: &str, modality: Modality) -> Option<usize> {
    let lower = name.to_lowercase();
    global_channels(modality).iter().position(|&c| c == lower)
}

/// Look up indices for multiple channel names.
/// Returns indices as i64 for burn Int tensors.
pub fn channel_indices(names: &[&str], modality: Modality) -> Vec<i64> {
    names
        .iter()
        .map(|n| {
            channel_index(n, modality)
                .unwrap_or_else(|| panic!("Unknown {modality:?} channel: {n}")) as i64
        })
        .collect()
}

/// Create temporal and spatial embedding indices.
///
/// Python: `create_embedding_ix` in inference modules.
///
/// temporal_ix: for each electrode, [max_patches - n_time .. max_patches-1] repeated
/// spatial_ix: for each electrode, its global index repeated n_time times
pub fn create_embedding_ix(
    n_time: usize,
    max_n_patches: usize,
    channel_names: &[&str],
    modality: Modality,
) -> (Vec<i64>, Vec<i64>) {
    let n_channels = channel_names.len();

    // Temporal: for each channel, [max_patches - n_time, ..., max_patches - 1]
    let mut temporal_ix = Vec::with_capacity(n_channels * n_time);
    let start = max_n_patches - n_time;
    for _ch in 0..n_channels {
        for t in start..max_n_patches {
            temporal_ix.push(t as i64);
        }
    }

    // Spatial: for each channel, its index repeated n_time times
    let mut spatial_ix = Vec::with_capacity(n_channels * n_time);
    for name in channel_names {
        let idx = channel_index(name, modality)
            .unwrap_or_else(|| panic!("Unknown {modality:?} channel: {name}"))
            as i64;
        for _ in 0..n_time {
            spatial_ix.push(idx);
        }
    }

    (temporal_ix, spatial_ix)
}

/// Filter channel names to only those present in the global vocabulary.
/// Returns (mask, filtered_names).
///
/// Python: `ch_mask = np.isin(ch_names, ch_names_global)`
pub fn filter_channels<'a>(
    channel_names: &[&'a str],
    modality: Modality,
) -> (Vec<bool>, Vec<&'a str>) {
    let vocab = global_channels(modality);
    let mut mask = Vec::with_capacity(channel_names.len());
    let mut filtered = Vec::new();
    for &name in channel_names {
        let lower = name.to_lowercase();
        let found = vocab.iter().any(|&c| c == lower);
        mask.push(found);
        if found {
            filtered.push(name);
        }
    }
    (mask, filtered)
}

/// Compute the number of time patches per channel given a maximum patch budget.
///
/// Python: `n_time = maximum_patches // len(channels_use)`
pub fn compute_n_time(max_n_patches: usize, n_channels: usize) -> usize {
    max_n_patches / n_channels
}

/// Create patches from a signal: trim to fit n_time * patch_size.
///
/// Python: `create_patches(signal, maximum_patches, patch_size, channels_use)`
///
/// signal: [n_trials, n_channels, t_total] flattened row-major
/// Returns: (trimmed_signal, n_time)
pub fn create_patches(
    signal: &[f32],
    n_trials: usize,
    n_channels: usize,
    t_total: usize,
    max_n_patches: usize,
    patch_size: usize,
) -> (Vec<f32>, usize) {
    let n_time = max_n_patches / n_channels;
    let t_use = n_time * patch_size;
    let t_use = t_use.min(t_total);

    let mut out = Vec::with_capacity(n_trials * n_channels * t_use);
    for trial in 0..n_trials {
        for ch in 0..n_channels {
            let offset = trial * n_channels * t_total + ch * t_total;
            out.extend_from_slice(&signal[offset..offset + t_use]);
        }
    }

    (out, n_time)
}

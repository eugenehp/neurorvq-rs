use crate::config::Modality;
use crate::model::encoder_block::TransformerBlock;
use crate::model::multi_scale_conv::MultiScaleTemporalConv;
use crate::model::norm::NeuroLayerNorm;
use crate::model::patch_embed::PatchEmbed;
use burn::module::{Param, ParamId};
use burn::nn::{Linear, LinearConfig};
/// NeuroRVQ Foundation Model (NeuroRVQFM).
///
/// Python: `NeuroRVQFM` class in NeuroRVQ.py
///
/// Used as both:
///   - Encoder (use_as_encoder=true): takes raw signal, applies MultiScaleTemporalConv,
///     returns 4 branch outputs
///   - Decoder (use_as_encoder=false): takes quantized vectors via PatchEmbed,
///     returns single branch output
///
/// Architecture:
///   1. Patch embedding (multi-scale conv or 1x1 conv per branch)
///   2. Spatial + temporal positional embeddings
///   3. Transformer blocks
///   4. Output head per branch (LayerNorm + Linear)
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct NeuroRVQFM<B: Backend> {
    // Patch embedding — encoder uses multi-scale conv, decoder uses per-branch PatchEmbed
    pub multi_scale_conv: Option<MultiScaleTemporalConv<B>>,
    pub patch_embed_1: Option<PatchEmbed<B>>,
    pub patch_embed_2: Option<PatchEmbed<B>>,
    pub patch_embed_3: Option<PatchEmbed<B>>,
    pub patch_embed_4: Option<PatchEmbed<B>>,

    /// CLS token (legacy, kept for weight compatibility).
    pub cls_token: Param<Tensor<B, 3>>,
    /// Spatial positional embedding: [n_global_electrodes + 1, embed_dim].
    pub pos_embed: Param<Tensor<B, 2>>,
    /// Temporal positional embedding: [n_patches, embed_dim].
    pub time_embed: Param<Tensor<B, 2>>,

    /// Transformer blocks.
    pub blocks: Vec<TransformerBlock<B>>,

    /// Per-branch output heads.
    pub fc_norm_1: NeuroLayerNorm<B>,
    pub head_1: Linear<B>,
    pub fc_norm_2: NeuroLayerNorm<B>,
    pub head_2: Linear<B>,
    pub fc_norm_3: NeuroLayerNorm<B>,
    pub head_3: Linear<B>,
    pub fc_norm_4: NeuroLayerNorm<B>,
    pub head_4: Linear<B>,

    pub embed_dim: usize,
    pub patch_size: usize,
    pub use_as_encoder: bool,
}

impl<B: Backend> NeuroRVQFM<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_patches: usize,
        patch_size: usize,
        in_chans: usize,
        out_chans: usize,
        num_classes: usize,
        embed_dim: usize,
        depth: usize,
        num_heads: usize,
        mlp_ratio: f64,
        qkv_bias: bool,
        init_values: f64,
        n_global_electrodes: usize,
        use_as_encoder: bool,
        device: &B::Device,
    ) -> Self {
        Self::new_with_modality(
            n_patches,
            patch_size,
            in_chans,
            out_chans,
            num_classes,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            init_values,
            n_global_electrodes,
            use_as_encoder,
            Modality::EEG,
            device,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_modality(
        n_patches: usize,
        patch_size: usize,
        in_chans: usize,
        out_chans: usize,
        num_classes: usize,
        embed_dim: usize,
        depth: usize,
        num_heads: usize,
        mlp_ratio: f64,
        qkv_bias: bool,
        init_values: f64,
        n_global_electrodes: usize,
        use_as_encoder: bool,
        modality: Modality,
        device: &B::Device,
    ) -> Self {
        let (multi_scale_conv, pe1, pe2, pe3, pe4) = if use_as_encoder {
            (
                Some(MultiScaleTemporalConv::new_with_modality(
                    1, out_chans, modality, device,
                )),
                None,
                None,
                None,
                None,
            )
        } else {
            (
                None,
                Some(PatchEmbed::new(in_chans, embed_dim, device)),
                Some(PatchEmbed::new(in_chans, embed_dim, device)),
                Some(PatchEmbed::new(in_chans, embed_dim, device)),
                Some(PatchEmbed::new(in_chans, embed_dim, device)),
            )
        };

        let cls_token =
            Param::initialized(ParamId::new(), Tensor::zeros([1, 1, embed_dim], device));
        let pos_embed = Param::initialized(
            ParamId::new(),
            Tensor::zeros([n_global_electrodes + 1, embed_dim], device),
        );
        let time_embed = Param::initialized(
            ParamId::new(),
            Tensor::zeros([n_patches, embed_dim], device),
        );

        let norm_eps = 1e-6;
        let use_qk_norm = true; // NeuroRVQ uses qk_norm=partial(nn.LayerNorm, eps=1e-6)

        let blocks = (0..depth)
            .map(|_| {
                TransformerBlock::new(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    use_qk_norm,
                    init_values,
                    norm_eps,
                    device,
                )
            })
            .collect();

        let out_dim = if num_classes > 0 {
            num_classes
        } else {
            embed_dim
        };

        Self {
            multi_scale_conv,
            patch_embed_1: pe1,
            patch_embed_2: pe2,
            patch_embed_3: pe3,
            patch_embed_4: pe4,
            cls_token,
            pos_embed,
            time_embed,
            blocks,
            fc_norm_1: NeuroLayerNorm::new(embed_dim, norm_eps, device),
            head_1: LinearConfig::new(embed_dim, out_dim)
                .with_bias(true)
                .init(device),
            fc_norm_2: NeuroLayerNorm::new(embed_dim, norm_eps, device),
            head_2: LinearConfig::new(embed_dim, out_dim)
                .with_bias(true)
                .init(device),
            fc_norm_3: NeuroLayerNorm::new(embed_dim, norm_eps, device),
            head_3: LinearConfig::new(embed_dim, out_dim)
                .with_bias(true)
                .init(device),
            fc_norm_4: NeuroLayerNorm::new(embed_dim, norm_eps, device),
            head_4: LinearConfig::new(embed_dim, out_dim)
                .with_bias(true)
                .init(device),
            embed_dim,
            patch_size,
            use_as_encoder,
        }
    }

    /// Encoder forward: raw signal → 4 branch outputs.
    ///
    /// x: [B, N, A, T]
    /// temporal_embedding_ix: [1, n_patches_total] (i32 indices)
    /// spatial_embedding_ix: [1, n_patches_total] (i32 indices)
    ///
    /// Returns: 4 tensors, each [B, seq_len, out_dim]
    pub fn forward_encoder(
        &self,
        x: Tensor<B, 4>,
        temporal_ix: Tensor<B, 2, Int>,
        spatial_ix: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        assert!(self.use_as_encoder, "forward_encoder called on decoder");

        let conv = self.multi_scale_conv.as_ref().unwrap();
        let (x1, x2, x3, x4) = conv.forward(x);

        let device = x1.device();
        let [_b, _, _] = x1.dims();

        // Process each branch through transformer
        let process_branch = |x_branch: Tensor<B, 3>,
                              fc_norm: &NeuroLayerNorm<B>,
                              head: &Linear<B>|
         -> Tensor<B, 3> {
            let [b, seq_len, _] = x_branch.dims();

            // Prepend cls token
            let cls = self.cls_token.val().expand([b, 1, self.embed_dim]);
            let x = Tensor::cat(vec![cls, x_branch], 1); // [B, 1+seq_len, D]

            // Add spatial embeddings
            let spat_ix = Self::pad_spatial_ix(spatial_ix.clone(), &device);
            let spatial_emb = Self::gather_embeddings_2d(&self.pos_embed.val(), spat_ix, &device);
            let x = x + spatial_emb;

            // Add temporal embeddings (skip cls position)
            let temporal_emb =
                Self::gather_embeddings_2d(&self.time_embed.val(), temporal_ix.clone(), &device);
            let mut x = x;
            // x[:, 1:, :] += temporal_emb
            let x_cls = x.clone().narrow(1, 0, 1);
            let x_rest = x.narrow(1, 1, seq_len) + temporal_emb;
            x = Tensor::cat(vec![x_cls, x_rest], 1);

            // Transformer blocks
            for blk in &self.blocks {
                x = blk.forward(x);
            }

            // Remove cls token, apply head
            let x = x.narrow(1, 1, seq_len);
            head.forward(fc_norm.forward(x))
        };

        let o1 = process_branch(x1, &self.fc_norm_1, &self.head_1);
        let o2 = process_branch(x2, &self.fc_norm_2, &self.head_2);
        let o3 = process_branch(x3, &self.fc_norm_3, &self.head_3);
        let o4 = process_branch(x4, &self.fc_norm_4, &self.head_4);

        (o1, o2, o3, o4)
    }

    /// Decoder forward: quantized vectors for a single branch → output.
    ///
    /// x: [B, in_chans, H, W] (quantized codebook vectors)
    /// branch_idx: 0-3
    pub fn forward_decoder(
        &self,
        x: Tensor<B, 4>,
        temporal_ix: Tensor<B, 2, Int>,
        spatial_ix: Tensor<B, 2, Int>,
        branch_idx: usize,
    ) -> Tensor<B, 3> {
        assert!(!self.use_as_encoder, "forward_decoder called on encoder");

        let device = x.device();

        let pe = match branch_idx {
            0 => self.patch_embed_1.as_ref().unwrap(),
            1 => self.patch_embed_2.as_ref().unwrap(),
            2 => self.patch_embed_3.as_ref().unwrap(),
            3 => self.patch_embed_4.as_ref().unwrap(),
            _ => panic!("Invalid branch_idx: {}", branch_idx),
        };

        let x = pe.forward(x); // [B, seq_len, D]
        let [b, seq_len, _] = x.dims();

        // Prepend cls token
        let cls = self.cls_token.val().expand([b, 1, self.embed_dim]);
        let x = Tensor::cat(vec![cls, x], 1);

        // Add spatial + temporal embeddings
        let spat_ix = Self::pad_spatial_ix(spatial_ix, &device);
        let spatial_emb = Self::gather_embeddings_2d(&self.pos_embed.val(), spat_ix, &device);
        let x = x + spatial_emb;

        let temporal_emb = Self::gather_embeddings_2d(&self.time_embed.val(), temporal_ix, &device);
        let x_cls = x.clone().narrow(1, 0, 1);
        let x_rest = x.narrow(1, 1, seq_len) + temporal_emb;
        let mut x = Tensor::cat(vec![x_cls, x_rest], 1);

        for blk in &self.blocks {
            x = blk.forward(x);
        }

        // Remove cls token
        let x = x.narrow(1, 1, seq_len);

        let (fc_norm, head) = match branch_idx {
            0 => (&self.fc_norm_1, &self.head_1),
            1 => (&self.fc_norm_2, &self.head_2),
            2 => (&self.fc_norm_3, &self.head_3),
            3 => (&self.fc_norm_4, &self.head_4),
            _ => unreachable!(),
        };

        head.forward(fc_norm.forward(x))
    }

    /// Pad spatial_embedding_ix with 0 for cls token.
    fn pad_spatial_ix(spatial_ix: Tensor<B, 2, Int>, device: &B::Device) -> Tensor<B, 2, Int> {
        let [b, _n] = spatial_ix.dims();
        let zero_col = Tensor::<B, 2, Int>::zeros([b, 1], device);
        Tensor::cat(vec![zero_col, spatial_ix], 1) // [B, n+1]
    }

    /// Gather embedding vectors by integer indices.
    /// embed_table: [vocab_size, D]
    /// indices: [B, N] (int)
    /// Returns: [B, N, D]
    fn gather_embeddings_2d(
        embed_table: &Tensor<B, 2>,
        indices: Tensor<B, 2, Int>,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let [b, n] = indices.dims();
        let [_vocab, d] = embed_table.dims();

        // Flatten indices, gather, reshape
        let flat_indices = indices.reshape([b * n]); // [B*N]

        // One-hot matmul approach for backend compatibility
        let num_idx = b * n;
        let vocab = embed_table.dims()[0];
        let one_hot = Tensor::<B, 2>::zeros([num_idx, vocab], device);
        let idx_2d = flat_indices.unsqueeze_dim::<2>(1); // [num_idx, 1]
        let ones = Tensor::<B, 2, Int>::ones([num_idx, 1], device);
        let one_hot = one_hot.scatter(1, idx_2d, ones.float(), burn::tensor::IndexingUpdateOp::Add);
        let gathered = one_hot.matmul(embed_table.clone()); // [num_idx, D]

        gathered.reshape([b, n, d])
    }
}

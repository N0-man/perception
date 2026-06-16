"""HuggingFace config for the Mistral two-encoder contrastive (CLIP/SigLIP-style) model.

This file is self-contained (only depends on ``transformers``) so it can be shipped
alongside the weights and loaded via ``trust_remote_code=True``.

The model is a dual encoder where both towers are Mistral-style transformers
(RoPE + SwiGLU + RMSNorm, no biases):

- text tower: causal transformer, EOS-token pooling, then a linear projection.
- vision tower: bidirectional ViT with 2D RoPE, 2x2 patch merging, ``amax`` pooling
  over the merged tokens, then a linear projection.

Image and text embeddings are L2-normalized; ``logits_per_image`` is
``logit_scale.exp() * image_embeds @ text_embeds.T``.
"""

import math

from transformers.configuration_utils import PretrainedConfig


class MistralContrastiveConfig(PretrainedConfig):
    model_type = "mistral_contrastive"

    def __init__(
        self,
        # --- text tower ---
        text_vocab_size: int = 131072,
        text_dim: int = 1792,
        text_num_hidden_layers: int = 14,
        text_num_attention_heads: int = 14,
        text_head_dim: int = 128,
        text_intermediate_size: int = 4864,
        text_rope_theta: float = 1_000_000.0,
        text_norm_eps: float = 1e-5,
        eos_token_id: int = 2,
        # --- vision tower ---
        vision_dim: int = 1024,
        vision_num_hidden_layers: int = 24,
        vision_num_attention_heads: int = 8,
        vision_head_dim: int = 128,
        vision_intermediate_size: int = 4096,
        vision_rope_theta: float = 10_000.0,
        vision_norm_eps: float = 1e-5,
        num_channels: int = 3,
        patch_size: int = 14,
        image_size: int = 336,
        spatial_merge_size: int = 2,
        ln_pre: bool = True,
        # --- shared / contrastive head ---
        projection_dim: int = 1024,
        logit_scale_init: float = math.log(1 / 0.07),
        **kwargs,
    ) -> None:
        self.text_vocab_size = text_vocab_size
        self.text_dim = text_dim
        self.text_num_hidden_layers = text_num_hidden_layers
        self.text_num_attention_heads = text_num_attention_heads
        self.text_head_dim = text_head_dim
        self.text_intermediate_size = text_intermediate_size
        self.text_rope_theta = text_rope_theta
        self.text_norm_eps = text_norm_eps

        self.vision_dim = vision_dim
        self.vision_num_hidden_layers = vision_num_hidden_layers
        self.vision_num_attention_heads = vision_num_attention_heads
        self.vision_head_dim = vision_head_dim
        self.vision_intermediate_size = vision_intermediate_size
        self.vision_rope_theta = vision_rope_theta
        self.vision_norm_eps = vision_norm_eps
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.spatial_merge_size = spatial_merge_size
        self.ln_pre = ln_pre

        self.projection_dim = projection_dim
        self.logit_scale_init = logit_scale_init

        # `eos_token_id` is consumed by PretrainedConfig too; pass it through.
        super().__init__(eos_token_id=eos_token_id, **kwargs)


__all__ = ["MistralContrastiveConfig"]

"""HuggingFace modeling code for the Mistral two-encoder contrastive model.

Self-contained (only depends on ``torch`` and ``transformers``) so it can be shipped
alongside the weights and loaded via ``trust_remote_code=True``.

The forward passes replicate the Mistral training code exactly:

- RoPE uses the ``view_as_complex`` (interleaved adjacent-pair) convention from
  ``mistral/model/rope.py``, so converted q/k weights need **no** permutation.
- RMSNorm matches ``mistral/model/rms_norm.py``: normalize in fp32, cast back, then
  scale by ``weight``.
- SwiGLU FFN: ``down(silu(gate(x)) * up(x))``.
- Text tower: causal attention, final RMSNorm, EOS-token pooling, linear projection.
- Vision tower: bidirectional 2D-RoPE ViT, ``pre_mm_projector_norm``, 2x2 patch merge,
  ``contrastive_pre_norm``, ``amax`` over merged tokens, linear projection.

Both embeddings are L2-normalized.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.generic import ModelOutput

from .configuration_mistral_contrastive import MistralContrastiveConfig


def precompute_freqs_cis_1d(dim: int, end: int, theta: float, device: torch.device) -> torch.Tensor:
    """1D RoPE frequencies, shape (end, dim // 2), complex64. Matches mistral/model/rope.py."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(end, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def precompute_freqs_cis_2d(dim: int, height: int, width: int, theta: float, device: torch.device) -> torch.Tensor:
    """2D RoPE frequencies flattened to (height * width, dim // 2), complex64.

    Matches ``mistral/model/multimodal/rope.py::precompute_freqs_cis_2d`` (option 1):
    the first ``dim // 4`` frequencies index the height axis, the next ``dim // 4``
    index the width axis. Tokens are ordered row-major (height outer, width inner),
    matching the patch-conv ``flatten(2)`` order.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    h = torch.arange(height, device=device, dtype=torch.float32)
    w = torch.arange(width, device=device, dtype=torch.float32)
    freqs_h = torch.outer(h, freqs[::2])
    freqs_w = torch.outer(w, freqs[1::2])
    freqs_2d = torch.cat(
        [
            freqs_h[:, None, :].repeat(1, width, 1),
            freqs_w[None, :, :].repeat(height, 1, 1),
        ],
        dim=-1,
    )
    freqs_cis = torch.polar(torch.ones_like(freqs_2d), freqs_2d)
    return freqs_cis.reshape(height * width, dim // 2)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to a (batch, seq, n_heads, head_dim) tensor. ``freqs_cis`` is (seq, head_dim // 2)."""
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2))
    freqs_cis = freqs_cis.view(1, x.shape[1], 1, x.shape[-1] // 2)
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return x_out.type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)
        return output * self.weight


class SwiGLUMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)  # w1
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)  # w3
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)  # w2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        is_causal: bool,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q = self.q_proj(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seqlen, self.n_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # sdpa does not allow both is_causal and an explicit mask; an explicit mask
        # (used when padding is present) already encodes causality.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal and attn_mask is None)
        out = out.transpose(1, 2).reshape(bsz, seqlen, self.n_heads * self.head_dim)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int, hidden_dim: int, eps: float) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(dim, eps)
        self.self_attn = Attention(dim, n_heads, head_dim)
        self.post_attention_layernorm = RMSNorm(dim, eps)
        self.mlp = SwiGLUMLP(dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        is_causal: bool,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), freqs_cis, is_causal, attn_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class MistralContrastiveTextModel(nn.Module):
    def __init__(self, config: MistralContrastiveConfig) -> None:
        super().__init__()
        self.config = config
        assert config.eos_token_id is not None, "eos_token_id is required for EOS pooling"
        self.eos_token_id: int = config.eos_token_id
        self.head_dim: int = config.text_head_dim
        self.rope_theta: float = config.text_rope_theta

        self.embed_tokens = nn.Embedding(config.text_vocab_size, config.text_dim)
        self.layers = nn.ModuleList(
            TransformerBlock(
                config.text_dim,
                config.text_num_attention_heads,
                config.text_head_dim,
                config.text_intermediate_size,
                config.text_norm_eps,
            )
            for _ in range(config.text_num_hidden_layers)
        )
        self.norm = RMSNorm(config.text_dim, config.text_norm_eps)
        self.text_projection = nn.Linear(config.text_dim, config.projection_dim, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        x = self.embed_tokens(input_ids)
        freqs_cis = precompute_freqs_cis_1d(self.head_dim, seqlen, self.rope_theta, x.device)

        attn_mask: torch.Tensor | None = None
        if attention_mask is not None:
            causal = torch.tril(torch.ones(seqlen, seqlen, dtype=torch.bool, device=x.device))
            key_pad = attention_mask.to(torch.bool)[:, None, None, :]
            attn_mask = causal[None, None, :, :] & key_pad

        for layer in self.layers:
            x = layer(x, freqs_cis, is_causal=True, attn_mask=attn_mask)

        x = self.norm(x)

        # EOS pooling: take the hidden state at the first eos token of each row.
        matches = input_ids == self.eos_token_id
        assert bool(matches.any(dim=-1).all()), (
            f"every row must contain eos_token_id={self.eos_token_id} for EOS pooling"
        )
        eos_idx = matches.int().argmax(dim=-1)
        pooled = x[torch.arange(bsz, device=x.device), eos_idx]

        embeds = self.text_projection(pooled)
        return F.normalize(embeds, p=2, dim=-1)


class PatchMerger(nn.Module):
    def __init__(self, dim: int, spatial_merge_size: int) -> None:
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.merging_layer = nn.Linear(dim * spatial_merge_size**2, dim, bias=False)

    def forward(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        """x: (batch, grid_h * grid_w, dim) -> (batch, merged_tokens, dim)."""
        bsz, _, dim = x.shape
        s = self.spatial_merge_size
        img = x.view(bsz, grid_h, grid_w, dim).permute(0, 3, 1, 2)  # (B, dim, H, W)
        # F.unfold channel order is (dim, kh, kw) flattened as dim*kh*kw, which matches
        # the Mistral PatchMerger `get_sub_grids` ordering exactly.
        unfolded = F.unfold(img, kernel_size=s, stride=s)  # (B, dim * s * s, L)
        unfolded = unfolded.transpose(1, 2)  # (B, L, dim * s * s)
        return self.merging_layer(unfolded)


class MistralContrastiveVisionModel(nn.Module):
    def __init__(self, config: MistralContrastiveConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_size: int = config.patch_size
        self.head_dim: int = config.vision_head_dim
        self.rope_theta: float = config.vision_rope_theta
        self.spatial_merge_size: int = config.spatial_merge_size

        self.patch_conv = nn.Conv2d(
            config.num_channels,
            config.vision_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.ln_pre: nn.Module = RMSNorm(config.vision_dim, config.vision_norm_eps) if config.ln_pre else nn.Identity()
        self.layers = nn.ModuleList(
            TransformerBlock(
                config.vision_dim,
                config.vision_num_attention_heads,
                config.vision_head_dim,
                config.vision_intermediate_size,
                config.vision_norm_eps,
            )
            for _ in range(config.vision_num_hidden_layers)
        )
        self.pre_mm_projector_norm = RMSNorm(config.vision_dim, config.vision_norm_eps)
        self.patch_merger = PatchMerger(config.vision_dim, config.spatial_merge_size)
        self.contrastive_pre_norm = RMSNorm(config.vision_dim, config.vision_norm_eps)
        self.visual_projection = nn.Linear(config.vision_dim, config.projection_dim, bias=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.patch_conv(pixel_values)  # (B, dim, gh, gw)
        grid_h, grid_w = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)  # (B, gh * gw, dim)
        x = self.ln_pre(x)

        freqs_cis = precompute_freqs_cis_2d(self.head_dim, grid_h, grid_w, self.rope_theta, x.device)
        for layer in self.layers:
            x = layer(x, freqs_cis, is_causal=False)

        x = self.pre_mm_projector_norm(x)
        x = self.patch_merger(x, grid_h, grid_w)
        x = self.contrastive_pre_norm(x)
        x = x.amax(dim=1)  # amax pooling over the merged tokens

        embeds = self.visual_projection(x)
        return F.normalize(embeds, p=2, dim=-1)


@dataclass
class MistralContrastiveOutput(ModelOutput):
    logits_per_image: torch.Tensor | None = None
    logits_per_text: torch.Tensor | None = None
    image_embeds: torch.Tensor | None = None
    text_embeds: torch.Tensor | None = None


class MistralContrastiveModel(PreTrainedModel):
    config_class = MistralContrastiveConfig
    base_model_prefix = "mistral_contrastive"
    main_input_name = "pixel_values"
    _no_split_modules: ClassVar[list[str]] = ["TransformerBlock"]
    supports_gradient_checkpointing = False

    def __init__(self, config: MistralContrastiveConfig) -> None:
        super().__init__(config)
        self.text_model = MistralContrastiveTextModel(config)
        self.vision_model = MistralContrastiveVisionModel(config)
        self.logit_scale = nn.Parameter(torch.tensor([config.logit_scale_init], dtype=torch.float32))
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            bias = getattr(module, "bias", None)
            if bias is not None:
                bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def get_text_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.text_model(input_ids, attention_mask)

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_model(pixel_values)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> MistralContrastiveOutput:
        assert input_ids is not None and pixel_values is not None, (
            "MistralContrastiveModel.forward requires both input_ids and pixel_values; "
            "use get_text_features / get_image_features for a single tower."
        )
        image_embeds = self.get_image_features(pixel_values)
        text_embeds = self.get_text_features(input_ids, attention_mask)

        logit_scale = self.logit_scale.exp().to(image_embeds.device)
        logits_per_text = logit_scale * text_embeds @ image_embeds.t()
        logits_per_image = logits_per_text.t()

        return MistralContrastiveOutput(
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            image_embeds=image_embeds,
            text_embeds=text_embeds,
        )


MistralContrastiveConfig.register_for_auto_class()
MistralContrastiveModel.register_for_auto_class("AutoModel")

__all__ = ["MistralContrastiveModel", "MistralContrastiveOutput"]

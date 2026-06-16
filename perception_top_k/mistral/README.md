# Mistral two-encoder contrastive model (HF export)

A CLIP/SigLIP-style **dual encoder** (separate image + text towers) exported to HuggingFace
format. Converted from the Arsenal checkpoint `checkpoint_00275000` of the 336px 4T contrastive-only
run, using `morph convert-arsenal-contrastive-to-hf`.

- **Image embedding dim / Text embedding dim:** 1024 (shared contrastive space, L2-normalized)
- **Vision tower:** ViT, hidden 1024, 24 layers, patch 14, 2D-RoPE, 2×2 patch-merge, `amax` pooling
- **Text tower:** 1792-dim, 14 layers, causal, RoPE, EOS-token pooling (eos id 2)
- **Score:** `logits_per_image = logit_scale.exp() * image_embeds @ text_embeds.T`

Both towers are Mistral-style (RoPE + SwiGLU + RMSNorm, no biases). The model is loaded via
`trust_remote_code` (custom code in `modeling_mistral_contrastive.py` / `configuration_mistral_contrastive.py`).

## Files

| File | Purpose |
|------|---------|
| `model.safetensors`, `config.json` | weights + config |
| `configuration_mistral_contrastive.py`, `modeling_mistral_contrastive.py` | custom model code (`trust_remote_code`) |
| `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` | text tokenizer (adds `<s> … </s>` automatically) |
| `preprocessor_config.json` | `CLIPImageProcessor` (squash to 336², CLIP normalization) |
| `image_transforms.py` | squash + keep_ratio transforms (see below) |

## Image normalization

OpenAI-CLIP statistics (this is `centering="default"` in the training code):

```
mean = (0.48145466, 0.4578275, 0.40821073)
std  = (0.26862954, 0.26130258, 0.27577711)
```

These are baked into both `preprocessor_config.json` and `image_transforms.py`.

## Which image transform to use

The model is **variable-resolution** (2D RoPE, no learned position embeddings): it accepts square
**or** rectangular inputs, as long as height and width are multiples of `patch_size * spatial_merge_size`
= 14 × 2 = **28**.

| Transform | What it does | Use for |
|-----------|--------------|---------|
| **squash** | resize directly to `S×S` (distorts aspect ratio, no crop) | low / fixed resolution (224, 336) |
| **keep_ratio** | resize longest side to `S` keeping aspect ratio, round each side down to a multiple of 28 → rectangular, variable size | high resolution (e.g. 672) |

The bundled `CLIPImageProcessor` (`preprocessor_config.json`) implements **squash** at 336². Both
transforms (and the exact normalization) are also provided as plain torchvision in
`image_transforms.py`, so you can pick per resolution. Use `use_cv2_resize=True` to match the
training-time cv2 `INTER_AREA` resize exactly; the default (torchvision bicubic) is a close approximation.

> keep_ratio outputs vary in size per image, so encode images **one at a time** (or batch only
> same-size images).

## Usage

### Zero-shot classification (squash, low/fixed res)

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor

path = "/mnt/vast/runs/umberto.tomasini/hf_exports/ctr_2enc_275k"
model = AutoModel.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16).eval().cuda()
tok = AutoTokenizer.from_pretrained(path)
proc = CLIPImageProcessor.from_pretrained(path)  # squash to 336², CLIP normalization

classes = ["a photo of a cat", "a photo of a dog", "a photo of a car"]
text = tok(classes, return_tensors="pt", padding=True).to("cuda")           # adds <s> … </s>
pixel_values = proc(Image.open("img.jpg"), return_tensors="pt").pixel_values.to("cuda", torch.bfloat16)

with torch.no_grad():
    out = model(input_ids=text.input_ids, attention_mask=text.attention_mask, pixel_values=pixel_values)
print(out.logits_per_image.softmax(dim=-1))   # (1, n_classes)
```

### High resolution (keep_ratio) and raw embeddings

```python
import sys, torch
from PIL import Image
sys.path.insert(0, path)                        # to import image_transforms.py
from image_transforms import build_keep_ratio_transform

tf = build_keep_ratio_transform(size=672)       # rectangular, sides multiple of 28
pixel_values = tf(Image.open("img.jpg")).unsqueeze(0).to("cuda", torch.bfloat16)

with torch.no_grad():
    image_embeds = model.get_image_features(pixel_values)            # (1, 1024), L2-normalized
    text_embeds = model.get_text_features(text.input_ids, text.attention_mask)  # (n, 1024)
similarity = image_embeds @ text_embeds.T       # cosine similarity (already normalized)
```

`build_squash_transform(size=...)` is also available in `image_transforms.py` if you prefer to
control the squash resize (e.g. cv2 parity) instead of the bundled `CLIPImageProcessor`.

## Notes

- **Text must end with EOS** (id 2) — the tokenizer's post-processor wraps inputs as `<s> … </s>`
  automatically, so use `AutoTokenizer` as shown rather than tokenizing manually.
- `image_embeds` and `text_embeds` are already L2-normalized; their dot product is cosine similarity.
- Weights are bf16. `logit_scale` is stored in fp32.
- **Caveats:** the image preprocessing is a close approximation of the training cv2/variable-size
  pipeline (use `image_transforms.py` with `use_cv2_resize=True` for the closest match). Conversion
  of the model weights themselves is exact.

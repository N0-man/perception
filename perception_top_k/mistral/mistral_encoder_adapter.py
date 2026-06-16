"""Adapter wrapping the Mistral two-encoder contrastive model for image/text retrieval.

Exposes a unified interface:
    adapter.encode_images(pil_images_or_paths, batch_size=64)  -> np.ndarray (N, D)
    adapter.encode_texts(texts, batch_size=64)                  -> np.ndarray (N, D)

Both methods return L2-normalized float32 embeddings.
The Mistral model already L2-normalizes its output internally; the adapter re-applies
normalization as a safety net so callers can treat the outputs as unit vectors.

Model loading uses HuggingFace ``AutoModel.from_pretrained`` with
``trust_remote_code=True``, so the model directory must contain:
    - config.json
    - modeling_mistral_contrastive.py
    - configuration_mistral_contrastive.py
    - model.safetensors (or pytorch_model.bin / shards)
    - tokenizer.json + tokenizer_config.json
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_image(src: Union[str, Path, Image.Image]) -> Image.Image:
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    return Image.open(src).convert("RGB")


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True).clip(min=eps)
    return x / norms


# ---------------------------------------------------------------------------
# Public adapter class
# ---------------------------------------------------------------------------

class MistralEncoderAdapter:
    """Wraps the Mistral contrastive model for batch image and text encoding.

    Parameters
    ----------
    model_dir : str | Path
        Directory containing the HuggingFace checkpoint files
        (config.json, model.safetensors, tokenizer.json, …).
    device : str
        ``"cuda"`` or ``"cpu"``.  Defaults to CUDA if available.
    dtype : torch.dtype
        Weight/compute dtype.  Default ``torch.bfloat16`` on CUDA, ``float32`` on CPU.
    image_size : int
        Square size used for the squash-resize transform.  Must match the checkpoint
        (default 336 for the bundled weights).
    max_text_length : int
        Token budget for text encoding (including BOS + EOS).
    normalize : bool
        Whether to re-apply L2 normalization on the adapter side.  The Mistral model
        normalizes internally; this is a defensive belt-and-suspenders guard.
    """

    SUPPORTS_TEXT = True  # Mistral has a full text tower

    def __init__(
        self,
        model_dir: Union[str, Path],
        device: str | None = None,
        dtype: torch.dtype | None = None,
        image_size: int = 336,
        max_text_length: int = 77,
        normalize: bool = True,
    ) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise FileNotFoundError(
                f"Mistral model directory not found: {self.model_dir}\n"
                "Make sure the path points to the folder with config.json and model weights."
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        else:
            self.dtype = dtype

        self.image_size = image_size
        self.max_text_length = max_text_length
        self.normalize = normalize

        # Add the model dir to sys.path so trust_remote_code can import the local files
        model_dir_str = str(self.model_dir.resolve())
        if model_dir_str not in sys.path:
            sys.path.insert(0, model_dir_str)

        self._model: AutoModel | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._preprocess = None  # set after model load
        self._embed_dim: int | None = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def load(self) -> "MistralEncoderAdapter":
        """Load model and tokenizer onto the target device.  Safe to call multiple times."""
        if self._model is not None:
            return self

        print(f"[MistralEncoderAdapter] Loading model from: {self.model_dir}")
        t0 = time.time()

        self._model = AutoModel.from_pretrained(
            str(self.model_dir),
            trust_remote_code=True,
            torch_dtype=self.dtype,
        )
        self._model = self._model.to(self.device).eval()

        print(f"[MistralEncoderAdapter] Model loaded in {time.time() - t0:.1f}s")
        print(f"[MistralEncoderAdapter] Device: {self.device}  Dtype: {self.dtype}")

        # Projection dim (embedding dimension)
        self._embed_dim = self._model.config.projection_dim
        print(f"[MistralEncoderAdapter] Embedding dimension: {self._embed_dim}")

        # Tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            trust_remote_code=True,
        )
        # Ensure pad_token is set (Mistral tokenizer may not have one by default)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Image transform — import from the Mistral model directory
        from image_transforms import build_squash_transform  # type: ignore[import]
        self._preprocess = build_squash_transform(size=self.image_size)
        print(f"[MistralEncoderAdapter] Image size: {self.image_size}x{self.image_size}")

        return self

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self.load()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def embed_dim(self) -> int:
        self._ensure_loaded()
        return self._embed_dim  # type: ignore[return-value]

    @property
    def model(self) -> AutoModel:
        self._ensure_loaded()
        return self._model  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def encode_images(
        self,
        images: Sequence[Union[str, Path, Image.Image]],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode a list of image paths / PIL images.

        Parameters
        ----------
        images : sequence of path-like strings, Path objects, or PIL Images
        batch_size : int
        show_progress : bool

        Returns
        -------
        np.ndarray of shape (N, embed_dim), dtype float32, L2-normalized.
        """
        self._ensure_loaded()

        if len(images) == 0:
            raise ValueError("encode_images received an empty list.")

        all_embeddings: list[np.ndarray] = []
        failed_count = 0

        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding images", unit="batch")

        for start in iterator:
            batch_srcs = images[start : start + batch_size]
            tensors: list[torch.Tensor] = []

            for src in batch_srcs:
                try:
                    img = _load_image(src)
                    tensors.append(self._preprocess(img))
                except Exception as exc:
                    failed_count += 1
                    print(f"[MistralEncoderAdapter] WARNING: failed to load image "
                          f"'{src}': {exc}. Substituting zero tensor.")
                    tensors.append(torch.zeros(3, self.image_size, self.image_size))

            pixel_values = torch.stack(tensors).to(self.device, non_blocking=True)

            with torch.autocast(
                device_type="cuda" if self.device == "cuda" else "cpu",
                dtype=self.dtype,
                enabled=(self.device == "cuda"),
            ):
                embeds = self._model.get_image_features(pixel_values)

            embeds = embeds.float().cpu().numpy()

            if len(all_embeddings) == 0:
                print(f"[MistralEncoderAdapter] First batch embedding shape: {embeds.shape}, "
                      f"dtype: {embeds.dtype}")

            all_embeddings.append(embeds)

        if failed_count > 0:
            print(f"[MistralEncoderAdapter] WARNING: {failed_count} image(s) failed to load "
                  "and were replaced with zero vectors.")

        result = np.vstack(all_embeddings).astype("float32")
        if self.normalize:
            result = _l2_normalize(result)
        return result

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def encode_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 256,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode a list of text strings.

        The Mistral text tower uses EOS-token pooling.  Every tokenized sequence is
        given BOS at the start and EOS at the end before being passed to the model.

        Parameters
        ----------
        texts : sequence of str
        batch_size : int
        show_progress : bool

        Returns
        -------
        np.ndarray of shape (N, embed_dim), dtype float32, L2-normalized.
        """
        self._ensure_loaded()

        if isinstance(texts, str):
            texts = [texts]

        if len(texts) == 0:
            raise ValueError("encode_texts received an empty list.")

        all_embeddings: list[np.ndarray] = []
        tok = self._tokenizer
        eos_id: int = tok.eos_token_id

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding texts", unit="batch")

        for start in iterator:
            batch_texts = list(texts[start : start + batch_size])

            # Tokenize: add BOS (via add_special_tokens=True for Mistral),
            # then manually append EOS so the model can pool from it.
            encoded = tok(
                batch_texts,
                add_special_tokens=True,
                padding=False,
                truncation=True,
                max_length=self.max_text_length - 1,  # leave room for EOS
                return_tensors=None,  # list of lists
            )

            # Append EOS to each sequence and build padded batch
            all_ids: list[list[int]] = []
            all_masks: list[list[int]] = []
            for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"]):
                ids_with_eos = ids + [eos_id]
                mask_with_eos = mask + [1]
                all_ids.append(ids_with_eos)
                all_masks.append(mask_with_eos)

            # Pad to the longest sequence in this batch
            max_len = max(len(ids) for ids in all_ids)
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

            padded_ids = [ids + [pad_id] * (max_len - len(ids)) for ids in all_ids]
            padded_masks = [m + [0] * (max_len - len(m)) for m in all_masks]

            input_ids = torch.tensor(padded_ids, dtype=torch.long).to(self.device)
            attention_mask = torch.tensor(padded_masks, dtype=torch.long).to(self.device)

            with torch.autocast(
                device_type="cuda" if self.device == "cuda" else "cpu",
                dtype=self.dtype,
                enabled=(self.device == "cuda"),
            ):
                embeds = self._model.get_text_features(input_ids, attention_mask)

            embeds = embeds.float().cpu().numpy()

            if len(all_embeddings) == 0:
                print(f"[MistralEncoderAdapter] First batch text embedding shape: {embeds.shape}, "
                      f"dtype: {embeds.dtype}")

            all_embeddings.append(embeds)

        result = np.vstack(all_embeddings).astype("float32")
        if self.normalize:
            result = _l2_normalize(result)
        return result

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def summary(self) -> str:
        self._ensure_loaded()
        lines = [
            "MistralEncoderAdapter",
            f"  model_dir     : {self.model_dir}",
            f"  device        : {self.device}",
            f"  dtype         : {self.dtype}",
            f"  embed_dim     : {self._embed_dim}",
            f"  image_size    : {self.image_size}",
            f"  max_text_len  : {self.max_text_length}",
            f"  normalize     : {self.normalize}",
            f"  supports_text : {self.SUPPORTS_TEXT}",
        ]
        return "\n".join(lines)

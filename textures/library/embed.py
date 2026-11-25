"""Text embeddings for comparing descriptions by meaning.

BAAI/bge-small-en-v1.5 (the model the work journal's search uses), on the CPU: a description
is a sentence or two, and the GPU is left to klein and CHORD. Loaded the first time it is
needed, from the Hugging Face cache (about 130 MB, downloaded once).
"""

from __future__ import annotations

import threading

import numpy as np

MODEL = "BAAI/bge-small-en-v1.5"
_LOCK = threading.Lock()
_LOADED = None


def embed(texts: list[str]) -> np.ndarray:
    """Unit vectors for `texts`, one row each (float32)."""
    global _LOADED
    import torch
    from transformers import AutoModel, AutoTokenizer

    with _LOCK:
        if _LOADED is None:
            _LOADED = (AutoTokenizer.from_pretrained(MODEL), AutoModel.from_pretrained(MODEL).eval())
        tokenizer, model = _LOADED
        with torch.no_grad():
            batch = tokenizer(list(texts), padding=True, truncation=True, max_length=256, return_tensors="pt")
            out = model(**batch).last_hidden_state[:, 0]
            out = torch.nn.functional.normalize(out, dim=-1)
    return out.numpy().astype(np.float32)

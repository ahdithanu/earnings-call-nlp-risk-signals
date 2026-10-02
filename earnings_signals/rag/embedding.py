"""Dense embedders behind one interface.

The production embedder is model2vec's ``potion-base-8M``: static token
embeddings distilled from a sentence transformer. The choice is a
deployment trade-off made explicit — ~30MB of weights and pure-numpy
inference instead of a ~2GB torch stack, so the Docker image stays small
enough for the free-tier App Runner target. Static embeddings are weaker
than a full transformer at word-order nuance; the hybrid index compensates
by fusing them with BM25, and scripts/eval_rag.py measures exactly how
much each half contributes. A cross-encoder reranker would slot in after
retrieval if quality ever justifies the weight.

``HashingEmbedder`` is a deterministic, dependency-free fallback used by
the unit tests (no model download in CI) — a bag of hashed token and
character-trigram features. It captures lexical similarity only, which is
all the tests need.

Vectors from both are L2-normalized so dot product == cosine similarity.
"""

import hashlib
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "minishlab/potion-base-8M"


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """(len(texts), dim) float32 array, rows L2-normalized."""
        ...


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class HashingEmbedder:
    """Deterministic lexical embedder: hashed tokens + char trigrams."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _features(self, text: str) -> list[str]:
        tokens = text.lower().split()
        feats = list(tokens)
        for tok in tokens:
            padded = f"#{tok}#"
            feats.extend(padded[i : i + 3] for i in range(len(padded) - 2))
        return feats

    def encode(self, texts: list[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat in self._features(text):
                digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                mat[row, idx] += sign
        return _l2_normalize(mat)


class Model2VecEmbedder:
    """Static sentence embeddings via model2vec (lazy model load)."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.name = model_name.split("/")[-1]
        self._model = None
        self.dim = 256  # potion-base-8M; corrected on first encode

    def _load(self):
        if self._model is None:
            from model2vec import StaticModel  # deferred: optional [rag] extra

            self._model = StaticModel.from_pretrained(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        mat = np.asarray(model.encode(texts), dtype=np.float32)
        if mat.ndim == 1:  # single text
            mat = mat.reshape(1, -1)
        self.dim = mat.shape[1]
        return _l2_normalize(mat)


def get_embedder(name: str) -> Embedder:
    """Embedder by the name stored in an index's meta.json."""
    if name.startswith("hashing-"):
        return HashingEmbedder(dim=int(name.split("-")[1]))
    if name == DEFAULT_MODEL.split("/")[-1]:
        return Model2VecEmbedder(DEFAULT_MODEL)
    return Model2VecEmbedder(name)

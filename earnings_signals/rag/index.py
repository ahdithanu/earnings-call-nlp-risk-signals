"""Hybrid retrieval index: BM25 + dense cosine, fused with RRF.

Earnings-call questions mix two kinds of matching. "What did NVDA say
about export controls?" needs exact terms (tickers, product names,
'tariffs') — that's BM25's home turf. "Which companies sounded worried
about demand?" needs semantic matching — the dense half. Neither alone is
right, so the index keeps both and fuses their rankings with reciprocal
rank fusion (RRF), which needs no score calibration between the two
systems. ``mode`` exposes each half separately so scripts/eval_rag.py can
measure what fusion buys.

Metadata filters (tickers) are applied BEFORE ranking — the permission
model: a caller entitled to a subset of the corpus has everything else
masked out of retrieval, not filtered from the response afterward.

The index persists as a directory: meta.json (embedder identity — a query
must be embedded by the same model that built the matrix), chunks.jsonl.gz
and embeddings.npz (fp16 on disk). BM25 statistics are rebuilt from the
chunk texts at load time; they're cheap and derivable.
"""

import gzip
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from earnings_signals.rag.chunking import Chunk
from earnings_signals.rag.embedding import Embedder
from earnings_signals.uncertainty import tokenize

BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60  # standard fusion constant; rank 0 contributes 1/60


class _BM25:
    """Okapi BM25 over tokenized chunk texts (inverted index in memory)."""

    def __init__(self, docs: list[list[str]]):
        self.n_docs = len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for doc_id, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                self.postings.setdefault(term, []).append((doc_id, tf))

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        out = np.zeros(self.n_docs, dtype=np.float32)
        for term in set(query_tokens):
            plist = self.postings.get(term)
            if not plist:
                continue
            df = len(plist)
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for doc_id, tf in plist:
                denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * self.doc_len[doc_id] / self.avgdl)
                out[doc_id] += idf * tf * (BM25_K1 + 1) / denom
        return out


@dataclass
class SearchResult:
    chunk: Chunk
    score: float  # fused (or single-mode) score used for the final ranking
    dense_score: float  # cosine similarity
    bm25_score: float


class HybridIndex:
    def __init__(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        embedder_name: str,
        built: str = "",
        scope: str = "",
    ):
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(f"{len(chunks)} chunks but {embeddings.shape[0]} embedding rows")
        self.chunks = chunks
        self.embeddings = embeddings.astype(np.float32)
        self.embedder_name = embedder_name
        self.built = built
        self.scope = scope
        self._bm25 = _BM25([tokenize(c.text) for c in chunks])
        self._tickers = np.array([c.ticker for c in chunks])

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        embedder: Embedder,
        scope: str = "",
        batch_size: int = 512,
    ) -> "HybridIndex":
        parts = [
            embedder.encode([c.text for c in chunks[i : i + batch_size]])
            for i in range(0, len(chunks), batch_size)
        ]
        embeddings = np.vstack(parts) if parts else np.zeros((0, embedder.dim), dtype=np.float32)
        return cls(
            chunks,
            embeddings,
            embedder.name,
            built=datetime.now(UTC).strftime("%Y-%m-%d"),
            scope=scope,
        )

    def search(
        self,
        query: str,
        embedder: Embedder,
        k: int = 10,
        tickers: list[str] | None = None,
        mode: str = "hybrid",  # "hybrid" | "dense" | "bm25"
    ) -> list[SearchResult]:
        if embedder.name != self.embedder_name:
            raise ValueError(
                f"index built with {self.embedder_name!r}, queried with {embedder.name!r}"
            )
        n = len(self.chunks)
        if n == 0:
            return []

        mask = np.ones(n, dtype=bool)
        if tickers:
            mask = np.isin(self._tickers, [t.upper() for t in tickers])
            if not mask.any():
                return []

        dense = self.embeddings @ embedder.encode([query])[0]
        bm25 = self._bm25.scores(tokenize(query))
        neg = np.float32(-1e9)
        dense_m = np.where(mask, dense, neg)
        bm25_m = np.where(mask, bm25, neg)

        if mode == "dense":
            fused = dense_m
        elif mode == "bm25":
            fused = bm25_m
        else:
            # RRF over the two rankings, restricted to unmasked docs
            fused = np.full(n, neg, dtype=np.float32)
            fused[mask] = 0.0
            for scores_m in (dense_m, bm25_m):
                order = np.argsort(-scores_m)
                ranks = np.empty(n, dtype=np.int64)
                ranks[order] = np.arange(n)
                fused[mask] += 1.0 / (RRF_K + ranks[mask])

        top = np.argsort(-fused)[: min(k, int(mask.sum()))]
        return [
            SearchResult(
                chunk=self.chunks[i],
                score=float(fused[i]),
                dense_score=float(dense[i]),
                bm25_score=float(bm25[i]),
            )
            for i in top
            if mask[i]
        ]

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "embedder": self.embedder_name,
            "dim": int(self.embeddings.shape[1]),
            "n_chunks": len(self.chunks),
            "built": self.built,
            "scope": self.scope,
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        with gzip.open(path / "chunks.jsonl.gz", "wt", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        np.savez_compressed(path / "embeddings.npz", embeddings=self.embeddings.astype(np.float16))

    @classmethod
    def load(cls, path: Path | str) -> "HybridIndex":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        with gzip.open(path / "chunks.jsonl.gz", "rt", encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f]
        embeddings = np.load(path / "embeddings.npz")["embeddings"].astype(np.float32)
        return cls(
            chunks,
            embeddings,
            meta["embedder"],
            built=meta.get("built", ""),
            scope=meta.get("scope", ""),
        )

"""Retrieval-augmented access to the earnings-call corpus.

The pipeline's other modules reduce transcripts to numbers; this package
keeps the words. It chunks executive Q&A speech into speaker turns, embeds
them, and serves grounded question-answering over the result:

  chunking   — speaker-turn chunks with ticker/sector/quarter metadata
  embedding  — dense embedders (static model2vec by default; a hashing
               embedder for hermetic tests)
  index      — hybrid retrieval: BM25 + dense cosine, fused with RRF,
               with metadata filtering applied before ranking
  answer     — extractive grounded answers: every answer sentence is
               verbatim corpus text with a citation, or the question is
               refused as outside the knowledge base

Build the index artifact with ``python -m scripts.build_rag_index``;
measure retrieval quality with ``python -m scripts.eval_rag``.
"""

from earnings_signals.rag.answer import Answer, Citation, answer_question
from earnings_signals.rag.chunking import Chunk, chunk_transcript
from earnings_signals.rag.embedding import Embedder, HashingEmbedder, get_embedder
from earnings_signals.rag.index import HybridIndex, SearchResult

__all__ = [
    "Answer",
    "Chunk",
    "Citation",
    "Embedder",
    "HashingEmbedder",
    "HybridIndex",
    "SearchResult",
    "answer_question",
    "chunk_transcript",
    "get_embedder",
]

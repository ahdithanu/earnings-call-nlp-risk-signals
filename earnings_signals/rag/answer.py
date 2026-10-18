"""Grounded extractive answers over the retrieval index.

The grounding guarantee is structural, not behavioral: the answer is
assembled ONLY from verbatim corpus sentences, each tagged with the chunk
it came from, so there is nothing a model could hallucinate — the same
philosophy as the insight readouts (composed from data, never generated).
The generative mode (earnings_signals/rag/generate.py — Claude over the
retrieved excerpts, cite-or-refuse, checked for groundedness) layers on
top of exactly this retrieval contract; the extractive mode here is the
deterministic, CI-testable floor and the fallback when generation cannot
be verified.

Refusal is the other half of grounding. A question whose best retrieval
is weak on BOTH halves of the hybrid — no meaningful lexical overlap
(BM25 ~ 0) and dense similarity below ``REFUSE_DENSE_SIM`` — is outside
the knowledge base, and the honest answer is to say so rather than return
the least-irrelevant excerpt.
"""

import re
from dataclasses import asdict, dataclass, field

import numpy as np

from earnings_signals.rag.embedding import Embedder
from earnings_signals.rag.index import HybridIndex

K_RETRIEVE = 20  # cast wide (recall)...
K_KEEP = 5  # ...then keep few (precision); see results/rag_eval.txt
MAX_ANSWER_SENTENCES = 3
MIN_SENT_TOKENS = 6
MAX_SENT_TOKENS = 80

# Refusal floor, calibrated on the production index (55k chunks, potion
# embeddings): best-match dense cosine for scrambled-word junk spans
# 0.34–0.48 (a max over 55k docs is never near zero), real questions
# 0.53–0.66. 0.50 splits the measured distributions; any real BM25 term
# match overrides it so terse rare-term queries ("chartplotters?") still
# answer. Re-measure via scripts/eval_rag.py if the embedder changes.
REFUSE_DENSE_SIM = 0.50
REFUSE_BM25 = 1e-6

CAVEAT = (
    "Answers are verbatim excerpts from executive Q&A answers in historical "
    "earnings calls, selected by retrieval — not investment advice."
)

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$")


@dataclass
class Citation:
    chunk_id: str
    ticker: str
    company: str
    quarter: str
    speaker: str
    score: float


@dataclass
class Answer:
    question: str
    refused: bool
    reason: str | None  # set when refused
    answer: str | None  # the answer text (cited verbatim lines, or generated prose)
    citations: list[Citation] = field(default_factory=list)
    caveat: str = CAVEAT
    mode: str = "extractive"  # "extractive" | "generative" | "extractive_fallback"
    fallback_reason: str | None = None  # why generation fell back, when it did

    def to_dict(self) -> dict:
        return asdict(self)


def citation_for(result, score: float | None = None) -> Citation:
    c = result.chunk
    return Citation(
        chunk_id=c.chunk_id,
        ticker=c.ticker,
        company=c.company,
        quarter=c.quarter_label,
        speaker=c.speaker,
        score=round(score if score is not None else result.score, 4),
    )


def retrieve(
    question: str,
    index: HybridIndex,
    embedder: Embedder,
    *,
    k_retrieve: int = K_RETRIEVE,
    k_keep: int = K_KEEP,
    tickers: list[str] | None = None,
) -> tuple[list, Answer | None]:
    """The shared retrieval + refusal stage of both answer modes.

    Returns (kept results, None) when the question is answerable, or
    ([], refusal Answer) when it is out of scope / out of knowledge base.
    """
    results = index.search(question, embedder, k=k_retrieve, tickers=tickers)
    if not results:
        return [], Answer(
            question=question,
            refused=True,
            reason="no documents in scope (empty index or ticker filter matched nothing)",
            answer=None,
        )
    best_dense = max(r.dense_score for r in results)
    best_bm25 = max(r.bm25_score for r in results)
    if best_dense < REFUSE_DENSE_SIM and best_bm25 < REFUSE_BM25:
        return [], Answer(
            question=question,
            refused=True,
            reason="question does not match the knowledge base (executive Q&A "
            "answers from S&P 500 earnings calls); refusing rather than "
            "returning unrelated excerpts",
            answer=None,
        )
    return results[:k_keep], None


def answer_question(
    question: str,
    index: HybridIndex,
    embedder: Embedder,
    *,
    k_retrieve: int = K_RETRIEVE,
    k_keep: int = K_KEEP,
    tickers: list[str] | None = None,
) -> Answer:
    kept, refusal = retrieve(
        question, index, embedder, k_retrieve=k_retrieve, k_keep=k_keep, tickers=tickers
    )
    if refusal is not None:
        return refusal

    # Rank every sentence in the kept chunks by similarity to the question.
    candidates: list[tuple[int, str]] = []  # (kept-chunk idx, sentence)
    for ci, r in enumerate(kept):
        for m in _SENTENCE_RE.finditer(r.chunk.text):
            sentence = m.group(0).strip()
            if MIN_SENT_TOKENS <= len(sentence.split()) <= MAX_SENT_TOKENS:
                candidates.append((ci, sentence))
    if not candidates:
        candidates = [(0, kept[0].chunk.text)]

    sent_vecs = embedder.encode([s for _, s in candidates])
    q_vec = embedder.encode([question])[0]
    sims = sent_vecs @ q_vec
    order = np.argsort(-sims)[:MAX_ANSWER_SENTENCES]

    lines = []
    cited: list[int] = []  # kept-chunk indices in citation order
    for i in order:
        ci, sentence = candidates[int(i)]
        if ci not in cited:
            cited.append(ci)
        n = cited.index(ci) + 1
        c = kept[ci].chunk
        lines.append(f'- "{sentence}" — {c.ticker} {c.quarter_label} executive Q&A [{n}]')

    citations = [citation_for(kept[ci]) for ci in cited]
    return Answer(
        question=question,
        refused=False,
        reason=None,
        answer="\n".join(lines),
        citations=citations,
    )

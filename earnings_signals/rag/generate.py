"""Generative answers: Claude over the retrieved excerpts, cite-or-refuse.

The retrieval contract is identical to the extractive mode (same hybrid
search, same calibrated refusal floor — rag/answer.py); this module adds
the generation step, with grounding enforced in BOTH directions:

  before   — the prompt contains ONLY the retrieved excerpts, numbered,
             with an explicit instruction set: answer from the excerpts
             alone, cite every sentence with [n] markers, and reply with
             the literal token NOT_IN_CONTEXT when they don't contain the
             answer (the model-side refusal, on top of the retrieval-side
             one).
  after    — ``check_grounded`` verifies the output: every [n] marker
             must name a real excerpt, every sentence must carry at least
             one marker, and each sentence's content words must actually
             appear in the excerpts it cites (a fabricated claim fails
             this lexical-support test). Any violation discards the
             generation and falls back to the deterministic extractive
             answer, with the reason reported — the service degrades to
             verbatim quotes, never to unverified prose.

The Claude call (Opus 5.5 by default; override with ASK_LLM_MODEL) uses
low effort — synthesis over five short excerpts is not a reasoning task —
and server-side refusal fallbacks so a safety-classifier decline reroutes
instead of erroring; if the account/platform rejects that beta, the call
retries plain. Requires ANTHROPIC_API_KEY; the API returns 503 for
generate=true without it, and the extractive mode never needs it.
"""

import os
import re

from earnings_signals.rag.answer import (
    K_KEEP,
    K_RETRIEVE,
    Answer,
    citation_for,
    retrieve,
)
from earnings_signals.rag.answer import (
    answer_question as extractive_answer,
)
from earnings_signals.rag.embedding import Embedder
from earnings_signals.rag.index import HybridIndex
from earnings_signals.uncertainty import tokenize

DEFAULT_MODEL = "claude-opus-5-5"
MAX_ANSWER_TOKENS = 2048  # answers are a short cited paragraph by design
NOT_IN_CONTEXT = "NOT_IN_CONTEXT"

# Lexical-support floor: at least this share of a sentence's content words
# must appear in the excerpts it cites. Verbatim quotes score ~1.0; honest
# paraphrase stays well above 0.5 because the subject matter words come
# from the excerpt; a fabricated claim pulls vocabulary from nowhere.
MIN_SUPPORT = 0.4

_MARKER_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$")
_STOP = set(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "as",
        "at",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "we",
        "our",
        "you",
        "your",
        "they",
        "their",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "have",
        "has",
        "had",
        "will",
        "would",
        "could",
        "may",
        "might",
        "can",
        "not",
        "no",
        "than",
        "then",
        "there",
        "here",
        "what",
        "which",
        "who",
        "said",
        "says",
        "say",
        "about",
        "into",
        "over",
        "under",
        "also",
        "very",
        "more",
        "most",
        "some",
        "all",
        "any",
        "one",
        "two",
        "three",
    ]
)

SYSTEM_PROMPT = (
    "You answer questions about S&P 500 earnings calls using ONLY the numbered "
    "excerpts provided in the user message. The excerpts are verbatim executive "
    "answers from the Q&A sections of real calls.\n"
    "Rules:\n"
    "1. Use no knowledge beyond the excerpts — not even well-known facts.\n"
    "2. Every sentence you write must end with the [n] marker(s) of the "
    "excerpt(s) that support it.\n"
    f"3. If the excerpts do not contain the answer, reply with exactly "
    f"{NOT_IN_CONTEXT} and nothing else.\n"
    "4. Be concise: 2-4 sentences of synthesis, no preamble, no caveats of "
    "your own (the service appends its own disclaimer).\n"
    "5. Attribute claims to the company and quarter they come from."
)


def _excerpts_block(kept: list) -> str:
    lines = []
    for n, r in enumerate(kept, start=1):
        c = r.chunk
        lines.append(f"[{n}] {c.company} ({c.ticker}), {c.quarter_label}, {c.speaker}:\n{c.text}")
    return "\n\n".join(lines)


def _content_words(text: str) -> set[str]:
    return {t for t in tokenize(text) if len(t) > 3 and t not in _STOP}


def check_grounded(text: str, kept: list) -> tuple[bool, str]:
    """Verify a generated answer against the excerpts it was given."""
    markers_seen = {int(m) for m in _MARKER_RE.findall(text)}
    if not markers_seen:
        return False, "no citations in generated answer"
    valid = set(range(1, len(kept) + 1))
    if not markers_seen <= valid:
        return False, f"citation of nonexistent excerpt {sorted(markers_seen - valid)}"

    excerpt_words = {n: _content_words(kept[n - 1].chunk.text) for n in valid}
    for m in _SENTENCE_RE.finditer(text):
        sentence = m.group(0).strip()
        if not sentence:
            continue
        cited = [int(x) for x in _MARKER_RE.findall(sentence)]
        if not cited:
            return False, f"uncited sentence: {sentence[:60]!r}"
        words = _content_words(_MARKER_RE.sub("", sentence))
        if not words:
            continue
        support: set[str] = set()
        for n in cited:
            support |= excerpt_words[n]
        if len(words & support) / len(words) < MIN_SUPPORT:
            return False, f"low lexical support for: {sentence[:60]!r}"
    return True, ""


def _call_model(question: str, kept: list, model: str) -> str:
    """One Claude call over the excerpts. Isolated so tests can mock it."""
    import anthropic

    client = anthropic.Anthropic()
    user = f"Excerpts:\n\n{_excerpts_block(kept)}\n\nQuestion: {question}"
    kwargs: dict = dict(
        model=model,
        max_tokens=MAX_ANSWER_TOKENS,
        output_config={"effort": "low"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
    )
    try:
        # Server-side refusal fallbacks: a safety-classifier decline is
        # rerouted instead of surfacing as a refused response.
        response = client.beta.messages.create(
            betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
        )
    except (anthropic.BadRequestError, TypeError):
        # Platform/SDK without the fallback beta: plain call.
        response = client.messages.create(**kwargs)
    if response.stop_reason == "refusal":
        return NOT_IN_CONTEXT
    return "".join(b.text for b in response.content if b.type == "text").strip()


def generate_answer(
    question: str,
    index: HybridIndex,
    embedder: Embedder,
    *,
    k_retrieve: int = K_RETRIEVE,
    k_keep: int = K_KEEP,
    tickers: list[str] | None = None,
    model: str | None = None,
) -> Answer:
    """Grounded generative answer, falling back to extractive on any doubt."""
    model = model or os.environ.get("ASK_LLM_MODEL", DEFAULT_MODEL)
    kept, refusal = retrieve(
        question, index, embedder, k_retrieve=k_retrieve, k_keep=k_keep, tickers=tickers
    )
    if refusal is not None:
        refusal.mode = "generative"
        return refusal

    def fallback(reason: str) -> Answer:
        ans = extractive_answer(
            question, index, embedder, k_retrieve=k_retrieve, k_keep=k_keep, tickers=tickers
        )
        ans.mode = "extractive_fallback"
        ans.fallback_reason = reason
        return ans

    try:
        text = _call_model(question, kept, model)
    except Exception as exc:  # API/network failure: degrade, don't 500
        return fallback(f"generation failed ({type(exc).__name__}); served extractive answer")

    if text.strip() == NOT_IN_CONTEXT:
        return Answer(
            question=question,
            refused=True,
            reason="the retrieved excerpts do not contain an answer (model-side "
            "refusal on top of the retrieval-side threshold)",
            answer=None,
            mode="generative",
        )

    ok, why = check_grounded(text, kept)
    if not ok:
        return fallback(
            f"generated answer failed the groundedness check ({why}); served extractive answer"
        )

    cited_ns = sorted({int(m) for m in _MARKER_RE.findall(text)})
    return Answer(
        question=question,
        refused=False,
        reason=None,
        answer=text,
        citations=[citation_for(kept[n - 1]) for n in cited_ns],
        mode="generative",
    )

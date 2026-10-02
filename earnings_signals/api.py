"""FastAPI scoring service.

Exposes the pipeline's text-scoring machinery over HTTP:

  POST /score            — score arbitrary transcript text: uncertainty + LM
                           tone densities (negation-aware) for every derivable
                           scope (full; Q&A; executive-only; CEO / CFO)
  POST /ask              — grounded Q&A over the executive-answer corpus:
                           hybrid retrieval, verbatim cited excerpts, refusal
                           for out-of-KB questions (earnings_signals/rag/)
  GET  /signals          — the latest out-of-sample hedging watchlist
  GET  /insight/{ticker} — plain-English readout of a company's latest call
  GET  /healthz          — liveness + lexicon sanity

Run locally:  uvicorn earnings_signals.api:app --reload
The service is stateless (lexicons load once at startup); CORS is open
because /score is a read-only demo endpoint over caller-supplied text.
"""

import csv
import json
import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from earnings_signals.exec_roles import exec_qa_by_role
from earnings_signals.lexicon import CONTROL_LOADERS, REPO_ROOT, load_uncertainty_terms
from earnings_signals.qa_extract import extract_qa
from earnings_signals.qa_isolation import isolate_executive_qa
from earnings_signals.uncertainty import count_uncertainty

SIGNALS_CSV = REPO_ROOT / "results" / "latest_uncertainty_signals.csv"
INSIGHTS_JSON = REPO_ROOT / "data" / "processed" / "latest_insights.json"
MAX_TEXT_BYTES = 1_000_000  # one transcript is ~50KB; 1MB is generous

# Retrieval index for /ask (built by scripts/build_rag_index.py). Loaded
# lazily on first request so every other endpoint works without the [rag]
# extra or the artifact; cached per directory.
_RAG_CACHE: dict[str, tuple] = {}


def _rag_index_dir() -> Path:
    return Path(
        os.environ.get("RAG_INDEX_DIR", str(REPO_ROOT / "data" / "processed" / "rag_index"))
    )


def _load_rag():
    path = _rag_index_dir()
    key = str(path.resolve())
    if key not in _RAG_CACHE:
        if not (path / "meta.json").exists():
            raise HTTPException(
                status_code=503,
                detail="retrieval index not built — run `python -m scripts.build_rag_index`",
            )
        from earnings_signals.rag import HybridIndex, get_embedder

        index = HybridIndex.load(path)
        try:
            embedder = get_embedder(index.embedder_name)
            embedder.encode(["warmup"])  # fail fast if the model can't load
        except Exception as exc:  # model2vec missing or model not cached
            raise HTTPException(
                status_code=503,
                detail=f"embedding model {index.embedder_name!r} unavailable: {exc}",
            ) from exc
        _RAG_CACHE[key] = (index, embedder)
    return _RAG_CACHE[key]


try:
    _VERSION = pkg_version("earnings-signals")
except PackageNotFoundError:  # running from a bare checkout
    _VERSION = "0.0.0+uninstalled"

_LEXICON = load_uncertainty_terms()
_CONTROLS = {cat: loader() for cat, loader in CONTROL_LOADERS.items()}


class ScoreRequest(BaseModel):
    text: str = Field(min_length=1, description="Transcript (or any) text to score")


class ScopeScore(BaseModel):
    total_tokens: int
    uncertainty_count: int
    negation_excluded: int
    uncertainty_density: float | None  # per 100 tokens; None for empty text
    tone_density: dict[str, float | None]  # negative/positive/litigious/constraining


class ScoreResponse(BaseModel):
    scopes: dict[str, ScopeScore]
    qa_isolated: bool
    exec_isolated: bool
    role_source: str | None  # "roster" | "intro" | None


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    tickers: list[str] | None = Field(
        default=None,
        max_length=10,
        description="Restrict retrieval to these tickers (masked before ranking)",
    )
    k: int = Field(default=5, ge=1, le=10, description="Excerpt chunks to keep after fusion")


class CitationModel(BaseModel):
    chunk_id: str
    ticker: str
    company: str
    quarter: str
    speaker: str
    score: float


class AskResponse(BaseModel):
    question: str
    refused: bool
    reason: str | None
    answer: str | None  # verbatim cited excerpt sentences, or None when refused
    citations: list[CitationModel]
    caveat: str
    index: dict  # embedder, n_chunks, scope, built


class HealthResponse(BaseModel):
    status: str
    version: str
    lexicon_terms: int


app = FastAPI(
    title="Earnings Call NLP Risk Signals",
    description="Negation-aware Loughran-McDonald uncertainty scoring for "
    "earnings-call text, plus the latest hedging watchlist.",
    version=_VERSION,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _score_scope(text: str) -> ScopeScore:
    r = count_uncertainty(text, _LEXICON)
    return ScopeScore(
        total_tokens=r.total_tokens,
        uncertainty_count=r.uncertainty_count,
        negation_excluded=r.negation_excluded,
        uncertainty_density=r.density,
        tone_density={cat: count_uncertainty(text, lex).density for cat, lex in _CONTROLS.items()},
    )


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    if len(req.text.encode("utf-8", errors="ignore")) > MAX_TEXT_BYTES:
        raise HTTPException(status_code=413, detail=f"text exceeds {MAX_TEXT_BYTES} bytes")

    scopes = {"full": _score_scope(req.text)}

    qa = extract_qa(req.text)
    if qa is not None:
        scopes["qa"] = _score_scope(qa)

    iso = isolate_executive_qa(req.text)
    if iso.mode == "exec_turns":
        scopes["execqa"] = _score_scope(iso.text)

    by_role, role_source = exec_qa_by_role(req.text)
    for role in ("ceo", "cfo"):
        role_text = (by_role or {}).get(role)
        if role_text:
            scopes[role] = _score_scope(role_text)

    return ScoreResponse(
        scopes=scopes,
        qa_isolated=qa is not None,
        exec_isolated=iso.mode == "exec_turns",
        role_source=role_source,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Grounded question-answering over executive Q&A answers.

    Hybrid retrieval (BM25 + dense, RRF-fused) over speaker-turn chunks;
    the answer is composed ONLY of verbatim corpus sentences with
    citations, and out-of-knowledge-base questions are refused rather than
    answered badly. Ticker filters are applied before ranking — the
    entitlement model for callers scoped to a subset of the corpus."""
    index, embedder = _load_rag()
    from earnings_signals.rag import answer_question

    ans = answer_question(req.question, index, embedder, k_keep=req.k, tickers=req.tickers)
    return AskResponse(
        question=ans.question,
        refused=ans.refused,
        reason=ans.reason,
        answer=ans.answer,
        citations=[CitationModel(**c.__dict__) for c in ans.citations],
        caveat=ans.caveat,
        index={
            "embedder": index.embedder_name,
            "n_chunks": len(index.chunks),
            "scope": index.scope,
            "built": index.built,
        },
    )


@app.get("/signals")
def signals(limit: int = 25) -> list[dict[str, str]]:
    """Latest hedging watchlist, highest z-score first (as exported by
    scripts/latest_signals.py; regenerated by the weekly refresh)."""
    if not SIGNALS_CSV.exists():
        raise HTTPException(status_code=503, detail="watchlist not generated yet")
    limit = max(1, min(limit, 500))
    with SIGNALS_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit]


@app.get("/insight/{ticker}")
def insight(ticker: str) -> dict:
    """Plain-English readout of a company's latest call: level vs its own
    history, CEO-vs-CFO driver, tone character, driving terms, and the
    highest-density executive sentences — composed deterministically from
    the data by scripts/build_insights.py (see earnings_signals/insights.py)."""
    if not INSIGHTS_JSON.exists():
        raise HTTPException(status_code=503, detail="insights not generated yet")
    with INSIGHTS_JSON.open(encoding="utf-8") as f:
        payload = json.load(f)
    readout = payload.get("insights", {}).get(ticker.upper())
    if readout is None:
        raise HTTPException(
            status_code=404,
            detail=f"no readout for {ticker.upper()!r} — ticker not in the panel "
            "or too little history to benchmark",
        )
    return {"generated": payload.get("generated"), **readout}


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(status="ok", version=_VERSION, lexicon_terms=len(_LEXICON))

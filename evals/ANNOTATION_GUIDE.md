# Q&A Boundary Annotation Guide

Rules for labeling the gold Q&A boundary in an earnings-call transcript.
Annotations live in `evals/annotations.json`; the benchmark manifest is
`evals/manifest.json`. Labels must come from **reading the source text**
— never from the extractor's output. Use
`python -m evals.show_candidates <id>` to surface candidate regions; its
cue set is deliberately broader than the extractor's so your attention is
not anchored to what the extractor already finds.

## What counts as the Q&A section

The Q&A section is the portion of the call from the **Q&A-opening
element** to the end of call content. The Q&A-opening element is the
FIRST of:

1. an explicit section heading (`Question-and-Answer Session`,
   `Questions & Answers:`, ...), when the transcript has one; else
2. the start of the **utterance that initiates the live Q&A** — the
   operator (or moderating IR host) turn that gives Q&A instructions,
   announces the queue, or hands off to the first analyst.

The boundary offset (`qa_start_char`) is the character index of the first
character of that element, **including its speaker label** ("Operator :",
"Operator\n", ...).

Disambiguation rules, in order:

- An executive's or IR's *handoff sentence* ("With that, let's move on to
  Q&A", "The operator will take your questions now") belongs to prepared
  remarks: it closes them, it does not open the Q&A. The boundary is the
  *next* speaker's utterance (or the heading).
- Intro announcements ("Later, we will conduct a question-and-answer
  session") are prepared-remarks content, never the boundary.
- If the operator's Q&A-opening utterance contains both a thank-you to
  the last speaker and the Q&A instructions, the boundary is the start of
  the whole utterance.

## When there is no Q&A

Label `qa_start_char: null` when no Q&A **content** exists — including
calls where questions are invited but none are asked. Record the evidence
in `notes`. If a reasonable annotator could disagree (e.g., an empty Q&A
section was formally opened), set `review_status: "ambiguous"`.

## End of the Q&A span

`qa_end_char` defaults to `null` = end of transcript. Set it only when
the transcript carries **trailing non-call content** (Motley Fool
"Call participants" footers, duration lines, site boilerplate): the
offset where call content ends. Leading scraper junk needs no annotation
(it precedes the boundary and is handled by the metrics).

## Format of an annotation

```json
{
  "id": "glopardo-HLT-2022Q4",
  "qa_start_char": 14511,
  "qa_end_char": null,
  "anchor_text": "<exact transcript text at the offset, 100 chars>",
  "review_status": "unreviewed | ambiguous | human_reviewed",
  "notes": "evidence for the decision, quoted from the transcript"
}
```

`anchor_text` must be copied from the transcript at the offset — the
runner verifies it and reports a mismatch as a technical failure, so a
silently shifted transcript can never be scored against stale offsets.

## Provenance and review status

Every annotation records how it was produced. The current labels are
**model-annotated and unreviewed** (see `provenance` in
`annotations.json`) — they are provisional. A human reviewer promotes an
annotation by checking the boundary against this guide and setting
`review_status: "human_reviewed"`. Only metrics computed over
human-reviewed labels may be described as accuracy claims; until then,
benchmark numbers are informational.

## Freeze discipline

Annotations are frozen before any change to extraction behavior. If a
held-out transcript's failure influences a code change, that transcript
has become development data: disclose it and replace it in the holdout
split before quoting holdout numbers again.

"""Hermetic unit tests for the extraction-eval metric definitions.

These test the MEASUREMENT code on constructed text, so a metric bug
can't masquerade as an extractor result. No network, no benchmark cache.
"""

from evals.metrics import TOLERANCE_SENTENCES, evaluate_transcript, sentence_delta

# Prepared remarks (3 sentences), then Q&A opened by an operator sentence.
PREPARED = (
    "Welcome to the Acme earnings call and thank you all for joining today. "
    "During the call we will conduct a question-and-answer session as usual. "
    "Management will first walk through the quarterly results in detail now. "
) + ("Prepared remarks sentence with plenty of further detail follows here. " * 30)
QA_OPEN = "Question-and-Answer Session. Operator: Our first question comes from Amy Wu. "
QA_BODY = "Jane Doe: We expect supply to improve. " * 40
TEXT = PREPARED + QA_OPEN + QA_BODY
GOLD = len(PREPARED)


def ann(start, end=None, status="unreviewed"):
    return {
        "qa_start_char": start,
        "qa_end_char": end,
        "review_status": status,
        "anchor_text": TEXT[start : start + 40] if start is not None else None,
    }


def test_sentence_delta_signs_and_zero():
    assert sentence_delta(TEXT, GOLD, GOLD) == 0
    # a point three sentences earlier: positive delta means pred later
    early = TEXT.index("Management will first")
    d = sentence_delta(TEXT, GOLD, early)
    assert d > 0
    assert sentence_delta(TEXT, early, GOLD) == -d


def test_exact_boundary_found():
    r = evaluate_transcript(TEXT, ann(GOLD), id="t", split="dev", bucket="b")
    assert r.outcome == "found_vs_gold"
    # the extractor finds the in-window header at exactly the gold offset
    assert r.exact and r.within_tolerance and r.sentence_delta == 0
    assert r.precision == 1.0 and r.recall == 1.0 and r.contamination == 0.0


def test_early_cut_is_contamination_late_cut_is_recall_loss():
    # gold moved later than the match -> extractor "cuts early": contamination
    later_gold = TEXT.index("Operator: Our first question")
    r = evaluate_transcript(TEXT, ann(later_gold), id="t", split="dev", bucket="b")
    assert r.outcome == "found_vs_gold"
    assert r.char_delta < 0 and r.contamination > 0 and r.recall == 1.0
    # gold earlier than the match -> extractor "cuts late": recall loss, no contamination
    earlier_gold = TEXT.index("Management will first")
    r2 = evaluate_transcript(TEXT, ann(earlier_gold), id="t", split="dev", bucket="b")
    assert r2.char_delta > 0 and r2.contamination == 0.0 and r2.recall < 1.0
    assert not r2.within_tolerance or abs(r2.sentence_delta) <= TOLERANCE_SENTENCES


def test_gold_end_trims_tail_junk_from_precision():
    tail = " Call participants: Jane Doe; Amy Wu. Scraper footer text here." * 20
    text2 = TEXT + tail
    a = {
        "qa_start_char": GOLD,
        "qa_end_char": len(TEXT),
        "review_status": "unreviewed",
        "anchor_text": text2[GOLD : GOLD + 40],
    }
    r = evaluate_transcript(text2, a, id="t", split="dev", bucket="b")
    assert r.outcome == "found_vs_gold"
    assert r.contamination == 0.0 and r.recall == 1.0
    assert r.precision is not None and r.precision < 1.0  # tail junk counted against precision


def test_abstention_classes():
    no_qa = PREPARED  # markers only in the intro -> extractor abstains
    r = evaluate_transcript(
        no_qa,
        {
            "qa_start_char": None,
            "qa_end_char": None,
            "review_status": "unreviewed",
            "anchor_text": None,
        },
        id="t",
        split="dev",
        bucket="b",
    )
    assert r.outcome == "correct_abstain"
    r2 = evaluate_transcript(
        no_qa,
        {
            "qa_start_char": 10,
            "qa_end_char": None,
            "review_status": "unreviewed",
            "anchor_text": no_qa[10:50],
        },
        id="t",
        split="dev",
        bucket="b",
    )
    assert r2.outcome == "false_abstain"
    r3 = evaluate_transcript(
        TEXT,
        {
            "qa_start_char": None,
            "qa_end_char": None,
            "review_status": "ambiguous",
            "anchor_text": None,
        },
        id="t",
        split="dev",
        bucket="b",
    )
    assert r3.outcome == "false_find" and r3.ambiguous

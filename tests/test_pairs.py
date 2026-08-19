import json
import pathlib
import random

import pytest

ROOT = pathlib.Path(__file__).parent.parent
PAIRS = json.loads((ROOT / "docs" / "pairs.json").read_text())
RAW = json.loads((ROOT / "data" / "raw_pairs.json").read_text())


def test_schema():
    assert isinstance(PAIRS, list) and PAIRS
    ids = set()
    for p in PAIRS:
        assert set(p) == {"id", "prompt", "textA", "textB", "watermarked", "scoreW", "scoreC", "pW", "pC"}
        assert isinstance(p["id"], int) and p["id"] not in ids
        ids.add(p["id"])
        assert isinstance(p["prompt"], str) and p["prompt"]
        assert isinstance(p["textA"], str) and p["textA"]
        assert isinstance(p["textB"], str) and p["textB"]
        assert p["watermarked"] in ("A", "B")
        for k in ("scoreW", "scoreC", "pW", "pC"):
            assert isinstance(p[k], float)


def test_watermarked_scores_higher():
    for p in PAIRS:
        assert p["scoreW"] > p["scoreC"], f"pair {p['id']}"


def test_texts_match_raw_verbatim():
    raw_by_id = {r["id"]: r for r in RAW["pairs"]}
    for p in PAIRS:
        r = raw_by_id[p["id"]]
        wm, clean = (p["textA"], p["textB"]) if p["watermarked"] == "A" else (p["textB"], p["textA"])
        assert wm == r["text_watermarked"], f"pair {p['id']}"
        assert clean == r["text_clean"], f"pair {p['id']}"


def test_random_guessers_center_on_half():
    rng = random.Random(0)
    n_questions = 10
    scores = [sum(rng.random() < 0.5 for _ in range(n_questions)) for _ in range(1000)]
    mean = sum(scores) / len(scores)
    assert mean == pytest.approx(5.0, abs=0.15)

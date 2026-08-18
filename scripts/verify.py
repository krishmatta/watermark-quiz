# Label verification. Pre-registered: this script and its criteria were
# committed before any generated data was looked at, and are applied
# identically to every pair. The ONLY filtering in the experiment:
#
#   (a) the watermarked side's detection p-value is below ALPHA
#   (b) the clean side's detection p-value is at or above ALPHA
#   (c) both sides have at least MIN_TOKENS generated tokens
#
# Passing pairs ship verbatim to docs/pairs.json (left/right order
# randomized deterministically). Failing pairs go to
# data/rejected_pairs.json with reasons. No other screens, ever.

import json
import math
import pathlib
import random
import sys

import torch

sys.path.append(str(pathlib.Path(__file__).parent))
from generate_modal import MODEL, NGRAM_LEN, WATERMARK_KEYS

ALPHA = 0.01
MIN_TOKENS = 50

ROOT = pathlib.Path(__file__).parent.parent


def score(processor, special_ids, token_ids):
    ids = list(token_ids)
    while ids and ids[-1] in special_ids:
        ids.pop()
    if len(ids) < MIN_TOKENS:
        return {"tokens": len(ids), "mean_g": None, "p": None}
    t = torch.tensor([ids])
    g = processor.compute_g_values(t)[0]
    mask = processor.compute_context_repetition_mask(t)[0].bool()
    valid = g[mask].float()
    n = valid.numel()
    mean_g = valid.mean().item()
    # Under no watermark each g-value is Bernoulli(1/2); normal
    # approximation gives a one-sided p-value for "mean above 1/2".
    z = (mean_g - 0.5) * 2.0 * math.sqrt(n)
    p = 0.5 * math.erfc(z / math.sqrt(2.0))
    return {"tokens": len(ids), "mean_g": mean_g, "p": p, "z": z}


def main():
    from transformers import AutoTokenizer
    from transformers.generation import SynthIDTextWatermarkLogitsProcessor

    raw = json.loads((ROOT / "data" / "raw_pairs.json").read_text())
    assert raw["watermark"] == {"keys": WATERMARK_KEYS, "ngram_len": NGRAM_LEN}
    assert raw["model"] == MODEL

    processor = SynthIDTextWatermarkLogitsProcessor(
        keys=WATERMARK_KEYS, ngram_len=NGRAM_LEN, device="cpu"
    )
    special_ids = set(AutoTokenizer.from_pretrained(MODEL).all_special_ids)

    shipped, rejected = [], []
    for pair in raw["pairs"]:
        s_wm = score(processor, special_ids, pair["token_ids_watermarked"])
        s_clean = score(processor, special_ids, pair["token_ids_clean"])

        reasons = []
        if s_wm["tokens"] < MIN_TOKENS:
            reasons.append(f"watermarked side has {s_wm['tokens']} tokens (< {MIN_TOKENS})")
        if s_clean["tokens"] < MIN_TOKENS:
            reasons.append(f"clean side has {s_clean['tokens']} tokens (< {MIN_TOKENS})")
        if not reasons:
            if not s_wm["p"] < ALPHA:
                reasons.append(f"watermarked side not detected (p={s_wm['p']:.3g} >= {ALPHA})")
            if not s_clean["p"] >= ALPHA:
                reasons.append(f"clean side detected as watermarked (p={s_clean['p']:.3g} < {ALPHA})")

        if reasons:
            rejected.append({"id": pair["id"], "reasons": reasons,
                             "score_watermarked": s_wm, "score_clean": s_clean})
            continue

        wm_is_a = random.Random(f"watermark-quiz-{pair['id']}").random() < 0.5
        shipped.append({
            "id": pair["id"],
            "prompt": pair["prompt"],
            "textA": pair["text_watermarked"] if wm_is_a else pair["text_clean"],
            "textB": pair["text_clean"] if wm_is_a else pair["text_watermarked"],
            "watermarked": "A" if wm_is_a else "B",
            "scoreW": s_wm["mean_g"],
            "scoreC": s_clean["mean_g"],
            "pW": s_wm["p"],
            "pC": s_clean["p"],
        })

    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "pairs.json").write_text(
        json.dumps(shipped, indent=2, ensure_ascii=False))
    (ROOT / "data" / "rejected_pairs.json").write_text(
        json.dumps(rejected, indent=2, ensure_ascii=False))
    print(f"{len(shipped)} shipped, {len(rejected)} rejected")
    for r in rejected:
        print(f"  pair {r['id']}: {'; '.join(r['reasons'])}")


if __name__ == "__main__":
    main()

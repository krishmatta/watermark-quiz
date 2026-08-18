# Generates watermarked/clean text pairs on a Modal A100.
#
# For each prompt, two generations with identical sampling settings:
# one with a SynthID-Text watermark, one without. Everything is written
# verbatim to data/raw_pairs.json; no output is ever edited or filtered
# here. The watermark key is committed on purpose — the experiment is
# public and reproducibility matters more than key secrecy.

import json
import pathlib

import modal

MODEL = "Qwen/Qwen2.5-32B-Instruct"
TRANSFORMERS_VERSION = "5.15.0"

WATERMARK_KEYS = [
    913, 4408, 306, 2550, 1247, 1218, 2739, 3892, 728, 1793,
    3521, 4477, 665, 2004, 3644, 158, 2708, 3980, 1105, 2295,
    3241, 461, 1521, 2870, 3013, 3722, 4192, 819, 1391, 2011,
]
NGRAM_LEN = 5

SAMPLING = {
    "do_sample": True,
    "temperature": 0.9,
    "top_k": 64,
    # Headroom, not a target: prompts ask for ~120 words and verification
    # rejects any generation that hit this cap instead of ending naturally.
    "max_new_tokens": 400,
}

# Appended to every prompt so generations end on their own well before
# the token cap; truncated text would be unreadable in the quiz.
PROMPT_SUFFIX = " Keep it to a single paragraph of about 120 words."

# High-entropy genres only: creative/opinion writing gives the sampler
# room to embed the watermark. No math, code, or factual recall.
PROMPTS = [
    # Fiction openings
    "Write the opening paragraph of a mystery novel set in a small coastal town.",
    "Write the first paragraph of a science fiction story about a translator who works between humans and an alien species.",
    "Write the opening paragraph of a story that begins with someone missing their train.",
    "Write the first paragraph of a ghost story set in a modern apartment building.",
    "Write the opening paragraph of a heist story where the narrator is the getaway driver.",
    "Write the first paragraph of a story about two strangers stuck in an airport overnight.",
    # Opinion / blog paragraphs
    "Write a paragraph arguing that breakfast is overrated.",
    "Write a blog paragraph about why paper books beat e-readers.",
    "Write an opinion paragraph on whether tipping culture has gone too far.",
    "Write a paragraph arguing that group chats are the best social network.",
    "Write a blog paragraph about why everyone should learn to cook one great dish.",
    "Write an opinion paragraph about whether open offices help or hurt work.",
    # Product blurbs
    "Write a product description for a mechanical keyboard aimed at writers.",
    "Write a product blurb for a thermos that keeps coffee hot for 24 hours.",
    "Write a product description for a minimalist wallet.",
    "Write a product blurb for noise-cancelling headphones marketed to commuters.",
    "Write a product description for a beginner-friendly houseplant subscription box.",
    "Write a product blurb for a weighted blanket.",
    # Advice
    "Give advice to someone starting their first job out of college.",
    "Give advice on how to make friends after moving to a new city.",
    "Give advice to someone nervous about public speaking.",
    "Give advice on keeping a New Year's resolution past January.",
    "Give advice to someone learning to drive later in life than their peers.",
    "Give advice on how to run a good book club.",
    # Casual explainers
    "Explain to a curious friend why the sky changes color at sunset.",
    "Explain casually why sourdough bread needs a starter.",
    "Explain to a friend why cats knead with their paws.",
    "Explain casually how noise-cancelling headphones work.",
    "Explain to a curious kid why onions make you cry.",
    "Explain casually why some songs get stuck in your head.",
]

app = modal.App("watermark-quiz-gen")
cache = modal.Volume.from_name("watermark-quiz-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(f"transformers=={TRANSFORMERS_VERSION}", "torch", "accelerate")
    .env({"HF_HOME": "/cache"})
)


@app.function(image=image, gpu="A100-80GB", volumes={"/cache": cache}, timeout=2 * 60 * 60)
def generate() -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.generation import SynthIDTextWatermarkingConfig

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    cache.commit()

    watermarking_config = SynthIDTextWatermarkingConfig(keys=WATERMARK_KEYS, ngram_len=NGRAM_LEN)

    pairs = []
    for i, base_prompt in enumerate(PROMPTS):
        prompt = base_prompt + PROMPT_SUFFIX
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")
        prompt_len = inputs["input_ids"].shape[1]
        seed_watermarked, seed_clean = 2 * i, 2 * i + 1

        torch.manual_seed(seed_watermarked)
        out_watermarked = model.generate(**inputs, watermarking_config=watermarking_config, **SAMPLING)
        torch.manual_seed(seed_clean)
        out_clean = model.generate(**inputs, **SAMPLING)

        ids_watermarked = out_watermarked[0][prompt_len:].tolist()
        ids_clean = out_clean[0][prompt_len:].tolist()
        pairs.append({
            "id": i,
            "prompt": prompt,
            "text_watermarked": tokenizer.decode(ids_watermarked, skip_special_tokens=True),
            "text_clean": tokenizer.decode(ids_clean, skip_special_tokens=True),
            # Exact generated ids, so verification doesn't depend on a
            # decode/re-tokenize round trip.
            "token_ids_watermarked": ids_watermarked,
            "token_ids_clean": ids_clean,
            "seed_watermarked": seed_watermarked,
            "seed_clean": seed_clean,
        })
        print(f"pair {i}: {len(ids_watermarked)} wm tokens, {len(ids_clean)} clean tokens")

    return {
        "model": MODEL,
        "transformers_version": TRANSFORMERS_VERSION,
        "watermark": {"keys": WATERMARK_KEYS, "ngram_len": NGRAM_LEN},
        "sampling": SAMPLING,
        "pairs": pairs,
    }


@app.local_entrypoint()
def main():
    result = generate.remote()
    out = pathlib.Path(__file__).parent.parent / "data" / "raw_pairs.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {out} ({len(result['pairs'])} pairs)")

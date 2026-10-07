"""RouteLLM's BERT router as a baseline feature for the learned router.

    pip install torch transformers
    python bench/routellm_baseline.py                       # adds "routellm:bert" to the features file
    python bench/router_train.py --features routellm:bert   # evaluate it like any judge

RouteLLM (Ong et al., ICLR 2025; Apache-2.0) trains routers on Chatbot Arena preferences to
decide between a strong model (GPT-4) and a weak one (Mixtral 8x7B). Its BERT router outputs
the strong model's win rate for a prompt. We score each problem with the published checkpoint
and the scoring code copied verbatim from routellm/routers/routers.py (BERTRouter), and use
that win rate as a difficulty feature, exactly like a judge's rating. Its matrix-factorization
router, their recommended one, needs OpenAI embeddings, so it's out of a $0 budget.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from router_features import problems

CHECKPOINT = "routellm/bert_gpt4_augmented"  # RouteLLM's config.example.yaml


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/cache/router_features.json")
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model.eval()

    def strong_win_rate(prompt: str) -> float:  # BERTRouter.calculate_strong_win_rate, verbatim logic
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits.numpy()[0]
        exp_scores = np.exp(logits - np.max(logits))
        softmax_scores = exp_scores / np.sum(exp_scores)
        return float(1 - np.sum(softmax_scores[-2:]))

    out = Path(args.out)
    cache = json.loads(out.read_text()) if out.exists() else {}
    store = cache.setdefault("routellm:bert", {})
    todo = [(pid, t) for ps in problems().values() for pid, t in ps.items() if pid not in store]
    print(f"scoring {len(todo)} problems with {CHECKPOINT} ...", flush=True)
    for pid, text in todo:
        store[pid] = round(strong_win_rate(text), 5)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cache))
    v = np.array(list(store.values()))
    print(f"done: strong win rate mean {v.mean():.3f}, min {v.min():.3f}, max {v.max():.3f}")


if __name__ == "__main__":
    main()

"""Measure retrieval with real customer questions from the intent test set.
A question counts as answered if a section that answers its intent is retrieved."""
import time

import pandas as pd


def build_eval_set(data_csv="data/banking_en_ar.csv",
                   coverage_csv="knowledge_base/intent_coverage.csv",
                   per_intent=4, seed=42, split="val") -> pd.DataFrame:
    """split="val" while we improve things; split="test" once, at the very end."""
    data = pd.read_csv(data_csv)
    gold = pd.read_csv(coverage_csv).set_index("intent")["sections"].str.split("|")
    pool = data[data["split"] == split]
    sample = pool.sample(frac=1, random_state=seed).groupby(["intent", "lang"]).head(per_intent)
    sample = sample[["text", "lang", "intent"]].reset_index(drop=True)
    sample["gold"] = sample["intent"].map(lambda i: set(gold[i]))
    return sample


def evaluate(retriever, queries: pd.DataFrame, mode: str, k_values=(1, 3, 5), depth=10) -> dict:
    hits = {k: 0 for k in k_values}
    reciprocal_ranks = []
    start = time.perf_counter()
    for text, gold in zip(queries["text"], queries["gold"]):
        found = [c.section_id for c in retriever.search(text, top_k=depth, mode=mode)]
        rank = next((i + 1 for i, s in enumerate(found) if s in gold), None)
        for k in k_values:
            hits[k] += rank is not None and rank <= k
        reciprocal_ranks.append(1 / rank if rank else 0)
    n = len(queries)
    row = {"mode": mode, **{f"hit@{k}": round(hits[k] / n, 3) for k in k_values},
           "mrr@10": round(sum(reciprocal_ranks) / n, 3),
           "ms_per_query": round(1000 * (time.perf_counter() - start) / n, 1)}
    return row

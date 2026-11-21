# %% [markdown]
# # 06 · RAG part 1: finding the right document section (retrieval)
#
# **RAG = Retrieval-Augmented Generation.** Before the LLM answers, we *retrieve*
# the bank's text that contains the answer and give it to the LLM. The LLM then
# answers from that text instead of from memory, so it can't invent fees or rules,
# and it can cite its source.
#
# This notebook builds and measures the **retrieval** half. Generation comes next.
#
# Pipeline:
# 1. **Chunking**: cut documents into small pieces (here: one section = one chunk).
# 2. **BM25**: keyword search (classic search-engine scoring).
# 3. **Dense search**: meaning search with an embedding model (BGE-M3).
# 4. **Hybrid**: combine both with Reciprocal Rank Fusion (RRF).
# 5. **Reranking**: a cross-encoder re-reads the top results and reorders them.
#
# We measure each step on real customer questions from the intent validation set.

# %%
import sys
sys.path.append("..")

import pandas as pd

from src.rag.knowledge_base import load_chunks
from src.rag.retriever import Retriever
from src.rag.evaluation import build_eval_set, evaluate

KB_DIR = "../knowledge_base"
chunks = load_chunks(KB_DIR)
print(len(chunks), "chunks")
pd.Series([c.lang for c in chunks]).value_counts()

# %% [markdown]
# ## 1. Chunking
# Our documents already have one topic per section (`## 5. Activating your card`),
# so each section is one chunk (about 50 to 150 words). Small, single-topic chunks
# make search precise; chunks that are too big mix topics, and chunks that are too
# small lose context. We also add the document and section titles to each chunk's
# text, so a chunk still makes sense on its own.
#
# `section_id` (like `cards#5`) is the same in English and Arabic, so citations line up.

# %%
example = chunks[4]
print(example.chunk_id, "|", example.citation)
print(example.text)

# %% [markdown]
# ## 2. BM25 (keyword search)
# BM25 scores a chunk higher when it contains the query's words, especially rare
# words ("CVV", "SWIFT", "IBAN"). Fast and exact, but it fails when the customer
# uses different words for the same idea ("can't find my card" vs "lost").

# %%
retriever = Retriever(chunks)   # BM25 only for now

for q in ["my card still has not arrived",
          "I can't find my card anywhere",
          "كم رسوم السحب من الصراف؟"]:
    print(q, "->", [c.chunk_id for c in retriever.search(q, top_k=3, mode="bm25")])

# %% [markdown]
# ## 3. Dense search (embeddings)
# An **embedding model** turns a text into a list of numbers (a vector) so that
# texts with similar *meaning* get similar vectors. We use **BGE-M3**: multilingual
# (100+ languages, including Arabic), strong on retrieval benchmarks, free.
# Similarity between two vectors = **cosine similarity** (1 = same meaning direction).
#
# The first run downloads the model (about 2.3 GB) into your user folder; later
# runs load it from disk.

# %%
from src.rag.models import Embedder

embedder = Embedder("BAAI/bge-m3")
retriever = Retriever(chunks, embedder=embedder)

for q in ["I can't find my card anywhere",
          "someone used my card details online",
          "ما المدة التي يستغرقها وصول البطاقة؟"]:
    print(q)
    print("  bm25 :", [c.chunk_id for c in retriever.search(q, top_k=3, mode="bm25")])
    print("  dense:", [c.chunk_id for c in retriever.search(q, top_k=3, mode="dense")])

# %% [markdown]
# ## 4. Measure: which search method finds the answer?
# Questions: 4 per intent per language from the intent **validation** set
# (about 600 real customer questions).
# Same rule as in part 1: we improve the system using validation questions, and use
# the test questions only once at the end. Otherwise we would tune the documents
# to the exam questions and our score would be too optimistic.
# `knowledge_base/intent_coverage.csv` says which section(s) answer each intent.
#
# Metrics:
# - **hit@k**: share of questions where a correct section is in the top k results.
#   The LLM will see the top 3 to 5, so hit@3 and hit@5 matter most.
# - **MRR@10** (mean reciprocal rank): 1 if the correct section is first, 1/2 if
#   second, 1/3 if third, and so on, averaged. Rewards putting it near the top.
# - **ms_per_query**: speed.
#
# **Hybrid = Reciprocal Rank Fusion (RRF):** each method votes for each chunk with
# 1 / (60 + its position). Chunks that BOTH methods rank high win.

# %%
queries = build_eval_set("../data/banking_en_ar.csv", f"{KB_DIR}/intent_coverage.csv", per_intent=4, split="val")
print(len(queries), "validation questions")

rows = [evaluate(retriever, queries, mode) for mode in ["bm25", "dense", "hybrid"]]
pd.DataFrame(rows)

# %% [markdown]
# ## 4b. Weighted hybrid
# Equal-weight hybrid was worse than dense alone, because BM25 is weak on our data.
# Here BM25's vote counts less (weight 0 = dense only, weight 1 = equal vote).
# We pick the weight on VALIDATION questions.

# %%
weight_rows = []
for w in [0.0, 0.1, 0.25, 0.5, 1.0]:
    retriever.bm25_weight = w
    row = evaluate(retriever, queries, "hybrid")
    row["bm25_weight"] = w
    weight_rows.append(row)

best_weight = max(weight_rows, key=lambda r: (r["hit@3"], r["mrr@10"]))["bm25_weight"]
retriever.bm25_weight = best_weight          # keep the best weight for the next steps
print("Best BM25 weight on validation:", best_weight)
pd.DataFrame(weight_rows)

# %% [markdown]
# ## 5. Reranking (cross-encoder)
# Embeddings encode the question and the chunk **separately**, then compare.
# A **cross-encoder reranker** reads the question and the chunk **together**,
# so it understands how well this chunk answers this question. Much more accurate,
# but much slower, so we only use it to reorder the top results.
# Model: **BGE-reranker-v2-m3** (multilingual, same family as our embedder).
#
# Another ~2.3 GB download the first time. The evaluation takes several minutes on CPU.

# %%
from src.rag.models import Reranker

retriever.reranker = Reranker("BAAI/bge-reranker-v2-m3")
rows.append(evaluate(retriever, queries, "hybrid+rerank"))
results = pd.DataFrame(rows)
results.to_csv("../results/rag_retrieval.csv", index=False)
results

# %% [markdown]
# ## 6. Look at the mistakes
# Which questions does the best method still miss? Read them: is the question
# unclear, is the label wrong, or is our document missing information?

# %%
best_mode = results.sort_values("hit@3", ascending=False).iloc[0]["mode"]
print("Best mode:", best_mode)

misses = []
for text, lang, intent, gold in queries.itertuples(index=False):
    found = [c.section_id for c in retriever.search(text, top_k=3, mode=best_mode)]
    if not set(found) & gold:
        misses.append({"question": text, "intent": intent, "expected": "|".join(gold), "got": found})
misses = pd.DataFrame(misses)
print(len(misses), "questions without a correct section in the top 3")
misses.head(20)
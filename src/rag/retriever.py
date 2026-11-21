"""Hybrid search: BM25 (exact words) + dense embeddings (meaning), fused with RRF,
then (optionally) reranked by a cross-encoder."""
import re

import numpy as np
from rank_bm25 import BM25Okapi

from src.rag.knowledge_base import Chunk, detect_lang
from src.text_utils import normalize_arabic


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", normalize_arabic(text.lower()))


class BM25Search:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def rank(self, query: str) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return list(np.argsort(-scores))


class DenseSearch:
    """`embedder` is anything with .encode(list_of_texts) -> normalized vectors."""
    def __init__(self, chunks: list[Chunk], embedder):
        self.chunks = chunks
        self.embedder = embedder
        self.vectors = np.asarray(embedder.encode([c.text for c in chunks]))

    def rank(self, query: str) -> list[int]:
        q = np.asarray(self.embedder.encode([query]))[0]
        return list(np.argsort(-(self.vectors @ q)))


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60, weights=None) -> list[int]:
    """Each list votes weight/(k + position). Items ranked high by both lists win."""
    weights = weights or [1.0] * len(rankings)
    scores = {}
    for ranking, weight in zip(rankings, weights):
        for position, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + weight / (k + position + 1)
    return sorted(scores, key=scores.get, reverse=True)


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder=None, reranker=None):
        self.chunks = chunks
        self.bm25 = BM25Search(chunks)
        self.dense = DenseSearch(chunks, embedder) if embedder is not None else None
        self.reranker = reranker
        self.bm25_weight = 1.0   # how much BM25's vote counts in hybrid (dense counts 1.0)

    def search(self, query: str, top_k: int = 5, mode: str = "hybrid",
               lang: str | None = "auto", rerank_pool: int = 20) -> list[Chunk]:
        """mode: 'bm25', 'dense', 'hybrid', or 'hybrid+rerank'.
        lang: 'auto' = only search documents in the query's language; None = all."""
        if lang == "auto":
            lang = detect_lang(query)
        if mode == "bm25":
            ranking = self.bm25.rank(query)
        elif mode == "dense":
            ranking = self.dense.rank(query)
        else:
            ranking = reciprocal_rank_fusion([self.bm25.rank(query), self.dense.rank(query)],
                                              weights=[self.bm25_weight, 1.0])
        ranking = [i for i in ranking if lang is None or self.chunks[i].lang == lang]

        if mode == "hybrid+rerank":
            pool = ranking[:rerank_pool]
            scores = self.reranker.predict([(query, self.chunks[i].text) for i in pool])
            ranking = [pool[j] for j in np.argsort(-np.asarray(scores))]
        return [self.chunks[i] for i in ranking[:top_k]]

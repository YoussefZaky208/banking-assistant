"""Wrappers around the two downloaded models, so the rest of the code doesn't
depend on a specific library (we could swap them later)."""


class Embedder:
    """Turns texts into vectors that capture meaning (BGE-M3: 100+ languages)."""
    def __init__(self, name: str = "BAAI/bge-m3"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name)
        self._cache = {}

    def encode(self, texts: list[str]):
        new = [t for t in texts if t not in self._cache]
        if new:
            vectors = self.model.encode(new, normalize_embeddings=True, batch_size=16)
            self._cache.update(zip(new, vectors))
        return [self._cache[t] for t in texts]


class Reranker:
    """Reads (question, passage) TOGETHER and scores how well the passage answers it."""
    def __init__(self, name: str = "BAAI/bge-reranker-v2-m3"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(name, max_length=512)

    def predict(self, pairs):
        return self.model.predict(pairs, batch_size=16)

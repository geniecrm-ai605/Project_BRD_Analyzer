"""Text -> vector.

Default  : BAAI/bge-small-en-v1.5 (pretrained, frozen, ~130MB, CPU is fine).
Fallback : TF-IDF + SVD (set EMBED_BACKEND=tfidf) - only for offline testing, weaker than bge.
"""
import os
import numpy as np

BGE_NAME = "BAAI/bge-small-en-v1.5"


class BgeEmbedder:
    kind = "bge"

    def __init__(self, name=BGE_NAME):
        self.name = name
        self._model = None

    def fit(self, texts):
        return self  # frozen pretrained model

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.name, device="cpu")
        return self._model

    def transform(self, texts):
        return np.asarray(self._load().encode(list(texts), normalize_embeddings=True, batch_size=16))

    def __getstate__(self):  # don't pickle the 130MB model, reload by name
        d = self.__dict__.copy()
        d["_model"] = None
        return d


class TfidfEmbedder:
    kind = "tfidf"

    def __init__(self, dims=64):
        self.dims = dims

    def fit(self, texts):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        X = self.vec.fit_transform(texts)
        self.svd = TruncatedSVD(min(self.dims, X.shape[1] - 1), random_state=0).fit(X)
        return self

    def transform(self, texts):
        Z = self.svd.transform(self.vec.transform(texts))
        return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


def make_embedder():
    return TfidfEmbedder() if os.getenv("EMBED_BACKEND", "bge").lower() == "tfidf" else BgeEmbedder()

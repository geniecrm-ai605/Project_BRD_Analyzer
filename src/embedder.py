import os
import numpy as np

BGE_NAME = "BAAI/bge-small-en-v1.5"


def _model_dir():
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "bge_model")
    )


class BgeEmbedder:
    kind = "bge"

    def __init__(self, name=BGE_NAME):
        self.name = name
        self._model = None

    def fit(self, texts):
        return self

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(
                model_name=self.name,
                specific_model_path=_model_dir()
            )
        return self._model

    def transform(self, texts):
        vectors = list(
            self._load().embed(list(texts), batch_size=16)
        )

        embeddings = np.asarray(
            vectors, dtype=np.float32
        )

        # Ensure unit-normalized vectors.
        norms = np.linalg.norm(
            embeddings, axis=1, keepdims=True
        )
        return embeddings / np.maximum(norms, 1e-12)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        return state


class TfidfEmbedder:
    kind = "tfidf"

    def __init__(self, dims=64):
        self.dims = dims

    def fit(self, texts):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )
        X = self.vec.fit_transform(texts)
        self.svd = TruncatedSVD(
            min(self.dims, X.shape[1] - 1),
            random_state=0
        ).fit(X)
        return self

    def transform(self, texts):
        Z = self.svd.transform(self.vec.transform(texts))
        return Z / (
            np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
        )


def make_embedder():
    return (
        TfidfEmbedder()
        if os.getenv("EMBED_BACKEND", "bge").lower() == "tfidf"
        else BgeEmbedder()
    )

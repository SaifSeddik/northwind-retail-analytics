"""Simple TF-IDF retriever over local docs/ directory. Produces chunk ids like filename::chunk0."""
import os
from typing import List, Tuple, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Retriever:
    def __init__(self, docs_path: str = None):
        self.docs_path = docs_path or os.path.join(os.getcwd(), "docs")
        self.chunks: List[Dict] = []  # each: {id, content, source}
        self._vectorizer = None
        self._matrix = None
        self._build_index()

    def _load_docs(self) -> List[Tuple[str, str]]:
        pairs = []
        if not os.path.exists(self.docs_path):
            return pairs
        for fn in sorted(os.listdir(self.docs_path)):
            path = os.path.join(self.docs_path, fn)
            if not os.path.isfile(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            # For now, single chunk per file
            chunk_id = f"{os.path.splitext(fn)[0]}::chunk0"
            pairs.append((chunk_id, content))
        return pairs

    def _build_index(self):
        pairs = self._load_docs()
        self.chunks = [
            {"id": cid, "content": text, "source": cid.split("::")[0]} for cid, text in pairs
        ]
        docs = [c["content"] for c in self.chunks]
        if docs:
            self._vectorizer = TfidfVectorizer(stop_words='english')
            self._matrix = self._vectorizer.fit_transform(docs)
        else:
            self._vectorizer = None
            self._matrix = None

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[Dict, float]]:
        if self._matrix is None:
            return []
        qv = self._vectorizer.transform([query])
        scores = (self._matrix @ qv.T).toarray().ravel()
        idx = np.argsort(scores)[::-1][:k]
        results = []
        for i in idx:
            if scores[i] <= 0:
                continue
            results.append((self.chunks[i], float(scores[i])))
        return results

"""Dense retrieval with bge-m3. Optional: needs ``sentence-transformers``.

Embeddings are cached to a .npy file next to the index so the corpus is only
encoded once.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple


class DenseIndex:
    def __init__(self, ids: Sequence[str], texts: Sequence[str], model: str = "BAAI/bge-m3",
                 cache: Optional[str | Path] = None, batch_size: int = 32, device: Optional[str] = None) -> None:
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("dense retrieval needs `pip install sentence-transformers`") from e
        self._np = np
        self.ids = list(ids)
        self.encoder = SentenceTransformer(model, device=device)
        path = Path(cache) if cache else None
        if path is not None and path.exists():
            self.emb = np.load(path)
        else:
            self.emb = self.encoder.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                                           show_progress_bar=True).astype("float32")
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, self.emb)

    def search_many(self, queries: Sequence[str], k: int = 50) -> List[List[Tuple[str, float]]]:
        q = self.encoder.encode(list(queries), normalize_embeddings=True).astype("float32")
        sims = q @ self.emb.T
        out = []
        for row in sims:
            top = self._np.argsort(-row)[:k]
            out.append([(self.ids[i], float(row[i])) for i in top])
        return out

    def search(self, query: str, k: int = 50) -> List[Tuple[str, float]]:
        return self.search_many([query], k)[0]

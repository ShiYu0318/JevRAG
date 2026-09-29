"""bge-reranker-v2-m3 as a cross-encoder baseline. Needs sentence-transformers."""
from __future__ import annotations

import math
from typing import List, Optional, Sequence


class CrossEncoderReranker:
    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3", device: Optional[str] = None,
                 max_length: int = 512, batch_size: int = 16) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("the reranker baseline needs `pip install sentence-transformers`") from e
        self.model = CrossEncoder(model, device=device, max_length=max_length)
        self.batch_size = batch_size

    def score(self, query: str, passages: Sequence[str]) -> List[float]:
        """Raw logits; larger is more relevant."""
        import torch

        # activation_fn=None would fall back to the model default (sigmoid), so pass identity explicitly
        out = self.model.predict([(query, p) for p in passages], batch_size=self.batch_size,
                                 activation_fn=torch.nn.Identity(), show_progress_bar=False)
        return [float(x) for x in out]

    def prob(self, query: str, passages: Sequence[str]) -> List[float]:
        return [1 / (1 + math.exp(-s)) for s in self.score(query, passages)]

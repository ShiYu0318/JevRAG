"""Multilingual NLI as an evidence-sufficiency baseline. Needs transformers.

Each passage is a premise; the hypothesis states the question together with a
candidate answer. Passage-level entailment / contradiction scores are then
aggregated into a set-level verdict (after SURE-RAG).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"


def hypothesis(question: str, answer: str) -> str:
    return f"{question.rstrip('？?')}？答案是{answer}。"


class NLIScorer:
    def __init__(self, model: str = DEFAULT_MODEL, device: Optional[str] = None, max_length: int = 512,
                 batch_size: int = 16) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("the NLI baseline needs `pip install transformers torch`") from e
        self._torch = torch
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(device).eval()
        self.labels = {i: l.lower() for i, l in self.model.config.id2label.items()}
        self.max_length = max_length
        self.batch_size = batch_size

    def score(self, pairs: Sequence[Tuple[str, str]]) -> List[Dict[str, float]]:
        """(premise, hypothesis) -> {entailment, neutral, contradiction} probabilities."""
        out: List[Dict[str, float]] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            enc = self.tok([p for p, _ in batch], [h for _, h in batch], truncation="only_first",
                           max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
            with self._torch.no_grad():
                probs = self._torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu().tolist()
            out += [{self.labels[j]: p for j, p in enumerate(row)} for row in probs]
        return out


def verdict(entail: Sequence[float], contra: Sequence[float], sub_entail: Sequence[float],
            tau_e: float, tau_c: float) -> str:
    """Set-level verdict from passage scores.

    entail / contra:  per passage, for the full answer (single questions)
    sub_entail:       best entailment per sub-answer for conjunction questions, empty otherwise
    """
    if sub_entail:
        covered = [e >= tau_e for e in sub_entail]
        if all(covered):
            return "sufficient"
        return "partial" if any(covered) else "insufficient"
    if max(entail, default=0.0) >= tau_e:
        return "conflicting" if max(contra, default=0.0) >= tau_c else "sufficient"
    return "insufficient"

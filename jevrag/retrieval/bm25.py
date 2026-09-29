"""BM25 over character bigrams (plus unigrams), suited to Chinese without a
word segmenter."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

from ..data.normalize import normalize


def char_ngrams(text: str, unigrams: bool = True) -> List[str]:
    s = normalize(text)
    grams = [s[i:i + 2] for i in range(len(s) - 1)]
    if unigrams:
        grams += list(s)
    return grams


class BM25:
    def __init__(self, ids: Sequence[str], texts: Sequence[str], k1: float = 1.2, b: float = 0.75,
                 unigrams: bool = True) -> None:
        self.ids = list(ids)
        self.k1, self.b, self.unigrams = k1, b, unigrams
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.doc_len: List[int] = []
        for i, t in enumerate(texts):
            tf = Counter(char_ngrams(t, unigrams))
            self.doc_len.append(sum(tf.values()))
            for g, c in tf.items():
                self.postings[g].append((i, c))
        n = len(self.ids)
        self.avgdl = sum(self.doc_len) / max(n, 1)
        self.idf = {g: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for g, p in self.postings.items()}

    def scores(self, query: str) -> Dict[int, float]:
        acc: Dict[int, float] = defaultdict(float)
        k1, b, avgdl = self.k1, self.b, self.avgdl
        for g, qtf in Counter(char_ngrams(query, self.unigrams)).items():
            idf = self.idf.get(g)
            if idf is None:
                continue
            for i, tf in self.postings[g]:
                denom = tf + k1 * (1 - b + b * self.doc_len[i] / avgdl)
                acc[i] += qtf * idf * tf * (k1 + 1) / denom
        return acc

    def search(self, query: str, k: int = 50) -> List[Tuple[str, float]]:
        s = self.scores(query)
        top = sorted(s.items(), key=lambda x: -x[1])[:k]
        return [(self.ids[i], v) for i, v in top]

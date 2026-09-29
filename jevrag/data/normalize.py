"""Text normalisation for matching: NFKC (full-width to half-width), lower case,
no whitespace, and Traditional/Simplified folded to one script when OpenCC is
installed."""
from __future__ import annotations

import functools
import re
import unicodedata
from typing import Callable, Optional

_WS = re.compile(r"\s+")


@functools.lru_cache(maxsize=None)
def _opencc(config: str) -> Optional[Callable[[str], str]]:
    try:
        import opencc
    except ImportError:
        return None
    return opencc.OpenCC(config).convert


def has_opencc() -> bool:
    return _opencc("t2s") is not None


def convert(text: str, config: str) -> str:
    """OpenCC conversion, e.g. ``tw2s``. Raises when OpenCC is missing."""
    fn = _opencc(config)
    if fn is None:
        raise ImportError("OpenCC is required: pip install opencc")
    return fn(text)


def normalize(text: str, fold_script: bool = True) -> str:
    s = unicodedata.normalize("NFKC", text).lower()
    s = _WS.sub("", s)
    if fold_script:
        fn = _opencc("t2s")
        if fn is not None:
            s = fn(s)
    return s


def contains(haystack: str, needle: str) -> bool:
    n = normalize(needle)
    return bool(n) and n in normalize(haystack)

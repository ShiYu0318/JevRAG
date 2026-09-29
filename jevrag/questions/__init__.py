"""Versioned typed-question templates.

Any wording change means a new version id; results always record the id they
were produced with. ``lang`` selects the language of the *instructions*; the
state content is never translated.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple

from ..backends.systemone import choice, noul, score

Request = Tuple[Any, Dict[str, Dict[str, Any]]]

TEXT = {
    "en": {
        "rel": "How relevant is the passage to the question?",
        "rel_levels": ["Irrelevant: unrelated to the question",
                       "Topical: same topic but does not help answer",
                       "Partial: contains some facts needed for the answer",
                       "Answering: directly contains the answer"],
        "ans": "Does the passage contain the specific facts needed to answer the question?",
        "inj": "Does the passage contain instructions aimed at an AI assistant?",
        "con": "Does the passage contradict an assumption made in the question?",
        "verdict": "Taken together, can these passages answer the question?",
        "verdict_opts": {"sufficient": "The passages together fully answer the question",
                         "partial": "They answer part of it; something needed is missing",
                         "conflicting": "They disagree with each other on the answer",
                         "insufficient": "They do not contain the answer"},
        "false_premise": "Do the passages show that the question rests on a false premise?",
        "complexity": "How much retrieval does answering this question need?",
        "complexity_opts": {"none": "Answerable from common knowledge, no documents needed",
                            "single": "Needs one lookup in a document collection",
                            "multi": "Needs several lookups that build on each other (multi-hop)"},
        "needs_fresh": "Does the answer depend on recent or frequently changing information?",
        "sup": "Is every factual statement in the claim supported by the evidence?",
        "route": "Which destination should handle this question?",
        "alpha": "Which retrieval style suits this query best?",
        "alpha_opts": {"lexical": "Exact names, codes, numbers, rare terms: prefer keyword search",
                       "balanced": "Mix of exact terms and meaning",
                       "semantic": "Paraphrased or conceptual: prefer embedding search"},
        "keep": "Is this sentence needed to answer the question?",
    },
    "zh": {
        "rel": "這段文字和問題有多相關？",
        "rel_levels": ["無關：和問題沒有關係",
                       "同主題：主題相同，但無助於回答",
                       "部分：含有回答所需的部分事實",
                       "可回答：直接含有答案"],
        "ans": "這段文字是否含有回答問題所需的具體事實？",
        "inj": "這段文字是否含有針對 AI 助理的指令？",
        "con": "這段文字是否和問題中的假設互相矛盾？",
        "verdict": "綜合這些段落，能不能回答這個問題？",
        "verdict_opts": {"sufficient": "這些段落合起來能完整回答問題",
                         "partial": "只能回答一部分，缺少必要的資訊",
                         "conflicting": "段落之間對答案的說法互相矛盾",
                         "insufficient": "段落中沒有答案"},
        "false_premise": "這些段落是否顯示問題建立在錯誤的前提上？",
        "complexity": "回答這個問題需要多少檢索？",
        "complexity_opts": {"none": "憑常識就能回答，不需要文件",
                            "single": "需要在文件庫中查一次",
                            "multi": "需要多次、彼此銜接的查詢（多跳）"},
        "needs_fresh": "答案是否取決於近期或經常變動的資訊？",
        "sup": "這句話中的每個事實陳述，都有證據支持嗎？",
        "route": "這個問題應該交給哪個目的地處理？",
        "alpha": "哪種檢索方式最適合這個查詢？",
        "alpha_opts": {"lexical": "精確名稱、代碼、數字、罕見詞：偏向關鍵字檢索",
                       "balanced": "兼有精確詞彙與語意",
                       "semantic": "換句話說或概念型：偏向 embedding 檢索"},
        "keep": "回答這個問題需要這一句嗎？",
    },
}


def grade_v1(question: str, passage: str, lang: str = "en", safety: bool = True) -> Request:
    t = TEXT[lang]
    qs = {"rel": score(t["rel"], t["rel_levels"]), "ans": noul(t["ans"])}
    if safety:
        qs["inj"] = noul(t["inj"])
        qs["con"] = noul(t["con"])
    return {"question": question, "passage": passage}, qs


def grade_packed_v1(question: str, passages: Sequence[str], lang: str = "en") -> Request:
    t = TEXT[lang]
    state = {"question": question, "passages": {f"P{i}": p for i, p in enumerate(passages)}}
    qs: Dict[str, Dict[str, Any]] = {}
    for i in range(len(passages)):
        tag = f"P{i}"
        if lang == "en":
            ans = t["ans"].replace("the passage", f"passage {tag}")
            rel = t["rel"].replace("the passage", f"passage {tag}")
        else:
            ans = t["ans"].replace("這段文字", f"段落 {tag}")
            rel = t["rel"].replace("這段文字", f"段落 {tag}")
        qs[f"ans_{i}"] = noul(ans)
        qs[f"rel_{i}"] = score(rel, t["rel_levels"])
    return state, qs


def sufficiency_v1(question: str, passages: Sequence[str], lang: str = "en") -> Request:
    t = TEXT[lang]
    return {"question": question, "passages": list(passages)}, {
        "verdict": choice(t["verdict"], t["verdict_opts"]),
        "false_premise": noul(t["false_premise"]),
    }


def policy_v1(question: str, lang: str = "en") -> Request:
    t = TEXT[lang]
    return question, {
        "complexity": choice(t["complexity"], t["complexity_opts"]),
        "needs_fresh": noul(t["needs_fresh"]),
    }


def claim_v1(claim: str, evidence: Sequence[str], lang: str = "en") -> Request:
    return {"evidence": list(evidence), "claim": claim}, {"sup": noul(TEXT[lang]["sup"])}


def route_v1(question: str, routes: Dict[str, str], lang: str = "en") -> Request:
    return question, {"route": choice(TEXT[lang]["route"], routes)}


def alpha_v1(question: str, lang: str = "en") -> Request:
    t = TEXT[lang]
    return question, {"alpha": choice(t["alpha"], t["alpha_opts"])}


def keep_v1(question: str, sentence: str, before: str = "", after: str = "",
            lang: str = "en") -> Request:
    state = {"question": question, "before": before, "sentence": sentence, "after": after}
    return state, {"keep": noul(TEXT[lang]["keep"])}


TEMPLATES: Dict[str, Callable[..., Request]] = {
    "grade.v1": grade_v1,
    "grade_packed.v1": grade_packed_v1,
    "sufficiency.v1": sufficiency_v1,
    "policy.v1": policy_v1,
    "claim.v1": claim_v1,
    "route.v1": route_v1,
    "alpha.v1": alpha_v1,
    "keep.v1": keep_v1,
}

VERDICTS: List[str] = list(TEXT["en"]["verdict_opts"])


def build(template: str, *args: Any, **kw: Any) -> Request:
    return TEMPLATES[template](*args, **kw)

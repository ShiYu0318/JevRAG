"""Routing, retrieval policy and per-query hybrid weight."""
from __future__ import annotations

from typing import Dict, Mapping

from ..backends.systemone import Response, SystemOneClient
from ..questions import alpha_v1, policy_v1, route_v1

ALPHA_MAP = {"lexical": 0.2, "balanced": 0.5, "semantic": 0.8}  # weight on dense


def route_query(client: SystemOneClient, query: str, routes: Dict[str, str]) -> Response:
    """Keep ``routes`` under ~20 options for Laya; Jev accepts up to 255."""
    return client.ask(*route_v1(query, routes))


def retrieval_policy(client: SystemOneClient, query: str) -> Response:
    return client.ask(*policy_v1(query))


def expected_alpha(client: SystemOneClient, query: str) -> float:
    probs = client.ask(*alpha_v1(query))["alpha"].probs
    return sum(p * ALPHA_MAP.get(k, 0.5) for k, p in probs.items())


def min_expected_cost(probs: Mapping[str, float], cost: Mapping[str, Mapping[str, float]]) -> str:
    """argmin_r sum_t p(t) * cost[r][t]; cost[r][t] is the price of routing to r when t is right."""
    return min(cost, key=lambda r: sum(p * cost[r].get(t, 0.0) for t, p in probs.items()))

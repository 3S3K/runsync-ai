"""후보에 점수를 매겨 서로 다른 코스 상위 N개를 고른다. 점수는 낮을수록 좋다."""
from typing import List

from app import config
from app.routing import Candidate


def distance_error(candidate: Candidate, target_m: float) -> float:
    return abs(candidate.distance_m - target_m) / target_m


def intensity_error(candidate: Candidate, intensity: str) -> float:
    """예상 운동 강도가 목표 구간의 중앙에서 벗어난 정도(%p)."""
    low, high = config.INTENSITY_ZONE_PCT[intensity]
    return abs(candidate.predicted_hr_pct - (low + high) / 2)


def score(candidate: Candidate, target_m: float, intensity: str) -> float:
    return (config.W_DISTANCE * distance_error(candidate, target_m)
            + config.W_OVERLAP * candidate.overlap_ratio
            + config.W_INTENSITY * intensity_error(candidate, intensity))


def similarity(a: Candidate, b: Candidate) -> float:
    """두 코스가 공유하는 구간의 비율 (짧은 쪽 기준)."""
    edges_a, edges_b = set(a.edge_ids), set(b.edge_ids)
    return len(edges_a & edges_b) / min(len(edges_a), len(edges_b))


def select_top(candidates: List[Candidate], target_m: float, intensity: str) -> List[Candidate]:
    usable = [c for c in candidates if distance_error(c, target_m) <= config.MAX_DISTANCE_ERROR]
    selected: List[Candidate] = []
    for candidate in sorted(usable, key=lambda c: score(c, target_m, intensity)):
        if all(similarity(candidate, chosen) <= config.MAX_SIMILARITY for chosen in selected):
            selected.append(candidate)
        if len(selected) == config.TOP_N:
            break
    return selected

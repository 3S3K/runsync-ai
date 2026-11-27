"""후보에 점수를 매겨 서로 다른 코스 상위 N개를 고른다. 점수는 낮을수록 좋다."""
from typing import List

from app import config
from app.routing import Candidate


def distance_error(candidate: Candidate, target_m: float) -> float:
    return abs(candidate.distance_m - target_m) / target_m


def zone_position(predicted_hr_pct: float, intensity: str) -> int:
    """예상 강도가 목표 구간보다 낮으면 -1, 구간 안이면 0, 높으면 1.

    화면에 보이는 값(정수로 반올림)으로 판정한다. 70.4%는 70%로 표시되므로 60~70% 구간 안이다.
    """
    low, high = config.INTENSITY_ZONE_PCT[intensity]
    shown = round(predicted_hr_pct)
    if shown < low:
        return -1
    if shown > high:
        return 1
    return 0


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
    # 요청한 강도 구간에 드는 코스를 먼저, 그 안에서는 점수가 낮은 순으로 고른다.
    # 가중치만으로는 구간 밖 코스가 거리 오차가 작다는 이유로 앞설 수 있다.
    def sort_key(candidate: Candidate):
        in_zone = zone_position(candidate.predicted_hr_pct, intensity) == 0
        return (not in_zone, score(candidate, target_m, intensity))

    for candidate in sorted(usable, key=sort_key):
        if all(similarity(candidate, chosen) <= config.MAX_SIMILARITY for chosen in selected):
            selected.append(candidate)
        if len(selected) == config.TOP_N:
            break
    return selected

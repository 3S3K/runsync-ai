import pytest

from app import config, scorer
from app.routing import Candidate


def candidate(distance_m, overlap=0.0, edges=(1, 2, 3), ascent_m=0.0):
    return Candidate(bearing_deg=0, distance_m=distance_m, ascent_m=ascent_m, max_slope_pct=0,
                     overlap_ratio=overlap, edge_ids=list(edges), coordinates=[])


def test_score_combines_distance_error_and_overlap():
    c = candidate(5500, overlap=0.2)
    assert scorer.score(c, 5000, "EASY") == pytest.approx(config.W_DISTANCE * 0.1 + config.W_OVERLAP * 0.2)


def test_score_is_zero_for_exact_distance_without_overlap_on_flat_easy_course():
    assert scorer.score(candidate(5000), 5000, "EASY") == 0


def test_climb_error_is_gap_from_intensity_target():
    c = candidate(5000, ascent_m=100)  # 1km당 20m
    assert scorer.climb_error(c, "EASY") == pytest.approx(20 - config.TARGET_CLIMB_M_PER_KM["EASY"])
    assert scorer.climb_error(c, "HARD") == pytest.approx(abs(20 - config.TARGET_CLIMB_M_PER_KM["HARD"]))


def test_score_adds_weighted_climb_error():
    c = candidate(5000, ascent_m=100)
    assert scorer.score(c, 5000, "NORMAL") == pytest.approx(config.W_CLIMB * scorer.climb_error(c, "NORMAL"))


def test_easy_prefers_flatter_course_and_hard_prefers_hillier():
    flat = candidate(5000, edges=(1, 2), ascent_m=30)     # 1km당 6m
    hilly = candidate(5000, edges=(3, 4), ascent_m=150)   # 1km당 30m
    assert scorer.select_top([hilly, flat], 5000, "EASY") == [flat, hilly]
    assert scorer.select_top([flat, hilly], 5000, "HARD") == [hilly, flat]


def test_similarity_uses_shorter_course_as_base():
    a = candidate(5000, edges=range(1, 11))   # 1~10
    b = candidate(5000, edges=range(6, 26))   # 6~25, 겹치는 구간 5개
    assert scorer.similarity(a, b) == pytest.approx(0.5)


def test_select_top_orders_by_score():
    best = candidate(5000, edges=(1, 2))
    middle = candidate(5300, edges=(3, 4))
    worst = candidate(6200, edges=(5, 6))
    assert scorer.select_top([worst, best, middle], 5000, "EASY") == [best, middle, worst]


def test_select_top_skips_near_duplicate_course():
    best = candidate(5000, edges=range(1, 11))
    duplicate = candidate(5010, edges=range(1, 11))
    other = candidate(5400, edges=range(20, 30))
    assert scorer.select_top([best, duplicate, other], 5000, "EASY") == [best, other]


def test_select_top_returns_at_most_top_n():
    candidates = [candidate(5000 + i * 10, edges=(i * 10, i * 10 + 1)) for i in range(6)]
    assert len(scorer.select_top(candidates, 5000, "EASY")) == config.TOP_N


def test_select_top_excludes_course_far_from_target_distance():
    close = candidate(5200, edges=(1, 2))
    too_short = candidate(2300, edges=(3, 4))   # 목표의 46%
    too_long = candidate(7000, edges=(5, 6))    # 목표의 140%
    assert scorer.select_top([too_short, close, too_long], 5000, "EASY") == [close]


def test_select_top_with_no_candidates():
    assert scorer.select_top([], 5000, "EASY") == []

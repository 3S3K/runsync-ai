import pytest

from app import config, scorer
from app.routing import Candidate


EASY_MID = sum(config.INTENSITY_ZONE_PCT["EASY"]) / 2


def candidate(distance_m, overlap=0.0, edges=(1, 2, 3), hr_pct=EASY_MID):
    # 기본 예상 강도를 EASY 구간 중앙으로 두면 EASY 점수에서 강도 항이 0이 된다
    return Candidate(bearing_deg=0, distance_m=distance_m, ascent_m=0, max_slope_pct=0,
                     overlap_ratio=overlap, edge_ids=list(edges), coordinates=[],
                     predicted_hr_pct=hr_pct)


def test_score_combines_distance_error_and_overlap():
    c = candidate(5500, overlap=0.2)
    assert scorer.score(c, 5000, "EASY") == pytest.approx(config.W_DISTANCE * 0.1 + config.W_OVERLAP * 0.2)


def test_score_is_zero_for_exact_distance_without_overlap_at_zone_center():
    assert scorer.score(candidate(5000), 5000, "EASY") == 0


def test_intensity_error_is_gap_from_zone_center():
    c = candidate(5000, hr_pct=72)
    assert scorer.intensity_error(c, "EASY") == pytest.approx(7)     # 중앙 65
    assert scorer.intensity_error(c, "NORMAL") == pytest.approx(3)   # 중앙 75
    assert scorer.intensity_error(c, "HARD") == pytest.approx(13)    # 중앙 85


def test_score_adds_weighted_intensity_error():
    c = candidate(5000, hr_pct=72)
    assert scorer.score(c, 5000, "NORMAL") == pytest.approx(config.W_INTENSITY * 3)


def test_easy_prefers_lower_predicted_intensity_and_hard_prefers_higher():
    gentle = candidate(5000, edges=(1, 2), hr_pct=68)
    tough = candidate(5000, edges=(3, 4), hr_pct=82)
    assert scorer.select_top([tough, gentle], 5000, "EASY") == [gentle, tough]
    assert scorer.select_top([gentle, tough], 5000, "HARD") == [tough, gentle]


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

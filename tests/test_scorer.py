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


@pytest.mark.parametrize("predicted,expected", [
    (59.4, -1),   # 59%로 표시 → 구간보다 낮음
    (59.6, 0),    # 60%로 표시 → 구간 안
    (70.4, 0),    # 70%로 표시 → 구간 안
    (70.6, 1),    # 71%로 표시 → 구간보다 높음
])
def test_zone_position_uses_displayed_integer(predicted, expected):
    assert scorer.zone_position(predicted, "EASY") == expected


def test_in_zone_course_ranks_before_out_of_zone_course_with_better_distance():
    in_zone = candidate(4500, edges=(1, 2), hr_pct=70.0)       # 거리 10% 부족, 구간 안
    out_of_zone = candidate(5000, edges=(3, 4), hr_pct=70.6)   # 거리 정확, 71%로 표시돼 구간 밖
    # 점수만 보면 구간 밖 코스가 더 좋다
    assert scorer.score(out_of_zone, 5000, "EASY") < scorer.score(in_zone, 5000, "EASY")
    assert scorer.select_top([out_of_zone, in_zone], 5000, "EASY") == [in_zone, out_of_zone]


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


def test_ranking_score_adds_penalty_only_outside_zone():
    inside = candidate(5000, hr_pct=68)
    outside = candidate(5000, hr_pct=75)
    assert scorer.ranking_score(inside, 5000, "EASY") == pytest.approx(scorer.score(inside, 5000, "EASY"))
    assert scorer.ranking_score(outside, 5000, "EASY") == pytest.approx(
        scorer.score(outside, 5000, "EASY") + config.OUT_OF_ZONE_PENALTY)


def test_display_score_follows_ranking_order():
    best = candidate(5000, edges=(1, 2), hr_pct=65)        # 구간 중앙, 거리 정확
    good = candidate(5600, edges=(3, 4), hr_pct=69)        # 구간 안, 거리 12% 초과
    outside = candidate(5000, edges=(5, 6), hr_pct=75)     # 거리는 정확하지만 구간 밖
    scores = [scorer.display_score(c, 5000, "EASY") for c in (best, good, outside)]
    assert scores[0] == 100
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= value <= 100 for value in scores)
    assert scorer.select_top([outside, good, best], 5000, "EASY") == [best, good, outside]


@pytest.mark.parametrize("predicted,expected", [
    (55, "Z1"), (59.6, "Z2"), (69.4, "Z2"), (69.6, "Z3"), (79.4, "Z3"), (80, "Z4"), (90, "Z5"),
])
def test_hr_zone(predicted, expected):
    assert scorer.hr_zone(predicted) == expected

import pytest

from app import explain
from app.routing import Candidate
from app.schemas import RecentRecord

COURSE = Candidate(bearing_deg=0, distance_m=5120, ascent_m=38.4, max_slope_pct=6.42,
                   overlap_ratio=0.05, edge_ids=[1], coordinates=[], predicted_hr_pct=74.4)


@pytest.mark.parametrize("seconds,expected", [(378, "6'18\""), (300, "5'00\""), (359.6, "6'00\"")])
def test_format_pace(seconds, expected):
    assert explain.format_pace(seconds) == expected


def test_average_pace_is_total_time_over_total_distance():
    records = [RecentRecord(distance_km=5, duration_sec=1800), RecentRecord(distance_km=10, duration_sec=4200)]
    assert explain.average_pace_sec_per_km(records) == pytest.approx(400)


def test_average_pace_without_records():
    assert explain.average_pace_sec_per_km([]) is None


def test_reason_without_records_or_age():
    reason = explain.build_reason(COURSE, 5.0, "NORMAL", [])
    assert reason == ("목표 5km 기준 5.1km 순환 코스예요. 누적 오르막은 38m예요. "
                      "예상 운동 강도는 최대 심박의 74%로 '보통' 구간(70~80%)에 들어요.")


def test_reason_with_age_adds_bpm():
    reason = explain.build_reason(COURSE, 5.0, "NORMAL", [], age=24)
    assert "최대 심박의 74%(약 146bpm)로" in reason   # 196 × 0.744


@pytest.mark.parametrize("intensity,expected", [
    ("EASY", "'가볍게' 구간(60~70%)보다 높아요."),
    ("NORMAL", "'보통' 구간(70~80%)에 들어요."),
    ("HARD", "'힘들게' 구간(80~90%)보다 낮아요."),
])
def test_describe_intensity_says_whether_prediction_fits_zone(intensity, expected):
    assert explain.describe_intensity(74.4, intensity, None).endswith(expected)


def test_describe_intensity_judges_by_displayed_value():
    # 70.4%는 70%로 표시되므로 60~70% 구간 안으로 말해야 한다
    assert explain.describe_intensity(70.4, "EASY", None) == (
        "예상 운동 강도는 최대 심박의 70%로 '가볍게' 구간(60~70%)에 들어요.")


def test_reason_with_records_adds_estimated_time():
    records = [RecentRecord(distance_km=5, duration_sec=1800)]  # 6'00"/km → 5.12km에 약 31분
    reason = explain.build_reason(COURSE, 5.0, "EASY", records)
    assert reason.endswith("최근 1회 평균 페이스(6'00\"/km)로 달리면 약 31분 걸려요.")


def test_estimated_time_uses_recent_average_pace():
    records = [RecentRecord(distance_km=5, duration_sec=1800)]   # 6'00"/km
    assert explain.estimated_time_min(5120, records) == 31
    assert explain.estimated_time_min(5120, []) is None


def test_hr_bpm_range_spans_model_error_margin():
    # 24세 → 최대 심박 196. 75% ± 2.5%p → 72.5%~77.5% → 142~152bpm
    assert explain.hr_bpm_range(75.0, 24) == [142, 152]
    assert explain.hr_bpm_range(75.0, None) is None


def test_reason_mentions_longest_climb_when_big_enough():
    reason = explain.build_reason(COURSE, 5.0, "NORMAL", [], climb=(2810.0, 452.0, 11.3))
    assert "가장 긴 오르막은 2.8km 지점부터 450m 동안 11m 올라가요." in reason


def test_reason_skips_small_climb():
    reason = explain.build_reason(COURSE, 5.0, "NORMAL", [], climb=(100.0, 60.0, 2.0))
    assert "가장 긴 오르막" not in reason

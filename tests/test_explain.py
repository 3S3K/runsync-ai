import pytest

from app import explain
from app.routing import Candidate
from app.schemas import RecentRecord

COURSE = Candidate(bearing_deg=0, distance_m=5120, ascent_m=38.4, max_slope_pct=6.42,
                   overlap_ratio=0.05, edge_ids=[1], coordinates=[])


@pytest.mark.parametrize("seconds,expected", [(378, "6'18\""), (300, "5'00\""), (359.6, "6'00\"")])
def test_format_pace(seconds, expected):
    assert explain.format_pace(seconds) == expected


def test_average_pace_is_total_time_over_total_distance():
    records = [RecentRecord(distance_km=5, duration_sec=1800), RecentRecord(distance_km=10, duration_sec=4200)]
    assert explain.average_pace_sec_per_km(records) == pytest.approx(400)


def test_average_pace_without_records():
    assert explain.average_pace_sec_per_km([]) is None


def test_reason_without_records_describes_course_only():
    reason = explain.build_reason(COURSE, 5.0, "NORMAL", [])
    assert reason == ("목표 5km 기준 5.1km 순환 코스예요. "
                      "누적 오르막 38m, 최대 경사 6.4%이고, '보통' 강도라 가파른 오르막은 되도록 줄였어요.")


def test_reason_with_records_adds_estimated_time():
    records = [RecentRecord(distance_km=5, duration_sec=1800)]  # 6'00"/km → 5.12km에 약 31분
    reason = explain.build_reason(COURSE, 5.0, "EASY", records)
    assert "'가볍게' 강도라 가파른 오르막은 최대한 피했어요." in reason
    assert reason.endswith("최근 1회 평균 페이스(6'00\"/km)로 달리면 약 31분 걸려요.")

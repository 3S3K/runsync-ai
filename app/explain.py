"""추천 이유 문장. LLM 없이 계산된 값을 템플릿에 채운다."""
from typing import List, Optional

from app import config, scorer
from app.routing import Candidate
from app.schemas import RecentRecord

INTENSITY_LABEL = {"EASY": "가볍게", "NORMAL": "보통", "HARD": "힘들게"}


def average_pace_sec_per_km(records: List[RecentRecord]) -> Optional[float]:
    """총 시간 ÷ 총 거리. 기록이 없으면 None."""
    total_km = sum(r.distance_km for r in records)
    if total_km <= 0:
        return None
    return sum(r.duration_sec for r in records) / total_km


def format_pace(sec_per_km: float) -> str:
    minutes, seconds = divmod(round(sec_per_km), 60)
    return f"{minutes}'{seconds:02d}\""


def describe_intensity(predicted_hr_pct: float, intensity: str, age: Optional[int]) -> str:
    """예상 운동 강도가 요청한 강도 구간에 드는지 말한다. 나이를 알면 심박수(bpm)도 덧붙인다."""
    low, high = config.INTENSITY_ZONE_PCT[intensity]
    # 순위를 정할 때와 같은 기준(표시되는 정수 값)으로 판정한다
    fit = {-1: "보다 낮아요", 0: "에 들어요", 1: "보다 높아요"}[scorer.zone_position(predicted_hr_pct, intensity)]
    bpm = f"(약 {round(predicted_hr_pct / 100 * (220 - age))}bpm)" if age is not None else ""
    return (f"예상 운동 강도는 최대 심박의 {predicted_hr_pct:.0f}%{bpm}로 "
            f"'{INTENSITY_LABEL[intensity]}' 구간({low:.0f}~{high:.0f}%){fit}.")


def estimated_time_min(distance_m: float, records: List[RecentRecord]) -> Optional[int]:
    """최근 기록의 평균 페이스로 달릴 때 걸리는 시간(분). 기록이 없으면 None."""
    pace = average_pace_sec_per_km(records)
    if pace is None:
        return None
    return round(pace * distance_m / 1000 / 60)


def hr_bpm_range(predicted_hr_pct: float, age: Optional[int]) -> Optional[List[int]]:
    """예상 심박수 [최저, 최고]. 예측값에서 모델 오차만큼 위아래로 둔다. 나이를 모르면 None."""
    if age is None:
        return None
    max_hr = 220 - age
    margin = config.HR_RANGE_MARGIN_PCT
    return [round((predicted_hr_pct - margin) / 100 * max_hr), round((predicted_hr_pct + margin) / 100 * max_hr)]


def describe_climb(climb) -> str:
    """가장 많이 오르는 구간을 설명한다. climb은 (시작 거리 m, 길이 m, 오르는 높이 m)."""
    start_m, length_m, gain_m = climb
    return (f" 가장 큰 오르막은 {start_m / 1000:.1f}km 지점부터 "
            f"{round(length_m, -1):.0f}m 동안 {gain_m:.0f}m 올라가요.")


def build_reason(candidate: Candidate, target_km: float, intensity: str,
                 records: List[RecentRecord], age: Optional[int] = None, climb=None) -> str:
    distance_km = candidate.distance_m / 1000
    reason = (
        f"목표 {target_km:g}km 기준 {distance_km:.1f}km 순환 코스예요. "
        f"누적 오르막은 {candidate.ascent_m:.0f}m예요. "
        + describe_intensity(candidate.predicted_hr_pct, intensity, age)
    )
    if climb is not None and climb[2] >= config.MIN_CLIMB_GAIN_M:
        reason += describe_climb(climb)
    minutes = estimated_time_min(candidate.distance_m, records)
    if minutes is not None:
        pace = average_pace_sec_per_km(records)
        reason += f" 최근 {len(records)}회 평균 페이스({format_pace(pace)}/km)로 달리면 약 {minutes}분 걸려요."
    return reason

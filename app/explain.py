"""추천 이유 문장. LLM 없이 계산된 값을 템플릿에 채운다."""
from typing import List, Optional

from app import config
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
    if predicted_hr_pct < low:
        fit = "보다 낮아요"
    elif predicted_hr_pct > high:
        fit = "보다 높아요"
    else:
        fit = "에 들어요"
    bpm = f"(약 {round(predicted_hr_pct / 100 * (220 - age))}bpm)" if age is not None else ""
    return (f"예상 운동 강도는 최대 심박의 {predicted_hr_pct:.0f}%{bpm}로 "
            f"'{INTENSITY_LABEL[intensity]}' 구간({low:.0f}~{high:.0f}%){fit}.")


def build_reason(candidate: Candidate, target_km: float, intensity: str,
                 records: List[RecentRecord], age: Optional[int] = None) -> str:
    distance_km = candidate.distance_m / 1000
    reason = (
        f"목표 {target_km:g}km 기준 {distance_km:.1f}km 순환 코스예요. "
        f"누적 오르막은 {candidate.ascent_m:.0f}m예요. "
        + describe_intensity(candidate.predicted_hr_pct, intensity, age)
    )
    pace = average_pace_sec_per_km(records)
    if pace is not None:
        minutes = round(pace * distance_km / 60)
        reason += f" 최근 {len(records)}회 평균 페이스({format_pace(pace)}/km)로 달리면 약 {minutes}분 걸려요."
    return reason

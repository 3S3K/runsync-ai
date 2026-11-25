"""추천 이유 문장. LLM 없이 계산된 값을 템플릿에 채운다."""
from typing import List, Optional

from app.routing import Candidate
from app.schemas import RecentRecord

INTENSITY_LABEL = {"EASY": "가볍게", "NORMAL": "보통", "HARD": "힘들게"}
# 강도별로 경로 탐색이 실제로 한 일 (경사 회피 계수 k의 크기)
SLOPE_POLICY = {
    "EASY": "가파른 오르막은 최대한 피했어요.",
    "NORMAL": "가파른 오르막은 되도록 줄였어요.",
    "HARD": "오르막을 굳이 피하지 않았어요.",
}


def average_pace_sec_per_km(records: List[RecentRecord]) -> Optional[float]:
    """총 시간 ÷ 총 거리. 기록이 없으면 None."""
    total_km = sum(r.distance_km for r in records)
    if total_km <= 0:
        return None
    return sum(r.duration_sec for r in records) / total_km


def format_pace(sec_per_km: float) -> str:
    minutes, seconds = divmod(round(sec_per_km), 60)
    return f"{minutes}'{seconds:02d}\""


def build_reason(candidate: Candidate, target_km: float, intensity: str,
                 records: List[RecentRecord]) -> str:
    distance_km = candidate.distance_m / 1000
    reason = (
        f"목표 {target_km:g}km 기준 {distance_km:.1f}km 순환 코스예요. "
        f"누적 오르막 {candidate.ascent_m:.0f}m, 최대 경사 {candidate.max_slope_pct:.1f}%이고, "
        f"'{INTENSITY_LABEL[intensity]}' 강도라 {SLOPE_POLICY[intensity]}"
    )
    pace = average_pace_sec_per_km(records)
    if pace is not None:
        minutes = round(pace * distance_km / 60)
        reason += f" 최근 {len(records)}회 평균 페이스({format_pace(pace)}/km)로 달리면 약 {minutes}분 걸려요."
    return reason

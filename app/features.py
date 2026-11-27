"""심박 예측 모델의 피처. 학습과 예측이 반드시 이 모듈을 같이 쓴다.

따로 구현하면 단위나 순서가 어긋나서 예측이 조용히 틀어진다.
모르는 값은 NaN으로 둔다. Ridge는 중앙값으로 채우고 LightGBM은 NaN을 그대로 다룬다.
"""
import math
from typing import List, Optional, Sequence

from app.schemas import RecentRecord, UserInfo

FEATURE_NAMES = (
    "age",
    "is_male",                 # 남 1, 여 0
    "recent_pace_sec_per_km",  # 최근 기록의 평균 페이스 (총 시간 ÷ 총 거리)
    "recent_distance_km",      # 최근 기록의 평균 거리
    "recent_cadence",          # 최근 기록의 평균 케이던스
    "course_distance_km",
    "climb_m_per_km",          # 코스의 1km당 누적 오르막
    "max_slope_pct",
)

NAN = float("nan")


def _mean(values: Sequence[Optional[float]]) -> float:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else NAN


def user_features(user: Optional[UserInfo], records: Sequence[RecentRecord]) -> List[float]:
    """사용자와 최근 기록에서 나오는 앞쪽 5개 값."""
    age = float(user.age) if user and user.age is not None else NAN
    gender = user.gender.upper() if user and user.gender else None
    is_male = {"MALE": 1.0, "FEMALE": 0.0}.get(gender, NAN)

    total_km = sum(r.distance_km for r in records)
    pace = sum(r.duration_sec for r in records) / total_km if total_km > 0 else NAN
    distance = total_km / len(records) if records else NAN
    cadence = _mean([r.cadence for r in records])
    return [age, is_male, pace, distance, cadence]


def course_features(distance_km: float, climb_m_per_km: float, max_slope_pct: float) -> List[float]:
    """코스에서 나오는 뒤쪽 3개 값."""
    return [distance_km, climb_m_per_km, max_slope_pct]


def build_row(user_part: Sequence[float], course_part: Sequence[float]) -> List[float]:
    row = list(user_part) + list(course_part)
    assert len(row) == len(FEATURE_NAMES)
    return row


def hr_pct(average_heart_rate: float, age: float) -> float:
    """타깃: 평균 심박 ÷ (220 − 나이) × 100."""
    return average_heart_rate / (220 - age) * 100


def is_missing(value: float) -> bool:
    return math.isnan(value)

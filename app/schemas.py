"""요청·응답 모델. JSON은 백엔드와 맞춰 camelCase, 코드 안에서는 snake_case."""
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from app import config


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class Intensity(str, Enum):
    EASY = "EASY"
    NORMAL = "NORMAL"
    HARD = "HARD"


class Point(CamelModel):
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)


class UserInfo(CamelModel):
    user_id: int
    age: Optional[int] = None
    gender: Optional[str] = None


class RecentRecord(CamelModel):
    distance_km: float = Field(gt=0)
    duration_sec: int = Field(gt=0)
    avg_heart_rate: Optional[int] = None
    cadence: Optional[int] = None
    elevation_gain_m: Optional[float] = None


class RecommendReq(CamelModel):
    start: Point
    target_distance_km: float = Field(ge=config.MIN_DISTANCE_KM, le=config.MAX_DISTANCE_KM)
    intensity: Intensity
    user: Optional[UserInfo] = None
    recent_records: List[RecentRecord] = Field(default_factory=list)


class Geometry(CamelModel):
    type: str = "LineString"
    coordinates: List[List[float]]  # [lng, lat]


class ElevationPoint(CamelModel):
    distance_m: int       # 출발점부터의 거리
    elevation_m: float    # 해발 고도


class SlopeSegment(CamelModel):
    grade: str                      # FLAT(3% 미만) | MODERATE(3~6%) | STEEP(6% 이상)
    coordinates: List[List[float]]  # [lng, lat]


class Climb(CamelModel):
    start_m: int    # 출발점부터 오르막이 시작하는 지점까지의 거리
    length_m: int   # 오르막의 길이
    gain_m: int     # 오르는 높이


class CourseRes(CamelModel):
    rank: int
    name: str                       # 예: "용운로 순환"
    score: int                      # 추천 점수 (0~100)
    distance_km: float
    elevation_gain_m: float
    max_slope_pct: float
    predicted_hr_pct: float  # 예상 운동 강도 (최대 심박 대비 %)
    intensity: Intensity
    reason: str
    geometry: Geometry
    elevation_profile: List[ElevationPoint]
    estimated_time_min: Optional[int] = None    # 최근 기록의 평균 페이스 기준. 기록이 없으면 null
    stairs_count: int                           # 계단 구간 수
    hr_zone: str                                # 예상 심박존 (Z1~Z5)
    hr_bpm_range: Optional[List[int]] = None    # 예상 심박수 [최저, 최고]. 나이를 모르면 null
    biggest_climb: Optional[Climb] = None       # 가장 많이 오르는 구간(높이 기준). 오르막이 거의 없으면 null
    slope_segments: List[SlopeSegment]          # 경사 등급별로 나눈 경로 (지도에 색을 칠할 때 쓴다)


class RecommendRes(CamelModel):
    courses: List[CourseRes]


class ErrorRes(BaseModel):
    code: str
    message: str

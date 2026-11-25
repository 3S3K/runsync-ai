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


class CourseRes(CamelModel):
    rank: int
    distance_km: float
    elevation_gain_m: float
    max_slope_pct: float
    predicted_hr_pct: Optional[float] = None  # 심박 예측 모델을 붙이기 전까지는 null
    intensity: Intensity
    reason: str
    geometry: Geometry


class RecommendRes(CamelModel):
    courses: List[CourseRes]


class ErrorRes(BaseModel):
    code: str
    message: str

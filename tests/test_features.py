import math

import pytest

from app import features
from app.schemas import RecentRecord, UserInfo

RECORDS = [
    RecentRecord(distance_km=5, duration_sec=1800, cadence=170),
    RecentRecord(distance_km=10, duration_sec=4200),            # 케이던스 없음
]


def test_user_features_from_user_and_records():
    age, is_male, pace, distance, cadence = features.user_features(
        UserInfo(user_id=1, age=24, gender="MALE"), RECORDS)
    assert (age, is_male) == (24.0, 1.0)
    assert pace == pytest.approx(400)       # 6000초 ÷ 15km
    assert distance == pytest.approx(7.5)
    assert cadence == pytest.approx(170)    # 있는 값만 평균


def test_user_features_female():
    assert features.user_features(UserInfo(user_id=1, age=30, gender="female"), [])[1] == 0.0


def test_unknown_values_become_nan():
    row = features.user_features(None, [])
    assert all(math.isnan(v) for v in row)


def test_missing_age_and_gender_become_nan():
    age, is_male, *_ = features.user_features(UserInfo(user_id=1), RECORDS)
    assert math.isnan(age) and math.isnan(is_male)


def test_build_row_matches_feature_names_order():
    user_part = features.user_features(UserInfo(user_id=1, age=24, gender="MALE"), RECORDS)
    row = features.build_row(user_part, features.course_features(5.1, 12.0, 6.4))
    assert len(row) == len(features.FEATURE_NAMES)
    assert row[features.FEATURE_NAMES.index("course_distance_km")] == 5.1
    assert row[features.FEATURE_NAMES.index("climb_m_per_km")] == 12.0
    assert row[features.FEATURE_NAMES.index("max_slope_pct")] == 6.4


def test_hr_pct_target():
    assert features.hr_pct(147, 24) == pytest.approx(75.0)   # 147 ÷ 196 × 100

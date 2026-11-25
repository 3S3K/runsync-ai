"""합성 학습 데이터 생성기.

서비스 초기에는 심박이 기록된 러닝이 충분하지 않다(콜드 스타트). 운동생리학에서 알려진
관계를 식으로 옮겨 가상의 러너와 러닝을 만들고, 그 데이터로 초기 모델을 학습한다.
실제 기록이 쌓이면 generate() 대신 실제 기록에서 같은 모양의 (X, y)를 만들어 학습하면 된다.

반영한 관계 (모두 "평소 페이스로 뛴다"는 전제):
- 오르막이 많을수록 심박이 오른다. 체력이 낮을수록 더 크게 오른다.
- 평소보다 긴 거리를 뛰면 후반에 심박이 올라간다(cardiac drift).
- 평소 페이스가 느린 러너일수록 같은 '평소 페이스 러닝'이 상대적으로 더 힘들다.
- 가파른 구간(8% 초과)이 있으면 추가로 오른다.
- 케이던스가 낮으면 조금 더 오른다.
계수는 생리학 문헌의 정밀한 값이 아니라 방향과 대략의 크기를 맞춘 가정이다.
"""
from typing import Tuple

import numpy as np

from app.features import FEATURE_NAMES

BASE_HR_PCT = 70.0     # 평균적인 러너가 평지에서 평소 거리를 평소 페이스로 뛸 때
NOISE_STD = 3.0        # 컨디션·날씨·측정 오차
MISSING_RATE = {"age": 0.10, "is_male": 0.10, "recent_cadence": 0.25}


def generate(n_rows: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """(X, y)를 만든다. X의 열 순서는 FEATURE_NAMES, y는 최대 심박 대비 %."""
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 61, n_rows).astype(float)
    is_male = rng.integers(0, 2, n_rows).astype(float)

    # 평소 페이스(초/km): 나이가 많을수록, 여성일수록 평균적으로 조금 느리다. 개인차가 가장 크다.
    pace = 375 + 1.2 * (age - 30) - 25 * (is_male - 0.5) * 2 + rng.normal(0, 45, n_rows)
    pace = np.clip(pace, 240, 540)
    usual_km = np.clip(rng.lognormal(np.log(5.5), 0.45, n_rows), 1.5, 21)
    # 케이던스: 빠른 러너일수록 높다
    cadence = np.clip(172 - 0.06 * (pace - 360) + rng.normal(0, 6, n_rows), 140, 200)

    course_km = np.clip(usual_km * rng.lognormal(0, 0.3, n_rows), 1, 21)
    climb = np.clip(rng.gamma(2.0, 8.0, n_rows), 0, 70)                 # 1km당 오르막(m)
    max_slope = np.clip(4 + 0.35 * climb + rng.normal(0, 3, n_rows), 0, 40)

    slowness = (pace - 360) / 60                                        # 6'00"/km보다 1분 느리면 1
    y = (
        BASE_HR_PCT
        + 3.0 * slowness
        + 0.22 * climb * (1 + 0.25 * np.clip(slowness, -1, 2))          # 체력이 낮으면 오르막이 더 힘들다
        + 6.0 * np.log(course_km / usual_km)                            # 평소보다 길면 오른다
        + 0.15 * np.clip(max_slope - 8, 0, None)
        - 0.04 * (cadence - 170)
        + 0.05 * (age - 30)
        - 1.0 * (is_male - 0.5)
        + rng.normal(0, NOISE_STD, n_rows)
    )
    y = np.clip(y, 50, 98)

    X = np.column_stack([age, is_male, pace, usual_km, cadence, course_km, climb, max_slope])
    for name, rate in MISSING_RATE.items():
        X[rng.random(n_rows) < rate, FEATURE_NAMES.index(name)] = np.nan
    return X, y

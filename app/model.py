"""심박 예측 모델을 불러와 예측한다.

모델은 scikit-learn 방식의 predict(X)를 가진 객체면 무엇이든 된다.
Ridge 파이프라인과 LGBMRegressor가 같은 인터페이스라 파일만 바꾸면 교체된다.
"""
import warnings
from typing import List, Optional, Sequence

import joblib
import numpy as np

from app import config

_model = None


def load(path: Optional[str] = None) -> None:
    global _model
    path = path or config.MODEL_PATH
    try:
        _model = joblib.load(path)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"모델 파일이 없습니다: {path}. 먼저 `python -m app.train`으로 학습하세요.") from exc


def replace(new_model) -> None:
    """재학습한 모델로 바꾼다. 서버를 다시 시작하지 않아도 다음 예측부터 새 모델을 쓴다."""
    global _model
    _model = new_model


def predict_hr_pct(rows: Sequence[Sequence[float]]) -> List[float]:
    """피처 행들(app.features.FEATURE_NAMES 순서)에 대한 예상 운동 강도(최대 심박 대비 %)."""
    if _model is None:
        raise RuntimeError("모델이 로드되지 않았습니다. model.load()를 먼저 호출하세요.")
    # macOS(Apple Silicon)의 numpy 2.0.x는 행렬 곱에서 가짜 RuntimeWarning을 낸다.
    # 결과는 정상임을 수동 계산과 비교해 확인했다. 대신 결과가 유한한지 직접 검사한다.
    # LightGBM은 numpy 배열로 학습해도 열 이름을 만들어 붙여서, 배열로 예측하면 scikit-learn이
    # 이름이 없다는 경고를 낸다. 열 순서는 features.FEATURE_NAMES로 고정돼 있어 무시해도 된다.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        predictions = _model.predict(np.asarray(rows, dtype=float))
    if not np.isfinite(predictions).all():
        raise RuntimeError("모델이 유한하지 않은 값을 예측했습니다.")
    return [float(np.clip(p, 40, 100)) for p in predictions]

"""심박 예측 모델 학습.

    python -m app.train                  # LightGBM (기본)
    python -m app.train --model ridge

학습 데이터의 20%를 검증용으로 떼어 MAE를 출력하고, 모델을 파일로 저장한다.
"""
import argparse
import os
import warnings
from typing import Optional

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app import config, synthetic


def build_model(kind: str):
    if kind == "ridge":
        # Ridge는 NaN을 못 다루므로 중앙값으로 채운다
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    if kind == "lightgbm":
        from lightgbm import LGBMRegressor  # macOS에서는 libomp가 있어야 해서 필요할 때만 불러온다
        return LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15,
                             min_child_samples=30, random_state=42, verbose=-1)
    raise ValueError(f"알 수 없는 모델: {kind}")


def fit_model(kind: str, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
    """모델을 만들어 학습한다. 표본 가중치는 모델 종류에 맞는 방식으로 넘긴다."""
    model = build_model(kind)
    if sample_weight is None:
        return model.fit(X, y)
    if kind == "ridge":
        return model.fit(X, y, ridge__sample_weight=sample_weight)
    return model.fit(X, y, sample_weight=sample_weight)


def train(kind: str, n_rows: int, seed: int, out_path: str) -> float:
    X, y = synthetic.generate(n_rows, seed)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=seed)

    model = build_model(kind).fit(X_train, y_train)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        predicted = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, predicted)
    baseline = mean_absolute_error(y_valid, np.full_like(y_valid, y_train.mean()))
    print(f"{kind}: 검증 MAE {mae:.2f}%p (평균만 찍는 기준선 {baseline:.2f}%p, 학습 {len(y_train)}건 / 검증 {len(y_valid)}건)")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    joblib.dump(model, out_path)
    print(f"저장: {out_path}")
    return mae


def main() -> None:
    parser = argparse.ArgumentParser(description="심박 예측 모델 학습")
    parser.add_argument("--model", choices=("ridge", "lightgbm"), default=config.MODEL_KIND)
    parser.add_argument("--rows", type=int, default=config.SYNTHETIC_ROWS)
    parser.add_argument("--seed", type=int, default=config.SYNTHETIC_SEED)
    parser.add_argument("--out", default=config.MODEL_PATH)
    args = parser.parse_args()
    train(args.model, args.rows, args.seed, args.out)


if __name__ == "__main__":
    main()

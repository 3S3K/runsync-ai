import math

import joblib
import numpy as np
import pytest

from app import features, model, synthetic, train

NAN = float("nan")
# 24세 남성, 6'00"/km, 평소 5km, 케이던스 170 / 5km, 1km당 10m, 최대 경사 6%
BASE = [24, 1, 360, 5, 170, 5, 10, 6]


def changed(name, value):
    row = list(BASE)
    row[features.FEATURE_NAMES.index(name)] = value
    return row


def test_synthetic_shape_range_and_missing_values():
    X, y = synthetic.generate(2000, seed=1)
    assert X.shape == (2000, len(features.FEATURE_NAMES))
    assert y.min() >= 50 and y.max() <= 98
    cadence = X[:, features.FEATURE_NAMES.index("recent_cadence")]
    assert 0.15 < np.isnan(cadence).mean() < 0.35      # 일부는 일부러 비운다
    assert not np.isnan(X[:, features.FEATURE_NAMES.index("climb_m_per_km")]).any()


def test_synthetic_is_reproducible_by_seed():
    X1, y1 = synthetic.generate(100, seed=3)
    X2, y2 = synthetic.generate(100, seed=3)
    assert np.array_equal(X1, X2, equal_nan=True) and np.array_equal(y1, y2)


def test_training_beats_mean_baseline(tmp_path):
    path = str(tmp_path / "m.joblib")
    mae = train.train("ridge", n_rows=3000, seed=7, out_path=path)
    X, y = synthetic.generate(3000, seed=7)
    baseline = np.abs(y - y.mean()).mean()
    assert mae < baseline * 0.7
    assert hasattr(joblib.load(path), "predict")


def test_prediction_is_in_percent_range():
    [predicted] = model.predict_hr_pct([BASE])
    assert 55 < predicted < 90


@pytest.mark.parametrize("name,low,high", [
    ("climb_m_per_km", 5, 35),             # 오르막이 많으면
    ("recent_pace_sec_per_km", 300, 450),  # 평소 페이스가 느리면
    ("course_distance_km", 3, 10),         # 평소보다 길게 뛰면
])
def test_prediction_rises_with(name, low, high):
    lower, higher = model.predict_hr_pct([changed(name, low), changed(name, high)])
    assert higher > lower


def test_prediction_handles_unknown_user():
    # 나이·성별·최근 기록을 전혀 모르는 사용자
    [predicted] = model.predict_hr_pct([[NAN, NAN, NAN, NAN, NAN, 5, 10, 6]])
    assert math.isfinite(predicted) and 55 < predicted < 90


def test_predict_without_loaded_model_raises(monkeypatch):
    monkeypatch.setattr(model, "_model", None)
    with pytest.raises(RuntimeError):
        model.predict_hr_pct([BASE])


def test_load_missing_file_gives_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(model, "_model", None)
    with pytest.raises(RuntimeError, match="app.train"):
        model.load(str(tmp_path / "none.joblib"))

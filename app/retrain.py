"""실제 러닝 기록을 반영한 심박 예측 모델 재학습.

백엔드가 보내 준 기록을 선별하고, 합성 데이터와 함께 학습한 모델이
합성 데이터만으로 학습한 모델보다 실제 기록을 더 잘 맞힐 때만 서비스 중인 모델을 바꾼다.

비교 대상을 "서비스 중인 모델"이 아니라 "합성 데이터만으로 학습한 모델"로 둔 이유:
서비스 중인 모델은 이전 재학습에서 같은 기록을 이미 본 상태라, 그 기록으로 재면 오차가 실제보다
작게 나온다. 처음 보는 기록으로 재는 새 모델과 공정하게 비교할 수 없다.
"""
import json
import logging
import os
import shutil
import threading
import warnings
from collections import defaultdict
from datetime import datetime
from itertools import groupby
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.model_selection import KFold

from app import config, features, model, synthetic, train
from app.schemas import KST, RecentRecord, TrainingRecord, UserInfo

logger = logging.getLogger(__name__)

# 재학습이 겹쳐 실행되지 않게 한다 (모델 파일과 이력 파일을 같이 쓴다)
_lock = threading.Lock()

REASON_NO_RECORDS = "학습에 쓸 수 있는 기록이 없음"
REASON_NOT_ENOUGH_FOR_VALIDATION = "교차 검증에는 기록이 2건 이상 필요함"
REASON_IMPROVED = "실제 기록을 반영한 모델의 검증 오차가 더 작음"
REASON_NOT_IMPROVED = "실제 기록을 반영해도 검증 오차가 줄지 않음"
REASON_FAILED = "재학습 중 오류가 발생함"


def is_usable(record: TrainingRecord) -> bool:
    """학습의 정답으로 쓸 수 있는 기록인지. 사용자가 손으로 입력한 값이라 범위를 벗어나면 뺀다."""
    if record.avg_heart_rate is None or record.age is None:
        return False
    if not config.TRAIN_MIN_AGE <= record.age <= config.TRAIN_MAX_AGE:
        return False
    if record.distance_km < config.TRAIN_MIN_DISTANCE_KM or record.duration_sec <= 0:
        return False
    pace = record.duration_sec / record.distance_km
    if not config.TRAIN_MIN_PACE_SEC_PER_KM <= pace <= config.TRAIN_MAX_PACE_SEC_PER_KM:
        return False
    hr_pct = features.hr_pct(record.avg_heart_rate, record.age)
    if not config.TRAIN_MIN_HR_PCT <= hr_pct <= config.TRAIN_MAX_HR_PCT:
        return False
    if record.cadence is not None and not config.TRAIN_MIN_CADENCE <= record.cadence <= config.TRAIN_MAX_CADENCE:
        return False
    return True


def _to_recent(record: TrainingRecord) -> RecentRecord:
    return RecentRecord(
        distance_km=record.distance_km,
        duration_sec=record.duration_sec,
        avg_heart_rate=record.avg_heart_rate,
        cadence=record.cadence,
        elevation_gain_m=record.elevation_gain_m,
    )


def _record_order(record: TrainingRecord) -> tuple:
    """시작 시각 순. 시각이 같으면 나머지 모든 값으로 순서를 정해 입력 순서에 좌우되지 않게 한다.

    피처나 정답에 쓰이는 값을 하나라도 빼면, 그 값만 다른 기록들의 순서가 입력 순서를 따르게 된다.
    그래서 기록의 모든 필드를 넣는다. 모든 값이 같은 기록은 순서가 바뀌어도 결과가 같다.
    """
    def slot(value, placeholder):
        # 값이 없는 것과 실제 값을 따로 구분한다. 자리 채움 값만 쓰면 그 값과 같은 실제 입력과 겹친다
        return (value is None, placeholder if value is None else value)

    return (
        record.start_time, record.distance_km, record.duration_sec,
        slot(record.avg_heart_rate, 0), slot(record.cadence, 0),
        slot(record.age, 0), slot(record.gender, ""), slot(record.elevation_gain_m, 0.0),
    )


def _climb_m_per_km(record: TrainingRecord) -> float:
    # 음수는 잘못 입력된 값이라 모르는 값으로 둔다
    if record.elevation_gain_m is None or record.elevation_gain_m < 0:
        return features.NAN
    return record.elevation_gain_m / record.distance_km


def build_dataset(records: Sequence[TrainingRecord]) -> Tuple[np.ndarray, np.ndarray]:
    """선별을 통과한 기록마다 (피처, 정답)을 만든다.

    "평소 페이스·거리·케이던스"는 같은 사용자의 그 기록 직전 기록들로 계산한다. 추천 요청에서
    백엔드가 최근 기록을 넘겨주는 것과 같은 기준이다(거리와 시간이 0보다 큰 기록, 최근 5회).
    코스의 최대 경사는 기록에 없어 NaN으로 둔다.
    """
    by_user: Dict[int, List[TrainingRecord]] = defaultdict(list)
    for record in records:
        by_user[record.user_id].append(record)

    rows, targets = [], []
    # 같은 기록 묶음이면 보낸 순서와 무관하게 같은 행 순서가 나오도록, 사용자와 기록의 순서를 고정한다.
    # 행 순서가 바뀌면 교차 검증의 묶음 구성이 달라져 결과가 실행마다 달라진다
    for user_id in sorted(by_user):
        user_records = sorted(by_user[user_id], key=_record_order)
        history: List[TrainingRecord] = []      # 이 기록보다 먼저 달린, 평소 값 계산에 쓸 수 있는 기록
        # 시작 시각이 같은 기록끼리는 서로의 "직전 기록"이 아니다. 같은 이전 기록으로 계산한 뒤 한꺼번에 더한다
        for _, same_time in groupby(user_records, key=lambda record: record.start_time):
            group = list(same_time)
            recent = [_to_recent(r) for r in history[-config.RECENT_RECORD_COUNT:]]
            for record in group:
                if is_usable(record):
                    user = UserInfo(user_id=record.user_id, age=record.age, gender=record.gender)
                    rows.append(features.build_row(
                        features.user_features(user, recent),
                        features.course_features(record.distance_km, _climb_m_per_km(record), features.NAN),
                    ))
                    targets.append(features.hr_pct(record.avg_heart_rate, record.age))
            history.extend(r for r in group if r.distance_km > 0 and r.duration_sec > 0)

    X = np.asarray(rows, dtype=float).reshape(-1, len(features.FEATURE_NAMES))
    return X, np.asarray(targets, dtype=float)


def _predict(fitted, X: np.ndarray) -> np.ndarray:
    # 경고를 끄는 이유는 app.model.predict_hr_pct 의 주석과 같다
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        return fitted.predict(X)


def _fit_mixed(X_synthetic: np.ndarray, y_synthetic: np.ndarray, X_real: np.ndarray, y_real: np.ndarray):
    """합성 데이터와 실제 기록을 함께 학습한다. 실제 기록에 더 큰 비중을 둔다."""
    X = np.vstack([X_synthetic, X_real])
    y = np.concatenate([y_synthetic, y_real])
    weight = np.concatenate([
        np.ones(len(y_synthetic)),
        np.full(len(y_real), config.REAL_RECORD_WEIGHT),
    ])
    return train.fit_model(config.MODEL_KIND, X, y, weight)


def baseline_mae(X_synthetic: np.ndarray, y_synthetic: np.ndarray, X_real: np.ndarray, y_real: np.ndarray) -> float:
    """합성 데이터만으로 학습한 모델이 실제 기록을 얼마나 틀리는지."""
    fitted = train.fit_model(config.MODEL_KIND, X_synthetic, y_synthetic)
    return float(np.mean(np.abs(_predict(fitted, X_real) - y_real)))


def cross_validated_mae(X_synthetic: np.ndarray, y_synthetic: np.ndarray, X_real: np.ndarray, y_real: np.ndarray) -> float:
    """실제 기록을 묶음으로 나눠, 학습에 쓰지 않은 묶음을 얼마나 틀리는지 돌아가며 잰다.

    기록이 적어도 모든 기록이 한 번씩 검증에 쓰인다.
    """
    folds = KFold(n_splits=min(config.CV_MAX_FOLDS, len(y_real)), shuffle=True, random_state=config.SYNTHETIC_SEED)
    errors = []
    for train_index, valid_index in folds.split(X_real):
        fitted = _fit_mixed(X_synthetic, y_synthetic, X_real[train_index], y_real[train_index])
        errors.extend(np.abs(_predict(fitted, X_real[valid_index]) - y_real[valid_index]))
    return float(np.mean(errors))


def _replace_model(new_model) -> None:
    """모델 파일을 바꾸고(이전 파일은 .prev 로 보관) 메모리의 모델도 바꾼다."""
    path = config.MODEL_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        shutil.copy2(path, path + ".prev")
    # 쓰다가 실패해도 기존 파일이 깨지지 않도록 임시 파일에 쓴 뒤 바꿔치기한다
    temporary_path = path + ".tmp"
    joblib.dump(new_model, temporary_path)
    os.replace(temporary_path, path)
    model.replace(new_model)


def _append_history(result: dict) -> None:
    try:
        os.makedirs(os.path.dirname(config.TRAINING_LOG_PATH) or ".", exist_ok=True)
        with open(config.TRAINING_LOG_PATH, "a", encoding="utf-8") as file:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
    except OSError:
        # 이력을 남기지 못해도 재학습 결과는 돌려준다
        logger.warning("재학습 이력을 기록하지 못했습니다: %s", config.TRAINING_LOG_PATH, exc_info=True)


def history(limit: Optional[int] = None) -> List[dict]:
    """재학습 이력. 최근 실행부터."""
    limit = limit or config.TRAINING_HISTORY_LIMIT
    try:
        with open(config.TRAINING_LOG_PATH, encoding="utf-8") as file:
            lines = [line for line in file if line.strip()]
    except FileNotFoundError:
        return []
    items = []
    for line in reversed(lines[-limit:]):
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            # 쓰다가 끊긴 줄 하나 때문에 이력 전체를 못 보는 일이 없게 건너뛴다
            logger.warning("읽을 수 없는 재학습 이력 줄을 건너뜁니다")
    return items


def _result(received: int, used: int, replaced: bool, reason: str,
            baseline: Optional[float] = None, candidate: Optional[float] = None) -> dict:
    return {
        "trained_at": datetime.now(KST).isoformat(timespec="seconds"),
        "received_count": received,
        "used_count": used,
        "baseline_mae": None if baseline is None else round(baseline, 2),
        "candidate_mae": None if candidate is None else round(candidate, 2),
        "replaced": replaced,
        "reason": reason,
    }


def _run(records: Sequence[TrainingRecord]) -> dict:
    X_real, y_real = build_dataset(records)
    used = len(y_real)
    if used == 0:
        return _result(len(records), used, False, REASON_NO_RECORDS)
    if used < 2:
        return _result(len(records), used, False, REASON_NOT_ENOUGH_FOR_VALIDATION)

    X_synthetic, y_synthetic = synthetic.generate(config.SYNTHETIC_ROWS, config.SYNTHETIC_SEED)
    baseline = baseline_mae(X_synthetic, y_synthetic, X_real, y_real)
    candidate = cross_validated_mae(X_synthetic, y_synthetic, X_real, y_real)
    # 오차가 같으면 나아진 것이 아니므로 교체하지 않는다
    if candidate >= baseline:
        return _result(len(records), used, False, REASON_NOT_IMPROVED, baseline, candidate)

    # 검증을 통과하면 모든 실제 기록을 넣어 다시 학습한 모델을 쓴다
    _replace_model(_fit_mixed(X_synthetic, y_synthetic, X_real, y_real))
    return _result(len(records), used, True, REASON_IMPROVED, baseline, candidate)


def run(records: Sequence[TrainingRecord]) -> dict:
    """재학습을 한 번 실행하고 결과를 이력에 남긴다."""
    with _lock:
        try:
            result = _run(records)
        except Exception:
            # 실패한 실행도 이력에 남긴다. 서비스 중인 모델은 그대로다
            _append_history(_result(len(records), 0, False, REASON_FAILED))
            raise
        _append_history(result)
    logger.info("재학습: %s", result)
    return result

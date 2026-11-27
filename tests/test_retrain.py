import os
from datetime import datetime, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import config, features, model, retrain
from app.main import app
from app.schemas import TrainingRecord

FIRST_RUN = datetime(2026, 11, 1, 7, 0)
INTERNAL_KEY = "test-internal-key"
AUTH = {"X-Internal-Api-Key": INTERNAL_KEY}


def record(user_id=1, days=0, distance_km=5.0, duration_sec=1900, hr=152, cadence=165,
           elevation_gain_m=40.0, age=25, gender="MALE"):
    return TrainingRecord(
        user_id=user_id, age=age, gender=gender, start_time=FIRST_RUN + timedelta(days=days),
        distance_km=distance_km, duration_sec=duration_sec, avg_heart_rate=hr, cadence=cadence,
        elevation_gain_m=elevation_gain_m,
    )


@pytest.fixture(autouse=True)
def retrain_settings(monkeypatch, tmp_path):
    """테스트는 빠른 Ridge 와 작은 합성 데이터를 쓰고, 이력은 임시 폴더에 남긴다."""
    monkeypatch.setattr(config, "MODEL_KIND", "ridge")
    monkeypatch.setattr(config, "SYNTHETIC_ROWS", 1500)
    monkeypatch.setattr(config, "TRAINING_LOG_PATH", str(tmp_path / "trainings.jsonl"))
    monkeypatch.setattr(config, "INTERNAL_API_KEY", INTERNAL_KEY)


@pytest.fixture
def isolated_model(monkeypatch, tmp_path, trained_model):
    """모델을 교체하는 테스트가 다른 테스트의 모델을 바꾸지 않게, 복사본을 쓰고 끝나면 되돌린다."""
    import shutil
    path = str(tmp_path / "hr_model.joblib")
    shutil.copy2(trained_model, path)
    monkeypatch.setattr(config, "MODEL_PATH", path)
    yield path
    model.load(trained_model)


# ----- 기록 선별 -----

def test_plausible_record_is_usable():
    assert retrain.is_usable(record())


@pytest.mark.parametrize("changes", [
    {"hr": None},                               # 심박이 없으면 정답을 만들 수 없다
    {"age": None},                              # 나이를 모르면 최대 심박 대비 비율을 계산할 수 없다
    {"distance_km": 0.9, "duration_sec": 350},  # 1km 미만
    {"duration_sec": 0},
    {"duration_sec": 800},                      # 2'40"/km: 너무 빠름
    {"duration_sec": 3700},                     # 12'20"/km: 너무 느림
    {"hr": 70},                                 # 최대 심박의 36%
    {"hr": 210},                                # 최대 심박의 108%
    {"cadence": 90},
    {"cadence": 260},
    {"age": 5},
])
def test_out_of_range_record_is_not_usable(changes):
    assert not retrain.is_usable(record(**changes))


def test_record_without_cadence_is_usable():
    assert retrain.is_usable(record(cadence=None))


# ----- 학습 데이터 구성 -----

def test_recent_features_come_from_earlier_records_of_the_same_user():
    records = [
        record(days=0, distance_km=4.0, duration_sec=1600, cadence=160),
        record(days=2, distance_km=6.0, duration_sec=2400, cadence=170),
        record(days=4, distance_km=5.0, duration_sec=1900),
        record(user_id=2, days=1, distance_km=3.0, duration_sec=1380),  # 다른 사용자의 기록은 섞이지 않는다
    ]
    X, y = retrain.build_dataset(records)

    assert X.shape == (4, len(features.FEATURE_NAMES))
    pace, distance, cadence = (features.FEATURE_NAMES.index(name) for name in
                               ("recent_pace_sec_per_km", "recent_distance_km", "recent_cadence"))
    first, _, third, other_user = X
    # 첫 기록은 그 전 기록이 없어 평소 값이 비어 있다
    assert np.isnan(first[pace]) and np.isnan(first[distance])
    # 세 번째 기록의 평소 값은 앞의 두 기록으로 계산한다: (1600+2400)초 ÷ (4+6)km
    assert third[pace] == pytest.approx(400)
    assert third[distance] == pytest.approx(5.0)
    assert third[cadence] == pytest.approx(165)
    assert np.isnan(other_user[pace])
    assert y[0] == pytest.approx(152 / 195 * 100)


def test_dataset_does_not_depend_on_input_order():
    records = [record(days=day, distance_km=4.0 + day, duration_sec=400 * (4 + day)) for day in range(4)]
    X_sorted, y_sorted = retrain.build_dataset(records)
    X_shuffled, y_shuffled = retrain.build_dataset(list(reversed(records)))

    assert np.array_equal(X_sorted, X_shuffled, equal_nan=True)
    assert np.array_equal(y_sorted, y_shuffled)


def test_dataset_does_not_depend_on_order_across_users():
    # 사용자가 여럿일 때도 행 순서가 같아야 교차 검증의 묶음 구성이 실행마다 같다
    records = [record(user_id=user_id, days=day, hr=150 + user_id + day)
               for user_id in (3, 1, 2) for day in range(3)]
    X_first, y_first = retrain.build_dataset(records)
    X_second, y_second = retrain.build_dataset(list(reversed(records)))

    assert np.array_equal(X_first, X_second, equal_nan=True)
    assert np.array_equal(y_first, y_second)
    assert list(y_first[:3]) == [features.hr_pct(151 + day, 25) for day in range(3)]   # 사용자 1부터


def test_records_with_the_same_start_time_are_not_each_others_history():
    earlier = record(days=0, distance_km=4.0, duration_sec=1600)
    same_time = [record(days=1, distance_km=5.0, duration_sec=1900), record(days=1, distance_km=8.0, duration_sec=3200)]
    pace = features.FEATURE_NAMES.index("recent_pace_sec_per_km")

    for ordered in (same_time, list(reversed(same_time))):
        X, _ = retrain.build_dataset([earlier] + ordered)
        # 같은 시각의 두 기록은 모두 그 전 기록(400초/km)만 평소 값으로 본다
        assert X[1][pace] == pytest.approx(400) and X[2][pace] == pytest.approx(400)


def test_start_times_with_and_without_offset_can_be_mixed():
    # 07:00+09:00 과 시간대 없는 07:00 은 같은 한국 시각이다. 섞여 있어도 정렬에서 오류가 나지 않는다
    with_offset = TrainingRecord(
        user_id=1, age=25, gender="MALE", start_time="2026-11-02T07:00:00+09:00",
        distance_km=5.0, duration_sec=1900, avg_heart_rate=152)
    utc = TrainingRecord(
        user_id=1, age=25, gender="MALE", start_time="2026-11-02T22:00:00Z",
        distance_km=5.0, duration_sec=1900, avg_heart_rate=152)

    assert with_offset.start_time == datetime(2026, 11, 2, 7, 0)
    assert utc.start_time == datetime(2026, 11, 3, 7, 0)
    _, y = retrain.build_dataset([record(days=0), with_offset, utc])
    assert len(y) == 3


def test_negative_elevation_gain_is_treated_as_unknown():
    X, y = retrain.build_dataset([record(elevation_gain_m=-5.0)])

    assert len(y) == 1
    assert np.isnan(X[0][features.FEATURE_NAMES.index("climb_m_per_km")])


def test_only_the_last_five_earlier_records_count():
    # 처음 2회는 10km, 이후 5회는 4km. 여덟 번째 기록의 평소 거리는 직전 5회(4km)만 본다
    records = [record(days=day, distance_km=10.0, duration_sec=4000) for day in range(2)]
    records += [record(days=day, distance_km=4.0, duration_sec=1600) for day in range(2, 7)]
    records.append(record(days=7))
    X, _ = retrain.build_dataset(records)

    assert X[-1][features.FEATURE_NAMES.index("recent_distance_km")] == pytest.approx(4.0)


def test_unusable_record_still_counts_as_earlier_record():
    # 심박이 없는 기록은 정답으로는 못 쓰지만, 다음 기록의 "평소 페이스"에는 들어간다 (추천 요청과 같은 기준)
    records = [record(days=0, hr=None, distance_km=5.0, duration_sec=2000), record(days=1)]
    X, y = retrain.build_dataset(records)

    assert len(y) == 1
    assert X[0][features.FEATURE_NAMES.index("recent_pace_sec_per_km")] == pytest.approx(400)


def test_missing_course_values_are_nan():
    X, _ = retrain.build_dataset([record(elevation_gain_m=None)])

    assert np.isnan(X[0][features.FEATURE_NAMES.index("climb_m_per_km")])
    assert np.isnan(X[0][features.FEATURE_NAMES.index("max_slope_pct")])


# ----- 재학습 실행 -----

def many_records(count=12):
    return [record(user_id=1 + day % 3, days=day, hr=148 + day % 9) for day in range(count)]


def test_no_usable_records_keeps_the_model(isolated_model):
    result = retrain.run([record(hr=None)])

    assert (result["received_count"], result["used_count"]) == (1, 0)
    assert result["replaced"] is False
    assert result["reason"] == retrain.REASON_NO_RECORDS
    assert not os.path.exists(isolated_model + ".prev")


def test_single_record_cannot_be_validated(isolated_model):
    result = retrain.run([record()])

    assert result["used_count"] == 1
    assert result["replaced"] is False
    assert result["reason"] == retrain.REASON_NOT_ENOUGH_FOR_VALIDATION


def test_model_is_replaced_when_validation_error_improves(monkeypatch, isolated_model):
    monkeypatch.setattr(retrain, "baseline_mae", lambda *_: 5.0)
    monkeypatch.setattr(retrain, "cross_validated_mae", lambda *_: 3.0)

    result = retrain.run(many_records())

    assert result["replaced"] is True
    assert (result["baseline_mae"], result["candidate_mae"]) == (5.0, 3.0)
    assert os.path.exists(isolated_model + ".prev")        # 이전 모델을 보관한다
    assert not os.path.exists(isolated_model + ".tmp")
    # 새 모델로 바로 예측할 수 있다
    row = features.build_row([25, 1, 380, 5, 165], [5, 10, 6])
    assert 40 <= model.predict_hr_pct([row])[0] <= 100


def test_model_is_kept_when_validation_error_gets_worse(monkeypatch, isolated_model):
    monkeypatch.setattr(retrain, "baseline_mae", lambda *_: 3.0)
    monkeypatch.setattr(retrain, "cross_validated_mae", lambda *_: 3.5)

    result = retrain.run(many_records())

    assert result["replaced"] is False
    assert result["reason"] == retrain.REASON_NOT_IMPROVED
    assert not os.path.exists(isolated_model + ".prev")


def test_model_is_kept_when_validation_error_is_the_same(monkeypatch, isolated_model):
    monkeypatch.setattr(retrain, "baseline_mae", lambda *_: 3.0)
    monkeypatch.setattr(retrain, "cross_validated_mae", lambda *_: 3.0)

    result = retrain.run(many_records())

    assert result["replaced"] is False
    assert result["reason"] == retrain.REASON_NOT_IMPROVED


def test_failed_run_is_recorded_and_keeps_the_model(monkeypatch, isolated_model):
    def fail(*_):
        raise RuntimeError("학습 실패")
    monkeypatch.setattr(retrain, "baseline_mae", fail)

    with pytest.raises(RuntimeError):
        retrain.run(many_records())

    assert retrain.history()[0]["reason"] == retrain.REASON_FAILED
    assert not os.path.exists(isolated_model + ".prev")


def test_unreadable_history_line_is_skipped(isolated_model):
    retrain.run([record()])
    with open(config.TRAINING_LOG_PATH, "a", encoding="utf-8") as file:
        file.write("{끊긴 줄\n")

    assert len(retrain.history()) == 1


def test_validation_runs_end_to_end(isolated_model):
    """실제 계산 경로(합성 데이터 생성, 가중치 학습, 교차 검증)가 끝까지 돈다."""
    result = retrain.run(many_records())

    assert result["used_count"] == 12
    assert result["baseline_mae"] > 0 and result["candidate_mae"] > 0
    assert result["replaced"] == (result["candidate_mae"] <= result["baseline_mae"])


def test_history_lists_latest_first(isolated_model):
    retrain.run([record(hr=None)])
    retrain.run([record()])

    reasons = [item["reason"] for item in retrain.history()]
    assert reasons == [retrain.REASON_NOT_ENOUGH_FOR_VALIDATION, retrain.REASON_NO_RECORDS]


def test_history_is_empty_before_first_run():
    assert retrain.history() == []


# ----- API -----

# with 블록 없이 만들면 앱의 시작·종료 절차(DB 연결 풀 열기)가 돌지 않는다. 재학습 API는 DB를 쓰지 않고,
# 연결 풀은 한 번 닫히면 다시 열 수 없어 다른 테스트 모듈과 충돌한다. 모델은 conftest 가 이미 불러 두었다
client = TestClient(app)

TRAINING_BODY = {"records": [{
    "userId": 1, "age": 25, "gender": "MALE", "startTime": "2026-11-20T07:00:00",
    "distanceKm": 5.0, "durationSec": 1900, "avgHeartRate": 152, "cadence": 165, "elevationGainM": 40,
}]}


def test_training_api_uses_camel_case(isolated_model):
    response = client.post("/model/trainings", json=TRAINING_BODY, headers=AUTH)
    history = client.get("/model/trainings", headers=AUTH)

    assert response.status_code == 200
    result = response.json()
    assert result["receivedCount"] == 1 and result["usedCount"] == 1
    assert result["replaced"] is False
    assert history.json()["trainings"][0]["trainedAt"] == result["trainedAt"]


@pytest.mark.parametrize("headers", [{}, {"X-Internal-Api-Key": "wrong-key"}])
def test_training_api_requires_the_internal_key(isolated_model, headers):
    created = client.post("/model/trainings", json=TRAINING_BODY, headers=headers)
    listed = client.get("/model/trainings", headers=headers)

    assert created.status_code == 401 and listed.status_code == 401
    assert created.json()["code"] == "INVALID_INTERNAL_KEY"
    assert retrain.history() == []      # 거절된 요청은 학습을 시작하지 않는다


def test_training_api_is_closed_when_the_server_has_no_key(monkeypatch, isolated_model):
    monkeypatch.setattr(config, "INTERNAL_API_KEY", "")
    response = client.post("/model/trainings", json=TRAINING_BODY, headers=AUTH)

    assert response.status_code == 503
    assert response.json()["code"] == "TRAINING_DISABLED"


def test_training_api_rejects_non_finite_numbers(isolated_model):
    body = {"records": [dict(TRAINING_BODY["records"][0], distanceKm="NaN")]}
    response = client.post("/model/trainings", json=body, headers=AUTH)

    assert response.status_code == 422

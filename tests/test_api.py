"""API 테스트. 코스 생성 테스트는 전처리가 끝난 로컬 DB가 있어야 하고, 없으면 건너뛴다."""
import re

import psycopg
import pytest
from fastapi.testclient import TestClient

from app import config, routing
from app.db import pool
from app.main import app

DAEJEON_UNIV = {"lat": 36.3321, "lng": 127.4562}


def _has_routing_db() -> bool:
    try:
        with psycopg.connect(config.DATABASE_URL, connect_timeout=2) as conn:
            return conn.execute("SELECT to_regclass('routing.ways')").fetchone()[0] is not None
    except psycopg.Error:
        return False


needs_db = pytest.mark.skipif(not _has_routing_db(), reason="routing 스키마가 있는 로컬 DB 필요")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_out_of_service_area_returns_400(client):
    body = {"start": {"lat": 37.5665, "lng": 126.9780}, "targetDistanceKm": 5, "intensity": "NORMAL"}
    response = client.post("/recommend", json=body)
    assert response.status_code == 400
    assert response.json()["code"] == "OUT_OF_SERVICE_AREA"


@pytest.mark.parametrize("body", [
    {"start": DAEJEON_UNIV, "targetDistanceKm": 0.5, "intensity": "NORMAL"},   # 너무 짧음
    {"start": DAEJEON_UNIV, "targetDistanceKm": 50, "intensity": "NORMAL"},    # 너무 김
    {"start": DAEJEON_UNIV, "targetDistanceKm": 5, "intensity": "VERY_HARD"},  # 없는 강도
    {"targetDistanceKm": 5, "intensity": "NORMAL"},                            # 출발 좌표 없음
])
def test_invalid_request_returns_422(client, body):
    assert client.post("/recommend", json=body).status_code == 422


@needs_db
@pytest.mark.parametrize("lat,lng", [
    (36.3321, 127.4562), (36.3323, 127.4344), (36.3105, 127.4720), (36.3500, 127.4400), (36.2900, 127.4600),
    (36.3700, 127.4800), (36.3200, 127.4450), (36.3400, 127.4650), (36.3050, 127.4500), (36.3600, 127.4300),
])
def test_nearest_node_is_closest_in_meters(client, lat, lng):
    # 도(degree) 기준이 아니라 미터 기준으로 가장 가까운 노드여야 한다 (전체를 훑어 구한 정답과 비교)
    with pool.connection() as conn:
        node, distance_m = routing.nearest_node(conn, lat, lng)
        expected = conn.execute(
            """
            SELECT id, ST_Distance(the_geom::geography, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography) AS d
            FROM routing.ways_vertices_pgr ORDER BY d LIMIT 1
            """, (lng, lat)).fetchone()
    assert node == expected[0]
    assert distance_m == pytest.approx(expected[1])


@needs_db
def test_no_usable_course_returns_404(client):
    # 대청호 한가운데 근처: 도로가 드물어 5km 순환 코스를 만들 수 없다
    body = {"start": {"lat": 36.405, "lng": 127.515}, "targetDistanceKm": 5, "intensity": "NORMAL"}
    response = client.post("/recommend", json=body)
    assert response.status_code == 404
    assert response.json()["code"] == "COURSE_NOT_FOUND"


def _average_predicted(client, intensity, records):
    body = {"start": DAEJEON_UNIV, "targetDistanceKm": 5, "intensity": intensity,
            "user": {"userId": 1, "age": 24, "gender": "MALE"}, "recentRecords": records}
    courses = client.post("/recommend", json=body).json()["courses"]
    return sum(c["predictedHrPct"] for c in courses) / len(courses)


@needs_db
def test_hard_recommends_higher_predicted_intensity_than_easy(client):
    records = [{"distanceKm": 5, "durationSec": 1800, "cadence": 168}]
    assert _average_predicted(client, "HARD", records) > _average_predicted(client, "EASY", records)


@needs_db
def test_slower_runner_gets_higher_predicted_intensity_on_same_request(client):
    fast = [{"distanceKm": 5, "durationSec": 1400, "cadence": 180}]   # 4'40"/km
    slow = [{"distanceKm": 5, "durationSec": 2300, "cadence": 158}]   # 7'40"/km
    assert _average_predicted(client, "EASY", slow) > _average_predicted(client, "EASY", fast) + 5


@needs_db
@pytest.mark.parametrize("target_km", [3, 5, 10])
def test_recommend_returns_distinct_loop_courses(client, target_km):
    body = {
        "start": DAEJEON_UNIV, "targetDistanceKm": target_km, "intensity": "NORMAL",
        "user": {"userId": 1, "age": 24, "gender": "MALE"},
        "recentRecords": [{"distanceKm": 5.2, "durationSec": 1890, "avgHeartRate": 158}],
    }
    response = client.post("/recommend", json=body)
    assert response.status_code == 200
    courses = response.json()["courses"]

    assert len(courses) == config.TOP_N
    assert [c["rank"] for c in courses] == [1, 2, 3]
    best = courses[0]
    assert 50 <= best["predictedHrPct"] <= 100
    assert "예상 운동 강도" in best["reason"]
    # 응답값이 정수여야 화면과 문장이 같은 숫자를 쓴다
    assert best["elevationGainM"] == int(best["elevationGainM"])
    assert best["predictedHrPct"] == int(best["predictedHrPct"])
    # 카드에 보이는 숫자와 문장 속 숫자가 같아야 한다
    assert f"누적 오르막은 {best['elevationGainM']:.0f}m" in best["reason"]
    assert f"최대 심박의 {best['predictedHrPct']:.0f}%" in best["reason"]
    assert "평균 페이스" in best["reason"]

    # 순위가 높을수록 추천 점수가 높거나 같다
    scores = [course["score"] for course in courses]
    assert scores == sorted(scores, reverse=True)
    assert len({course["name"] for course in courses}) == len(courses)

    for course in courses:
        assert abs(course["distanceKm"] - target_km) / target_km <= config.MAX_DISTANCE_ERROR
        assert re.search(r"순환( \d+)?$", course["name"])   # "용운로 순환", "용운로 동쪽 순환 2"
        assert 0 <= course["score"] <= 100
        assert course["estimatedTimeMin"] > 0                      # 최근 기록을 보냈으므로 값이 있다
        assert course["stairsCount"] >= 0
        assert course["hrZone"] in ("Z1", "Z2", "Z3", "Z4", "Z5")
        low_bpm, high_bpm = course["hrBpmRange"]                   # 나이를 보냈으므로 값이 있다
        assert 80 < low_bpm < high_bpm < 220
        assert 0 <= course["maxSlopePct"] < 30

        # 경사 등급 구간을 이으면 전체 경로와 같은 출발·도착점을 가진다
        segments = course["slopeSegments"]
        assert all(segment["grade"] in ("FLAT", "MODERATE", "STEEP") for segment in segments)
        assert segments[0]["coordinates"][0] == course["geometry"]["coordinates"][0]
        assert segments[-1]["coordinates"][-1] == course["geometry"]["coordinates"][-1]
        for previous, following in zip(segments, segments[1:]):
            assert previous["coordinates"][-1] == following["coordinates"][0]

        profile = course["elevationProfile"]
        distances = [point["distanceM"] for point in profile]
        assert 2 <= len(profile) <= config.ELEVATION_PROFILE_MAX_POINTS
        assert distances[0] == 0
        assert distances == sorted(distances)
        # 프로필의 끝 거리는 코스 총 거리와 같아야 한다 (반올림 오차 허용)
        assert distances[-1] == pytest.approx(course["distanceKm"] * 1000, rel=0.02)
        # 순환 코스라 출발점과 도착점의 고도가 같다
        assert profile[0]["elevationM"] == pytest.approx(profile[-1]["elevationM"], abs=0.5)
        coordinates = course["geometry"]["coordinates"]
        assert course["geometry"]["type"] == "LineString"
        # 순환 코스: 출발점으로 돌아온다
        assert coordinates[0] == coordinates[-1]
        # 출발점이 요청 좌표 근처 (약 300m 이내)
        assert abs(coordinates[0][0] - DAEJEON_UNIV["lng"]) < 0.004
        assert abs(coordinates[0][1] - DAEJEON_UNIV["lat"]) < 0.003

"""API 테스트. 코스 생성 테스트는 전처리가 끝난 로컬 DB가 있어야 하고, 없으면 건너뛴다."""
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


@needs_db
def test_hard_recommends_more_climb_than_easy(client):
    def average_climb(intensity):
        body = {"start": DAEJEON_UNIV, "targetDistanceKm": 5, "intensity": intensity}
        courses = client.post("/recommend", json=body).json()["courses"]
        return sum(c["elevationGainM"] / c["distanceKm"] for c in courses) / len(courses)

    assert average_climb("HARD") > average_climb("EASY")


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
    assert best["predictedHrPct"] is None
    assert "평균 페이스" in best["reason"]

    for course in courses:
        assert abs(course["distanceKm"] - target_km) / target_km <= config.MAX_DISTANCE_ERROR
        coordinates = course["geometry"]["coordinates"]
        assert course["geometry"]["type"] == "LineString"
        # 순환 코스: 출발점으로 돌아온다
        assert coordinates[0] == coordinates[-1]
        # 출발점이 요청 좌표 근처 (약 300m 이내)
        assert abs(coordinates[0][0] - DAEJEON_UNIV["lng"]) < 0.004
        assert abs(coordinates[0][1] - DAEJEON_UNIV["lat"]) < 0.003

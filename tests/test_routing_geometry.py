import math

import pytest

from app import config, routing

START = (36.3321, 127.4562)


def distance_m(a, b):
    """두 (lat, lng) 사이 거리. 테스트용 하버사인."""
    lat1, lng1, lat2, lng2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lng2 - lng1) / 2) ** 2
    return 2 * 6_371_000 * math.asin(math.sqrt(h))


@pytest.mark.parametrize("bearing", [0, 45, 90, 180, 270])
def test_offset_point_moves_requested_distance(bearing):
    moved = routing.offset_point(START[0], START[1], bearing, 1000)
    assert distance_m(START, moved) == pytest.approx(1000, rel=0.01)


def test_offset_point_direction():
    north = routing.offset_point(START[0], START[1], 0, 500)
    east = routing.offset_point(START[0], START[1], 90, 500)
    assert north[0] > START[0] and north[1] == pytest.approx(START[1])
    assert east[1] > START[1] and east[0] == pytest.approx(START[0])


@pytest.mark.parametrize("bearing", config.CANDIDATE_BEARINGS_DEG)
def test_loop_waypoints_form_equilateral_triangle(bearing):
    radius = 800
    a, b = routing.loop_waypoints(START[0], START[1], bearing, radius)
    side = math.sqrt(3) * radius
    assert distance_m(START, a) == pytest.approx(side, rel=0.01)
    assert distance_m(a, b) == pytest.approx(side, rel=0.01)
    assert distance_m(b, START) == pytest.approx(side, rel=0.01)


def test_initial_radius_makes_triangle_match_target():
    target_m = 5000
    radius = routing.initial_radius_m(target_m)
    assert 3 * math.sqrt(3) * radius * config.DETOUR_FACTOR == pytest.approx(target_m)


def test_search_box_contains_waypoints_even_after_radius_retry():
    target_m = 5000
    box = routing.search_box(START[0], START[1], target_m)
    for bearing in config.CANDIDATE_BEARINGS_DEG:
        for point in routing.loop_waypoints(START[0], START[1], bearing, routing.initial_radius_m(target_m) * 1.8):
            assert box["south"] <= point[0] <= box["north"]
            assert box["west"] <= point[1] <= box["east"]


@pytest.mark.parametrize("lat,lng,expected", [
    (36.3321, 127.4562, True),    # 대전대
    (37.5665, 126.9780, False),   # 서울시청
    (36.3504, 127.3845, False),   # 대전시청 (서구)
])
def test_is_in_service_area(lat, lng, expected):
    assert routing.is_in_service_area(lat, lng) is expected

import json

import pytest

from app import detail


def row(length_m, name, is_steps, up_and_down_m, coordinates):
    return (length_m, name, is_steps, up_and_down_m, json.dumps({"type": "LineString", "coordinates": coordinates}))


@pytest.mark.parametrize("up_and_down_m,expected", [
    (2.0, detail.FLAT),       # 100m에 2m → 2%
    (4.9, detail.FLAT),
    (5.0, detail.MODERATE),   # 5%
    (9.9, detail.MODERATE),
    (10.0, detail.STEEP),     # 10%
])
def test_slope_grade_boundaries(up_and_down_m, expected):
    assert detail.slope_grade(100, up_and_down_m) == expected


def test_main_road_is_the_longest_named_road():
    info = detail.build_detail([
        row(300, "용운로", False, 1, [[0, 0], [1, 0]]),
        row(500, "대학로", False, 1, [[1, 0], [2, 0]]),
        row(400, "용운로", False, 1, [[2, 0], [3, 0]]),   # 용운로 합계 700m
        row(900, None, False, 1, [[3, 0], [4, 0]]),      # 이름 없는 길은 세지 않는다
    ])
    assert info.main_road == "용운로"


def test_main_road_is_none_without_named_roads():
    info = detail.build_detail([row(300, None, False, 1, [[0, 0], [1, 0]])])
    assert info.main_road is None


def test_consecutive_stairs_count_as_one():
    info = detail.build_detail([
        row(100, None, False, 1, [[0, 0], [1, 0]]),
        row(10, None, True, 3, [[1, 0], [2, 0]]),
        row(10, None, True, 3, [[2, 0], [3, 0]]),    # 이어진 계단
        row(100, None, False, 1, [[3, 0], [4, 0]]),
        row(10, None, True, 3, [[4, 0], [5, 0]]),    # 떨어진 계단
    ])
    assert info.stairs_count == 2


def test_slope_segments_merge_same_grade_and_stay_connected():
    info = detail.build_detail([
        row(100, None, False, 1, [[0, 0], [1, 0]]),          # 완만
        row(100, None, False, 2, [[1, 0], [2, 0]]),          # 완만 → 앞과 합친다
        row(100, None, False, 12, [[2, 0], [3, 0], [4, 0]]), # 급경사
    ])
    assert [segment.grade for segment in info.slope_segments] == [detail.FLAT, detail.STEEP]
    assert info.slope_segments[0].coordinates == [[0, 0], [1, 0], [2, 0]]
    # 다음 구간은 앞 구간이 끝난 점에서 시작해야 지도에서 끊기지 않는다
    assert info.slope_segments[1].coordinates[0] == info.slope_segments[0].coordinates[-1]


@pytest.mark.parametrize("bearing,expected", [(0, "북쪽"), (45, "북동쪽"), (180, "남쪽"), (337.5, "북쪽"), (292.5, "북서쪽"), (22.4, "북쪽")])
def test_compass(bearing, expected):
    assert detail.compass(bearing) == expected


def test_course_name_uses_main_road():
    assert detail.course_name("용운로", 90, []) == "용운로 순환"


def test_course_name_adds_direction_when_name_is_taken():
    assert detail.course_name("용운로", 90, ["용운로 순환"]) == "용운로 동쪽 순환"


def test_course_name_adds_number_when_road_and_direction_are_taken():
    used = ["용운로 순환", "용운로 동쪽 순환"]
    assert detail.course_name("용운로", 90, used) == "용운로 동쪽 순환 2"


def test_course_name_without_named_road_uses_direction():
    assert detail.course_name(None, 180, []) == "남쪽 방향 순환"

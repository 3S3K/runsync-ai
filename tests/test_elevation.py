import pytest

from app import elevation


def rows(order, length_m, zs):
    """한 구간의 표본 행들 (구간 순번, 구간 길이, 구간 안에서의 거리, 고도). 표본은 고른 간격이다."""
    seg_len = length_m / (len(zs) - 1)
    return [(order, length_m, index * seg_len, z) for index, z in enumerate(zs)]


def test_build_profile_accumulates_distance_along_one_edge():
    profile = elevation.build_profile(rows(1, 60, [10.0, 11.0, 13.0]))
    assert profile == [(0, 10.0), (30, 11.0), (60, 13.0)]


def test_build_profile_joins_edges_without_repeating_shared_node():
    # 첫 구간의 끝(12m)과 둘째 구간의 시작(12m)은 같은 지점이다
    profile = elevation.build_profile(rows(1, 60, [10.0, 11.0, 12.0]) + rows(2, 40, [12.0, 15.0]))
    assert profile == [(0, 10.0), (30, 11.0), (60, 12.0), (100, 15.0)]


def test_build_profile_keeps_distance_for_edge_without_samples():
    # 가운데 구간은 고도 표본이 없다(DEM 범위 밖). 점은 없어도 거리는 흘러야 한다
    no_samples = [(2, 50, None, None)]
    profile = elevation.build_profile(rows(1, 30, [10.0, 11.0]) + no_samples + rows(3, 20, [14.0, 15.0]))
    # 셋째 구간의 첫 표본(80m)은 앞 구간에 점이 없었으므로 중복이 아니다
    assert profile == [(0, 10.0), (30, 11.0), (80, 14.0), (100, 15.0)]


def test_build_profile_keeps_edge_length_when_some_samples_are_missing():
    # 60m 구간의 가운데 표본이 빠졌다. 구간 끝은 30m가 아니라 60m여야 하고, 다음 구간도 60m에서 시작한다
    profile = elevation.build_profile(rows(1, 60, [10.0, None, 12.0]) + rows(2, 40, [12.0, 15.0]))
    assert profile == [(0, 10.0), (60, 12.0), (100, 15.0)]


def test_build_profile_places_remaining_sample_at_its_real_position():
    # 구간의 첫 표본이 빠져도 남은 표본은 구간 안의 실제 위치(30m, 60m)에 놓인다
    profile = elevation.build_profile(rows(1, 60, [None, 11.0, 12.0]))
    assert profile == [(30, 11.0), (60, 12.0)]


def test_build_profile_with_no_rows():
    assert elevation.build_profile([]) == []


def test_downsample_keeps_short_profile_as_is():
    points = [(i * 30.0, float(i)) for i in range(10)]
    assert elevation.downsample(points, 100) == points


def test_downsample_keeps_first_and_last_and_limits_count():
    points = [(i * 30.0, float(i)) for i in range(501)]
    result = elevation.downsample(points, 100)
    assert len(result) == 100
    assert result[0] == points[0]
    assert result[-1] == points[-1]
    distances = [distance for distance, _ in result]
    assert distances == sorted(distances)

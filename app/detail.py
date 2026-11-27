"""추천으로 뽑힌 코스의 상세 정보: 코스 이름, 계단 구간 수, 경사 등급별 구간.

경로의 구간(edge)을 지나는 순서대로 읽어 만든다. 후보 전체가 아니라 뽑힌 코스에만 계산한다.
"""
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from psycopg import Connection

from app import config
from app.routing import Candidate

_DETAIL_SQL = """
SELECT w.length_m, w.name, w.is_steps, w.ascent_m + w.descent_m,
       ST_AsGeoJSON(CASE WHEN s.fwd THEN w.the_geom ELSE ST_Reverse(w.the_geom) END, 6)
FROM unnest(%(edges)s::bigint[], %(forward)s::boolean[]) WITH ORDINALITY AS s(gid, fwd, ord)
JOIN routing.ways w ON w.gid = s.gid
ORDER BY s.ord
"""

FLAT, MODERATE, STEEP = "FLAT", "MODERATE", "STEEP"
_COMPASS = ("북쪽", "북동쪽", "동쪽", "남동쪽", "남쪽", "남서쪽", "서쪽", "북서쪽")


@dataclass
class SlopeSegment:
    grade: str                       # FLAT | MODERATE | STEEP
    coordinates: List[List[float]]   # [lng, lat]


@dataclass
class CourseDetail:
    main_road: Optional[str]         # 가장 길게 지나는 도로 이름. 이름 있는 도로가 없으면 None
    stairs_count: int                # 계단 구간 수 (이어진 계단은 한 번으로 센다)
    slope_segments: List[SlopeSegment]


def slope_grade(length_m: float, up_and_down_m: float) -> str:
    """구간의 평균 기울기(오르내린 높이 ÷ 길이)로 등급을 매긴다. 방향과 무관한 가파름이다."""
    steepness_pct = up_and_down_m / length_m * 100 if length_m > 0 else 0.0
    if steepness_pct >= config.SLOPE_GRADE_STEEP_PCT:
        return STEEP
    if steepness_pct >= config.SLOPE_GRADE_MODERATE_PCT:
        return MODERATE
    return FLAT


def build_detail(rows: Iterable[Sequence]) -> CourseDetail:
    """(길이, 도로 이름, 계단 여부, 오르내린 높이, GeoJSON) 행들을 지나는 순서대로 받아 정리한다."""
    road_length = defaultdict(float)
    segments: List[SlopeSegment] = []
    stairs_count = 0
    on_stairs = False
    for length_m, name, is_steps, up_and_down_m, geojson in rows:
        if name:
            road_length[name] += length_m
        if is_steps and not on_stairs:
            stairs_count += 1
        on_stairs = bool(is_steps)

        grade = slope_grade(length_m, up_and_down_m or 0.0)
        coordinates = json.loads(geojson)["coordinates"]
        if segments and segments[-1].grade == grade:
            segments[-1].coordinates.extend(coordinates[1:])  # 첫 점은 앞 구간의 끝과 같다
        else:
            segments.append(SlopeSegment(grade=grade, coordinates=list(coordinates)))

    main_road = max(road_length, key=road_length.get) if road_length else None
    return CourseDetail(main_road=main_road, stairs_count=stairs_count, slope_segments=segments)


def course_detail(conn: Connection, candidate: Candidate) -> CourseDetail:
    rows = conn.execute(
        _DETAIL_SQL, {"edges": candidate.edge_ids, "forward": candidate.edge_forward}).fetchall()
    return build_detail(rows)


def compass(bearing_deg: float) -> str:
    """방위각(북 0°, 시계 방향)을 8방위 이름으로. 경계(22.5° 등)는 시계 방향 쪽으로 넘긴다."""
    return _COMPASS[int((bearing_deg + 22.5) // 45) % 8]


def course_name(main_road: Optional[str], bearing_deg: float, used_names: Sequence[str]) -> str:
    """'용운로 순환'처럼 짓는다. 도로 이름이 없거나 이미 쓴 이름이면 방향을 붙여 구분한다."""
    direction = compass(bearing_deg)
    base = f"{main_road} 순환" if main_road else f"{direction} 방향 순환"
    options = [base, f"{main_road} {direction} 순환"] if main_road else [base]
    for name in options:
        if name not in used_names:
            return name
    # 같은 도로·같은 방향의 코스가 또 있으면 번호로 구분한다
    number = 2
    while f"{options[-1]} {number}" in used_names:
        number += 1
    return f"{options[-1]} {number}"

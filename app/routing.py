"""pgRouting으로 순환 코스 후보를 만든다.

출발지를 지나는 원을 그리고 그 원 위에 경유점 A, B를 둔다 (출발지·A·B가 정삼각형).
출발 → A → B → 출발을 pgr_dijkstraVia로 이으면 순환 코스가 된다.
원의 중심 방향을 일정 각도씩 돌려 서로 다른 방향의 후보를 만든다.
"""
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from psycopg import Connection

from app import config

M_PER_DEG_LAT = 111_320.0


@dataclass
class Candidate:
    bearing_deg: float
    distance_m: float
    ascent_m: float
    max_slope_pct: float
    overlap_ratio: float  # 같은 구간을 두 번 이상 지나는 거리의 비율
    edge_ids: List[int]
    coordinates: List[List[float]]  # [lng, lat]
    predicted_hr_pct: Optional[float] = None  # 심박 예측 모델이 채운다

    @property
    def climb_m_per_km(self) -> float:
        return self.ascent_m / (self.distance_m / 1000)


def is_in_service_area(lat: float, lng: float) -> bool:
    return (config.SERVICE_MIN_LAT <= lat <= config.SERVICE_MAX_LAT
            and config.SERVICE_MIN_LNG <= lng <= config.SERVICE_MAX_LNG)


def offset_point(lat: float, lng: float, bearing_deg: float, distance_m: float) -> Tuple[float, float]:
    """(lat, lng)에서 bearing 방향(북 0°, 시계 방향)으로 distance_m 떨어진 점. 수 km 범위라 평면 근사로 충분하다."""
    rad = math.radians(bearing_deg)
    dlat = distance_m * math.cos(rad) / M_PER_DEG_LAT
    dlng = distance_m * math.sin(rad) / (M_PER_DEG_LAT * math.cos(math.radians(lat)))
    return lat + dlat, lng + dlng


def initial_radius_m(target_m: float) -> float:
    """정삼각형 둘레(3√3 × 반지름)에 도로 우회 비율을 곱한 값이 목표 거리가 되도록 반지름을 잡는다."""
    return target_m / (3 * math.sqrt(3) * config.DETOUR_FACTOR)


def loop_waypoints(lat: float, lng: float, bearing_deg: float,
                   radius_m: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    center = offset_point(lat, lng, bearing_deg, radius_m)
    a = offset_point(center[0], center[1], bearing_deg - 60, radius_m)
    b = offset_point(center[0], center[1], bearing_deg + 60, radius_m)
    return a, b


def search_box(lat: float, lng: float, target_m: float) -> Dict[str, float]:
    half = target_m * config.SEARCH_BOX_RATIO
    south, west = offset_point(lat, lng, 225, half * math.sqrt(2))
    north, east = offset_point(lat, lng, 45, half * math.sqrt(2))
    return {"west": west, "south": south, "east": east, "north": north}


def nearest_node(conn: Connection, lat: float, lng: float) -> Tuple[int, float]:
    """가장 가까운 도로 노드의 id와 거리(m).

    geometry끼리의 <->는 도(degree) 단위라 동서 방향이 실제보다 약 1.24배 가깝게 계산된다.
    인덱스로 가까운 20개를 먼저 추린 뒤, 거리 제한과 같은 미터 기준으로 다시 정렬한다.
    (geography로 바로 정렬하면 정확하지만 쿼리당 20ms로 35배 느렸다)
    """
    row = conn.execute(
        """
        SELECT id, distance_m
        FROM (
            SELECT v.id, ST_Distance(v.the_geom::geography, p.geom::geography) AS distance_m
            FROM routing.ways_vertices_pgr v,
                 (SELECT ST_SetSRID(ST_MakePoint(%(lng)s, %(lat)s), 4326) AS geom) p
            ORDER BY v.the_geom <-> p.geom
            LIMIT 20
        ) nearest
        ORDER BY distance_m
        LIMIT 1
        """,
        {"lat": lat, "lng": lng},
    ).fetchone()
    return row[0], row[1]


# 구간 비용은 전처리에서 만들어 둔 컬럼에 k만 곱해 계산한다.
# pgr_dijkstraVia는 SQL 문자열을 받으므로, 값을 float8로 바인딩한 뒤 DB의 format()으로 끼워 넣는다.
_ROUTE_SQL = """
WITH route AS (
    SELECT seq, node, edge
    FROM pgr_dijkstraVia(
        format($q$
            SELECT gid AS id, source, target,
                   CASE WHEN is_steps THEN length_m * %%1$s
                        ELSE length_m + %%2$s * up_pen_fwd + %%3$s * down_len_fwd END AS cost,
                   CASE WHEN is_steps THEN length_m * %%1$s
                        ELSE length_m + %%2$s * up_pen_rev + %%3$s * down_len_rev END AS reverse_cost
            FROM routing.ways
            WHERE the_geom && ST_MakeEnvelope(%%4$s, %%5$s, %%6$s, %%7$s, 4326)
        $q$, %(steps)s::float8, %(k)s::float8, %(down)s::float8,
             %(west)s::float8, %(south)s::float8, %(east)s::float8, %(north)s::float8),
        %(via)s::bigint[], directed := true, strict := true)
    WHERE edge > 0
), steps AS (
    SELECT r.seq, w.gid, w.length_m,
           CASE WHEN r.node = w.source THEN w.ascent_m ELSE w.descent_m END AS ascent_m,
           CASE WHEN r.node = w.source THEN w.max_slope_fwd ELSE w.max_slope_rev END AS max_slope,
           CASE WHEN r.node = w.source THEN w.the_geom ELSE ST_Reverse(w.the_geom) END AS geom
    FROM route r
    JOIN routing.ways w ON w.gid = r.edge
)
SELECT sum(length_m),
       sum(ascent_m),
       max(max_slope),
       (SELECT sum(length_m) FROM (SELECT DISTINCT gid, length_m FROM steps) u),
       array_agg(gid ORDER BY seq),
       ST_AsGeoJSON(ST_RemoveRepeatedPoints(ST_MakeLine(geom ORDER BY seq)), 6)
FROM steps
"""


def _route(conn: Connection, bearing_deg: float, via: List[int], k: float,
           box: Dict[str, float]) -> Optional[Candidate]:
    params = {"steps": config.STEPS_COST_FACTOR, "k": k, "down": config.STEEP_DOWNHILL_EXTRA,
              "via": via, **box}
    distance_m, ascent_m, max_slope, unique_m, edge_ids, geojson = conn.execute(_ROUTE_SQL, params).fetchone()
    if not distance_m:
        return None
    return Candidate(
        bearing_deg=bearing_deg,
        distance_m=distance_m,
        ascent_m=ascent_m,
        max_slope_pct=max_slope,
        overlap_ratio=1 - unique_m / distance_m,
        edge_ids=edge_ids,
        coordinates=json.loads(geojson)["coordinates"],
    )


def _build_candidate(conn: Connection, start_node: int, lat: float, lng: float, bearing_deg: float,
                     target_m: float, k: float, box: Dict[str, float]) -> Optional[Candidate]:
    radius = initial_radius_m(target_m)
    best: Optional[Candidate] = None
    for _ in range(2):  # 첫 시도 + 반지름 조정 후 1회 재시도
        a, b = loop_waypoints(lat, lng, bearing_deg, radius)
        a_node, _ = nearest_node(conn, a[0], a[1])
        b_node, _ = nearest_node(conn, b[0], b[1])
        if len({start_node, a_node, b_node}) < 3:
            break
        candidate = _route(conn, bearing_deg, [start_node, a_node, b_node, start_node], k, box)
        if candidate is None:
            break
        if best is None or abs(candidate.distance_m - target_m) < abs(best.distance_m - target_m):
            best = candidate
        if abs(candidate.distance_m - target_m) / target_m <= config.DISTANCE_TOLERANCE:
            break
        # 탐색 범위를 벗어나지 않도록 조정 폭을 제한한다
        radius *= min(1.8, max(0.5, target_m / candidate.distance_m))
    return best


def generate_candidates(conn: Connection, start_node: int, lat: float, lng: float,
                        target_km: float, intensity: str) -> List[Candidate]:
    target_m = target_km * 1000
    k = config.SLOPE_AVOIDANCE_K[intensity]
    box = search_box(lat, lng, target_m)
    candidates = []
    for bearing_deg in config.CANDIDATE_BEARINGS_DEG:
        candidate = _build_candidate(conn, start_node, lat, lng, bearing_deg, target_m, k, box)
        if candidate is not None:
            candidates.append(candidate)
    return candidates

"""코스의 고도 프로필.

전처리에서 구간(edge)마다 약 30m 간격으로 읽어 둔 고도 표본(routing.edge_samples)을
경로의 구간 순서와 진행 방향에 맞춰 이어 붙인다. 추천으로 뽑힌 코스에만 계산한다.
"""
from itertools import groupby
from typing import Iterable, List, Sequence, Tuple

from psycopg import Connection

from app import config
from app.routing import Candidate

# 표본마다 "구간에 들어선 지점부터의 거리(offset_m)"를 구한다.
# 표본은 구간의 source 쪽이 seq 0이므로, 역방향으로 지나는 구간은 구간 길이에서 뺀다.
# 고도 표본이 없는 구간(DEM 범위 밖)도 거리는 더해야 하므로 LEFT JOIN으로 남긴다.
_PROFILE_SQL = """
SELECT s.ord, w.length_m, o.offset_m, e.z
FROM unnest(%(edges)s::bigint[], %(forward)s::boolean[]) WITH ORDINALITY AS s(gid, fwd, ord)
JOIN routing.ways w ON w.gid = s.gid
LEFT JOIN routing.edge_samples e ON e.gid = s.gid
CROSS JOIN LATERAL (
    SELECT CASE WHEN s.fwd THEN e.seq * e.seg_len ELSE w.length_m - e.seq * e.seg_len END AS offset_m
) o
ORDER BY s.ord, o.offset_m
"""

Point = Tuple[float, float]  # (출발점부터의 거리 m, 고도 m)


# 두 점의 거리가 이보다 가까우면 같은 지점으로 본다 (구간이 만나는 교차점)
_SAME_POINT_M = 0.01


def build_profile(rows: Iterable[Sequence]) -> List[Point]:
    """(구간 순번, 구간 길이, 구간 안에서의 거리, 고도) 행들을 거리-고도 점으로 잇는다.

    각 구간은 표본이 몇 개 남았든 구간 길이만큼 거리를 진행한다.
    표본이 일부 빠져도(DEM 경계) 남은 점의 거리와 뒤따르는 구간의 거리가 어긋나지 않는다.
    """
    points: List[Point] = []
    edge_start_m = 0.0
    for _, group in groupby(rows, key=lambda row: row[0]):
        edge_rows = list(group)
        for _, _, offset_m, z in edge_rows:
            if z is None:
                continue
            distance_m = edge_start_m + offset_m
            # 구간의 첫 표본은 앞 구간의 끝과 같은 지점이라 한 번만 넣는다
            if points and abs(distance_m - points[-1][0]) < _SAME_POINT_M:
                continue
            points.append((distance_m, z))
        edge_start_m += edge_rows[0][1]
    return points


def downsample(points: List[Point], max_points: int) -> List[Point]:
    """처음과 끝을 지키면서 고르게 솎아 낸다. 화면에 그릴 때 수백 개까지는 필요 없다."""
    if len(points) <= max_points:
        return points
    step = (len(points) - 1) / (max_points - 1)
    return [points[round(i * step)] for i in range(max_points)]


def elevation_profile(conn: Connection, candidate: Candidate) -> List[Point]:
    rows = conn.execute(
        _PROFILE_SQL, {"edges": candidate.edge_ids, "forward": candidate.edge_forward}).fetchall()
    return downsample(build_profile(rows), config.ELEVATION_PROFILE_MAX_POINTS)

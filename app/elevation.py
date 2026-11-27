"""코스의 고도 프로필.

전처리에서 구간(edge)마다 약 30m 간격으로 읽어 둔 고도 표본(routing.edge_samples)을
경로의 구간 순서와 진행 방향에 맞춰 이어 붙인다. 추천으로 뽑힌 코스에만 계산한다.
"""
from itertools import groupby
from typing import Iterable, List, Sequence, Tuple

from psycopg import Connection

from app import config
from app.routing import Candidate

# 표본은 구간의 source 쪽이 seq 0이다. 역방향으로 지나는 구간은 seq를 거꾸로 읽는다.
# 고도 표본이 없는 구간(DEM 범위 밖)도 거리는 더해야 하므로 LEFT JOIN으로 남긴다.
_PROFILE_SQL = """
SELECT s.ord, w.length_m, e.seg_len, e.z
FROM unnest(%(edges)s::bigint[], %(forward)s::boolean[]) WITH ORDINALITY AS s(gid, fwd, ord)
JOIN routing.ways w ON w.gid = s.gid
LEFT JOIN routing.edge_samples e ON e.gid = s.gid
ORDER BY s.ord, CASE WHEN s.fwd THEN e.seq ELSE -e.seq END
"""

Point = Tuple[float, float]  # (출발점부터의 거리 m, 고도 m)


def build_profile(rows: Iterable[Sequence]) -> List[Point]:
    """(구간 순번, 구간 길이, 표본 간격, 고도) 행들을 거리-고도 점으로 잇는다."""
    points: List[Point] = []
    distance_m = 0.0
    for _, group in groupby(rows, key=lambda row: row[0]):
        edge_rows = list(group)
        samples = [(seg_len, z) for _, _, seg_len, z in edge_rows if z is not None]
        if not samples:
            distance_m += edge_rows[0][1]
            continue
        for index, (seg_len, z) in enumerate(samples):
            if index > 0:
                distance_m += seg_len
            elif points:
                continue  # 구간의 첫 표본은 앞 구간의 끝과 같은 지점이다
            points.append((distance_m, z))
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

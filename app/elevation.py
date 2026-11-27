"""코스의 고도 프로필.

전처리에서 구간(edge)마다 약 30m 간격으로 읽어 둔 고도 표본(routing.edge_samples)을
경로의 구간 순서와 진행 방향에 맞춰 이어 붙인다. 추천으로 뽑힌 코스에만 계산한다.
"""
from itertools import groupby
from typing import Iterable, List, Optional, Sequence, Tuple

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


def max_slope_pct(points: List[Point], window_m: float) -> float:
    """window_m 이상 떨어진 두 점 사이에서 가장 가파른 오르막 경사(%). 오르막이 없으면 0.

    최소 길이를 처음 넘는 끝점만 보면, 그 뒤로 더 가팔라지는 구간(예: 0→100m 1%, 100→150m 10%)을
    놓친다. 그래서 시작점마다 window_m 이상인 끝점을 모두 본다. 점이 수백 개라 부담이 없다.
    """
    steepest = 0.0
    first_end = 0
    for start in range(len(points)):
        while first_end < len(points) and points[first_end][0] - points[start][0] < window_m:
            first_end += 1
        for end in range(first_end, len(points)):
            run_m = points[end][0] - points[start][0]
            steepest = max(steepest, (points[end][1] - points[start][1]) / run_m * 100)
    return steepest


def biggest_climb(points: List[Point], dip_tolerance_m: float) -> Optional[Tuple[float, float, float]]:
    """가장 많이 오르는 구간의 (시작 거리 m, 길이 m, 오르는 높이 m). 오르막이 없으면 None.

    길이가 아니라 오르는 높이로 고른다. 길고 완만한 오르막보다 짧아도 많이 오르는 구간이 더 힘들다.

    dip_tolerance_m까지 잠깐 내려갔다 다시 오르는 것은 같은 오르막으로 본다.
    """
    best: Optional[Tuple[float, float, float]] = None
    if not points:
        return None
    low = peak = points[0]
    for point in points[1:]:
        if point[1] > peak[1]:
            peak = point
        elif point[1] < peak[1] - dip_tolerance_m or point[1] < low[1]:
            # 정점에서 충분히 내려왔거나 시작점보다 낮아졌다 → 지금까지의 오르막을 마감한다
            if peak[1] - low[1] > 0 and (best is None or peak[1] - low[1] > best[2]):
                best = (low[0], peak[0] - low[0], peak[1] - low[1])
            low = peak = point
    if peak[1] - low[1] > 0 and (best is None or peak[1] - low[1] > best[2]):
        best = (low[0], peak[0] - low[0], peak[1] - low[1])
    return best


def full_profile(conn: Connection, candidate: Candidate) -> List[Point]:
    """솎아 내기 전의 전체 프로필. 최대 경사와 오르막 구간은 이것으로 계산한다."""
    rows = conn.execute(
        _PROFILE_SQL, {"edges": candidate.edge_ids, "forward": candidate.edge_forward}).fetchall()
    return build_profile(rows)

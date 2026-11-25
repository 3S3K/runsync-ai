-- 노드 고도와 구간별 경사 지표 계산.
-- 구간(edge)은 교차로 사이라 수백 m가 넘는 것도 많다. 양 끝 고도만 쓰면 중간의 언덕이 사라지므로
-- 구간을 약 30m(DEM 원본 해상도) 간격으로 잘라 표본 지점마다 고도를 읽는다.

-- 0) 달릴 수 없는 도로 제거: 보행 금지, 사유지·출입 금지 (보행 허용이 명시된 경우는 남긴다)
DELETE FROM routing.ways w
USING routing.way_tags t
WHERE t.osm_id = w.osm_id
  AND (t.foot = 'no'
       OR (t.access IN ('no', 'private') AND t.foot NOT IN ('yes', 'designated', 'permissive')));

-- 0-1) 가장 큰 연결 덩어리만 남긴다. 외딴 조각에 출발점이 붙으면 코스를 만들 수 없다.
WITH comp AS (
    SELECT component, node
    FROM pgr_connectedComponents('SELECT gid AS id, source, target, length_m AS cost, length_m AS reverse_cost FROM routing.ways')
), main AS (
    SELECT component FROM comp GROUP BY component ORDER BY count(*) DESC LIMIT 1
)
DELETE FROM routing.ways w
WHERE w.source NOT IN (SELECT node FROM comp WHERE component = (SELECT component FROM main));

DELETE FROM routing.ways_vertices_pgr v
WHERE NOT EXISTS (SELECT 1 FROM routing.ways w WHERE w.source = v.id OR w.target = v.id);

-- 1) 노드 고도
ALTER TABLE routing.ways_vertices_pgr ADD COLUMN IF NOT EXISTS elevation double precision;

UPDATE routing.ways_vertices_pgr v
SET elevation = ST_Value(r.rast, v.the_geom)
FROM routing.dem_raster r
WHERE ST_Intersects(r.rast, v.the_geom);

-- 2) 구간별 표본 지점 고도 (seq 0 = source 쪽 끝, seq n = target 쪽 끝)
DROP TABLE IF EXISTS routing.edge_samples;
CREATE TABLE routing.edge_samples AS
SELECT w.gid,
       s.seq,
       w.length_m / n.n AS seg_len,
       ST_Value(r.rast, p.geom) AS z
FROM routing.ways w
LEFT JOIN routing.way_tags t ON t.osm_id = w.osm_id
-- 다리·터널은 DEM이 노면이 아니라 강바닥·산 위를 읽으므로 양 끝만 써서 일정한 경사로 본다
CROSS JOIN LATERAL (SELECT CASE
        WHEN COALESCE(t.bridge, '') NOT IN ('', 'no') OR COALESCE(t.tunnel, '') NOT IN ('', 'no') THEN 1
        ELSE GREATEST(1, CEIL(w.length_m / 30.0))::int END AS n) n
CROSS JOIN LATERAL generate_series(0, n.n) AS s(seq)
CROSS JOIN LATERAL (SELECT ST_LineInterpolatePoint(w.the_geom, s.seq::float8 / n.n) AS geom) p
JOIN routing.dem_raster r ON ST_Intersects(r.rast, p.geom)
WHERE w.length_m > 0;

CREATE INDEX ON routing.edge_samples (gid, seq);

-- 3) 구간별 경사 지표
--    slope_pct      : 양 끝 기준 순경사(%), source→target 방향
--    ascent_m/descent_m : source→target 방향 누적 오르막/내리막(m)
--    up_pen_fwd/rev : Σ 조각길이 × max(0, 경사 − 8)²   → 요청 시 k를 곱해 오르막 비용으로 쓴다
--    down_len_fwd/rev : 경사 −5%보다 가파른 내리막 조각의 길이 합(m)
--    max_slope_fwd/rev : 해당 방향으로 갈 때 가장 가파른 오르막 조각의 경사(%)
ALTER TABLE routing.ways
    ADD COLUMN IF NOT EXISTS slope_pct double precision,
    ADD COLUMN IF NOT EXISTS ascent_m double precision,
    ADD COLUMN IF NOT EXISTS descent_m double precision,
    ADD COLUMN IF NOT EXISTS up_pen_fwd double precision,
    ADD COLUMN IF NOT EXISTS up_pen_rev double precision,
    ADD COLUMN IF NOT EXISTS down_len_fwd double precision,
    ADD COLUMN IF NOT EXISTS down_len_rev double precision,
    ADD COLUMN IF NOT EXISTS max_slope_fwd double precision,
    ADD COLUMN IF NOT EXISTS max_slope_rev double precision;

WITH seg AS (
    SELECT gid,
           seg_len,
           z - LAG(z) OVER (PARTITION BY gid ORDER BY seq) AS dz,
           FIRST_VALUE(z) OVER (PARTITION BY gid ORDER BY seq) AS z_first,
           LAST_VALUE(z) OVER (PARTITION BY gid ORDER BY seq
                               ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS z_last
    FROM routing.edge_samples
), agg AS (
    SELECT gid,
           MIN(z_last - z_first) AS net_dz,
           SUM(GREATEST(dz, 0)) AS ascent_m,
           SUM(GREATEST(-dz, 0)) AS descent_m,
           SUM(seg_len * POWER(GREATEST(0, dz / seg_len * 100 - 8), 2)) AS up_pen_fwd,
           SUM(seg_len * POWER(GREATEST(0, -dz / seg_len * 100 - 8), 2)) AS up_pen_rev,
           SUM(CASE WHEN dz / seg_len * 100 < -5 THEN seg_len ELSE 0 END) AS down_len_fwd,
           SUM(CASE WHEN dz / seg_len * 100 > 5 THEN seg_len ELSE 0 END) AS down_len_rev,
           MAX(dz / seg_len * 100) AS max_slope_fwd,
           MAX(-dz / seg_len * 100) AS max_slope_rev
    FROM seg
    WHERE dz IS NOT NULL
    GROUP BY gid
)
UPDATE routing.ways w
SET slope_pct = a.net_dz / w.length_m * 100,
    ascent_m = a.ascent_m,
    descent_m = a.descent_m,
    up_pen_fwd = a.up_pen_fwd,
    up_pen_rev = a.up_pen_rev,
    down_len_fwd = a.down_len_fwd,
    down_len_rev = a.down_len_rev,
    max_slope_fwd = GREATEST(a.max_slope_fwd, 0),
    max_slope_rev = GREATEST(a.max_slope_rev, 0)
FROM agg a
WHERE a.gid = w.gid;

-- 계단 여부 (요청 시 길이 × 100 비용 적용)
ALTER TABLE routing.ways ADD COLUMN IF NOT EXISTS is_steps boolean;
UPDATE routing.ways w
SET is_steps = (c.tag_value = 'steps')
FROM routing.configuration c
WHERE c.tag_id = w.tag_id;

-- DEM 범위 밖이라 고도를 못 읽은 극소수 구간은 평지로 둔다
UPDATE routing.ways
SET slope_pct = 0, ascent_m = 0, descent_m = 0, up_pen_fwd = 0, up_pen_rev = 0,
    down_len_fwd = 0, down_len_rev = 0, max_slope_fwd = 0, max_slope_rev = 0
WHERE slope_pct IS NULL;

ANALYZE routing.ways;
ANALYZE routing.ways_vertices_pgr;

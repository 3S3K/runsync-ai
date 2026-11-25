#!/usr/bin/env bash
# 그래프 전처리: OSM 도로망 + DEM 고도 → DB의 routing 스키마
# 한 번만 실행하고, 지도 데이터를 갱신했을 때 다시 실행한다. (약 4분)
#
# 필요한 파일
#   data/donggu.osm        Overpass API로 받은 도로망 (highway 전체)
#   data/dem_n36_e127.tif  Copernicus DEM 30m 타일
set -euo pipefail
cd "$(dirname "$0")/.."

CONTAINER=${DB_CONTAINER:-runsync-postgres}
# 대전 동구 행정구역 + 약 2km 여유 (서, 남, 동, 북)
OSM_BBOX="127.39 36.18 127.56 36.46"
# DEM은 도로망보다 조금 넓게 자른다
DEM_BBOX="127.38 36.17 127.57 36.47"

echo "[1/5] 파일 복사"
docker exec "$CONTAINER" mkdir -p /data
docker cp data/donggu.osm "$CONTAINER":/data/donggu.osm
docker cp data/dem_n36_e127.tif "$CONTAINER":/data/dem.tif
docker cp scripts/mapconfig_running.xml "$CONTAINER":/data/mapconfig_running.xml
python3 scripts/extract_way_tags.py data/donggu.osm > data/way_tags.csv
docker cp data/way_tags.csv "$CONTAINER":/data/way_tags.csv

run_sql() { docker exec -i "$CONTAINER" psql -q -v ON_ERROR_STOP=1 -U runsync -d runsync "$@"; }

echo "[2/5] 확장·스키마"
run_sql -c "CREATE EXTENSION IF NOT EXISTS postgis;
            CREATE EXTENSION IF NOT EXISTS postgis_raster;
            CREATE EXTENSION IF NOT EXISTS pgrouting;
            CREATE SCHEMA IF NOT EXISTS routing;"

echo "[3/5] 도로망 적재 (osm2pgrouting)"
docker exec "$CONTAINER" sh -c 'osm2pgrouting --file /data/donggu.osm --conf /data/mapconfig_running.xml \
    --schema routing --dbname "$POSTGRES_DB" --username "$POSTGRES_USER" --password "$POSTGRES_PASSWORD" \
    --host localhost --clean' | grep -E "Total processed|Elapsed" || true
run_sql -c "DROP TABLE IF EXISTS routing.way_tags;
            CREATE TABLE routing.way_tags (osm_id bigint PRIMARY KEY, bridge text, tunnel text, foot text, access text);
            COPY routing.way_tags FROM '/data/way_tags.csv' CSV HEADER;"

echo "[4/5] 고도 적재"
# Copernicus는 건물·수목 높이가 섞인 지표면 모델(DSM)이라 그대로 쓰면 평지 도로도 가파르게 나온다.
# 약 60m로 평균을 낸 뒤 10m 격자로 부드럽게 보간해서 넣는다.
docker exec "$CONTAINER" sh -c "
    rm -f /data/dem_avg.tif /data/dem_smooth.tif
    gdalwarp -q -te $DEM_BBOX -tr 0.0006 0.0006 -r average /data/dem.tif /data/dem_avg.tif
    gdalwarp -q -te $DEM_BBOX -tr 0.0001 0.0001 -r cubicspline /data/dem_avg.tif /data/dem_smooth.tif
    raster2pgsql -q -d -s 4326 -t 100x100 -I -C -M /data/dem_smooth.tif routing.dem_raster \
        | psql -q -U \"\$POSTGRES_USER\" -d \"\$POSTGRES_DB\"" > /dev/null

echo "[5/5] 노드 고도·구간 경사 계산 (약 3분)"
run_sql < scripts/slope.sql

run_sql -Atc "SELECT '구간 ' || count(*) || '개, 총 ' || round(sum(length_m)::numeric / 1000) || 'km' FROM routing.ways;
              SELECT '노드 ' || count(*) || '개' FROM routing.ways_vertices_pgr;"
echo "완료"

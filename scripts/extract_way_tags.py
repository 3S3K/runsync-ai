"""OSM 파일에서 도로(way)별 다리·터널·통행 제한 태그를 뽑아 CSV로 출력한다.

osm2pgrouting은 이 태그들을 테이블에 넣어 주지 않는다. 다리·터널은 DEM 고도가 실제 노면과
다르고(다리 밑 강바닥, 터널 위 산), foot=no / access=private 도로는 달릴 수 없어서 필요하다.

사용: python3 extract_way_tags.py donggu.osm > way_tags.csv
"""
import csv
import sys
import xml.etree.ElementTree as ET

KEYS = ("bridge", "tunnel", "foot", "access")


def main(path):
    writer = csv.writer(sys.stdout)
    writer.writerow(("osm_id",) + KEYS)
    for _, elem in ET.iterparse(path, events=("end",)):
        if elem.tag == "way":
            tags = {t.get("k"): t.get("v") for t in elem.findall("tag")}
            if any(k in tags for k in KEYS):
                writer.writerow([elem.get("id")] + [tags.get(k, "") for k in KEYS])
        if elem.tag in ("node", "way", "relation"):
            elem.clear()


if __name__ == "__main__":
    main(sys.argv[1])

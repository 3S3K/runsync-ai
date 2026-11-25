"""환경변수와 조정값. 경사 기준·k·가중치는 실제로 뛰어 보며 바꾸는 값이라 여기 한 곳에 모은다."""
import os

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://runsync:1234@localhost:5432/runsync")

# 서비스 범위: 대전 동구 행정구역 + 약 2km 여유 (전처리 범위와 같다)
SERVICE_MIN_LAT, SERVICE_MAX_LAT = 36.18, 36.46
SERVICE_MIN_LNG, SERVICE_MAX_LNG = 127.39, 127.56
# 출발 좌표에서 가장 가까운 도로 노드가 이보다 멀면 서비스 지역 밖으로 본다
MAX_SNAP_DISTANCE_M = 300.0

MIN_DISTANCE_KM = 1.0
MAX_DISTANCE_KM = 20.0

# 강도별 경사 회피 계수 k. 오르막 비용 = 길이 × (1 + k × max(0, 경사 − 8)²)
# 8% 기준은 전처리(scripts/slope.sql)에서 up_pen 컬럼에 이미 반영돼 있다.
SLOPE_AVOIDANCE_K = {"EASY": 0.5, "NORMAL": 0.2, "HARD": 0.05}
# 경사 −5%보다 가파른 내리막은 길이 × 1.5 (무릎 부담)
STEEP_DOWNHILL_EXTRA = 0.5
# 계단은 양방향 길이 × 100. 다른 길이 없을 때만 지나가게 한다
STEPS_COST_FACTOR = 100.0

# 순환 후보: 출발지를 지나는 원 위에 경유점 2개를 두고, 원의 방향을 22.5°씩 돌려 16개를 만든다.
# 8개(45°)로는 강도별로 고를 만한 후보가 부족했다.
CANDIDATE_BEARINGS_DEG = tuple(i * 22.5 for i in range(16))
# 실제 도로 거리 ÷ 직선 거리의 초기 추정값. 원의 반지름을 정할 때 쓴다
DETOUR_FACTOR = 1.3
# 목표 거리에서 이 비율 넘게 벗어나면 반지름을 조정해 1회 재시도
DISTANCE_TOLERANCE = 0.15
# 재시도 후에도 목표 거리에서 이 비율 넘게 벗어난 후보는 추천하지 않는다
MAX_DISTANCE_ERROR = 0.3
# 탐색 범위: 출발지 기준 목표 거리의 이 비율만큼 사방으로
SEARCH_BOX_RATIO = 0.6

# 점수(낮을수록 좋음)
#   = W_DISTANCE × |거리 − 목표|/목표 + W_OVERLAP × 왕복 중복률
#   + W_CLIMB × |1km당 오르막 − 강도별 목표|
W_DISTANCE = 1.0
W_OVERLAP = 0.5
# 강도별로 선호하는 1km당 누적 오르막(m). 심박 예측 모델을 붙이면 이 항은
# |예상 강도 − 목표 강도|로 바뀐다. 그 전까지 강도 차이를 내기 위한 대체 항이다.
TARGET_CLIMB_M_PER_KM = {"EASY": 0.0, "NORMAL": 15.0, "HARD": 30.0}
W_CLIMB = 0.02
# 이미 고른 코스와 겹치는 구간 비율이 이보다 크면 같은 코스로 보고 건너뛴다
MAX_SIMILARITY = 0.7
TOP_N = 3

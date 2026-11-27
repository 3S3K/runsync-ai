"""환경변수와 조정값. 경사 기준·k·가중치는 실제로 뛰어 보며 바꾸는 값이라 여기 한 곳에 모은다."""
import os

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://runsync:1234@localhost:5432/runsync")
# 심박 예측 모델 파일. `python -m app.train`으로 만든다
MODEL_PATH = os.environ.get("MODEL_PATH", "models/hr_model.joblib")
# 모델 종류: lightgbm | ridge
MODEL_KIND = os.environ.get("MODEL_KIND", "lightgbm")
# 재학습 이력 파일 (한 줄에 한 건). 배포 서버에서는 logs 폴더가 컨테이너 밖에 보관된다
TRAINING_LOG_PATH = os.environ.get("TRAINING_LOG_PATH", "logs/trainings.jsonl")

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

# 강도별 목표 운동 강도 구간 (최대 심박 대비 %)
INTENSITY_ZONE_PCT = {"EASY": (60.0, 70.0), "NORMAL": (70.0, 80.0), "HARD": (80.0, 90.0)}

# 점수(낮을수록 좋음)
#   = W_DISTANCE × |거리 − 목표|/목표 + W_OVERLAP × 왕복 중복률
#   + W_INTENSITY × |예상 강도 − 목표 구간의 중앙| (%p)
W_DISTANCE = 1.0
W_OVERLAP = 0.5
W_INTENSITY = 0.1
# 요청한 강도 구간을 벗어난 코스에 더하는 값. 구간 안 코스가 낼 수 있는 가장 나쁜 점수
# (거리 0.3 + 중복 0.5 + 강도 0.5)보다 커서, 구간 안 코스가 항상 앞에 온다.
OUT_OF_ZONE_PENALTY = 1.5
# 화면에 보여 주는 추천 점수(100점 만점)로 바꿀 때의 기준. 이 값이면 0점
SCORE_SCALE = 3.0
# 이미 고른 코스와 겹치는 구간 비율이 이보다 크면 같은 코스로 보고 건너뛴다
MAX_SIMILARITY = 0.7
TOP_N = 3

# 고도 프로필로 내려주는 점의 최대 개수
ELEVATION_PROFILE_MAX_POINTS = 100

# 화면에 보여 주는 최대 경사는 이 거리 이상의 구간에서 잰다.
# 30m 표본 사이의 경사를 그대로 쓰면 고도 데이터 오차 때문에 20%가 넘는 값이 나온다.
MAX_SLOPE_WINDOW_M = 100.0
# 오르막 구간을 찾을 때, 이만큼까지 내려갔다 다시 오르는 것은 같은 오르막으로 본다
CLIMB_DIP_TOLERANCE_M = 2.0
# 추천 이유에 언급할 만한 오르막의 최소 높이 (가장 많이 오르는 구간 기준)
MIN_CLIMB_GAIN_M = 5.0

# 지도에 칠하는 경사 등급의 경계 (구간의 평균 기울기 %)
# 러닝 코칭에서 언덕 훈련에 권장하는 경사 범위(5~10%)를 가운데 등급으로 삼는다
SLOPE_GRADE_MODERATE_PCT = 5.0
SLOPE_GRADE_STEEP_PCT = 10.0

# 예상 심박수 범위의 폭. 모델의 검증 오차(MAE 약 2.5%p)만큼 위아래로 둔다
HR_RANGE_MARGIN_PCT = 2.5

# ----- 재학습 -----
# 합성 데이터의 크기와 시드. 최초 학습과 재학습이 같은 합성 데이터를 쓴다
SYNTHETIC_ROWS = 8000
SYNTHETIC_SEED = 42
# 실제 기록 1건을 합성 데이터 몇 건의 비중으로 볼지. 기록이 쌓일수록 실제 기록이 모델을 끌고 간다
REAL_RECORD_WEIGHT = 10.0
# 기록의 "평소 페이스·거리·케이던스"는 그 기록 직전 몇 회로 계산할지 (추천 요청과 같은 값)
RECENT_RECORD_COUNT = 5
# 학습에 쓰는 기록의 조건. 사용자가 손으로 입력한 값이라 범위를 벗어난 기록은 뺀다
TRAIN_MIN_DISTANCE_KM = 1.0
TRAIN_MIN_PACE_SEC_PER_KM, TRAIN_MAX_PACE_SEC_PER_KM = 180.0, 720.0   # 3'00" ~ 12'00"
TRAIN_MIN_HR_PCT, TRAIN_MAX_HR_PCT = 40.0, 100.0                      # 최대 심박 대비 %
TRAIN_MIN_CADENCE, TRAIN_MAX_CADENCE = 120, 220
TRAIN_MIN_AGE, TRAIN_MAX_AGE = 10, 90
# 교차 검증 묶음 수의 상한 (기록이 이보다 적으면 기록 수만큼 나눈다)
CV_MAX_FOLDS = 5
# 이력 조회 때 돌려주는 최근 건수
TRAINING_HISTORY_LIMIT = 50

# ==================================================================
# 0. 포즈 인식 모델 자동 준비
# ==================================================================
MODEL_PATH = "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

# ==================================================================
# 1. 게임 설정값
# ==================================================================
CHALLENGE_TIME = 8          # 도전 제한시간(초)
SUCCESS_THRESHOLD = 80      # 성공 기준 일치율(%)
COUNTDOWN_SEC = 3           # 시작 전 카운트다운(초)
COIN_PRICE = 500            # 원 (화면 표시용 - 실제 결제는 하드웨어 연동 필요)
RESULT_DISPLAY_SEC = 4      # 결과 화면 유지 시간(초)
HOLD_SEC = 0.5              # 성공 기준 이상을 이 시간만큼 유지해야 성공(초)
URGENT_TIME = 3             # 남은 시간이 이 이하면 타이머가 빨갛게 깜빡임(초)

# 점수 판정
SCORE_SMOOTHING = 0.35      # 일치율 지수이동평균 계수 (클수록 빠르게 반응, 작을수록 안정적)
MIN_VISIBILITY = 0.5        # 이보다 가시성이 낮은 관절은 '안 보임'으로 처리
ANGLE_TOLERANCE = 90.0      # 목표 각도와 이만큼 차이 나면 해당 관절 0점

# 카메라 / 창
CAMERA_INDEX = 1            # USB 웹캠 번호 (노트북 내장 카메라는 보통 0)
CAMERA_WIDTH, CAMERA_HEIGHT = 1280, 720
WINDOW_NAME = "포즈를 맞춰라!"
WINDOW_POS = (1920, 0)      # 보조 모니터 위치 (주 모니터 가로 해상도)

# ==================================================================
# 2. 랜드마크 인덱스 & 관절 각도 정의
#    (33점 포즈 토폴로지는 legacy/Tasks API 모두 동일합니다)
# ==================================================================
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

ANGLE_DEFS = {
    "left_elbow":     (LEFT_SHOULDER,  LEFT_ELBOW,     LEFT_WRIST),
    "right_elbow":    (RIGHT_SHOULDER, RIGHT_ELBOW,    RIGHT_WRIST),
    "left_shoulder":  (LEFT_HIP,       LEFT_SHOULDER,  LEFT_ELBOW),
    "right_shoulder": (RIGHT_HIP,      RIGHT_SHOULDER, RIGHT_ELBOW),
    "left_hip":       (LEFT_SHOULDER,  LEFT_HIP,       LEFT_KNEE),
    "right_hip":      (RIGHT_SHOULDER, RIGHT_HIP,      RIGHT_KNEE),
    "left_knee":      (LEFT_HIP,       LEFT_KNEE,      LEFT_ANKLE),
    "right_knee":     (RIGHT_HIP,      RIGHT_KNEE,     RIGHT_ANKLE),
}
WEIGHTS = {k: 1.0 for k in ANGLE_DEFS}

# 스켈레톤 그리기용 연결선 (표준 33점 포즈 토폴로지, 상반신+하반신 위주)
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

# ==================================================================
# 3. 목표 포즈 정의 (관절별 "이상적인 각도")
# ==================================================================
TARGET_POSES = {
    "만세! (양팔 위로)": {
        "left_elbow": 170, "right_elbow": 170,
        "left_shoulder": 170, "right_shoulder": 170,
        "left_hip": 175, "right_hip": 175,
        "left_knee": 175, "right_knee": 175,
    },
    "차렷 자세": {
        "left_elbow": 170, "right_elbow": 170,
        "left_shoulder": 15, "right_shoulder": 15,
        "left_hip": 175, "right_hip": 175,
        "left_knee": 175, "right_knee": 175,
    },
    "T-포즈 (양팔 벌리기)": {
        "left_elbow": 170, "right_elbow": 170,
        "left_shoulder": 90, "right_shoulder": 90,
        "left_hip": 175, "right_hip": 175,
        "left_knee": 175, "right_knee": 175,
    },
    "하트 만들기 (머리 위)": {
        "left_elbow": 60, "right_elbow": 60,
        "left_shoulder": 160, "right_shoulder": 160,
        "left_hip": 175, "right_hip": 175,
        "left_knee": 175, "right_knee": 175,
    },
    "짝다리 (오른쪽 무릎 들기)": {
        "left_elbow": 170, "right_elbow": 170,
        "left_shoulder": 15, "right_shoulder": 15,
        "left_hip": 175, "right_hip": 90,
        "left_knee": 175, "right_knee": 90,
    },
}

# ==================================================================
# 4. 게임 상태
# ==================================================================
STATE_WAIT_COIN = "WAIT_COIN"
STATE_COUNTDOWN = "COUNTDOWN"
STATE_PLAYING = "PLAYING"
STATE_RESULT = "RESULT"
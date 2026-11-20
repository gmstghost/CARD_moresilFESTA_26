import os
import time
import random
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import params as pr
import pose as ps
import utils as ut
from ui import TextLayer, screens, widgets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, pr.MODEL_PATH)
POSES_PATH = os.path.join(BASE_DIR, "pose", "poses.json")


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("포즈 인식 모델 다운로드 중... (최초 1회, 약 5MB)")
        try:
            urllib.request.urlretrieve(pr.MODEL_URL, MODEL_PATH)
            print("다운로드 완료:", MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"모델 자동 다운로드 실패: {e}")


class PoseGameController:
    def __init__(self):
        ensure_model()

        # 외부 JSON 파일에서 포즈 데이터를 관리하는 매니저 초기화
        self.pose_mgr = ps.PoseManager(POSES_PATH)

        # MediaPipe 설정
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self.text = TextLayer()
        self.last_timestamp_ms = -1

        # 상태 변수들 (ui.screens 가 읽어서 화면을 그림)
        self.state = pr.STATE_WAIT_COIN
        self.state_start = time.monotonic()
        self.current_pose_name = None
        self.target_angles = {}
        self.joint_scores = {}
        self.score = 0.0            # 평활화된 현재 일치율
        self.best_score = 0.0
        self.hold_start = None      # 성공 기준 이상을 유지하기 시작한 시각
        self.success = False
        self.final_score = 0.0
        self.player_status = "none"  # none / partial / ok

    # ------------------------------------------------------------------
    # 상태 전이 (게임 로직)
    # ------------------------------------------------------------------
    def _set_state(self, state, now):
        self.state = state
        self.state_start = now

    def remaining_time(self, now):
        return max(0.0, pr.CHALLENGE_TIME - (now - self.state_start))

    def hold_ratio(self, now):
        if self.hold_start is None:
            return 0.0
        return min(1.0, (now - self.hold_start) / pr.HOLD_SEC)

    def start_challenge(self, now):
        pose_names = self.pose_mgr.get_pose_names()
        if not pose_names:
            print("경고: 불러올 포즈 데이터가 없습니다.")
            return
        self.current_pose_name = random.choice(pose_names)
        self.target_angles = self.pose_mgr.get_pose_angles(self.current_pose_name)
        self.joint_scores = {}
        self._set_state(pr.STATE_COUNTDOWN, now)

    def _finish(self, now, success):
        self.success = success
        self.final_score = self.score if success else self.best_score
        self._set_state(pr.STATE_RESULT, now)

    def update(self, now, current_angles):
        # 카운트다운 중에도 관절 색 피드백을 보여주기 위해 미리 계산
        if self.state in (pr.STATE_COUNTDOWN, pr.STATE_PLAYING):
            self.joint_scores = (ut.calc_joint_scores(current_angles, self.target_angles)
                                 if current_angles is not None else {})

        if self.state == pr.STATE_COUNTDOWN:
            if now - self.state_start >= pr.COUNTDOWN_SEC:
                self.score = self.best_score = 0.0
                self.hold_start = None
                self._set_state(pr.STATE_PLAYING, now)

        elif self.state == pr.STATE_PLAYING:
            raw = ut.calc_match_score(self.joint_scores) if self.joint_scores else 0.0
            a = pr.SCORE_SMOOTHING
            self.score = a * raw + (1 - a) * self.score
            self.best_score = max(self.best_score, self.score)

            if self.score >= pr.SUCCESS_THRESHOLD:
                if self.hold_start is None:
                    self.hold_start = now
                if now - self.hold_start >= pr.HOLD_SEC:
                    self._finish(now, success=True)
                    return
            else:
                self.hold_start = None

            if self.remaining_time(now) <= 0:
                self._finish(now, success=False)

        elif self.state == pr.STATE_RESULT:
            if now - self.state_start > pr.RESULT_DISPLAY_SEC:
                self._set_state(pr.STATE_WAIT_COIN, now)

    def handle_key(self, key, now):
        """False 를 반환하면 게임 종료"""
        if key in (ord('q'), 27):  # q 또는 ESC
            return False
        if key == ord(' ') and self.state in (pr.STATE_WAIT_COIN, pr.STATE_RESULT):
            self.start_challenge(now)
        return True

    # ------------------------------------------------------------------
    # 화면 그리기
    # ------------------------------------------------------------------
    def render(self, frame, now, landmarks):
        if landmarks is not None:
            show_feedback = self.state in (pr.STATE_COUNTDOWN, pr.STATE_PLAYING)
            widgets.draw_skeleton(frame, landmarks, self.joint_scores if show_feedback else None)
        screens.DRAWERS[self.state](frame, self.text, self, now)
        self.text.render(frame)

    # ------------------------------------------------------------------
    # 메인 루프
    # ------------------------------------------------------------------
    def detect(self, frame, now):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # VIDEO 모드는 타임스탬프가 반드시 증가해야 함
        ts = max(int(now * 1000), self.last_timestamp_ms + 1)
        self.last_timestamp_ms = ts
        result = self.landmarker.detect_for_video(mp_image, ts)
        return result.pose_landmarks[0] if result.pose_landmarks else None

    def run(self):
        cv2.namedWindow(pr.WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.moveWindow(pr.WINDOW_NAME, *pr.WINDOW_POS)
        cv2.setWindowProperty(pr.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cap = cv2.VideoCapture(pr.CAMERA_INDEX)
        if not cap.isOpened():
            print(f"웹캠({pr.CAMERA_INDEX}번)을 열 수 없습니다. 카메라 연결이나 params.CAMERA_INDEX 를 확인하세요.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, pr.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, pr.CAMERA_HEIGHT)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)  # 거울 모드
                now = time.monotonic()

                landmarks = self.detect(frame, now)
                current_angles = None
                if landmarks is not None:
                    h, w = frame.shape[:2]
                    current_angles = ut.extract_angles(landmarks, w, h)
                    self.player_status = "ok" if len(current_angles) == len(pr.ANGLE_DEFS) else "partial"
                else:
                    self.player_status = "none"

                self.update(now, current_angles)
                self.render(frame, now, landmarks)
                cv2.imshow(pr.WINDOW_NAME, frame)

                if not self.handle_key(cv2.waitKey(1) & 0xFF, now):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.landmarker.close()


if __name__ == "__main__":
    PoseGameController().run()

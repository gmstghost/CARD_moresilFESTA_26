import os
import time
import random
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import params as pr
import pose as ps
import utils as ut


def ensure_model():
    if not os.path.exists(pr.MODEL_PATH):
        print("포즈 인식 모델 다운로드 중... (최초 1회, 약 5MB)")
        try:
            urllib.request.urlretrieve(pr.MODEL_URL, pr.MODEL_PATH)
            print("다운로드 완료:", pr.MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"모델 자동 다운로드 실패: {e}")


class PoseGameController:
    def __init__(self):
        ensure_model()
        
        # 외부 JSON 파일에서 포즈 데이터를 관리하는 매니저 초기화
        self.pose_mgr = ps.PoseManager("pose/poses.json")
        
        # MediaPipe 설정
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=pr.MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        
        # 상태 변수들
        self.state = pr.STATE_WAIT_COIN
        self.current_pose_name = None
        self.countdown_start = 0.0
        self.play_start = 0.0
        self.best_score = 0.0
        self.result_text = ""
        self.result_start = 0.0
        self.session_start = time.time()

    def handle_wait_coin(self, frame):
        """코인 대기 상태 렌더링"""
        ut.put_text_center(frame, "포즈를 맞춰라!", 80, 1.4, (0, 255, 255), 3)
        ut.put_text_center(frame, f"도전 비용: {pr.COIN_PRICE}원", 130, 0.9)
        ut.put_text_center(frame, "[SPACE] 코인 투입 / 게임 시작", 170, 0.8, (200, 200, 200))

    def handle_countdown(self, frame):
        """카운트다운 상태 렌더링 및 전환 처리"""
        elapsed = time.time() - self.countdown_start
        remain = pr.COUNTDOWN_SEC - int(elapsed)
        if remain > 0:
            ut.put_text_center(frame, str(remain), frame.shape[0] // 2, 3.0, (0, 255, 255), 5)
            ut.put_text_center(frame, f"목표 포즈: {self.current_pose_name}", 60, 1.0, (255, 255, 0))
        else:
            self.state = pr.STATE_PLAYING
            self.play_start = time.time()
            self.best_score = 0.0

    def handle_playing(self, frame, current_angles):
        """게임 플레이 상태 로직 (점수 계산 및 UI 렌더링)"""
        remain_time = pr.CHALLENGE_TIME - (time.time() - self.play_start)
        score = 0.0
        
        if current_angles:
            target_angles = self.pose_mgr.get_pose_angles(self.current_pose_name)
            score = ut.calc_match_score(current_angles, target_angles)
            self.best_score = max(self.best_score, score)

        # 게이지 바 UI
        bar_color = (0, 255, 0) if score >= pr.SUCCESS_THRESHOLD else (0, 165, 255)
        cv2.rectangle(frame, (20, 20), (20 + int(score * 3), 50), bar_color, -1)
        cv2.rectangle(frame, (20, 20), (320, 50), (255, 255, 255), 2)
        cv2.putText(frame, f"{score:.1f}%", (330, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        ut.put_text_center(frame, f"목표: {self.current_pose_name}", 90, 1.0, (255, 255, 0))
        cv2.putText(frame, f"남은시간: {max(0, remain_time):.1f}s", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 성공/실패 판정
        if score >= pr.SUCCESS_THRESHOLD:
            self.state = pr.STATE_RESULT
            self.result_text = f"성공! 일치율 {score:.1f}%  -> 상품 지급"
            self.result_start = time.time()
        elif remain_time <= 0:
            self.state = pr.STATE_RESULT
            self.result_text = f"실패... 최고 일치율 {self.best_score:.1f}%"
            self.result_start = time.time()

    def handle_result(self, frame):
        """결과 화면 상태 처리"""
        color = (0, 255, 0) if "성공" in self.result_text else (0, 0, 255)
        ut.put_text_center(frame, self.result_text, frame.shape[0] // 2, 1.1, color, 3)
        ut.put_text_center(frame, "[SPACE] 처음으로", frame.shape[0] // 2 + 50, 0.8, (200, 200, 200))
        
        if time.time() - self.result_start > pr.RESULT_DISPLAY_SEC:
            self.state = pr.STATE_WAIT_COIN

    def run(self):
        """메인 게임 루프"""
        cv2.namedWindow('포즈를 맞춰라!', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.moveWindow('포즈를 맞춰라!', 1920, 0)
        cv2.setWindowProperty('포즈를 맞춰라!', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("웹캠을 열 수 없습니다. 카메라 연결을 확인하세요.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int((time.time() - self.session_start) * 1000)
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            current_angles = None
            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                ut.draw_skeleton(frame, lm)
                current_angles = ut.extract_angles(lm)

            # 상태별 핸들러 호출 (if-elif 복잡도 제거)
            if self.state == pr.STATE_WAIT_COIN:
                self.handle_wait_coin(frame)
            elif self.state == pr.STATE_COUNTDOWN:
                self.handle_countdown(frame)
            elif self.state == pr.STATE_PLAYING:
                self.handle_playing(frame, current_angles)
            elif self.state == pr.STATE_RESULT:
                self.handle_result(frame)

            cv2.imshow("포즈를 맞춰라!", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                # 코인 대기 중이거나 결과 화면일 때 스페이스바로 게임(재)시작
                if self.state in (pr.STATE_WAIT_COIN, pr.STATE_RESULT):
                    # JSON 매니저를 통해 포즈 목록을 가져옴
                    pose_names = self.pose_mgr.get_pose_names()
                    if not pose_names:
                        print("경고: 불러올 포즈 데이터가 없습니다.")
                        continue
                        
                    self.current_pose_name = random.choice(pose_names)
                    self.state = pr.STATE_COUNTDOWN
                    self.countdown_start = time.time()

        # 자원 해제
        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()


if __name__ == "__main__":
    game = PoseGameController()
    game.run()
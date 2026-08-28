import os
import time
import random
import math
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import params as pr


def ensure_model():
    if not os.path.exists(pr.MODEL_PATH):
        print("포즈 인식 모델 다운로드 중... (최초 1회, 약 5MB)")
        try:
            urllib.request.urlretrieve(pr.MODEL_URL, pr.MODEL_PATH)
            print("다운로드 완료:", pr.MODEL_PATH)
        except Exception as e:
            raise RuntimeError(
                f"모델 자동 다운로드 실패: {e}\n"
                f"{pr.MODEL_URL} 을 직접 받아서 스크립트 폴더에 "
                f"'{pr.MODEL_PATH}' 이름으로 저장한 뒤 다시 실행하세요."
            )


def calc_angle(a, b, c):
    """세 점(a-b-c)이 이루는 각도를 0~180도로 계산"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def extract_angles(landmarks):
    """landmarks: PoseLandmarkerResult.pose_landmarks[0] (사람 1명의 33개 랜드마크)"""
    angles = {}
    for name, (i1, i2, i3) in pr.ANGLE_DEFS.items():
        a = (landmarks[i1].x, landmarks[i1].y)
        b = (landmarks[i2].x, landmarks[i2].y)
        c = (landmarks[i3].x, landmarks[i3].y)
        angles[name] = calc_angle(a, b, c)
    return angles


def draw_skeleton(frame, landmarks):
    """랜드마크를 이용해 화면에 직접 스켈레톤을 그림 (구버전 drawing_utils 미사용)"""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for i1, i2 in pr.POSE_CONNECTIONS:
        cv2.line(frame, pts[i1], pts[i2], (0, 200, 255), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 255, 0), -1)


def calc_match_score(current_angles, target_angles):
    total_score, total_weight = 0.0, 0.0
    for name, target in target_angles.items():
        current = current_angles.get(name)
        if current is None:
            continue
        diff = abs(current - target)
        similarity = max(0.0, 1 - diff / 90.0)
        w = pr.WEIGHTS[name]
        total_score += similarity * w
        total_weight += w
    return (total_score / total_weight) * 100 if total_weight else 0.0


def put_text_center(img, text, y, scale=1.2, color=(255, 255, 255), thickness=2):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (img.shape[1] - w) // 2
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def main():
    cv2.namedWindow('포즈를 맞춰라!', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.moveWindow('포즈를 맞춰라!', 1920, 0)
    cv2.setWindowProperty('포즈를 맞춰라!', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    ensure_model()

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=pr.MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다. 카메라 연결을 확인하세요.")
        return

    state = pr.STATE_WAIT_COIN
    current_pose_name = None
    countdown_start = 0.0
    play_start = 0.0
    best_score = 0.0
    result_text = ""
    result_start = 0.0
    session_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # 거울 모드
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((time.time() - session_start) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        current_angles = None
        if result.pose_landmarks:
            lm = result.pose_landmarks[0]  # 첫 번째로 감지된 사람
            draw_skeleton(frame, lm)
            current_angles = extract_angles(lm)

        # ---------------- 상태별 로직 ----------------
        if state == pr.STATE_WAIT_COIN:
            put_text_center(frame, "포즈를 맞춰라!", 80, 1.4, (0, 255, 255), 3)
            put_text_center(frame, f"도전 비용: {pr.COIN_PRICE}원", 130, 0.9)
            put_text_center(frame, "[SPACE] 코인 투입 / 게임 시작", 170, 0.8, (200, 200, 200))

        elif state == pr.STATE_COUNTDOWN:
            elapsed = time.time() - countdown_start
            remain = pr.COUNTDOWN_SEC - int(elapsed)
            if remain > 0:
                put_text_center(frame, str(remain), frame.shape[0] // 2, 3.0, (0, 255, 255), 5)
                put_text_center(frame, f"목표 포즈: {current_pose_name}", 60, 1.0, (255, 255, 0))
            else:
                state = pr.STATE_PLAYING
                play_start = time.time()
                best_score = 0.0

        elif state == pr.STATE_PLAYING:
            remain_time = pr.CHALLENGE_TIME - (time.time() - play_start)
            score = 0.0
            if current_angles:
                score = calc_match_score(current_angles, pr.TARGET_POSES[current_pose_name])
                best_score = max(best_score, score)

            bar_color = (0, 255, 0) if score >= pr.SUCCESS_THRESHOLD else (0, 165, 255)
            cv2.rectangle(frame, (20, 20), (20 + int(score * 3), 50), bar_color, -1)
            cv2.rectangle(frame, (20, 20), (320, 50), (255, 255, 255), 2)
            cv2.putText(frame, f"{score:.1f}%", (330, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            put_text_center(frame, f"목표: {current_pose_name}", 90, 1.0, (255, 255, 0))
            cv2.putText(frame, f"남은시간: {max(0, remain_time):.1f}s", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if score >= pr.SUCCESS_THRESHOLD:
                state = pr.STATE_RESULT
                result_text = f"성공! 일치율 {score:.1f}%  -> 상품 지급"
                result_start = time.time()
            elif remain_time <= 0:
                state = pr.STATE_RESULT
                result_text = f"실패... 최고 일치율 {best_score:.1f}%"
                result_start = time.time()

        elif state == pr.STATE_RESULT:
            color = (0, 255, 0) if "성공" in result_text else (0, 0, 255)
            put_text_center(frame, result_text, frame.shape[0] // 2, 1.1, color, 3)
            put_text_center(frame, "[SPACE] 처음으로", frame.shape[0] // 2 + 50, 0.8, (200, 200, 200))
            if time.time() - result_start > pr.RESULT_DISPLAY_SEC:
                state = pr.STATE_WAIT_COIN

        cv2.imshow("포즈를 맞춰라!", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            if state in (pr.STATE_WAIT_COIN, pr.STATE_RESULT):
                current_pose_name = random.choice(list(pr.TARGET_POSES.keys()))
                state = pr.STATE_COUNTDOWN
                countdown_start = time.time()

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
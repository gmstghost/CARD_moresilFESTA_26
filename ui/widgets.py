"""화면 구성 요소 (반투명 패널, 진행 바, 링, 스켈레톤, 포즈 그림)."""
import math

import cv2
import numpy as np

import params as pr

# 색상 (BGR)
WHITE = (255, 255, 255)
GRAY = (150, 150, 150)
BLACK = (0, 0, 0)
YELLOW = (0, 220, 255)
GREEN = (80, 220, 60)
ORANGE = (0, 150, 255)
RED = (70, 70, 240)

# 관절 이름 -> 각도의 꼭짓점이 되는 랜드마크 인덱스 (예: left_elbow -> 13)
JOINT_INDEX = {name: idx[1] for name, idx in pr.ANGLE_DEFS.items()}


def ui_scale(frame):
    """720p 기준 UI 배율. 해상도가 달라도 레이아웃 비율이 유지된다."""
    return frame.shape[0] / 720.0


def lerp_color(c1, c2, t):
    t = min(1.0, max(0.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def score_color(similarity):
    """0~1 유사도 -> 빨강/노랑/초록"""
    if similarity < 0.5:
        return lerp_color(RED, YELLOW, similarity / 0.5)
    return lerp_color(YELLOW, GREEN, (similarity - 0.5) / 0.5)


def blend_rect(frame, p1, p2, color=BLACK, alpha=0.5):
    """반투명 사각형"""
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(p1[0])), max(0, int(p1[1]))
    x2, y2 = min(w, int(p2[0])), min(h, int(p2[1]))
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    if color[0] == color[1] == color[2]:  # 무채색은 오버레이 이미지 없이 바로 계산 (훨씬 빠름)
        frame[y1:y2, x1:x2] = cv2.convertScaleAbs(roi, alpha=1 - alpha, beta=color[0] * alpha)
        return
    overlay = np.empty_like(roi)
    overlay[:] = color
    frame[y1:y2, x1:x2] = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)


def dim(frame, alpha, color=BLACK):
    blend_rect(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, alpha)


def progress_bar(frame, x, y, w, h, ratio, color, border=True):
    ratio = min(1.0, max(0.0, ratio))
    blend_rect(frame, (x, y), (x + w, y + h), (40, 40, 40), 0.6)
    if ratio > 0:
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w * ratio), int(y + h)), color, -1)
    if border:
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), WHITE, 2, cv2.LINE_AA)


def ring(frame, center, radius, ratio, color, thickness):
    center = (int(center[0]), int(center[1]))
    cv2.circle(frame, center, int(radius), (70, 70, 70), thickness, cv2.LINE_AA)
    if ratio > 0:
        cv2.ellipse(frame, center, (int(radius), int(radius)), -90, 0, 360 * min(1.0, ratio),
                    color, thickness, cv2.LINE_AA)


def _is_visible(lm):
    v = getattr(lm, "visibility", None)
    return v is None or v >= pr.MIN_VISIBILITY


def draw_skeleton(frame, landmarks, joint_scores=None):
    """플레이어 스켈레톤. joint_scores 가 있으면 관절별로 맞았는지 색으로 표시한다."""
    h, w = frame.shape[:2]
    s = ui_scale(frame)
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    vis = [_is_visible(lm) for lm in landmarks]

    for i1, i2 in pr.POSE_CONNECTIONS:
        if vis[i1] and vis[i2]:
            cv2.line(frame, pts[i1], pts[i2], (230, 230, 230), max(2, int(4 * s)), cv2.LINE_AA)

    if joint_scores:
        r = max(6, int(13 * s))
        for name, sim in joint_scores.items():
            idx = JOINT_INDEX.get(name)
            if idx is None or not vis[idx]:
                continue
            cv2.circle(frame, pts[idx], r, score_color(sim), -1, cv2.LINE_AA)
            cv2.circle(frame, pts[idx], r, WHITE, 2, cv2.LINE_AA)
    else:
        for i in {i for pair in pr.POSE_CONNECTIONS for i in pair}:
            if vis[i]:
                cv2.circle(frame, pts[i], max(3, int(5 * s)), GREEN, -1, cv2.LINE_AA)


def draw_pose_figure(frame, angles, box, color=YELLOW):
    """목표 관절 각도로 졸라맨 그림을 그린다 (화면은 거울 모드 기준).

    각도는 방향이 없는 값이라, 팔/다리는 몸 바깥쪽으로 벌어지고
    팔꿈치는 안쪽(머리 쪽)으로 굽는다고 가정한다.
    """
    x1, y1, x2, y2 = box
    L = min((y2 - y1) / 2.4, (x2 - x1) / 1.95)  # 기준 길이
    cx = (x1 + x2) / 2
    top = y1 + ((y2 - y1) - 2.36 * L) / 2         # 세로 가운데 정렬
    sy = top + 0.72 * L                             # 어깨 높이
    th = max(3, int(L * 0.09))

    def angle(name, default):
        return math.radians(angles.get(name, default))

    def step(origin, a, length, side):
        # a: 아래 방향(0)에서 몸 바깥쪽으로 회전한 각도
        return (origin[0] + side * math.sin(a) * length, origin[1] + math.cos(a) * length)

    segments, joints = [], []
    # 거울 화면에서는 MediaPipe 의 right_* 관절이 화면 왼쪽에 나타난다.
    for side, prefix in ((-1, "right"), (+1, "left")):
        shoulder = (cx + side * 0.22 * L, sy)
        hip = (cx + side * 0.13 * L, sy + 0.8 * L)

        a_arm = angle(f"{prefix}_shoulder", 15)
        elbow = step(shoulder, a_arm, 0.36 * L, side)
        bend = math.pi - angle(f"{prefix}_elbow", 170)
        # 팔이 아래를 향하면 몸쪽으로, 위를 향하면 머리 쪽으로 굽힌다
        a_fore = a_arm + bend if a_arm > math.pi / 2 else a_arm - bend
        wrist = step(elbow, a_fore, 0.34 * L, side)

        a_thigh = math.pi - angle(f"{prefix}_hip", 175)
        knee = step(hip, a_thigh, 0.42 * L, side)
        a_shin = a_thigh - (math.pi - angle(f"{prefix}_knee", 175))
        ankle = step(knee, a_shin, 0.42 * L, side)

        segments += [(shoulder, elbow), (elbow, wrist), (shoulder, hip), (hip, knee), (knee, ankle)]
        joints += [shoulder, elbow, wrist, hip, knee, ankle]

    segments += [((cx - 0.22 * L, sy), (cx + 0.22 * L, sy)),
                 ((cx - 0.13 * L, sy + 0.8 * L), (cx + 0.13 * L, sy + 0.8 * L)),
                 ((cx, sy), (cx, sy - 0.11 * L))]

    def ip(p):
        return (int(p[0]), int(p[1]))

    for a, b in segments:
        cv2.line(frame, ip(a), ip(b), color, th, cv2.LINE_AA)
    for p in joints:
        cv2.circle(frame, ip(p), int(th * 0.9), color, -1, cv2.LINE_AA)
    cv2.circle(frame, ip((cx, sy - 0.25 * L)), int(0.14 * L), color, th, cv2.LINE_AA)

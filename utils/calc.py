import math

import numpy as np

import params as pr


def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def _visibility(lm):
    v = getattr(lm, "visibility", None)
    return 1.0 if v is None else v


def extract_angles(landmarks, width, height):
    """화면에 보이는 관절의 각도만 계산한다.

    랜드마크 x, y 는 가로/세로 각각 0~1 로 정규화돼 있으므로 픽셀 좌표로
    되돌려야 16:9 화면에서 각도가 찌그러지지 않는다.
    """
    angles = {}
    for name, idxs in pr.ANGLE_DEFS.items():
        pts = [landmarks[i] for i in idxs]
        if min(_visibility(p) for p in pts) < pr.MIN_VISIBILITY:
            continue
        a, b, c = [(p.x * width, p.y * height) for p in pts]
        angles[name] = calc_angle(a, b, c)
    return angles


def calc_joint_scores(current_angles, target_angles):
    """관절별 유사도(0~1). 화면에 안 보이는 관절은 0점 처리한다."""
    scores = {}
    for name, target in target_angles.items():
        current = current_angles.get(name)
        if current is None:
            scores[name] = 0.0
        else:
            scores[name] = max(0.0, 1 - abs(current - target) / pr.ANGLE_TOLERANCE)
    return scores


def calc_match_score(joint_scores):
    """관절별 유사도의 가중 평균 (0~100)"""
    total_weight = sum(pr.WEIGHTS.get(n, 1.0) for n in joint_scores)
    if not total_weight:
        return 0.0
    total = sum(s * pr.WEIGHTS.get(n, 1.0) for n, s in joint_scores.items())
    return total / total_weight * 100

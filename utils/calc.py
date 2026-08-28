import numpy as np
import math

import params as pr

def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def extract_angles(landmarks):
    angles = {}
    for name, (i1, i2, i3) in pr.ANGLE_DEFS.items():
        a = (landmarks[i1].x, landmarks[i1].y)
        b = (landmarks[i2].x, landmarks[i2].y)
        c = (landmarks[i3].x, landmarks[i3].y)
        angles[name] = calc_angle(a, b, c)
    return angles


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
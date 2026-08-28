import cv2

import params as pr

def draw_skeleton(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for i1, i2 in pr.POSE_CONNECTIONS:
        cv2.line(frame, pts[i1], pts[i2], (0, 200, 255), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 255, 0), -1)


def put_text_center(img, text, y, scale=1.2, color=(255, 255, 255), thickness=2):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (img.shape[1] - w) // 2
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
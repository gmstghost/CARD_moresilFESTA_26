"""상태별 화면 그리기. 게임 상태(g)는 읽기만 하고 바꾸지 않는다."""
import math

import cv2

import params as pr
from . import widgets as wg
from .widgets import WHITE, GRAY, YELLOW, GREEN, ORANGE, RED


def _player_status_line(frame, text, g, y):
    h, w = frame.shape[:2]
    s = wg.ui_scale(frame)
    if g.player_status == "ok":
        msg, color = "플레이어 인식됨", GREEN
    elif g.player_status == "partial":
        msg, color = "전신이 다 보이도록 한 걸음 뒤로 가주세요", YELLOW
    else:
        msg, color = "카메라 앞에 서주세요", GRAY
    text.add(msg, (w // 2, y), 30 * s, color, outline=2)


def draw_wait(frame, text, g, now):
    h, w = frame.shape[:2]
    s = wg.ui_scale(frame)
    wg.dim(frame, 0.5)

    text.add("포즈를 맞춰라!", (w // 2, h * 0.17), 100 * s, YELLOW, outline=int(5 * s))
    text.add("그림과 똑같은 자세를 잡으면 상품 증정!", (w // 2, h * 0.17 + 85 * s), 36 * s, WHITE, outline=2)

    # 포즈 미리보기 (1.5초마다 바뀜)
    names = g.pose_mgr.get_pose_names()
    if names:
        name = names[int(now / 1.5) % len(names)]
        box = (w / 2 - 170 * s, h * 0.36, w / 2 + 170 * s, h * 0.68)
        wg.blend_rect(frame, (box[0], box[1]), (box[2], box[3]), alpha=0.35)
        wg.draw_pose_figure(frame, g.pose_mgr.get_pose_angles(name), box, WHITE)

    text.add(f"도전 비용  {pr.COIN_PRICE:,}원", (w // 2, h * 0.74), 40 * s, WHITE, outline=2)
    blink = 0.5 + 0.5 * math.sin(now * 5)
    text.add("SPACE 를 눌러 시작!", (w // 2, h * 0.83), 54 * s,
             wg.lerp_color((90, 90, 90), YELLOW, blink), outline=int(3 * s))
    _player_status_line(frame, text, g, int(h * 0.94))


def _target_panel(frame, text, box, angles, title="따라하세요"):
    s = wg.ui_scale(frame)
    x1, y1, x2, y2 = box
    wg.blend_rect(frame, (x1, y1), (x2, y2), alpha=0.55)
    text.add(title, ((x1 + x2) / 2, y1 + 24 * s), 26 * s, WHITE)
    pad = 12 * s
    wg.draw_pose_figure(frame, angles, (x1 + pad, y1 + 48 * s, x2 - pad, y2 - pad), YELLOW)


def _top_banner(frame, text, pose_name, height):
    w = frame.shape[1]
    s = wg.ui_scale(frame)
    wg.blend_rect(frame, (0, 0), (w, height), alpha=0.6)
    text.add(pose_name, (w // 2, height / 2), 50 * s, YELLOW, outline=int(3 * s))


def draw_countdown(frame, text, g, now):
    h, w = frame.shape[:2]
    s = wg.ui_scale(frame)
    wg.dim(frame, 0.3)
    _top_banner(frame, text, g.current_pose_name, 100 * s)

    _target_panel(frame, text, (w * 0.62, 130 * s, w - 30 * s, h - 40 * s),
                  g.target_angles, "거울처럼 따라하세요")

    elapsed = now - g.state_start
    remain = max(1, pr.COUNTDOWN_SEC - int(elapsed))
    frac = elapsed % 1.0
    size = 260 * s * (1.3 - 0.3 * min(1.0, frac * 3))  # 숫자가 바뀔 때 '쿵' 하는 느낌
    cx = w * 0.31
    text.add(str(remain), (cx, h * 0.5), size, YELLOW, outline=int(6 * s))
    text.add("자세를 준비하세요!", (cx, h * 0.5 + 170 * s), 42 * s, WHITE, outline=2)


def draw_playing(frame, text, g, now):
    h, w = frame.shape[:2]
    s = wg.ui_scale(frame)
    banner_h = 100 * s
    _top_banner(frame, text, g.current_pose_name, banner_h)

    # 남은 시간 (숫자 + 줄어드는 바). 급할 때는 빨갛게 깜빡임
    remain = g.remaining_time(now)
    urgent = remain <= pr.URGENT_TIME
    t_color = RED if urgent and (now * 4) % 2 < 1 else WHITE
    text.add(f"{remain:.1f}", (w - 40 * s, banner_h / 2 - 4 * s), 56 * s, t_color, anchor="rm", outline=2)
    text.add("초", (w - 16 * s, banner_h / 2 + 8 * s), 24 * s, t_color, anchor="rm")
    wg.progress_bar(frame, 0, banner_h, w, 10 * s, remain / pr.CHALLENGE_TIME,
                    RED if urgent else YELLOW, border=False)

    # 목표 포즈 그림
    _target_panel(frame, text, (w - 250 * s, banner_h + 30 * s, w - 20 * s, banner_h + 330 * s),
                  g.target_angles)

    # 플레이어 인식 경고
    if g.player_status != "ok":
        msg = ("카메라 앞에 서주세요!" if g.player_status == "none"
               else "전신이 보이도록 한 걸음 뒤로!")
        wg.blend_rect(frame, (w * 0.18, h * 0.42), (w * 0.82, h * 0.42 + 80 * s), RED, 0.6)
        text.add(msg, (w // 2, h * 0.42 + 40 * s), 44 * s, WHITE, outline=2)

    # 하단 일치율 게이지
    panel_y = h - 130 * s
    wg.blend_rect(frame, (0, panel_y), (w, h), alpha=0.6)
    holding = g.hold_start is not None
    bar_color = GREEN if holding else wg.score_color(g.score / 100)
    text.add("일치율", (40 * s, panel_y + 38 * s), 28 * s, WHITE, anchor="lm")
    text.add(f"{g.score:.0f}%", (40 * s, panel_y + 88 * s), 58 * s, bar_color, anchor="lm", outline=2)

    bx, bw, by, bh = 230 * s, w - 400 * s, panel_y + 62 * s, 40 * s
    wg.progress_bar(frame, bx, by, bw, bh, g.score / 100, bar_color)
    tx = int(bx + bw * pr.SUCCESS_THRESHOLD / 100)
    cv2.line(frame, (tx, int(by - 10 * s)), (tx, int(by + bh + 6 * s)), WHITE, max(2, int(3 * s)), cv2.LINE_AA)
    text.add(f"목표 {pr.SUCCESS_THRESHOLD}%", (tx, by - 26 * s), 24 * s, WHITE)
    if holding:
        text.add("좋아요! 그대로 유지!", (bx + bw / 2 - 120 * s, by - 26 * s), 30 * s, GREEN, outline=2)

    # 유지 진행 링
    rc = (w - 85 * s, panel_y + 65 * s)
    wg.ring(frame, rc, 45 * s, g.hold_ratio(now), GREEN, max(4, int(9 * s)))
    text.add("유지", rc, 26 * s, GREEN if holding else GRAY)


def draw_result(frame, text, g, now):
    h, w = frame.shape[:2]
    s = wg.ui_scale(frame)
    elapsed = now - g.state_start

    if g.success and elapsed < 0.5:  # 성공 순간 번쩍
        wg.dim(frame, 0.8 * (1 - elapsed / 0.5), WHITE)
    wg.dim(frame, 0.55)

    if g.success:
        pop = 1 + 0.25 * max(0.0, 1 - elapsed / 0.3)
        text.add("성공!", (w // 2, h * 0.3), 170 * s * pop, GREEN, outline=int(6 * s))
        text.add(f"일치율 {g.final_score:.0f}%", (w // 2, h * 0.3 + 135 * s), 50 * s, WHITE, outline=2)
        text.add("상품을 받아가세요!", (w // 2, h * 0.3 + 205 * s), 46 * s, YELLOW, outline=2)
    else:
        text.add("아쉬워요!", (w // 2, h * 0.3), 150 * s, ORANGE, outline=int(6 * s))
        text.add(f"최고 일치율 {g.final_score:.0f}%   (목표 {pr.SUCCESS_THRESHOLD}%)",
                 (w // 2, h * 0.3 + 125 * s), 44 * s, WHITE, outline=2)
        text.add("다시 도전해 보세요!", (w // 2, h * 0.3 + 195 * s), 46 * s, YELLOW, outline=2)

    blink = 0.5 + 0.5 * math.sin(now * 5)
    text.add("SPACE 를 눌러 다시 도전", (w // 2, h * 0.8), 40 * s,
             wg.lerp_color((120, 120, 120), WHITE, blink), outline=2)

    remain = max(0.0, pr.RESULT_DISPLAY_SEC - elapsed)
    bw = 360 * s
    wg.progress_bar(frame, (w - bw) / 2, h * 0.88, bw, 8 * s, remain / pr.RESULT_DISPLAY_SEC, GRAY, border=False)
    text.add(f"{math.ceil(remain)}초 후 처음 화면으로", (w // 2, h * 0.88 + 30 * s), 24 * s, GRAY)


DRAWERS = {
    pr.STATE_WAIT_COIN: draw_wait,
    pr.STATE_COUNTDOWN: draw_countdown,
    pr.STATE_PLAYING: draw_playing,
    pr.STATE_RESULT: draw_result,
}

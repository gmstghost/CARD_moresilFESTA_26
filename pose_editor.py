"""목표 포즈 에디터 (pose/poses.json 편집기)

    python pose_editor.py

- 왼쪽: 포즈 목록 (추가 / 복제 / 이름 변경 / 삭제)
- 가운데: 관절별 목표 각도 슬라이더 (좌우 대칭 편집 지원)
- 오른쪽: 게임과 같은 졸라맨 미리보기, 또는 웹캠 화면
  · 카메라를 켜면 지금 편집 중인 포즈로 일치율을 바로 테스트할 수 있고
  · '캡처' 버튼으로 실제 자세의 각도를 그대로 가져올 수 있다.
- Ctrl+S 저장
"""
import json
import os
import time
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import params as pr
import utils as ut
from ui import TextLayer, widgets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSES_PATH = os.path.join(BASE_DIR, "pose", "poses.json")

JOINT_LABELS = {
    "left_elbow": "왼쪽 팔꿈치", "right_elbow": "오른쪽 팔꿈치",
    "left_shoulder": "왼쪽 어깨", "right_shoulder": "오른쪽 어깨",
    "left_hip": "왼쪽 골반", "right_hip": "오른쪽 골반",
    "left_knee": "왼쪽 무릎", "right_knee": "오른쪽 무릎",
}
JOINTS = list(pr.ANGLE_DEFS.keys())
DEFAULT_POSE = {  # 새 포즈 기본값 = 차렷
    "left_elbow": 170, "right_elbow": 170,
    "left_shoulder": 15, "right_shoulder": 15,
    "left_hip": 175, "right_hip": 175,
    "left_knee": 175, "right_knee": 175,
}
PREVIEW_W, PREVIEW_H = 640, 480
CAPTURE_DELAY = 3  # 캡처 버튼을 누르고 실제로 찍기까지 (자리 잡을 시간)
FONT = ("Malgun Gothic", 10)
FONT_B = ("Malgun Gothic", 10, "bold")


def mirror_name(joint):
    if joint.startswith("left_"):
        return "right_" + joint[5:]
    return "left_" + joint[6:]


def dump_poses(poses):
    """기존 poses.json 과 같은 모양(좌/우 한 줄씩)으로 직렬화."""
    blocks = []
    for name, angles in poses.items():
        keys = [k for k in JOINTS if k in angles] + [k for k in angles if k not in JOINTS]
        items = [f"{json.dumps(k)}: {int(round(angles[k]))}" for k in keys]
        rows = [", ".join(items[i:i + 2]) for i in range(0, len(items), 2)]
        body = ",\n".join("        " + r for r in rows)
        blocks.append(f"    {json.dumps(name, ensure_ascii=False)}: {{\n{body}\n    }}")
    return "{\n" + ",\n".join(blocks) + "\n}\n"


class Camera:
    """웹캠 + MediaPipe. 카메라를 켤 때만 무거운 모듈을 불러온다."""

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        from game import MODEL_PATH, ensure_model

        ensure_model()
        self.mp = mp
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self.cap = cv2.VideoCapture(pr.CAMERA_INDEX)
        if not self.cap.isOpened():
            self.landmarker.close()
            raise RuntimeError(f"웹캠({pr.CAMERA_INDEX}번)을 열 수 없습니다. params.CAMERA_INDEX 를 확인하세요.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, pr.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, pr.CAMERA_HEIGHT)
        self.last_ts = -1

    def read(self):
        """(거울 모드 프레임, 랜드마크 or None)"""
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(time.monotonic() * 1000), self.last_ts + 1)
        self.last_ts = ts
        result = self.landmarker.detect_for_video(image, ts)
        return frame, (result.pose_landmarks[0] if result.pose_landmarks else None)

    def close(self):
        self.cap.release()
        self.landmarker.close()


class PoseEditor:
    def __init__(self, root):
        self.root = root
        self.poses = self._load()
        self.current = None          # 선택된 포즈 이름
        self.dirty = False
        self._syncing = False        # 슬라이더 연쇄 콜백 방지
        self.camera = None
        self.live_angles = {}
        self.capture_at = None       # 캡처 예정 시각
        self.text = TextLayer()

        root.title("포즈 에디터")
        root.option_add("*Font", FONT)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.bind("<Control-s>", lambda e: self.save())

        self._build_ui()
        self._refresh_list()
        if self.poses:
            self.select(next(iter(self.poses)))
        self._render_preview()

    # ------------------------------------------------------------------
    # 데이터
    # ------------------------------------------------------------------
    def _load(self):
        if not os.path.exists(POSES_PATH):
            return {}
        with open(POSES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self):
        text = dump_poses(self.poses)
        tmp = POSES_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, POSES_PATH)
        self._set_dirty(False)
        self.status.set(f"저장 완료: {os.path.relpath(POSES_PATH, BASE_DIR)} ({len(self.poses)}개 포즈)")

    def revert(self):
        if self.dirty and not messagebox.askyesno("되돌리기", "저장하지 않은 변경 사항을 버리고 파일에서 다시 불러올까요?"):
            return
        self.poses = self._load()
        self._set_dirty(False)
        self._refresh_list()
        self.select(self.current if self.current in self.poses else next(iter(self.poses), None))
        self.status.set("파일에서 다시 불러왔습니다.")

    def _set_dirty(self, dirty):
        self.dirty = dirty
        self.root.title("포즈 에디터" + (" *" if dirty else ""))

    # ------------------------------------------------------------------
    # UI 구성
    # ------------------------------------------------------------------
    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        # 왼쪽: 포즈 목록
        left = ttk.Frame(main)
        left.pack(side="left", fill="y")
        ttk.Label(left, text="포즈 목록", font=FONT_B).pack(anchor="w")
        self.listbox = tk.Listbox(left, width=26, height=20, exportselection=False, activestyle="none")
        self.listbox.pack(fill="y", expand=True, pady=(4, 6))
        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)
        self.listbox.bind("<Double-Button-1>", lambda e: self.rename_pose())
        for label, cmd in (("새 포즈", self.add_pose), ("복제", self.duplicate_pose),
                           ("이름 변경", self.rename_pose), ("삭제", self.delete_pose)):
            ttk.Button(left, text=label, command=cmd).pack(fill="x", pady=1)

        # 가운데: 각도 슬라이더
        mid = ttk.Frame(main, padding=(14, 0))
        mid.pack(side="left", fill="y")
        self.title_var = tk.StringVar()
        ttk.Label(mid, textvariable=self.title_var, font=("Malgun Gothic", 13, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        self.vars, self.scales, self.live_labels = {}, {}, {}
        for i, joint in enumerate(JOINTS, start=1):
            ttk.Label(mid, text=JOINT_LABELS.get(joint, joint), width=12).grid(row=i, column=0, sticky="w")
            var = tk.IntVar(value=0)
            scale = tk.Scale(mid, from_=0, to=180, orient="horizontal", length=260, variable=var,
                             resolution=1, command=lambda v, j=joint: self._on_scale(j))
            scale.grid(row=i, column=1, sticky="we")
            live = ttk.Label(mid, text="", width=10, foreground="#888")
            live.grid(row=i, column=2, sticky="w", padx=(6, 0))
            self.vars[joint], self.scales[joint], self.live_labels[joint] = var, scale, live

        row = len(JOINTS) + 1
        self.mirror_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(mid, text="좌우 대칭으로 같이 움직이기", variable=self.mirror_var).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(8, 2))
        sym = ttk.Frame(mid)
        sym.grid(row=row + 1, column=0, columnspan=3, sticky="w")
        ttk.Button(sym, text="왼쪽 → 오른쪽 복사", command=lambda: self.copy_side("left")).pack(side="left")
        ttk.Button(sym, text="오른쪽 → 왼쪽 복사", command=lambda: self.copy_side("right")).pack(side="left", padx=4)
        ttk.Label(mid, text="0° = 완전히 접힘, 180° = 쭉 폄\n어깨: 0° 팔 내림 / 90° 옆으로 / 180° 위로",
                  foreground="#666").grid(row=row + 2, column=0, columnspan=3, sticky="w", pady=(10, 0))

        # 오른쪽: 미리보기 / 카메라
        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)
        self.preview = tk.Label(right, bg="black", width=PREVIEW_W, height=PREVIEW_H)
        self.preview.pack()
        cam_bar = ttk.Frame(right)
        cam_bar.pack(fill="x", pady=(6, 0))
        self.cam_btn = ttk.Button(cam_bar, text="카메라 켜기", command=self.toggle_camera)
        self.cam_btn.pack(side="left")
        self.cap_btn = ttk.Button(cam_bar, text=f"현재 자세 캡처 ({CAPTURE_DELAY}초 후)",
                                  command=self.start_capture, state="disabled")
        self.cap_btn.pack(side="left", padx=4)

        # 하단
        bottom = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        bottom.pack(fill="x")
        ttk.Button(bottom, text="저장 (Ctrl+S)", command=self.save).pack(side="right")
        ttk.Button(bottom, text="파일에서 다시 불러오기", command=self.revert).pack(side="right", padx=4)
        self.status = tk.StringVar(value=os.path.relpath(POSES_PATH, BASE_DIR))
        ttk.Label(bottom, textvariable=self.status, foreground="#555").pack(side="left")

    # ------------------------------------------------------------------
    # 목록 조작
    # ------------------------------------------------------------------
    def _refresh_list(self):
        self.listbox.delete(0, "end")
        for name in self.poses:
            self.listbox.insert("end", name)

    def _on_list_select(self, _event):
        sel = self.listbox.curselection()
        if sel:
            self.select(self.listbox.get(sel[0]))

    def select(self, name):
        self.current = name
        names = list(self.poses)
        self.listbox.selection_clear(0, "end")
        state = "normal" if name else "disabled"
        if name:
            idx = names.index(name)
            self.listbox.selection_set(idx)
            self.listbox.see(idx)
        self.title_var.set(name or "(포즈 없음)")
        self._syncing = True
        for joint in JOINTS:
            self.vars[joint].set(int(round(self.poses[name].get(joint, 0))) if name else 0)
            self.scales[joint].configure(state=state)
        self._syncing = False
        self._render_preview()

    def _ask_name(self, title, initial=""):
        name = simpledialog.askstring(title, "포즈 이름:", initialvalue=initial, parent=self.root)
        if name is None:
            return None
        name = name.strip()
        if not name:
            return None
        if name in self.poses and name != initial:
            messagebox.showwarning(title, f"'{name}' 은(는) 이미 있는 이름입니다.")
            return None
        return name

    def add_pose(self):
        name = self._ask_name("새 포즈")
        if name:
            self.poses[name] = dict(DEFAULT_POSE)
            self._after_list_change(name)

    def duplicate_pose(self):
        if not self.current:
            return
        name = self._ask_name("포즈 복제", self.current + " (복사본)")
        if name:
            self.poses[name] = dict(self.poses[self.current])
            self._after_list_change(name)

    def rename_pose(self):
        if not self.current:
            return
        old = self.current
        name = self._ask_name("이름 변경", old)
        if name and name != old:
            # 순서를 유지한 채 키만 바꾼다
            self.poses = {(name if k == old else k): v for k, v in self.poses.items()}
            self._after_list_change(name)

    def delete_pose(self):
        if not self.current:
            return
        if not messagebox.askyesno("삭제", f"'{self.current}' 포즈를 삭제할까요?"):
            return
        names = list(self.poses)
        idx = names.index(self.current)
        del self.poses[self.current]
        rest = list(self.poses)
        self._after_list_change(rest[min(idx, len(rest) - 1)] if rest else None)

    def _after_list_change(self, select_name):
        self._set_dirty(True)
        self._refresh_list()
        self.select(select_name)

    # ------------------------------------------------------------------
    # 각도 편집
    # ------------------------------------------------------------------
    def _on_scale(self, joint):
        if self._syncing or not self.current:
            return
        value = self.vars[joint].get()
        self.poses[self.current][joint] = value
        if self.mirror_var.get():
            other = mirror_name(joint)
            if other in self.vars:
                self._syncing = True
                self.vars[other].set(value)
                self._syncing = False
                self.poses[self.current][other] = value
        self._set_dirty(True)
        self._render_preview()

    def copy_side(self, src):
        if not self.current:
            return
        pose = self.poses[self.current]
        for joint in JOINTS:
            if joint.startswith(src + "_"):
                pose[mirror_name(joint)] = pose.get(joint, 0)
        self._set_dirty(True)
        self.select(self.current)

    def apply_angles(self, angles):
        if not self.current:
            return
        pose = self.poses[self.current]
        for joint, value in angles.items():
            pose[joint] = int(round(value))
        self._set_dirty(True)
        self.select(self.current)

    # ------------------------------------------------------------------
    # 미리보기 / 카메라
    # ------------------------------------------------------------------
    def _current_angles(self):
        return self.poses.get(self.current, {}) if self.current else {}

    def _show(self, frame_bgr):
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        self._photo = ImageTk.PhotoImage(img)  # 참조를 붙잡아 두지 않으면 GC 로 사라짐
        self.preview.configure(image=self._photo, width=PREVIEW_W, height=PREVIEW_H)

    def _render_preview(self):
        if self.camera:
            return  # 카메라 루프가 그린다
        frame = np.full((PREVIEW_H, PREVIEW_W, 3), 30, np.uint8)
        if self.current:
            widgets.draw_pose_figure(frame, self._current_angles(), (40, 30, PREVIEW_W - 40, PREVIEW_H - 50))
            self.text.add("거울 모드 기준 (화면 왼쪽 = 플레이어 오른쪽)", (PREVIEW_W // 2, PREVIEW_H - 22),
                          16, widgets.GRAY, bold=False)
        self.text.render(frame)
        self._show(frame)

    def toggle_camera(self):
        if self.camera:
            self._stop_camera()
            return
        self.status.set("카메라 / 포즈 모델 준비 중...")
        self.root.update_idletasks()
        try:
            self.camera = Camera()
        except Exception as e:
            self.camera = None
            messagebox.showerror("카메라", str(e))
            self.status.set("카메라를 열지 못했습니다.")
            return
        self.cam_btn.configure(text="카메라 끄기")
        self.cap_btn.configure(state="normal")
        self.status.set("카메라 켜짐: 편집 중인 포즈로 일치율을 테스트할 수 있습니다.")
        self._camera_loop()

    def _stop_camera(self):
        if self.camera:
            self.camera.close()
        self.camera = None
        self.capture_at = None
        self.live_angles = {}
        for label in self.live_labels.values():
            label.configure(text="")
        self.cam_btn.configure(text="카메라 켜기")
        self.cap_btn.configure(state="disabled")
        self._render_preview()

    def start_capture(self):
        self.capture_at = time.monotonic() + CAPTURE_DELAY

    def _camera_loop(self):
        if not self.camera:
            return
        frame, landmarks = self.camera.read()
        if frame is None:
            self._stop_camera()
            messagebox.showerror("카메라", "카메라 프레임을 읽지 못했습니다.")
            return

        h, w = frame.shape[:2]
        s = widgets.ui_scale(frame)
        target = self._current_angles()
        self.live_angles = ut.extract_angles(landmarks, w, h) if landmarks is not None else {}
        scores = ut.calc_joint_scores(self.live_angles, target) if target and landmarks is not None else None
        if landmarks is not None:
            widgets.draw_skeleton(frame, landmarks, scores)

        # 오른쪽 위에 목표 포즈 그림, 왼쪽 위에 일치율
        bw, bh = int(220 * s), int(260 * s)
        widgets.blend_rect(frame, (w - bw - 20, 20), (w - 20, 20 + bh), alpha=0.55)
        widgets.draw_pose_figure(frame, target, (w - bw - 10, 30, w - 30, 10 + bh))
        if scores:
            match = ut.calc_match_score(scores)
            color = widgets.GREEN if match >= pr.SUCCESS_THRESHOLD else widgets.WHITE
            self.text.add(f"일치율 {match:.0f}%", (30, 50 * s), 44 * s, color, anchor="lm", outline=3)
        elif landmarks is None:
            self.text.add("사람이 안 보여요", (30, 50 * s), 36 * s, widgets.ORANGE, anchor="lm", outline=3)

        # 관절별 실시간 각도
        for joint, label in self.live_labels.items():
            v = self.live_angles.get(joint)
            label.configure(text=f"현재 {v:.0f}°" if v is not None else "현재 -")

        # 캡처 카운트다운
        if self.capture_at is not None:
            left = self.capture_at - time.monotonic()
            if left > 0:
                self.text.add(str(int(left) + 1), (w // 2, h // 2), 200 * s, widgets.YELLOW, outline=6)
            else:
                self.capture_at = None
                self._do_capture()
                widgets.dim(frame, 0.6, widgets.WHITE)  # 플래시

        self.text.render(frame)
        self._show(cv2.resize(frame, (PREVIEW_W, int(PREVIEW_W * h / w))))
        self.root.after(15, self._camera_loop)

    def _do_capture(self):
        if not self.live_angles:
            self.status.set("캡처 실패: 사람이 인식되지 않았습니다.")
            return
        missing = [JOINT_LABELS.get(j, j) for j in JOINTS if j not in self.live_angles]
        self.apply_angles(self.live_angles)
        msg = f"'{self.current}' 에 현재 자세를 캡처했습니다."
        if missing:
            msg += f"  (안 보여서 기존 값 유지: {', '.join(missing)})"
        self.status.set(msg)

    # ------------------------------------------------------------------
    def on_close(self):
        if self.dirty:
            ans = messagebox.askyesnocancel("종료", "변경 사항을 저장할까요?")
            if ans is None:
                return
            if ans:
                self.save()
        self._stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    PoseEditor(root)
    root.mainloop()

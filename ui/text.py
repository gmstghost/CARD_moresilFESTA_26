"""한글 텍스트 렌더링.

cv2.putText 의 Hershey 폰트는 ASCII 만 지원하므로 한글이 '???' 로 깨진다.
PIL + 시스템 한글 폰트로 그리되, 프레임당 BGR<->RGB 변환은 한 번만 하도록
텍스트를 모아 두었다가 render() 에서 일괄로 그린다.
"""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_PATHS = {
    True: [  # bold
        "C:/Windows/Fonts/malgunbd.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    ],
    False: [  # regular
        "C:/Windows/Fonts/malgun.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ],
}
_font_cache = {}


def get_font(size, bold=True):
    key = (size, bold)
    font = _font_cache.get(key)
    if font is None:
        for path in _FONT_PATHS[bold]:
            if os.path.exists(path):
                font = ImageFont.truetype(path, size)
                break
        else:
            print("경고: 한글 폰트를 찾지 못했습니다. 글자가 깨질 수 있습니다.")
            font = ImageFont.load_default()
        _font_cache[key] = font
    return font


def _bgr_to_rgb(color):
    return (color[2], color[1], color[0])


_measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))


class TextLayer:
    """한 프레임에 그릴 텍스트를 모았다가 render() 에서 그린다.
    전체 프레임 대신 글자 주변 영역만 PIL 로 변환해서 프레임 시간을 아낀다."""

    def __init__(self):
        self._items = []

    def add(self, text, xy, size, color=(255, 255, 255), anchor="mm", bold=True, outline=0):
        """xy 는 anchor 기준 좌표. anchor: 'mm'(가운데), 'lm'(왼쪽 가운데), 'rm'(오른쪽 가운데) 등.
        color 는 다른 cv2 그리기와 마찬가지로 BGR."""
        # 애니메이션 때문에 크기가 매 프레임 바뀌어도 폰트 로딩이 반복되지 않도록 4px 단위로 맞춤
        size = max(4, int(round(size / 4)) * 4)
        self._items.append((text, (int(xy[0]), int(xy[1])), size,
                            color, anchor, bold, int(outline)))

    def render(self, frame):
        fh, fw = frame.shape[:2]
        for text, xy, size, color, anchor, bold, outline in self._items:
            font = get_font(size, bold)
            try:
                l, t, r, b = _measure.textbbox(xy, text, font=font, anchor=anchor, stroke_width=outline)
            except ValueError:  # 기본 비트맵 폰트는 anchor 미지원
                anchor = None
                l, t, r, b = _measure.textbbox(xy, text, font=font, stroke_width=outline)
            l, t = max(0, int(l) - 2), max(0, int(t) - 2)
            r, b = min(fw, int(r) + 2), min(fh, int(b) + 2)
            if r <= l or b <= t:
                continue
            crop = Image.fromarray(cv2.cvtColor(frame[t:b, l:r], cv2.COLOR_BGR2RGB))
            kwargs = dict(font=font, fill=_bgr_to_rgb(color))
            if outline > 0:
                kwargs.update(stroke_width=outline, stroke_fill=(0, 0, 0))
            if anchor:
                kwargs["anchor"] = anchor
            ImageDraw.Draw(crop).text((xy[0] - l, xy[1] - t), text, **kwargs)
            frame[t:b, l:r] = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR)
        self._items.clear()

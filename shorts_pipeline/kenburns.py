"""4단계(편집): 켄 번즈(확대·이동) 장면 생성.

각 사진을 9:16 프레임에 맞게 배치한다.
- 배경: 사진을 화면 가득 채운 뒤 강하게 블러+어둡게 (숏폼에서 흔한 스타일)
- 전경: 사진 원본 비율을 유지한 채 화면에 맞추고, 시간에 따라 천천히 확대/이동
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


@dataclass
class KenBurnsParams:
    zoom_start: float = 1.0
    zoom_end: float = 1.15
    pan_start: Tuple[float, float] = (0.0, 0.0)  # -1..1 (남는 여백 대비 비율)
    pan_end: Tuple[float, float] = (0.0, 0.0)


def alternating_params(index: int, zoom_min: float, zoom_max: float, rng: np.random.Generator) -> KenBurnsParams:
    """장면마다 확대/축소, 좌→우/우→좌를 번갈아 써서 단조로움을 피한다."""
    jitter = lambda: float(rng.uniform(-0.15, 0.15))  # noqa: E731
    direction = 1 if index % 2 == 0 else -1
    if index % 4 in (0, 1):  # zoom in
        return KenBurnsParams(
            zoom_start=zoom_min,
            zoom_end=zoom_max,
            pan_start=(-0.5 * direction + jitter(), jitter()),
            pan_end=(0.5 * direction + jitter(), jitter()),
        )
    return KenBurnsParams(  # zoom out
        zoom_start=zoom_max,
        zoom_end=zoom_min,
        pan_start=(0.5 * direction + jitter(), jitter()),
        pan_end=(-0.5 * direction + jitter(), jitter()),
    )


class KenBurnsScene:
    """한 장의 사진에 대해 시각 t(초)의 프레임(RGB uint8)을 만들어 준다."""

    def __init__(
        self,
        image_rgb: np.ndarray,
        width: int,
        height: int,
        duration: float,
        params: KenBurnsParams,
        fg_box: Tuple[float, float] = (1.0, 0.72),  # 전경이 차지할 최대 비율(가로, 세로)
        fg_center_y: float = 0.46,  # 전경 중심의 세로 위치(비율). 아래쪽은 자막 공간
        bg_darken: float = 0.55,
    ):
        self.W, self.H = width, height
        self.duration = float(duration)
        self.params = params
        self.bg = self._make_background(image_rgb, bg_darken)

        ih, iw = image_rgb.shape[:2]
        box_w, box_h = int(width * fg_box[0]), int(height * fg_box[1])
        scale = min(box_w / iw, box_h / ih)
        self.win_w = max(2, int(iw * scale) // 2 * 2)
        self.win_h = max(2, int(ih * scale) // 2 * 2)
        self.zmax = max(params.zoom_start, params.zoom_end)
        src_w = int(math.ceil(self.win_w * self.zmax))
        src_h = int(math.ceil(self.win_h * self.zmax))
        interp = cv2.INTER_AREA if src_w < iw else cv2.INTER_CUBIC
        self.src = cv2.resize(image_rgb, (src_w, src_h), interpolation=interp)
        self.x0 = (width - self.win_w) // 2
        self.y0 = int(height * fg_center_y - self.win_h / 2)
        self.y0 = min(max(0, self.y0), height - self.win_h)

    def _make_background(self, img: np.ndarray, darken: float) -> np.ndarray:
        ih, iw = img.shape[:2]
        scale = max(self.W / iw, self.H / ih)
        rw, rh = int(math.ceil(iw * scale)), int(math.ceil(ih * scale))
        resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        x = (rw - self.W) // 2
        y = (rh - self.H) // 2
        crop = resized[y : y + self.H, x : x + self.W]
        small = cv2.resize(crop, (self.W // 8, self.H // 8), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 6)
        bg = cv2.resize(small, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return (bg.astype(np.float32) * (1.0 - darken)).astype(np.uint8)

    def frame(self, t: float) -> np.ndarray:
        p = smoothstep(t / self.duration) if self.duration > 0 else 0.0
        z = self.params.zoom_start + (self.params.zoom_end - self.params.zoom_start) * p
        px = self.params.pan_start[0] + (self.params.pan_end[0] - self.params.pan_start[0]) * p
        py = self.params.pan_start[1] + (self.params.pan_end[1] - self.params.pan_start[1]) * p

        src_h, src_w = self.src.shape[:2]
        cw = int(round(self.win_w * self.zmax / z))
        ch = int(round(self.win_h * self.zmax / z))
        cw, ch = min(cw, src_w), min(ch, src_h)
        slack_x, slack_y = src_w - cw, src_h - ch
        cx = int(round(slack_x / 2 * (1 + max(-1.0, min(1.0, px)))))
        cy = int(round(slack_y / 2 * (1 + max(-1.0, min(1.0, py)))))
        crop = self.src[cy : cy + ch, cx : cx + cw]
        fg = cv2.resize(crop, (self.win_w, self.win_h), interpolation=cv2.INTER_AREA if cw > self.win_w else cv2.INTER_LINEAR)

        out = self.bg.copy()
        out[self.y0 : self.y0 + self.win_h, self.x0 : self.x0 + self.win_w] = fg
        return out

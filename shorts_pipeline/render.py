"""4~6단계: 장면 합성 + 자막 애니메이션 + BGM → mp4 렌더링 (moviepy 2.x + ffmpeg)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from moviepy import AudioFileClip, ColorClip, CompositeVideoClip, ImageClip, VideoClip, afx, vfx

from .captions import CaptionPlan
from .config import ShortsConfig
from .kenburns import KenBurnsScene, alternating_params, smoothstep
from .subtitles import render_text_block

log = logging.getLogger(__name__)


def compute_scene_timing(n_scenes: int, total: float, transition: float) -> List[Tuple[float, float]]:
    """(시작 시각, 길이) 목록. 크로스페이드만큼 겹치게 배치해 총 길이가 total 이 되게 한다."""
    if n_scenes <= 0:
        return []
    if n_scenes == 1:
        return [(0.0, total)]
    transition = min(transition, total / n_scenes / 2)
    d = (total + (n_scenes - 1) * transition) / n_scenes
    return [(i * (d - transition), d) for i in range(n_scenes)]


def _overlay_clip(rgba: np.ndarray, start: float, duration: float, x: int, y: int, fade: float = 0.3, slide: int = 40) -> ImageClip:
    """RGBA 이미지를 (x, y)에 올리되, 아래에서 살짝 떠오르며 페이드인/아웃한다."""
    clip = ImageClip(rgba, transparent=True).with_start(start).with_duration(duration)

    def pos(t: float):
        p = smoothstep(t / 0.45)
        return (x, int(y + slide * (1 - p)))

    return clip.with_position(pos).with_effects([vfx.CrossFadeIn(fade), vfx.CrossFadeOut(fade)])


def _progress_bar_clip(width: int, height: int, total: float, thickness: int = 8, color=(255, 225, 80)) -> VideoClip:
    bar_h = thickness

    def frame(t: float):
        img = np.zeros((bar_h, width, 3), dtype=np.uint8)
        w = int(width * min(1.0, t / total))
        img[:, :w] = color
        return img

    def mask_frame(t: float):
        m = np.zeros((bar_h, width), dtype=np.float32)
        w = int(width * min(1.0, t / total))
        m[:, :w] = 1.0
        return m

    clip = VideoClip(frame, duration=total).with_position((0, height - bar_h))
    return clip.with_mask(VideoClip(mask_frame, is_mask=True, duration=total))


def render_video(
    images_rgb: Sequence[np.ndarray],
    plan: CaptionPlan,
    output: Path,
    cfg: ShortsConfig,
    font_path: Path,
    bgm_path: Optional[Path] = None,
    show_progress: bool = True,
) -> Path:
    W, H, T = cfg.width, cfg.height, cfg.duration
    n = len(images_rgb)
    if n == 0:
        raise ValueError("렌더링할 사진이 없습니다.")
    plan = plan.fitted(n)
    rng = np.random.default_rng(cfg.seed)
    timings = compute_scene_timing(n, T, cfg.transition)

    layers: List[VideoClip] = [ColorClip((W, H), color=(0, 0, 0)).with_duration(T)]

    # 장면(켄 번즈)
    for i, (img, (start, dur)) in enumerate(zip(images_rgb, timings)):
        scene = KenBurnsScene(img, W, H, dur, alternating_params(i, cfg.zoom_min, cfg.zoom_max, rng))
        clip = VideoClip(scene.frame, duration=dur).with_start(start)
        if i > 0:
            clip = clip.with_effects([vfx.CrossFadeIn(min(cfg.transition, dur / 2))])
        layers.append(clip)

    # 장면별 자막 (하단)
    cap_size = int(W * 0.052)
    for i, (text, (start, dur)) in enumerate(zip(plan.captions, timings)):
        rgba = render_text_block(text, font_path, cap_size, max_width=int(W * 0.88))
        x = (W - rgba.shape[1]) // 2
        y = int(H * 0.81) - rgba.shape[0] // 2
        lead = 0.35 if i == 0 else 0.2
        layers.append(_overlay_clip(rgba, start + lead, max(0.5, dur - lead - 0.15), x, y))

    # 제목 (상단, 전체 구간)
    title_rgba = render_text_block(
        plan.title, font_path, int(W * 0.072), max_width=int(W * 0.9),
        color=(255, 228, 90, 255), bg=(12, 12, 24, 175), padding=(36, 22), radius=30,
    )
    layers.append(_overlay_clip(title_rgba, 0.2, T - 0.2, (W - title_rgba.shape[1]) // 2, int(H * 0.11), fade=0.4, slide=-30))

    if show_progress:
        layers.append(_progress_bar_clip(W, H, T))

    video = CompositeVideoClip(layers, size=(W, H)).with_duration(T)

    audio = None
    if bgm_path:
        audio = AudioFileClip(str(bgm_path)).subclipped(0, T)
        audio = audio.with_effects([afx.AudioFadeOut(1.2)]).with_volume_scaled(cfg.bgm_volume)
        video = video.with_audio(audio)

    output.parent.mkdir(parents=True, exist_ok=True)
    log.info("[render] %dx%d %.1fs @%dfps → %s", W, H, T, cfg.fps, output)
    video.write_videofile(
        str(output),
        fps=cfg.fps,
        codec="libx264",
        audio_codec="aac",
        audio_bitrate="192k",
        bitrate=cfg.video_bitrate,
        preset=cfg.preset,
        threads=cfg.threads,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-profile:v", "high"],
        logger="bar" if show_progress else None,
    )
    video.close()
    if audio is not None:
        audio.close()
    return output

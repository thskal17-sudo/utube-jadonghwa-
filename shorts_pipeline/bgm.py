"""5단계: 저작권 프리 BGM 자체 합성 (numpy + scipy).

외부 음원을 전혀 쓰지 않고 코드 진행(I–V–vi–IV 등)을 기반으로
패드 + 베이스 + 아르페지오 + 가벼운 드럼을 합성한다.
스타일 프리셋은 Phase 3(활동 유형별 템플릿)의 확장 지점이다.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import signal
from scipy.io import wavfile

SAMPLE_RATE = 44100

STYLES: Dict[str, dict] = {
    # 밝고 경쾌 (일반 수업/행사)
    "bright": dict(
        bpm=112,
        chords=[[60, 64, 67], [67, 71, 74], [57, 60, 64], [65, 69, 72]],  # C G Am F
        arp=True, drums="light", cutoff=6500, pad_gain=0.16, arp_gain=0.20,
    ),
    # 잔잔함 (미술/독서/감성)
    "calm": dict(
        bpm=84,
        chords=[[60, 64, 67, 71], [57, 60, 64, 67], [65, 69, 72, 76], [67, 71, 74, 77]],
        arp=False, drums="none", cutoff=3800, pad_gain=0.18, arp_gain=0.0,
    ),
    # 활기참 (체육대회/실험)
    "energetic": dict(
        bpm=128,
        chords=[[57, 60, 64], [65, 69, 72], [60, 64, 67], [67, 71, 74]],  # Am F C G
        arp=True, drums="full", cutoff=8500, pad_gain=0.14, arp_gain=0.22,
    ),
}


def midi_to_hz(note: float) -> float:
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


def adsr(n: int, sr: int, a: float, d: float, s: float, r: float) -> np.ndarray:
    a_n, d_n, r_n = int(a * sr), int(d * sr), int(r * sr)
    a_n = min(a_n, n)
    d_n = min(d_n, max(0, n - a_n))
    r_n = min(r_n, max(0, n - a_n - d_n))
    s_n = max(0, n - a_n - d_n - r_n)
    env = np.concatenate([
        np.linspace(0, 1, a_n, endpoint=False),
        np.linspace(1, s, d_n, endpoint=False),
        np.full(s_n, s),
        np.linspace(s, 0, r_n),
    ])
    if len(env) < n:
        env = np.pad(env, (0, n - len(env)))
    return env[:n]


def pad_tone(freq: float, dur: float, sr: int) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    wave = (
        np.sin(2 * np.pi * freq * t)
        + 0.35 * np.sin(2 * np.pi * freq * 2 * t)
        + 0.12 * np.sin(2 * np.pi * freq * 3 * t)
        + 0.5 * np.sin(2 * np.pi * freq * 1.004 * t)  # 살짝 디튠 → 두께감
    )
    return wave * adsr(len(t), sr, 0.25, 0.3, 0.75, 0.5)


def pluck(freq: float, dur: float, sr: int, decay: float = 6.0) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    wave = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * freq * 2 * t)
    return wave * np.exp(-decay * t)


def kick(sr: int, dur: float = 0.2) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    freq = 130 * np.exp(-25 * t) + 45
    phase = 2 * np.pi * np.cumsum(freq) / sr
    return np.sin(phase) * np.exp(-14 * t)


def hat(sr: int, rng: np.random.Generator, dur: float = 0.06) -> np.ndarray:
    n = int(dur * sr)
    noise = rng.uniform(-1, 1, n)
    b, a = signal.butter(2, 7000 / (sr / 2), "high")
    return signal.lfilter(b, a, noise) * np.exp(-60 * np.arange(n) / sr)


def snare(sr: int, rng: np.random.Generator, dur: float = 0.16) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    noise = rng.uniform(-1, 1, n) * np.exp(-22 * t)
    body = np.sin(2 * np.pi * 190 * t) * np.exp(-30 * t)
    return 0.7 * noise + 0.5 * body


def _add(buf: np.ndarray, start: int, sig: np.ndarray, gain: float) -> None:
    end = min(len(buf), start + len(sig))
    if end > start:
        buf[start:end] += sig[: end - start] * gain


def synthesize_bgm(duration: float, style: str = "bright", seed: int = 0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """duration 초 길이의 mono float32 (-1..1) BGM을 합성한다."""
    if style not in STYLES:
        raise ValueError(f"알 수 없는 BGM 스타일: {style} (가능: {', '.join(STYLES)})")
    st = STYLES[style]
    rng = np.random.default_rng(seed)
    beat = 60.0 / st["bpm"]
    bar = 4 * beat
    n_bars = int(math.ceil(duration / bar)) + 1
    total = int((n_bars * bar + 2.0) * sr)
    buf = np.zeros(total, dtype=np.float64)

    chords: List[List[int]] = st["chords"]
    for b in range(n_bars):
        chord = chords[b % len(chords)]
        bar_start = int(b * bar * sr)

        # 패드(코드)
        for note in chord:
            _add(buf, bar_start, pad_tone(midi_to_hz(note), bar + 0.4, sr), st["pad_gain"] / len(chord))

        # 베이스: 1·3박 (energetic 은 매 박)
        bass_beats = range(4) if st["drums"] == "full" else (0, 2)
        for k in bass_beats:
            _add(buf, bar_start + int(k * beat * sr), pluck(midi_to_hz(chord[0] - 24), beat * 0.95, sr, 4.0), 0.28)

        # 아르페지오: 8분음표, 코드톤 한 옥타브 위를 순환 + 약간의 변주
        if st["arp"]:
            tones = [n + 12 for n in chord] + [chord[0] + 24]
            for k in range(8):
                if rng.random() < 0.12:  # 가끔 쉬어서 기계적인 느낌 완화
                    continue
                note = tones[(k + b) % len(tones)]
                _add(buf, bar_start + int(k * beat / 2 * sr), pluck(midi_to_hz(note), beat * 0.6, sr), st["arp_gain"])

        # 드럼
        if st["drums"] in ("light", "full"):
            kick_beats = range(4) if st["drums"] == "full" else (0, 2)
            for k in kick_beats:
                _add(buf, bar_start + int(k * beat * sr), kick(sr), 0.55)
            for k in range(8):
                if st["drums"] == "full" or k % 2 == 1:
                    _add(buf, bar_start + int(k * beat / 2 * sr), hat(sr, rng), 0.10)
            if st["drums"] == "full":
                for k in (1, 3):
                    _add(buf, bar_start + int(k * beat * sr), snare(sr, rng), 0.25)

    # 로우패스로 부드럽게, 길이 맞추고 페이드
    b_, a_ = signal.butter(4, st["cutoff"] / (sr / 2), "low")
    buf = signal.lfilter(b_, a_, buf)
    n = int(duration * sr)
    buf = buf[:n]
    fade_in, fade_out = int(0.5 * sr), int(min(1.5, duration / 3) * sr)
    buf[:fade_in] *= np.linspace(0, 1, fade_in)
    buf[-fade_out:] *= np.linspace(1, 0, fade_out)
    peak = np.max(np.abs(buf)) or 1.0
    return (buf / peak * 0.8).astype(np.float32)


def write_bgm(path: Path, duration: float, style: str = "bright", seed: int = 0, sr: int = SAMPLE_RATE) -> Path:
    audio = synthesize_bgm(duration, style=style, seed=seed, sr=sr)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), sr, (audio * 32767).astype(np.int16))
    return path

"""한글 자막용 폰트 탐색."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

# OS별로 흔히 설치돼 있는 한글 폰트 (앞에 있을수록 우선)
FONT_CANDIDATES = [
    # Windows
    "C:/Windows/Fonts/malgunbd.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/NanumGothicBold.ttf",
    "C:/Windows/Fonts/NanumGothic.ttf",
    # macOS
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/NanumGothicBold.ttf",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    # Linux
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansKR-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansKR-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/unifont/unifont.otf",
]


def font_supports_hangul(path: Path, size: int = 32) -> bool:
    """폰트가 한글 글리프를 실제로 갖고 있는지 렌더링해서 확인한다."""
    try:
        font = ImageFont.truetype(str(path), size)
    except OSError:
        return False

    def _render(ch: str):
        img = Image.new("L", (size * 2, size * 2), 0)
        ImageDraw.Draw(img).text((4, 4), ch, font=font, fill=255)
        return img.tobytes()

    hangul = _render("한")
    notdef = _render("\ue000")  # 사설 영역 문자 → 보통 .notdef 글리프
    return hangul != notdef and any(hangul)


def find_korean_font(explicit: Optional[Path] = None) -> Path:
    """사용할 한글 폰트 경로를 찾는다.

    우선순위: 명시 인자 → 환경변수 SHORTS_FONT → 알려진 경로 → fc-list.
    """
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"폰트 파일이 없습니다: {p}")
        return p

    env = os.environ.get("SHORTS_FONT")
    if env and Path(env).exists():
        return Path(env)

    for cand in FONT_CANDIDATES:
        p = Path(cand)
        if p.exists() and font_supports_hangul(p):
            return p

    if shutil.which("fc-list"):
        try:
            out = subprocess.run(
                ["fc-list", ":lang=ko", "file"], capture_output=True, text=True, timeout=10
            ).stdout
            for line in out.splitlines():
                p = Path(line.split(":")[0].strip())
                if p.suffix.lower() in {".ttf", ".otf", ".ttc"} and font_supports_hangul(p):
                    return p
        except (OSError, subprocess.SubprocessError):
            pass

    raise FileNotFoundError(
        "한글 폰트를 찾지 못했습니다. --font 옵션이나 SHORTS_FONT 환경변수로 "
        "TTF/OTF 경로를 지정하세요 (예: 나눔고딕, 맑은고딕, Noto Sans KR)."
    )

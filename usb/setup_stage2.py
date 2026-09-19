#!/usr/bin/env python3
"""USB 설치 2단계: 라이브러리 설치와 폴더 준비.

setup.bat 이 휴대용 파이썬과 pip 를 깔아 둔 뒤 이 파일을 부른다. 여기서부터는
파이썬이므로 한글 메시지와 세밀한 오류 처리가 가능하다.

설계 의도
- 모든 단계를 setup_log.txt 에 남긴다. 만든 사람이 윈도우에서 시험할 수 없어서,
  실패했을 때 로그가 유일한 단서다.
- 단계마다 실패 이유와 해결책을 한 줄로 알려준다. "오류 발생"만 띄우면
  아무것도 할 수 없다.
- 다시 실행해도 안전하다. 이미 된 단계는 건너뛴다.
- torch 는 CPU 전용으로 받는다. 기본 설치는 안 쓰는 GPU 라이브러리를 3GB 넘게
  끌고 온다.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# CPU 전용 PyTorch. 기본 저장소는 CUDA 빌드를 주는데 이 파이프라인은 CPU 로만 돈다.
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
TORCH_PACKAGES = ["torch", "torchvision"]

CORE_PACKAGES = [
    "opencv-python-headless>=4.8,<4.12",
    "moviepy>=2.1,<3",
    "imageio-ffmpeg>=0.5",
    "Pillow>=10",
    "numpy>=1.26,<2",
    "scipy>=1.11",
    "mediapipe>=0.10.9,<0.11",
    "ultralytics>=8.0,<9",
    # mediapipe 는 protobuf 5 이상에서 깨진다. ultralytics 가 실행 중에
    # protobuf 를 올려 버리는 일이 있어 여기서 못 박는다.
    "protobuf>=4.25.3,<5",
]

LOG_PATH: Path | None = None


def log(message: str, to_screen: bool = True) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    if to_screen:
        print(message, flush=True)
    if LOG_PATH:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def fail(what: str, why: str, fix: str) -> None:
    log("")
    log("=" * 52)
    log(f"  설치 실패: {what}")
    log("=" * 52)
    log(f"  원인: {why}")
    log(f"  해결: {fix}")
    log("")
    log(f"  이 화면과 {LOG_PATH} 파일을 보내주시면 고쳐 드립니다.")
    log("")
    sys.exit(1)


def run_pip(args: list, description: str) -> bool:
    """pip 을 돌리고 성공 여부를 돌려준다. 출력은 전부 로그에 남긴다."""
    cmd = [sys.executable, "-m", "pip", "install", "--no-warn-script-location", *args]
    log(f"    실행: pip install {' '.join(args[:3])}...", to_screen=False)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except OSError as e:
        log(f"    pip 을 실행할 수 없습니다: {e}", to_screen=False)
        return False
    if LOG_PATH:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(f"--- {description} (종료코드 {proc.returncode}) ---\n")
            f.write((proc.stdout or "")[-4000:] + "\n")
            f.write((proc.stderr or "")[-4000:] + "\n")
    return proc.returncode == 0


def check_disk_space(root: Path, need_gb: float = 2.0) -> None:
    try:
        free = shutil.disk_usage(root).free / 1024 ** 3
    except OSError:
        return
    log(f"  빈 공간: {free:.1f} GB")
    if free < need_gb:
        fail(
            "공간 부족",
            f"이 드라이브에 {free:.1f} GB 밖에 없습니다. {need_gb:.0f} GB 이상 필요합니다.",
            "USB 의 다른 파일을 지우거나 더 큰 USB 를 쓰세요.",
        )


def install_libraries() -> None:
    # ultralytics 가 실행 중에 제멋대로 패키지를 설치하지 못하게 막는다.
    os.environ["YOLO_AUTOINSTALL"] = "false"
    log("")
    log("  [3/5] 라이브러리를 설치합니다. 10~30분 걸립니다.")
    log("        (네트워크 속도에 따라 다릅니다. 창을 닫지 마세요)")

    log("")
    log("    - 사람 검출 엔진 (CPU 전용, 약 300MB)")
    if not run_pip(["--index-url", TORCH_INDEX, *TORCH_PACKAGES], "torch(cpu)"):
        log("      CPU 전용 설치에 실패했습니다. 일반 설치로 다시 시도합니다.")
        log("      (용량이 훨씬 커집니다)")
        if not run_pip(TORCH_PACKAGES, "torch(fallback)"):
            fail(
                "사람 검출 엔진 설치",
                "torch 를 받지 못했습니다. 네트워크가 막혔을 수 있습니다.",
                "학교 방화벽이 pypi.org 를 막는지 확인하거나, 집 네트워크에서 다시 해 보세요.",
            )

    log("    - 영상·이미지 처리 라이브러리")
    if not run_pip(CORE_PACKAGES, "core"):
        fail(
            "기본 라이브러리 설치",
            "설치 중 오류가 났습니다.",
            f"{LOG_PATH} 의 마지막 부분을 보내주세요. 어떤 패키지에서 막혔는지 적혀 있습니다.",
        )
    log("    설치 완료")


def verify_imports() -> None:
    log("")
    log("  [4/5] 설치된 것들을 확인합니다.")
    checks = [
        ("cv2", "영상 처리"),
        ("numpy", "수치 계산"),
        ("scipy", "음원 합성"),
        ("PIL", "자막 그리기"),
        ("moviepy", "영상 편집"),
        ("mediapipe", "얼굴 감지"),
        ("ultralytics", "사람 검출"),
    ]
    missing = []
    for module, purpose in checks:
        try:
            __import__(module)
            log(f"    OK  {purpose}")
        except Exception as e:
            log(f"    실패 {purpose} ({module}): {type(e).__name__}")
            log(f"      {e}", to_screen=False)
            missing.append((module, purpose, str(e)))

    if not missing:
        return

    # 전부 실패했다면 설치 자체가 아니라 검색 경로 문제일 가능성이 높다.
    # 실제로 겪은 일이다: pip 는 "설치 완료"라고 했는데 하나도 못 불러왔다.
    # 휴대용 파이썬의 ._pth 파일이 Lib\site-packages 를 빠뜨리면 이렇게 된다.
    if len(missing) == len(checks):
        log("")
        log("    하나도 불러오지 못했습니다. 설치가 아니라 경로 문제로 보입니다.")
        log("    파이썬이 찾고 있는 곳:")
        for entry in sys.path:
            log(f"      {entry}")
        site_packages = Path(sys.executable).parent / "Lib" / "site-packages"
        log(f"    라이브러리가 있어야 할 곳: {site_packages}")
        log(f"    그 폴더가 실제로 있는가: {site_packages.is_dir()}")
        if site_packages.is_dir():
            found = sorted(p.name for p in site_packages.iterdir() if p.is_dir())[:10]
            log(f"    그 안에 있는 것: {', '.join(found) if found else '(비어 있음)'}")
        pth = sorted(Path(sys.executable).parent.glob("python*._pth"))
        if pth:
            log(f"    설정 파일 {pth[0].name} 의 내용:")
            for line in pth[0].read_text(encoding="utf-8", errors="replace").splitlines():
                log(f"      {line}")
        fail(
            "라이브러리를 찾지 못함",
            "설치는 됐지만 파이썬이 그 폴더를 검색 경로에 넣지 않았습니다.",
            "setup.bat 을 다시 실행하면 설정 파일을 새로 씁니다. "
            "그래도 안 되면 위 내용을 그대로 보내주세요.",
        )

    names = ", ".join(m for m, _, _ in missing)
    if any("DLL" in e or "_ssl" in e or "msvc" in e.lower() for _, _, e in missing):
        fix = ("Microsoft Visual C++ 재배포 패키지가 필요할 수 있습니다. "
               "'Visual C++ Redistributable x64' 를 검색해 설치한 뒤 다시 실행하세요.")
    else:
        fix = f"setup.bat 을 다시 실행해 보세요. 그래도 안 되면 {LOG_PATH} 를 보내주세요."
    fail("라이브러리 확인", f"{names} 을(를) 불러오지 못했습니다.", fix)


def download_model(root: Path) -> None:
    log("")
    log("  [5/5] 사람 검출 모델을 내려받습니다 (약 7MB).")
    models = root / "models"
    models.mkdir(exist_ok=True)
    dest = models / "yolov8n.pt"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        log("    이미 있습니다. 건너뜁니다.")
        return
    try:
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
        src = Path(getattr(model, "ckpt_path", "") or "yolov8n.pt")
        if src.exists() and src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        log("    완료")
    except Exception as e:
        log(f"    실패: {type(e).__name__}: {e}", to_screen=False)
        log("    모델을 받지 못했습니다.")
        log("    영상은 만들 수 있지만 얼굴 감지 정확도가 크게 떨어집니다.")
        log("    나중에 인터넷이 되는 곳에서 setup.bat 을 다시 실행하세요.")


def make_folders(root: Path) -> None:
    for name in ("watch", "output"):
        (root / name).mkdir(exist_ok=True)
    sample = root / "watch" / "여기에_활동별_폴더를_만드세요.txt"
    if not sample.exists():
        sample.write_text(
            "이 폴더 안에 활동별로 폴더를 만들고 사진을 넣으세요.\n"
            "\n"
            "예)\n"
            "  watch\\과학실험\\1.jpg, 2.jpg, 3.jpg\n"
            "  watch\\운동회\\1.jpg, 2.jpg\n"
            "\n"
            "폴더 안에 description.txt 를 두고 활동 설명을 한 줄 적으면\n"
            "그 문장이 자막에 쓰입니다. 없으면 폴더 이름을 씁니다.\n"
            "\n"
            "사진을 다 넣은 뒤 watch_start.bat 을 실행하세요.\n"
            "완성된 영상은 output 폴더에 저장됩니다.\n",
            encoding="utf-8",
        )


def write_launchers(root: Path) -> None:
    """실행용 배치 파일. 한글은 파이썬 쪽에서만 쓰고 여기는 ASCII 로 둔다."""
    (root / "watch_start.bat").write_text(
        "@echo off\r\n"
        "chcp 65001 >nul 2>&1\r\n"
        'cd /d "%~dp0"\r\n'
        "echo.\r\n"
        "echo  Watching the 'watch' folder. Press Ctrl+C to stop.\r\n"
        "echo  Finished videos appear in the 'output' folder.\r\n"
        "echo.\r\n"
        'python\\python.exe watch_shorts.py watch -o output\r\n'
        "pause\r\n",
        encoding="ascii",
    )
    (root / "make_one.bat").write_text(
        "@echo off\r\n"
        "chcp 65001 >nul 2>&1\r\n"
        'cd /d "%~dp0"\r\n'
        'python\\python.exe usb\\make_one.py\r\n'
        "pause\r\n",
        encoding="ascii",
    )
    (root / "check_only.bat").write_text(
        "@echo off\r\n"
        "chcp 65001 >nul 2>&1\r\n"
        'cd /d "%~dp0"\r\n'
        'python\\python.exe watch_shorts.py watch -o output --dry-run\r\n'
        "pause\r\n",
        encoding="ascii",
    )


def main() -> int:
    global LOG_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    LOG_PATH = root / "setup_log.txt"

    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    log("")
    log("  설치 위치: " + str(root))
    check_disk_space(root)

    install_libraries()
    verify_imports()
    download_model(root)
    make_folders(root)
    write_launchers(root)

    log("")
    log("  " + "=" * 50)
    log("   설치가 끝났습니다.")
    log("  " + "=" * 50)
    log("")
    log("   1. watch 폴더 안에 활동 이름으로 폴더를 만드세요")
    log("   2. 그 안에 사진을 넣으세요")
    log("   3. watch_start.bat 을 더블클릭하세요")
    log("")
    log("   완성된 영상은 output 폴더에 저장됩니다.")
    log("   업로드 전에 output 폴더의 review_sheet.jpg 를 꼭 확인하세요.")
    log("")
    return 0


if __name__ == "__main__":
    sys.exit(main())

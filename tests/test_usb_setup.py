"""USB 배포용 설치 스크립트 테스트.

배치 파일은 리눅스에서 실행할 수 없다. 대신 실행 전에 확인할 수 있는 것을
전부 확인한다. 파일 인코딩, 줄바꿈, goto 라벨 정의, 파이썬 쪽 로직.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
USB = ROOT / "usb"
sys.path.insert(0, str(USB))


# ---------- 배치 파일 ----------
def test_setup_bat_exists():
    assert (USB / "setup.bat").exists()


def test_setup_bat_is_ascii_only():
    """cmd.exe 가 한글을 깨뜨린다. 한글 메시지는 파이썬 쪽에서만 쓴다."""
    data = (USB / "setup.bat").read_bytes()
    assert data.isascii(), "setup.bat 에 비ASCII 문자가 있습니다"


def test_setup_bat_every_goto_has_a_label():
    """정의되지 않은 라벨로 점프하면 cmd 가 조용히 종료한다."""
    text = (USB / "setup.bat").read_text(encoding="ascii")
    gotos = set(re.findall(r"goto :(\w+)", text))
    labels = set(re.findall(r"^:(\w+)", text, re.MULTILINE))
    assert gotos <= labels, f"정의되지 않은 라벨: {sorted(gotos - labels)}"


def _failure_blocks(text: str) -> dict:
    """라벨 '정의'(줄 맨 앞)부터 exit 까지를 잘라낸다.

    goto 줄에서 자르면 엉뚱한 곳을 검사하게 된다(실제로 그랬다).
    """
    blocks = {}
    for m in re.finditer(r"^:(fail_\w+)$", text, re.MULTILINE):
        rest = text[m.end():]
        blocks[m.group(1)] = rest.split("exit /b", 1)[0]
    return blocks


def test_setup_bat_has_failure_blocks():
    blocks = _failure_blocks((USB / "setup.bat").read_text(encoding="ascii"))
    assert len(blocks) >= 4, f"실패 처리가 너무 적습니다: {sorted(blocks)}"


def test_setup_bat_pauses_on_every_failure():
    """창이 바로 닫히면 사용자가 오류를 읽을 수 없다."""
    for label, block in _failure_blocks((USB / "setup.bat").read_text(encoding="ascii")).items():
        assert "pause" in block, f"{label} 에 pause 가 없습니다"


def test_setup_bat_mentions_the_log_file_on_failure():
    """만든 사람이 윈도우에서 시험할 수 없다. 로그가 유일한 단서다."""
    for label, block in _failure_blocks((USB / "setup.bat").read_text(encoding="ascii")).items():
        assert "%LOG%" in block, f"{label} 이 로그 파일 위치를 알려주지 않습니다"


def test_setup_bat_explains_what_to_do_on_failure():
    """'오류 발생' 만 띄우면 사용자가 할 수 있는 게 없다."""
    for label, block in _failure_blocks((USB / "setup.bat").read_text(encoding="ascii")).items():
        assert "FAILED" in block, f"{label} 이 무엇이 실패했는지 말하지 않습니다"


# ---------- 2단계 설치 스크립트 ----------
def test_stage2_imports():
    import setup_stage2  # noqa: F401


def test_stage2_pins_protobuf_below_5():
    """ultralytics 가 protobuf 를 올려 mediapipe 를 깨뜨린 적이 있다.

    실제로 겪은 문제다. ONNX 내보내기를 한 번 돌렸더니 protobuf 가 4 에서 7 로
    올라갔고 얼굴 감지와 자세 추정이 통째로 죽었다.
    """
    import setup_stage2

    pins = [p for p in setup_stage2.CORE_PACKAGES if p.startswith("protobuf")]
    assert pins, "protobuf 고정이 빠졌습니다"
    assert "<5" in pins[0]


def test_stage2_uses_cpu_only_torch():
    """기본 저장소는 쓰지도 않는 CUDA 라이브러리를 3GB 넘게 끌고 온다."""
    import setup_stage2

    assert "cpu" in setup_stage2.TORCH_INDEX


def test_stage2_disables_ultralytics_autoinstall():
    src = (USB / "setup_stage2.py").read_text(encoding="utf-8")
    assert "YOLO_AUTOINSTALL" in src


def test_stage2_creates_folders_and_launchers(tmp_path):
    import setup_stage2

    setup_stage2.LOG_PATH = tmp_path / "log.txt"
    setup_stage2.make_folders(tmp_path)
    setup_stage2.write_launchers(tmp_path)

    assert (tmp_path / "watch").is_dir()
    assert (tmp_path / "output").is_dir()
    for name in ("watch_start.bat", "make_one.bat", "check_only.bat"):
        assert (tmp_path / name).exists(), name


def test_launchers_are_ascii_with_windows_newlines(tmp_path):
    import setup_stage2

    setup_stage2.LOG_PATH = tmp_path / "log.txt"
    setup_stage2.write_launchers(tmp_path)
    for bat in tmp_path.glob("*.bat"):
        data = bat.read_bytes()
        assert data.isascii(), f"{bat.name} 에 비ASCII 문자"
        assert b"\r\n" in data, f"{bat.name} 이 윈도우 줄바꿈이 아님"


def test_make_folders_is_rerunnable(tmp_path):
    import setup_stage2

    setup_stage2.LOG_PATH = tmp_path / "log.txt"
    setup_stage2.make_folders(tmp_path)
    setup_stage2.make_folders(tmp_path)   # 두 번 돌려도 오류가 없어야 한다
    assert (tmp_path / "watch").is_dir()


def test_guide_file_is_written_into_watch(tmp_path):
    import setup_stage2

    setup_stage2.LOG_PATH = tmp_path / "log.txt"
    setup_stage2.make_folders(tmp_path)
    guides = list((tmp_path / "watch").glob("*.txt"))
    assert guides, "watch 폴더 안내 파일이 없습니다"
    assert "description.txt" in guides[0].read_text(encoding="utf-8")


# ---------- 대화식 실행 ----------
def test_make_one_imports():
    import make_one  # noqa: F401


def test_readme_exists_and_warns_about_review():
    text = (USB / "README.txt").read_text(encoding="utf-8")
    assert "review_sheet" in text
    assert "개인정보" in text


# ---------- 설치된 환경 ----------
def test_installed_protobuf_is_compatible_with_mediapipe():
    """이 환경이 실제로 깨지지 않았는지 확인한다."""
    pytest.importorskip("google.protobuf")
    import google.protobuf

    major = int(google.protobuf.__version__.split(".")[0])
    assert major < 5, f"protobuf {google.protobuf.__version__} 은 mediapipe 를 깨뜨립니다"


def test_mediapipe_actually_runs():
    """import 만으로는 부족하다. protobuf 가 어긋나면 호출할 때 죽는다."""
    mp = pytest.importorskip("mediapipe")
    import numpy as np

    fd = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    fd.process(np.zeros((240, 320, 3), dtype=np.uint8))  # 예외가 나면 실패


# ---------- 실제로 겪은 설치 실패 ----------
def test_setup_bat_rejects_bracket_paths():
    """대괄호가 든 경로에서 다운로드가 실패했다.

    실제로 G:\\[안전폴더]... 에서 막혔다. PowerShell 은 대괄호를 와일드카드로
    해석해서 -OutFile 이 조용히 실패한다. 미리 잡아서 무엇을 해야 하는지 알린다.
    """
    text = (USB / "setup.bat").read_text(encoding="ascii")
    assert 'findstr /C:"["' in text
    assert 'findstr /C:"]"' in text
    assert "fail_badpath" in text


def test_setup_bat_checks_path_length():
    """경로가 길면 라이브러리 폴더가 260자를 넘어 윈도우가 깨진다."""
    text = (USB / "setup.bat").read_text(encoding="ascii")
    assert "fail_longpath" in text
    assert "PLEN" in text


def test_bad_path_message_says_what_to_do():
    """'경로가 이상합니다' 만으로는 사용자가 할 수 있는 게 없다."""
    blocks = _failure_blocks((USB / "setup.bat").read_text(encoding="ascii"))
    for label in ("fail_badpath", "fail_longpath"):
        assert label in blocks, f"{label} 블록이 없습니다"
        assert "shorts" in blocks[label], f"{label} 이 대안 경로를 제시하지 않습니다"


def test_downloads_go_through_temp_not_the_usb_path():
    """USB 경로에 특수문자가 있어도 다운로드는 성공해야 한다."""
    text = (USB / "setup.bat").read_text(encoding="ascii")
    assert "%TEMP%" in text, "임시 폴더를 경유하지 않습니다"
    assert "-OutFile '%ROOT%" not in text, "USB 경로로 직접 내려받고 있습니다"


def test_download_failure_suggests_another_network():
    """학교 방화벽이 원인인 경우가 많다."""
    blocks = _failure_blocks((USB / "setup.bat").read_text(encoding="ascii"))
    assert "network" in blocks["fail_download"].lower()


def test_pth_rewrite_includes_site_packages():
    """실제 설치 실패의 원인이었다.

    휴대용 파이썬은 ._pth 파일에 적힌 경로만 검색한다. 거기에
    Lib\\site-packages 가 없으면 pip 는 "설치 완료"라고 하는데 정작
    import 는 전부 ModuleNotFoundError 가 난다.
    """
    text = (USB / "setup.bat").read_text(encoding="ascii")
    assert "Lib\\site-packages" in text
    assert "import site" in text


def test_pth_rewrite_runs_on_every_launch():
    """재실행으로 고칠 수 있어야 한다.

    다운로드 블록 안에 두면 파이썬이 이미 있을 때 건너뛴다. 그러면
    ._pth 가 깨진 상태로 영원히 남는다.
    """
    text = (USB / "setup.bat").read_text(encoding="ascii")
    after_haspython = text.split("\n:haspython", 1)[1]
    assert "_pth" in after_haspython, "._pth 수정이 재실행 경로에 없습니다"


def test_pth_content_is_written_whole_not_patched():
    """줄 단위 패치는 너무 깨지기 쉬웠다. 통째로 쓴다."""
    text = (USB / "setup.bat").read_text(encoding="ascii")
    assert "WriteAllLines" in text
    assert "-replace '^#" not in text, "아직 정규식 패치가 남아 있습니다"


def test_stage2_diagnoses_a_total_import_failure():
    """전부 실패하면 설치가 아니라 경로 문제다. 그걸 알려줘야 한다."""
    src = (USB / "setup_stage2.py").read_text(encoding="utf-8")
    assert "len(missing) == len(checks)" in src
    assert "sys.path" in src
    assert "site-packages" in src

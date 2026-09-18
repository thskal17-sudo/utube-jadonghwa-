# 교육사진 자동 숏폼 제작 파이프라인

사진 폴더와 활동 설명 한 줄을 넣으면 **얼굴을 가린 9:16 세로 mp4**(1080x1920)를 만들어 줍니다.
유튜브 숏폼 규격에 맞춰 15~20초 길이로 출력합니다.

현재 구현 범위는 **Phase 1 (MVP)** 입니다. 전체 로드맵은 [docs/DESIGN.md](docs/DESIGN.md)를 보세요.

```
사진 폴더 ─▶ 얼굴 블러 ─▶ 자막 계획 ─▶ 켄 번즈 편집 ─▶ BGM 합성 ─▶ mp4
  (입력)     mediapipe      템플릿        moviepy       numpy/scipy    ffmpeg
                 │                                                       │
                 └──────────▶ review_sheet.jpg (사람이 검수) ◀───────────┘
```

---

## 설치

```bash
pip install -r requirements.txt
```

ffmpeg 은 `imageio-ffmpeg` 가 함께 설치하므로 따로 준비하지 않아도 됩니다.

한글 자막을 쓰려면 한글 폰트가 하나 필요합니다. 윈도우(맑은 고딕)·맥(Apple SD Gothic)에는
기본 설치돼 있고, 리눅스라면 아래처럼 설치하거나 `--font` 로 직접 지정하세요.

```bash
sudo apt install fonts-nanum        # 또는 fonts-noto-cjk
python make_shorts.py ... --font /path/to/NanumGothicBold.ttf
```

## 사용법

```bash
python make_shorts.py ./photos/과학실험 \
    -d "3학년 과학 시간, 화산 폭발 실험을 했어요" \
    -o output/volcano.mp4
```

빠르게 확인만 하고 싶으면 `--preview` (540x960, 저화질)를 붙이세요. 약 15초면 끝납니다.

### 주요 옵션

| 옵션 | 설명 | 기본값 |
| --- | --- | --- |
| `-d, --description` | 활동 설명 한 줄. 제목과 첫 자막에 쓰입니다 | (필수) |
| `-o, --output` | 출력 mp4 경로 | `output/<폴더명>.mp4` |
| `--title` | 상단 제목 직접 지정 | 설명에서 자동 생성 |
| `--captions` | 장면별 자막 JSON (Phase 2 연결 지점) | 템플릿 자동 생성 |
| `--duration` | 영상 길이(초) | `18` |
| `--max-photos` | 사용할 최대 사진 수 | `8` |
| `--bgm` | `bright` / `calm` / `energetic` | `bright` |
| `--detector` | `auto` / `mediapipe` / `yunet` / `haar` | `auto` |
| `--blur-method` | `pixelate` / `gaussian` | `pixelate` |
| `--faces-json` | 놓친 얼굴 좌표를 직접 지정 | 없음 |
| `--preview` | 540x960 저화질 빠른 확인 | 꺼짐 |
| `--seed` | 효과·BGM 변주 시드 | `0` |

### 출력물

```
output/volcano.mp4              ← 완성본
output/volcano_work/
├── blurred/                    ← 얼굴을 가린 사진들
├── review_sheet.jpg            ← 검수용 격자 이미지 (초록 박스 = 가린 위치)
├── captions.json               ← 사용된 제목/자막
├── bgm_bright.wav              ← 합성된 BGM
└── report.json                 ← 실행 요약(사진 수, 감지 얼굴 수, 경고)
```

---

## ⚠️ 업로드 전에 반드시 할 일

**`review_sheet.jpg` 를 열어 모든 얼굴에 초록 박스가 쳐졌는지 눈으로 확인하세요.**

얼굴 감지는 100% 정확하지 않습니다. 다음 경우 놓칠 수 있습니다.

- 옆모습이거나 고개를 크게 숙인 경우
- 마스크·모자·손에 가려진 경우
- 역광이거나 아주 작게 찍힌 경우

놓친 얼굴이 있으면 좌표를 JSON 으로 직접 지정해 다시 실행하세요.

```json
{ "classroom_01.jpg": [[705, 399, 211, 211]] }
```

```bash
python make_shorts.py ./photos/과학실험 -d "..." --faces-json faces.json
```

좌표는 `[x, y, 너비, 높이]` (픽셀, 원본 사진 기준)입니다.

그 밖에 지켜야 할 것:

- **개인정보**: 학생 사진은 개인정보입니다. 보호자 동의와 학교 보안정책을 먼저 확인하세요.
- **자막 사실관계**: 자동 생성 문구는 사람이 한 번 읽고 확인하세요. 학교명·학년·이름은 넣지 마세요.
- **BGM**: 이 파이프라인이 만드는 BGM 은 코드로 직접 합성한 것이라 저작권 문제가 없습니다.
  외부 음원을 쓰려면 CC0 또는 로열티 프리인지 반드시 확인하세요.

---

## 얼굴 감지기

| 감지기 | 정확도 | 준비 |
| --- | --- | --- |
| `mediapipe` | 높음 (권장) | `pip install mediapipe` — 모델 내장, 다운로드 불필요 |
| `yunet` | 높음 | `python scripts/download_models.py` 로 ONNX 모델 필요 |
| `haar` | 낮음 | OpenCV 내장, 항상 사용 가능 |

`auto`(기본값)는 위 순서대로 사용 가능한 것을 고릅니다. MediaPipe 가 없으면 감지율이
크게 떨어지므로 경고를 출력합니다. 실제 측정에서 합성 테스트 사진 14개 얼굴 기준
MediaPipe 는 14개를 모두 찾았고 Haar 는 3개만 찾았습니다.

## 개발

```bash
python -m pytest tests/ -m "not slow"   # 빠른 테스트 (약 5초)
python -m pytest tests/                 # 실제 렌더링 포함 (약 20초)
```

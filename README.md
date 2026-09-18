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
python scripts/download_models.py    # 사람 검출 모델 (강력 권장)
```

`download_models.py` 는 인터넷 연결이 한 번 필요합니다. 건너뛰면 얼굴 전용 감지기로
동작하는데, 교실 사진에서는 누락이 크게 늘어납니다.

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

감지는 100% 정확하지 않습니다. 실측에서 `person` 방식도 45명 중 2명을 놓쳤습니다.
다음 경우 놓칠 수 있습니다.

- 사진 가장자리에서 몸이 크게 잘린 경우
- 다른 사람이나 사물에 거의 가려진 경우
- 아주 작게 찍혀 사람으로 인식되지 않는 경우

**촬영 팁**: 얼굴이 크게 나오는 근접 사진일수록 감지가 잘 되고 영상도 잘 나옵니다.
교실 전체를 멀리서 담은 사진은 감지도 어렵고 숏폼 화면에서 잘 보이지도 않습니다.

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

## 감지기

| 감지기 | 방식 | 준비 |
| --- | --- | --- |
| `person` | 사람 검출 + 자세 추정으로 머리를 찾음 (권장) | `pip install ultralytics` + `python scripts/download_models.py` |
| `mediapipe` | 얼굴만 감지 | `pip install mediapipe` — 모델 내장 |
| `yunet` | 얼굴만 감지 | `python scripts/download_models.py` |
| `haar` | 얼굴만 감지 | OpenCV 내장, 항상 사용 가능 |

`auto`(기본값)는 `person` 이 준비돼 있으면 그것을, 아니면 얼굴 전용 감지기를 씁니다.

### 왜 얼굴 감지기만으로는 부족한가

실제 교실 사진 5장(사람 45명)으로 측정한 결과입니다.

| 방식 | 찾은 사람 |
| --- | --- |
| `person` (사람 검출 + 자세) | 43 / 45 |
| `mediapipe` (얼굴만) | 13 / 45 |

얼굴 감지기는 **정면을 보고 있는 얼굴**만 찾습니다. 교실에서는 학생들이 고개를 숙이거나,
옆을 보거나, 뒤돌아 앉아 있어서 대부분 놓칩니다. `person` 방식은 사람을 먼저 찾고
자세 추정으로 머리 위치를 잡기 때문에 얼굴이 안 보여도 동작합니다.

`person` 방식은 일부러 과하게 가립니다. 얼굴이 아닌 곳이 가려지는 것은 보기에 아쉬울
뿐이지만, 놓치는 것은 개인정보 노출이라 비용이 전혀 다르기 때문입니다.

> **라이선스 주의**: `person` 방식이 쓰는 YOLOv8(Ultralytics)은 AGPL-3.0 입니다.
> 학교 내부에서 쓰는 데는 문제가 없지만, 이 파이프라인을 외부에 배포할 계획이라면
> 라이선스를 확인하세요.

## 개발

```bash
python -m pytest tests/ -m "not slow"   # 빠른 테스트 (약 5초)
python -m pytest tests/                 # 실제 렌더링 포함 (약 20초)
```

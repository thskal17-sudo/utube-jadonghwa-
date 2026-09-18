"""전체 파이프라인 오케스트레이션: 폴더 + 설명 → mp4."""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np

from .bgm import write_bgm
from .captions import CaptionPlan, TemplateCaptionGenerator
from .config import IMAGE_EXTENSIONS, SHORTS_MAX_SECONDS, SHORTS_MIN_SECONDS, ShortsConfig
from .face_blur import (
    FaceBlurResult, FaceDetector, blur_folder, load_image_rgb,
    load_manual_boxes, make_review_sheet,
)
from .fonts import find_korean_font
from .head_detect import PersonHeadDetector, person_detector_available
from .render import render_video

log = logging.getLogger(__name__)


@dataclass
class PipelineReport:
    output: str
    duration: float
    resolution: str
    photos_used: List[str]
    faces_per_photo: List[int]
    detector: str
    review_sheet: Optional[str]
    caption_plan: dict
    bgm: Optional[str]
    font: str
    seconds_elapsed: float
    warnings: List[str] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")


def _natural_key(p: Path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", p.name)]


def collect_images(input_dir: Path, max_photos: int) -> List[Path]:
    """폴더의 사진을 이름순으로 모으고, 너무 많으면 균등 간격으로 골라낸다."""
    files = sorted(
        (p for p in Path(input_dir).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=_natural_key,
    )
    if not files:
        raise FileNotFoundError(f"사진이 없습니다: {input_dir} (지원 확장자: {', '.join(sorted(IMAGE_EXTENSIONS))})")
    if len(files) > max_photos:
        idx = np.linspace(0, len(files) - 1, max_photos).round().astype(int)
        files = [files[i] for i in idx]
    return files


def _reinsert_kept(photos, keep, blurred_results, out_dir: Path):
    """블러를 건너뛴 사진을 원래 순서대로 다시 끼워 넣는다."""
    import shutil

    by_source = {r.source: r for r in blurred_results}
    out: List[FaceBlurResult] = []
    for i, src in enumerate(photos):
        if src.name in keep:
            dst = out_dir / f"{i:03d}_{src.stem}_keep{src.suffix.lower()}"
            shutil.copy2(src, dst)
            out.append(FaceBlurResult(source=src, output=dst, boxes=[], detector="keep"))
        else:
            out.append(by_source[src])
    return out


def build_detector(cfg: ShortsConfig):
    """설정에 맞는 감지기를 만든다.

    auto 는 사람 검출 기반(person)을 우선한다. 교실 사진에서 얼굴 전용
    감지기보다 누락이 훨씬 적기 때문이다. 준비돼 있지 않으면 얼굴 감지기로 내려간다.
    """
    if cfg.detector == "person":
        return PersonHeadDetector(weights=cfg.yolo_weights, tight=cfg.tight_blur)
    if cfg.detector == "auto" and person_detector_available(cfg.yolo_weights):
        return PersonHeadDetector(weights=cfg.yolo_weights, tight=cfg.tight_blur)
    backend = "auto" if cfg.detector == "auto" else cfg.detector
    return FaceDetector(backend=backend, yunet_model=cfg.yunet_model)


def run_pipeline(
    input_dir: Path,
    description: str,
    output: Path,
    cfg: Optional[ShortsConfig] = None,
    work_dir: Optional[Path] = None,
    caption_plan: Optional[CaptionPlan] = None,
) -> PipelineReport:
    cfg = cfg or ShortsConfig()
    t0 = time.time()
    input_dir, output = Path(input_dir), Path(output)
    work_dir = Path(work_dir) if work_dir else output.parent / f"{output.stem}_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []

    if not (SHORTS_MIN_SECONDS <= cfg.duration <= SHORTS_MAX_SECONDS):
        warnings.append(f"영상 길이 {cfg.duration:.1f}s 는 권장 범위({SHORTS_MIN_SECONDS:.0f}~{SHORTS_MAX_SECONDS:.0f}s) 밖입니다.")

    # 0. 입력 수집
    photos = collect_images(input_dir, cfg.max_photos)
    log.info("[input] 사진 %d장 사용: %s", len(photos), ", ".join(p.name for p in photos))
    font_path = find_korean_font(cfg.font_path)

    # 1. 얼굴 블러
    review_sheet: Optional[Path] = None
    if cfg.blur:
        detector = build_detector(cfg)
        if detector.backend == "haar":
            warnings.append(
                "Haar cascade 로 얼굴을 감지했습니다. 측면·작은 얼굴을 놓치기 쉬우니 "
                "`pip install mediapipe` 후 다시 실행하는 것을 권장합니다."
            )
        elif detector.backend != "person":
            warnings.append(
                "얼굴 전용 감지기로 동작했습니다. 교실 사진에서는 고개를 숙이거나 뒤돌아 앉은 "
                "학생을 놓칩니다. `pip install ultralytics` 후 "
                "`python scripts/download_models.py` 를 실행하면 사람 검출 기반으로 훨씬 잘 잡습니다."
            )
        review_sheet = work_dir / "review_sheet.jpg"
        manual = load_manual_boxes(cfg.manual_faces) if cfg.manual_faces else None
        keep = {Path(n).name for n in cfg.keep_faces}
        unknown = keep - {p.name for p in photos}
        if unknown:
            raise ValueError(
                f"--keep-faces 에 적은 파일이 폴더에 없습니다: {', '.join(sorted(unknown))}"
            )
        to_blur = [p for p in photos if p.name not in keep]
        results = blur_folder(
            to_blur, work_dir / "blurred", detector,
            method=cfg.blur_method, padding=cfg.blur_padding, strength=cfg.blur_strength,
            review_path=None, manual_boxes=manual,
        )
        if keep:
            results = _reinsert_kept(photos, keep, results, work_dir / "blurred")
            warnings.append(
                f"블러를 적용하지 않은 사진: {', '.join(sorted(keep))} — "
                "본인 또는 동의를 받은 사람만 지정했는지 확인하세요."
            )
        make_review_sheet(results, review_sheet)
        if sum(r.face_count for r in results) == 0:
            warnings.append("어떤 사진에서도 얼굴이 감지되지 않았습니다. 사진을 직접 확인하세요.")
        no_face = [r.source.name for r in results if r.face_count == 0 and r.detector != "keep"]
        if no_face and len(no_face) < len(results):
            warnings.append(f"얼굴이 하나도 감지되지 않은 사진: {', '.join(no_face)} — 검수 시트에서 확인하세요.")
    else:
        warnings.append("얼굴 블러를 건너뛰었습니다(--no-blur). 학생 얼굴이 그대로 노출됩니다.")
        (work_dir / "blurred").mkdir(exist_ok=True)
        results = []
        for i, src in enumerate(photos):
            dst = work_dir / "blurred" / f"{i:03d}_{src.stem}{src.suffix.lower()}"
            shutil.copy2(src, dst)
            results.append(FaceBlurResult(source=src, output=dst, boxes=[], detector="none"))

    # 2·3. 자막 계획
    if caption_plan is None:
        caption_plan = TemplateCaptionGenerator(title=cfg.title, seed=cfg.seed).generate(description, photos, len(photos))
    elif cfg.title:
        caption_plan = CaptionPlan(cfg.title, caption_plan.captions, caption_plan.hashtags)
    caption_plan = caption_plan.fitted(len(photos))
    caption_plan.to_json(work_dir / "captions.json")

    # 5. BGM
    bgm_path = write_bgm(work_dir / f"bgm_{cfg.bgm_style}.wav", cfg.duration + 1.0, style=cfg.bgm_style, seed=cfg.seed)

    # 4·6. 편집 + 렌더링
    images = [load_image_rgb(r.output) for r in results]
    render_video(images, caption_plan, output, cfg, font_path, bgm_path=bgm_path)

    report = PipelineReport(
        output=str(output),
        duration=cfg.duration,
        resolution=f"{cfg.width}x{cfg.height}",
        photos_used=[str(p) for p in photos],
        faces_per_photo=[r.face_count for r in results],
        detector=next((r.detector for r in results if r.detector != "keep"), "none"),
        review_sheet=str(review_sheet) if review_sheet else None,
        caption_plan=asdict(caption_plan),
        bgm=str(bgm_path),
        font=str(font_path),
        seconds_elapsed=round(time.time() - t0, 1),
        warnings=warnings,
    )
    report.save(work_dir / "report.json")
    return report

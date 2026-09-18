#!/usr/bin/env python3
"""Phase 1 → Phase 2 다리: Claude 에게 붙여넣을 자막 생성 프롬프트를 만든다.

사용법:
    python scripts/make_caption_prompt.py ./photos/과학실험 -d "화산 폭발 실험" > prompt.txt

출력된 프롬프트와 사진을 Claude 에 함께 주면 captions.json 형식의 결과가 나온다.
그 파일을 --captions 옵션으로 넘기면 맞춤 자막이 적용된다:

    python make_shorts.py ./photos/과학실험 -d "..." --captions captions.json

Phase 2 에서는 이 과정을 Anthropic API 호출로 자동화하면 된다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shorts_pipeline.config import ShortsConfig  # noqa: E402
from shorts_pipeline.pipeline import collect_images  # noqa: E402

TEMPLATE = """다음은 초등학교 교육 활동 사진 {n}장입니다 (순서대로 첨부).
선생님이 적은 활동 설명: "{description}"

이 사진들로 만들 유튜브 숏폼(세로 영상, {duration:.0f}초)의 제목과 장면별 자막을 만들어 주세요.

규칙:
1. 제목은 18자 이내, 클릭하고 싶어지는 문구. 낚시성 과장은 금지.
2. 자막은 정확히 {n}개. 사진 순서와 1:1로 대응.
3. 자막 한 개는 25자 이내. 화면에서 한두 줄로 읽히는 길이.
4. 학교명, 학년, 반, 학생 이름, 지역명 등 개인 식별 정보는 절대 넣지 말 것.
5. 사진에서 확실히 보이는 것만 쓸 것. 추측한 사실을 단정하지 말 것.
6. 마지막 자막은 마무리 느낌으로.
7. 해시태그 3~5개.

아래 JSON 형식으로만 답하세요:

{{
  "title": "제목",
  "captions": [{caption_slots}],
  "hashtags": ["#태그1", "#태그2", "#태그3"]
}}

사진 파일 순서:
{file_list}
"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Claude 용 자막 생성 프롬프트 출력")
    p.add_argument("input_dir", type=Path)
    p.add_argument("-d", "--description", required=True)
    p.add_argument("--duration", type=float, default=18.0)
    p.add_argument("--max-photos", type=int, default=ShortsConfig.max_photos)
    args = p.parse_args(argv)

    try:
        photos = collect_images(args.input_dir, args.max_photos)
    except FileNotFoundError as e:
        print(f"오류: {e}", file=sys.stderr)
        return 1

    n = len(photos)
    print(
        TEMPLATE.format(
            n=n,
            description=" ".join(args.description.split()),
            duration=args.duration,
            caption_slots=", ".join(f'"자막{i + 1}"' for i in range(n)),
            file_list="\n".join(f"{i + 1}. {p.name}" for i, p in enumerate(photos)),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

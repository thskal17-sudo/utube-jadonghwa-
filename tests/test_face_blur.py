import numpy as np
import pytest

from shorts_pipeline.face_blur import blur_faces, merge_boxes, load_manual_boxes


def test_merge_boxes_combines_overlaps():
    boxes = [(10, 10, 100, 100), (20, 20, 100, 100), (500, 500, 50, 50)]
    merged = merge_boxes(boxes)
    assert len(merged) == 2
    big = [b for b in merged if b[2] > 60][0]
    assert big == (10, 10, 110, 110)  # 합집합


def test_merge_boxes_keeps_disjoint():
    boxes = [(0, 0, 40, 40), (200, 200, 40, 40), (400, 50, 40, 40)]
    assert len(merge_boxes(boxes)) == 3


@pytest.mark.parametrize("method", ["pixelate", "gaussian"])
def test_blur_destroys_detail_inside_box(method):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (400, 400, 3), dtype=np.uint8)
    box = (150, 150, 100, 100)
    out = blur_faces(img, [box], method=method, padding=0.0)

    cx, cy = 200, 200  # 박스 중심 → 타원 마스크가 확실히 덮는 지점
    center_before = img[cy - 10 : cy + 10, cx - 10 : cx + 10].astype(float)
    center_after = out[cy - 10 : cy + 10, cx - 10 : cx + 10].astype(float)
    # 블러/모자이크 후에는 국소 분산이 크게 줄어든다
    assert center_after.std() < center_before.std() * 0.75


def test_blur_leaves_outside_untouched():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (400, 400, 3), dtype=np.uint8)
    out = blur_faces(img, [(150, 150, 60, 60)], padding=0.2)
    assert np.array_equal(img[:60, :60], out[:60, :60])
    assert np.array_equal(img[-60:, -60:], out[-60:, -60:])


def test_blur_no_boxes_is_identity():
    img = np.random.default_rng(2).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    assert np.array_equal(img, blur_faces(img, []))


def test_blur_box_at_image_edge_does_not_crash():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    out = blur_faces(img, [(-10, -10, 40, 40), (180, 180, 60, 60)], padding=0.5)
    assert out.shape == img.shape


def test_load_manual_boxes(tmp_path):
    f = tmp_path / "faces.json"
    f.write_text('{"a.jpg": [[1, 2, 3, 4]], "sub/b.jpg": [[5, 6, 7, 8]]}', encoding="utf-8")
    boxes = load_manual_boxes(f)
    assert boxes["a.jpg"] == [(1, 2, 3, 4)]
    assert boxes["b.jpg"] == [(5, 6, 7, 8)]  # 경로는 파일명으로 정규화


def test_load_manual_boxes_rejects_bad_shape(tmp_path):
    f = tmp_path / "faces.json"
    f.write_text('{"a.jpg": [[1, 2, 3]]}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_manual_boxes(f)


def test_load_manual_boxes_ignores_comment_keys(tmp_path):
    f = tmp_path / "faces.json"
    f.write_text('{"_설명": "메모", "a.jpg": [[1, 2, 3, 4]]}', encoding="utf-8")
    boxes = load_manual_boxes(f)
    assert set(boxes) == {"a.jpg"}


def test_example_faces_file_is_valid():
    from pathlib import Path

    example = Path(__file__).resolve().parent.parent / "examples" / "faces.example.json"
    boxes = load_manual_boxes(example)
    assert boxes
    for name, bs in boxes.items():
        assert not name.startswith("_")
        for b in bs:
            assert len(b) == 4 and all(isinstance(v, int) for v in b)

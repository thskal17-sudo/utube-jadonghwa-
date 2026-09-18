# models/

감지 모델을 두는 폴더입니다. 파일은 git 에 포함하지 않습니다(.gitignore).

```bash
python scripts/download_models.py
```

| 파일 | 용도 | 필요성 | 라이선스 |
| --- | --- | --- | --- |
| `yolov8n.pt` | 사람 검출 (`--detector person`/`auto`) | 강력 권장 | AGPL-3.0 |
| `face_detection_full_range_sparse.tflite` | 얼굴 감지 (MediaPipe Tasks) | mediapipe 0.10.30+ 에서 필요 | Apache-2.0 |
| `pose_landmarker_lite.task` | 자세 추정 → 머리 위치 (MediaPipe Tasks) | mediapipe 0.10.30+ 에서 필요 | Apache-2.0 |
| `face_detection_yunet_2023mar.onnx` | 얼굴 감지 대체 (`--detector yunet`) | 선택 | Apache-2.0 |

MediaPipe 가 0.10.21 이하(레거시 solutions API)면 모델이 패키지에 내장돼 있어
`.tflite` / `.task` 파일은 필요하지 않습니다.

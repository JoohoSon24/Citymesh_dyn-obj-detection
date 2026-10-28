최신 개선 실험: [PROJECT.md](PROJECT.md) · [개선안 HTML 보고서](harder_dataset/improved_baselines/REPORT.html) · [비교 CSV](harder_dataset/improved_baselines/results/combined_metrics.csv). 루트 REPORT/SUMMARY 파일은 첫 쉬운 장면의 선행 결과입니다.

# 다중 시기 항공영상의 차량 제거 베이킹 파일럿

**프로젝트 전체 정리는 [PROJECT.md](PROJECT.md)를 먼저 읽으면 된다.** 이 문서에서 이전의 “차량 내부” 지표를 “차량 위치 표본의 배경 복원”으로 명확히 정의했다.

짧은 보고는 `SUMMARY.md`, 실험 결과와 해석은 `REPORT.md`, 직접 볼 수 있는 결과 묶음은 `REPORT.html`, 전체 지표는 `results/metrics.csv` 및 `results/manual_metrics.csv`에 있다.

## 실행

Python 3.11 + CUDA GPU에서 실행했다. `requirements.lock`에 실행 패키지 버전을 저장했다.

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv/bin/python numpy scipy pillow matplotlib opencv-python scikit-image ultralytics lpips requests markdown
uv pip install --python .venv/bin/python git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf
mkdir -p weights
curl -L -o weights/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
curl -L -o weights/yolo11s-obb.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-obb.pt
bash run.sh
```

`download_selected.py`는 확정 장면을 받는다. `download.py`는 최초 Jaarbeurs 후보, `download_ikea.py`는 IKEA 후보이다. 이 두 후보는 장면 검토용이며 최종 학습에 쓰지 않았다. WMS가 PNG 요청에 JPEG 바이트를 반환하여 입력 확장자는 `.png`이지만 실제 디코더 형식은 JPEG이다. 응답과 체크섬은 `results/provenance.json`에 기록했다.

## 산출물

- `results/inputs.jpg`: 정합·색 정규화된 6개 연도.
- `results/cameras.jpg`: 실제 연도 사진에서 생성한 합성 카메라.
- `results/comparison.jpg`, `results/textures/*.png`: 전체 비교 및 개별 텍스처.
- `results/vehicle_audit_comparison.jpg`: 차량이 있었던 위치를 실제 빈 연도와 대조.
- `results/masks.jpg`: 2025년 영상의 S 마스크. 빨강은 제외 영역.
- `results/crop_*.jpg`, `results/residuals.jpg`: 확대 비교와 잔차.
- `results/*_fit.npz`: 최종 마스크·잔차·시점별 색보정 행렬.
- `results/*_unsupported.png`: 최종 S에서 어느 학습 관측도 지지하지 않는 텍셀. 구멍을 생성 모델로 채우지 않았다.
- `results/alignment.json`, `results/cameras.json`, `results/provenance.json`, `results/validation.json`: 정합·투영·출처·검산.
- `results/manual_annotations.json`: 시각적으로 확인해 표시한 차량 16대의 내부 점. 독립적인 사람 검수 라벨은 아니다.

`pilot.py hugs`는 SAM 계산 결과를 `results/sam_cache/`에 캐시한다. 입력·전처리·SIFT·SAM·임계값이 바뀌면 이 디렉터리를 다른 이름으로 옮긴 뒤 재생성해야 한다. 최종 텍스처는 1024²이며 모든 학습 기법에 같은 512² bilinear 기저를 적용한다. 학습은 기법마다 seed=42로 시작한다. L2 warm-up을 HuGS의 전처리로 재사용하지만 HuGS 텍스처는 공통 평균 초기화에서 다시 학습한다.

## 출처

항공영상: © Beeldmateriaal Nederland, PDOK 제공, [CC BY 4.0](https://www.beeldmateriaal.nl/dataroom). 정합·색 정규화·투영·베이킹으로 가공했다. 원 기법 전체 NeRF의 재현이 아니라, 이들에서 차용한 S의 평면 텍스처 베이킹 비교이다.

# 차량 점유가 높은 추가 데이터 비교

기존 쉬운 Jaarbeurs 파일럿을 보존하고, 실제 다중 연도 항공영상 두 장면에서 같은 베이킹을 추가 실행한다. 최종 설명은 `PROJECT.md`, 수치는 `results/metrics.csv`에 저장한다.

## 데이터와 선택 원칙

- PDOK HR 2021–2026, 장면마다 120×120m와 1500² WMS 격자를 요청했다. `candidates/source_manifest.json`에 URL과 SHA-256이 있다.
- industrial B: 52.0740, 5.0460. 포장된 고점유 주차장.
- industrial C: 52.0773, 5.04895. 차량 배치가 변하는 고점유 주차장. 토양·잔디 노출의 시기별 외관 차이가 남는다.
- 세 후보의 원본 연도별 사진을 보고 B/C를 선택했다. 차량 검출 마스크로 점유 빈도를 측정했고, 방법별 최종 결과를 보고 장면을 선택하지 않았다.
- 기존 데이터에서 한 번이라도 차량으로 검출된 지면 픽셀의 99.1%는 5개 학습 연도 중 한 해에만 가려졌다. 새 두 장면은 매년 지면 ROI의 약 절반 이상이 차량 검출 마스크로 덮인다. 이 값에는 마스크 팽창이 포함되며 실제 차량/주차면 점유율의 수작업 정답은 아니다.

## 기존 구현에서 유지한 것과 바꾼 것

`run_scene.py`는 상위 `pilot.py`의 카메라·렌더러·기저·초기화·색보정·S·600회 Adam 루프를 재사용한다. 원 결과 파일은 덮어쓰지 않는다. 영상 20장, 출력1024²/기저512², batch4, lr0.025/0.001도 같다.

주차된 차가 반복되어 전역 SIFT 정합이 퇴화하는 경우가 있어, 전처리 정합은 동일 WMS 좌표를 초기값으로 한 CLAHE+LK forward/backward 검증과 RANSAC 유사변환으로 바꿨다. `registration.py`에 구현했다. 모든 새 방법에 같은 정합 결과를 준다. HuGS의 SIFT 매칭은 유지하며, CPU 대신 CUDA에서 동일한 유클리드 거리를 전수 계산한다(`gpu_matcher.py`). 근사 FLANN 매칭으로 바꾼 것은 아니다.

각 장면의 원본 연도별 입력만 본 뒤 전체 차량 OBB 평가 후보를 검토했다. `verified_annotations.json`에 채택/기각 이유가 있다. 독립적인 사람 검수는 아니다. 점유 빈도는 고정 OBB의 55% 이상이 검출 마스크로 덮인 학습 연도 수이며, 정확한 실제 주차칸 점유 횟수가 아닌 검출기 기반 proxy다.

## 평가

- 학습: 2021–2025. 평가 기준: 학습에 넣지 않은 **2026년 실제 비차량 배경 관측**. 시간 중앙값을 GT로 사용하지 않는다.
- 33개 차량 전체 OBB를 검토해 채택했고, 이 중 21개는 학습 점유 proxy가 3/5 또는 4/5다. 나머지는 낮은 점유/5회 점유 진단에 쓴다. 작은 원형 중심 표본이 아니다.
- 기본 지표: 전체 OBB의 PSNR, 점유3–4회 OBB PSNR, 그중 픽셀별 검출 빈도3–4회인 부분의 PSNR, 표준 SSIM map을 OBB에서 평균한 값, ghost, 관측 지지 비율.
- LPIPS: 각 OBB 중심의 80² crop에서 OBB 밖을 2026 기준으로 치환한 뒤 AlexNet LPIPS를 계산해 객체별 평균. 표준 전체 이미지 LPIPS와 다르다.
- 정적 지면 보존은 학습 중 검출된 차량이 없고 RGB 표준편차<0.08이며 2026에도 검출된 차량이 없는 ROI를 보조 기준으로 삼는다. 사람 검수 static GT가 아니다.
- 2026년에도 차량이 남았거나 자재 더미가 있는 후보는 제외했다. 같은 검출기로 후보를 만들고 Block 마스크도 만들므로 편향이 남는다.
- seed42에 모든 방법과 Robust 변형을 실행한다. seed43/44에서는 L2/Block/HuGS/기본 Robust의 베이킹을 반복한다. SIFT/SAM/고정 S는 seed42 결과를 공유하므로 전체 파이프라인 반복의 분산은 아니다.

## 재실행

상위 프로젝트의 `.venv`, 가중치, `pilot.py`를 사용한다. 원본과 annotation의 출처를 먼저 확인한다. 입력이 바뀌면 `sam_cache`와 검토된 평가 annotation을 그대로 재사용하면 안 된다.

```bash
cd /root/sonjooho/baking_pilot/harder_dataset
bash run_all.sh
```

코드와 결과의 재생성을 위한 스크립트이며 과거의 장면 선택 판단이나 시각 검토를 자동 재현하는 것은 아니다. 고정된 원본과 `verified_annotations.json`을 평가 정의로 사용한다.

영상 출처: © Beeldmateriaal Nederland / PDOK, [CC BY 4.0](https://www.beeldmateriaal.nl/dataroom). 정합·노출 정규화·투영·베이킹으로 가공했다. [PDOK 소개](https://www.pdok.nl/introductie/-/article/pdok-luchtfoto-rgb-open-).

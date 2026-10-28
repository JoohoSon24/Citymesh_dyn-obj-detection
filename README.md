# CityMesh: Dynamic-object removal for texture baking

네덜란드 다중 시기 항공영상에서 차량 관측을 제외하고 지면 Texture를 만드는 평면 베이킹 파일럿입니다. 실제 mesh를 재구성한 실험이 아니라, 지면을 평면으로 가정한 homography 기반 비교입니다.

## 발표자료와 보고서

- [최신 PPT — AF가 2페이지인 최종본](dynamic_object_baking_v7.pptx)
- [방법별 쉬운 설명 — UV fusion 상세 예시 포함](dynamic_object_baking_method_reports.md)
- [이미지를 포함한 HTML 보고서](dynamic_object_baking_method_reports.html) — 다운로드 후 브라우저로 열기
- [레퍼런스 핵심 메소드와 실험 연관성](dynamic_object_baking_references_v3.md)
- [전체 프로젝트 개요](baking_pilot/PROJECT.md)
- [방법별 개별 보고서](baking_pilot/method_reports/individual/)
- [발표자료의 방법별 입력·최종 결과 이미지](baking_pilot/presentation_v5/assets/)

## 코드와 결과

- 기본 실험: `baking_pilot/pilot.py`, `baking_pilot/README.md`.
- 고점유 데이터: `baking_pilot/harder_dataset/`.
- Geometry mask / UV fusion / shadow guard: `baking_pilot/harder_dataset/improved_baselines/`.
- DINO·SAM의 S 이식 실험: `baking_pilot/reference_study_v3/`.
- [비교 지표 CSV](baking_pilot/reference_study_v3/results/combined_metrics.csv).
- [실행 조건](baking_pilot/reference_study_v3/protocol.json) 및 [검증 결과](baking_pilot/reference_study_v3/results/validation.json).

원 논문 전체 NeRF/3DGS의 재현과, 관측 선택 규칙 S를 평면 Texture에 옮긴 실험을 구분합니다. DINO·SAM 추가 실험은 24회 baking이며, 개선되지 않은 결과와 과도한 마스킹 실패도 포함합니다.

## 대용량 실험 자료 복원

PPT, 문서, 코드, 지표, 일반 이미지·최종 Texture는 Git에 보관했습니다. 큰 NumPy 배열, 모델, 압축 묶음, 동영상 및 외부 참조 코드의 스냅샷은 [동일 저장소의 Release](https://github.com/JoohoSon24/Citymesh_dyn-obj-detection/releases/tag/snapshot-2026-10-28)에 나누어 보관합니다. Release archive는 프로젝트 루트에 풀면 원래 경로가 복원됩니다.

```bash
git clone https://github.com/JoohoSon24/Citymesh_dyn-obj-detection.git
cd Citymesh_dyn-obj-detection
python3 scripts/restore_release_assets.py
```

복원 스크립트는 다운로드 SHA-256을 검사하고, 이미 존재하는 일반 파일은 덮어쓰지 않습니다. `UPLOAD_MANIFEST.json`에는 원본 일반 파일의 경로·크기·체크섬과 각 파일의 보관 위치를 기록했습니다. 기존 절대경로 심볼릭 링크 58개는 저장소 내 상대경로로 변환했습니다.

가상환경 `.venv`, Python 캐시, 외부 저장소의 `.git` 이력은 배포하지 않았습니다. 실행 환경은 `baking_pilot/requirements.lock` 및 원 프로젝트 README를 참조해 다시 구성해야 합니다. 기존 실행 스크립트 일부는 원래 작업 경로를 포함하므로 다른 위치에서 사용할 때 경로 조정이 필요할 수 있습니다. 일부 이전 PPT 버전은 원본 폴더에 이미 없어 포함되지 않았습니다.

## 자료 출처

항공영상: © Beeldmateriaal Nederland / PDOK, CC BY 4.0. 정합·색 정규화·투영·베이킹·피겨로 가공했습니다. 논문·모델·외부 참조 코드에는 각 원저자의 라이선스와 이용 조건이 적용됩니다. 이 저장소는 해당 자료에 새로운 일괄 라이선스를 부여하지 않습니다.

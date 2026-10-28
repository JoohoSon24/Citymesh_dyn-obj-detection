# 통합 LaTeX 보고서

`PROJECT_EASY_PILOT.md`, `PROJECT_HARD_BASELINES.md`, `PROJECT.md`와 저장된 코드·CSV·이미지를 통합했다. 원 문서·실험 결과는 수정하지 않았다.

- `PROJECT_INTEGRATED.tex`: Intro → Method → Result 본문. 본문은 S×L2 공통식 → 방법별 S 생성 → 공통 베이킹의 흐름이며, 상세 구현 수식은 부록에 분리했다. 평가 차이, 정량표, 정성 이미지와 한계를 포함한다.
- `PROJECT_INTEGRATED_SINGLE.tex`: 표 수식까지 한 파일에 포함한 대안 소스. 그림은 `assets/`가 필요하다.
- `BASELINE_MATH.md`: 수학적 설명만 읽을 수 있는 요약.
- `PROJECT_INTEGRATED.pdf`: XeLaTeX로 컴파일한 보고서.
- `*_table.tex`, `assets/`: 실제 저장 지표에서 생성한 표와 결과 이미지.
- `source_manifest.json`: 원 문서·코드·지표 SHA-256.

이 폴더에서 실행:

```sh
xelatex -interaction=nonstopmode -halt-on-error PROJECT_INTEGRATED.tex
xelatex -interaction=nonstopmode -halt-on-error PROJECT_INTEGRATED.tex
```

XeLaTeX, xeCJK, Noto Serif CJK KR / Noto Sans CJK KR 폰트가 필요하다. Overleaf에 ZIP 내용을 올리고 compiler를 XeLaTeX, main document를 `PROJECT_INTEGRATED.tex`로 선택해도 된다.

표와 이미지 재생성은 원 프로젝트 폴더 구조에서 `../.venv/bin/python build_tables.py`로 실행한다. 보고서 컴파일 자체에는 원 데이터·모델·Python이 필요하지 않다. ZIP에는 보고서 입력 파일과 PDF를 포함한다.

`build_mask_figures.py`는 저장된 S 마스크를 동일 입력 위에 빨간색으로 표시한다. 추가 추론·학습은 하지 않는다. 원본 매핑과 제외 비율은 `mask_figure_manifest.json`에 있다.

추가 채움 실험: `../harder_dataset/natural_fill/`. Gradient harmonization과 Patch transfer는 같은 shadow-guard S를 사용한다. 표·확대·전체 결과·빨간 제외 마스크를 본문에 추가했다. 전체 지표는58행이다.

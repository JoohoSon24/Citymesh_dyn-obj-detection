# Dynamic object 제거 레퍼런스와 평면 베이킹 파일럿의 연결

작성: 2026-10-28. 원문 Method와 공식 프로젝트/구현을 확인해 정리했다. 논문 설명과 우리 실험으로의 이식 제안·실행 결과를 구분한다.

## 먼저 구분할 점

이 파일럿은 사진으로 실제 mesh를 재구성한 실험이 아니다. 120m×120m의 지면을 z=0으로 가정하고, 영상 좌표를 동일 평면 좌표에 대응시킨 뒤 homography와 grid sampling으로 텍스처 베이킹을 모사했다. 정점·삼각형 mesh 파일 생성, NeRF geometry 학습, Gaussian densification은 수행하지 않았다. v3에서는 관련 표현을 수정했다. 구현 근거: `../baking_pilot/pilot.py`의 `prep()`과 `render()`.

`Robust Gaussian Splatting`이라는 정확한 제목은 Darmon 등의 블러·포즈·색 보정 연구다. 동적 물체 제거 논문인 Ungermann 등의 `Robust 3D Gaussian Splatting for Novel View Synthesis in Presence of Distractors`와 다르다. 요청한 여섯 편을 정리하고 후자를 보충했다.

동적 물체를 ‘제거’한다는 것은 대체로 그 관측이 **정적 장면을 학습시키지 않도록** 손실 기여를 줄인다는 뜻이다. 모든 시기에 가려진 배경을 실제 관측으로 복원한다는 뜻은 아니다.

## 1. NeRF On-the-go — CVPR 2024

핵심 메소드 3줄:

1. DINOv2 특징을 작은 MLP에 넣어 픽셀별 불확실성 β를 예측하고, 비슷한 특징 사이의 β를 일관되게 만듦.
2. RGB 오차 대신 변형 SSIM의 밝기·대비·구조 오차 곱으로 불확실성을 학습해, 배경과 색이 비슷한 distractor도 구분하려 함.
3. NeRF는 1/(2β²)로 가중한 RGB 손실로 학습하며, 불확실성 학습과 NeRF 사이의 gradient를 분리하고 dilated patch sampling을 사용함.

원문 근거: [§3.1–3.4, Eq. 8–11](https://arxiv.org/abs/2405.18715) · [공식 구현/프로젝트](https://github.com/cvg/nerf-on-the-go)

**우리 실험과의 관계 — 해석/적용 제안.** 우리 손실의 S를 이진 마스크 대신 학습된 연속 가중치로 바꾸는 직접적인 근거다. DINO가 차량 주변에서 일반화하는지와 배경의 큰 오차를 차량으로 오인하는지를 확인해야 한다.

**v3 실행 범위.** DINO-SSIM soft로 불확실성→S 부분만 차용했다. 고정 L2 warm-up의 box-window SSIM target을 사용하고 feature-neighbor 정규화·dilated sampler·온라인 공동 갱신은 생략했으므로 원 논문 재현이 아니다.

$$
\ell_{\mathrm{modSSIM}}=(1-L)(1-C)(1-Q),\quad\mathcal{L}_{u}=\frac{\ell_{\mathrm{modSSIM}}}{2\beta^2}+\lambda\log\beta,\quad S=\frac{1}{2\beta^2}
$$

## 2. RobustNeRF — CVPR 2023

핵심 메소드 3줄:

1. 동적 물체를 별도 3D 표현으로 모델링하지 않고, 정적 NeRF를 맞출 때의 큰 photometric residual을 outlier로 취급함.
2. 잔차 분위수와 주변 픽셀·패치의 일관성을 이용해 inlier 마스크를 만들고, 반복적으로 다시 계산하는 robust estimation을 수행함.
3. 채택된 관측만으로 정적 장면을 학습하므로 사전 정의된 차량·사람 클래스나 semantic detector 없이 distractor 영향을 줄임.

원문 근거: [§3 robust estimation / spatial filtering](https://arxiv.org/abs/2302.00833) · [공식 구현/프로젝트](https://github.com/google-research/multinerf/blob/main/internal/robustnerf.py)

**우리 실험과의 관계 — 해석/적용 제안.** 기존 파일럿의 Robust형 S가 이 계열이다. 고점유 장면에서 반복 차량의 잔차가 작아지는 실패는, 낮은 잔차가 곧 정적 배경이라는 보장이 없다는 점을 보여준다.

**v3 실행 범위.** 기존 B/C 3-seed 결과를 유지했다. 해당 점수는 평면 baking에 옮긴 Robust형 규칙의 결과이며 RobustNeRF 전체 모델의 성능이 아니다.

$$
a=\mathbf{1}[\varepsilon\le\tau],\quad W=\mathbf{1}[B_3(a)\ge0.5],\quad S=\operatorname{stopgrad}\!\left(\mathbf{1}[B_{16}(W)(p_8)\ge0.6]\right)
$$

위 식은 원 논문을 그대로 적은 것이 아니라 기존 파일럿의 S 구현을 요약한 식이다.

## 3. SpotLessSplats — arXiv 2024 / 프로젝트 표기 TOG 2025

핵심 메소드 3줄:

1. Stable Diffusion의 사전학습 특징에서 의미적으로 비슷한 픽셀을 묶어, RGB 잔차만으로 놓치는 distractor의 공간적 범위를 찾음.
2. 공간 clustering 투표(SLS-agg) 또는 잔차의 확실한 구간만 약한 supervision으로 쓰는 특징→inlier MLP(SLS-mlp)로 관측 선택을 학습함.
3. 점진적 masking, 여러 batch의 잔차 분포 추적, opacity reset을 대신하는 utilization-based pruning으로 3DGS 학습과 제거 규칙의 충돌을 줄임.

원문 근거: [§4.1–4.2](https://arxiv.org/abs/2406.20055) · [공식 구현/프로젝트](https://github.com/lilygoli/SpotLessSplats)

**우리 실험과의 관계 — 해석/적용 제안.** 우리 베이킹에도 특징 기반 영역 선택과 약한 잔차 supervision은 이식할 수 있다. Gaussian의 생성·제거가 없는 고정 Texture에서는 utilization pruning과 opacity-reset 대응은 직접 적용 대상이 아니다.

**v3 실행 범위.** 이번에는 Stable Diffusion 특징 추출 및 SLS의 bound-supervision·pruning을 구현하지 않았다. DINO 실험을 SpotLessSplats 재현 또는 대체 구현으로 부르지 않는다.

$$
S_{\mathrm{mlp}}=\mathcal H(\mathcal F;\theta),\qquad\mathcal L_{\mathrm{sup}}=\max(U-\mathcal H,0)+\max(\mathcal H-L,0)
$$

## 4. WildGaussians — NeurIPS 2024

핵심 메소드 3줄:

1. 입력 DINOv2 특징에서 affine+softplus로 불확실성 σ를 예측하고, 입력과 렌더링의 DINO cosine 차이로 학습해 외관 변화에 덜 민감하게 만듦.
2. 3DGS의 densification을 불안정하게 하는 큰 가중 gradient를 피하려고 1/(2σ²)>1인 픽셀만 채택하는 binary mask로 바꿈.
3. 이미지별·Gaussian별 appearance embedding과 색변환을 별도로 학습해, 조명 변화 보정과 transient 관측 제거를 함께 처리함.

원문 근거: [§3.2–3.3, Eq. 8–11](https://arxiv.org/abs/2407.08447) · [공식 구현/프로젝트](https://github.com/jkulhanek/wild-gaussians)

**우리 실험과의 관계 — 해석/적용 제안.** 우리 연도별 사진에서도 조명 차이와 차량 변화를 분리해야 한다는 근거다. 고정 Texture에는 densification이 없으므로 같은 σ의 soft/hard S를 나란히 비교할 수 있다.

**v3 실행 범위.** DINO-cosine soft/hard 두 변형에 같은 고정 warm-up 기반 σ를 적용했다. 원 논문의 온라인 uncertainty 갱신, Gaussian appearance, density control은 사용하지 않았다.

$$
d=\min(1,2-2\cos(D,\widehat D)),\quad\mathcal L_u=\frac{d}{2\sigma^2}+\lambda\log\sigma,\quad S=\mathbf{1}\!\left[\frac{1}{2\sigma^2}>1\right]
$$

## 5. GS-W (Gaussian in the Wild) — ECCV 2024

핵심 메소드 3줄:

1. Gaussian마다 고유한 intrinsic appearance와 조명·날씨에 따른 dynamic appearance를 분리하고, reference-image 특징을 적응적으로 샘플링해 색을 모델링함.
2. Transient object는 별도 U-Net의 2D visibility map VM으로 처리하며, 가시성이 낮은 위치의 영상 손실 기여를 줄임.
3. VM이 모든 픽셀을 가리지 못하도록 VM을 1로 유도하는 정규화를 더함. dynamic appearance는 이동 물체 자체와 같은 개념이 아님.

원문 근거: [§4.2–4.5, Eq. 12–14](https://arxiv.org/abs/2403.15704) · [공식 구현/프로젝트](https://github.com/EastbeanZhang/Gaussian-Wild)

**우리 실험과의 관계 — 해석/적용 제안.** 우리 S를 학습된 2D visibility로 만들 수 있지만, 전부 제외하는 퇴화를 막는 규제가 필요하다. 파일럿의 전역 affine 색보정은 GS-W의 국소 appearance 모델보다 제한적이다.

**v3 실행 범위.** U-Net VM 및 per-Gaussian appearance는 이번 실험에서 실행하지 않았다. DINO σ와 SAM 투표는 GS-W의 visibility network 구현이 아니다.

$$
VM=U_\psi(I),\quad\mathcal L_{vm}=\|VM-1\|_2^2,\quad\mathcal L_c\supset\lambda_1\|VM\odot I_r-VM\odot I_{gt}\|_1
$$

위 식은 원 논문 손실 중 visibility 관련 부분만 발췌한 개념식이다. 전체 목적함수에는 SSIM·LPIPS·appearance sampling 관련 항도 포함된다.

## 6. Robust Gaussian Splatting (Darmon et al.) — arXiv 2404.04211, 2024

핵심 메소드 3줄:

1. 핵심 대상은 동적 물체 제거가 아니라 카메라 포즈 오차, motion blur, defocus blur, 영상 간 색 불일치에 강한 3DGS임.
2. 카메라 포즈의 Gaussian 분포로 motion blur와 포즈 보정을 함께 다루고, 투영된 Gaussian의 covariance로 defocus를 보정함.
3. 이미지별 affine 색변환으로 외관 불일치를 보정함. 차량을 식별해 제외하는 S나 transient 분리 모듈을 제안한 논문은 아님.

원문 근거: [§4.1–4.3](https://arxiv.org/abs/2404.04211)

**우리 실험과의 관계 — 해석/적용 제안.** 정합·블러·색 차이도 큰 잔차를 만들 수 있으므로 잔차를 곧 차량으로 해석하면 안 된다는 점과 관련된다. 우리 공통 정합·색보정의 필요성을 설명하는 보조 레퍼런스로 적합하다.

**v3 실행 범위.** 새로운 동적 물체 제거 기준선으로 실행하지 않았다. 원 논문의 blur/pose 최적화는 이번 고정 homography 실험 범위와 다르다.

$$
I_{\mathrm{pred},i}=A_i\,\mathcal R(\Gamma,\pi_i)+b_i\qquad\text{(color correction; not a dynamic mask)}
$$

위 식은 색변환의 역할을 보여주는 요약식이며 동적 제거 공식이 아니다.

## 7. Robust 3D Gaussian Splatting (Ungermann et al., 보충) — GCPR 2024 / arXiv 2408.11697

핵심 메소드 3줄:

1. RGB 채널별 centered residual과 작은 이웃·패치 문맥을 합쳐 distractor 후보의 공간적 단서를 만듦.
2. 고정 임계값 대신 logistic regression의 학습 가능한 결정 경계와 마스킹 면적 정규화로 관측 채택 여부를 갱신함.
3. SAM object mask와 결합해 픽셀 단위의 조각난 판정을 object-aware mask로 정리하고 정적 3DGS 학습에서 distractor를 제외함.

원문 근거: [§3.1–3.3](https://arxiv.org/abs/2408.11697) · [공식 구현/프로젝트](https://paulungermann.github.io/Robust3DGaussians/)

**우리 실험과의 관계 — 해석/적용 제안.** 동적 물체 제거를 뜻한 레퍼런스가 이 논문이라면 직접 관련된다. 기존 HuGS형과 SAM 사용은 같지만, 후보를 고르는 잔차·학습 규칙이 다르므로 동일 기법으로 취급하면 안 된다.

**v3 실행 범위.** SAM-region median은 저장된 SAM 분리 영역의 residual 투표만 차용했다. Logistic boundary와 원 논문 SAM IoU 절차는 생략했으므로 단순 영역화 ablation이다.

$$
\widehat S=\operatorname{sigmoid}(W\omega_3+b),\quad\mathcal L_{mask}=\mathcal L_{GS}(\widehat S\odot R,\widehat S\odot G)+\lambda\operatorname{mean}(1-\widehat S)
$$

## S 관점의 비교

| 연구 | S를 결정하는 주된 정보 | S 형태 | 고정 Texture에 직접 옮길 부분 | 이번 실행 |
|---|---|---|---|---|
| On-the-go | DINO 특징 + 변형 SSIM 기반 uncertainty | 연속 inverse-variance | 특징→불확실성→손실 가중 | frozen SSIM-soft 변형 |
| RobustNeRF | RGB residual + 공간 문맥 | binary | 반복 robust masking | 기존 3-seed 결과 유지 |
| SpotLessSplats | Stable Diffusion 특징 + 약한 residual label | 영역/MLP inlier | semantic grouping, 약한 supervision | 전체 기법 미실행 |
| WildGaussians | 입력/렌더 DINO cosine + uncertainty | 최종 binary | appearance와 독립된 uncertainty 학습 | frozen cosine soft/hard |
| GS-W | U-Net visibility + appearance 분리 | 연속 visibility | visibility regularization | 미실행 |
| Robust Gaussian Splatting | 포즈·블러·색 보정 | dyn S 없음 | 잔차의 nuisance 원인 분리 | 미실행 |
| Robust 3D Gaussian Splatting | residual + logistic boundary + SAM | object-aware mask | residual의 영역 단위 보완 | SAM 투표 ablation |

## 추가 실험 설계와 재현 범위

실험 전 고정한 조건은 [protocol.json](protocol.json)에 있다. 전체 원 논문 실행이 아니라 **고정된 S 생성 규칙을 평면 베이킹에 옮긴 비교**다. 새로 실시한 네 방법을 논문 이름 자체로 점수 표기하지 않는다.

- 기존 B/C 장면, 학습 2021–2025, 평가 2026, 4개 합성 카메라/연도, 20개 학습 관측을 사용한다.
- 공통 저장 L2 warm-up(600회)의 학습 영상 예측에서 S를 학습하고 이후 고정한다. 실제 동시 학습·반복 갱신과 다르며 이 추가 비용이 있다.
- DINOv2는 공식 pretrained ViT-S/14, 입력 392×518, 28×37 patch grid를 사용한다. 모델 버전·리사이즈·σ floor는 원 논문별 설정과 같다고 가정하지 않는다.
- SSIM-soft: 11×11 box 통계로 (1−luminance)(1−contrast)(1−structure)를 계산해 patch 평균 target으로 사용한다. DINO 384→64→1 MLP, softplus, σ≥0.1.
- Cosine-soft/hard: d=min(1,2−2cos(DINO input,DINO warm-up))를 target으로 affine+softplus σ를 학습한다. 동일 σ에 soft S=1/(2σ²), hard S=1[1/(2σ²)>1]를 각각 적용한다.
- 두 uncertainty predictor 모두 학습 seed 42, Adam 1000회, lr 0.001, patch batch 4096, loss=d/(2σ²)+0.5 logσ. uncertainty predictor는 각 장면에서 따로 학습한다.
- SAM-region median: warm-up RMS residual의 valid 중앙값보다 큰 픽셀에 3×3 majority를 적용하고, 저장된 SAM 분리 영역의 제외 비율이 40% 초과이면 영역 전체를 제외한다. SAM이 덮지 않은 위치의 connected component도 캐시의 분리 영역에 포함된다.
- 모든 새 baking은 같은 학습 영상 평균 초기값, 512² 파라미터→1024² 출력, Adam 600회, batch 4, Texture lr 0.025, 색변환 lr 0.001, 색 정규화 0.001, seed 42/43/44를 사용한다.
- S와 uncertainty는 baking에서 gradient를 받지 않는다. Soft S는 1보다 클 수 있고 이번에는 clip하지 않는다. 데이터항 분모는 기존과 동일한 valid 픽셀 수이므로 평균 가중치가 데이터항/색정규화의 상대 크기를 바꾸는 교란도 보고한다.
- 2026 영상과 OBB 평가 주석은 특징 추출·S 학습·baking에 사용하지 않는다. 모든 방법과 실패를 보고하며 새 지표를 보고 hyperparameter를 조정하지 않는다.

평가에서는 기존의 고정 OBB 33개와 고점유 OBB 21개(B4/C17)를 재사용한다. PSNR은 2026 실제 배경과 비교하고, LPIPS는 전체 OBB 33개의 고정 crop, 정적 PSNR은 별도 대조영역에서 계산한다. 평가 참조도 연도별 변화·정합오차·그림자가 남을 수 있는 proxy다.

## 실제 실행 결과

24회 baking을 완료했다. 아래 새 방법은 **고정 S 이식 실험**이며 원 논문 이름의 모델 점수가 아니다. 학습형 PSNR은 3 seeds의 평균 ± 표본표준편차, LPIPS·정적 PSNR은 평균이다. 기존 직접 fusion은 결정적 1회다.

| 방법 | B 고점유 PSNR ↑ | C 고점유 PSNR ↑ | B / C LPIPS ↓ | B / C 정적 PSNR ↑ |
|---|---:|---:|---:|---:|
| L2 (existing) | 19.01 ± 0.18 | 18.69 ± 0.24 | 0.149 / 0.338 | 21.97 / 24.40 |
| robust | 14.49 ± 0.36 | 16.00 ± 0.40 | 0.173 / 0.342 | 19.71 / 23.83 |
| UV + shadow guard (existing) | 21.00 | 22.23 | 0.079 / 0.130 | 24.58 / 26.47 |
| DINO-SSIM soft | 18.99 ± 0.20 | 18.62 ± 0.24 | 0.150 / 0.341 | 21.85 / 24.30 |
| DINO-cosine soft | 18.65 ± 0.12 | 18.34 ± 0.28 | 0.153 / 0.353 | 21.64 / 24.97 |
| DINO-cosine hard | 19.60 ± 0.00 | 20.01 ± 0.00 | 0.116 / 0.283 | 23.29 / 26.27 |
| SAM-region median | 17.15 ± 0.19 | 19.92 ± 0.02 | 0.181 / 0.285 | 20.37 / 26.18 |

### 마스크·관측 지지 진단

| 규칙 | B / C 채택 픽셀 비율 | B / C 평균 S | B / C 지면 ROI 무지지 비율 |
|---|---:|---:|---:|
| DINO-SSIM soft | 100.0% / 100.0% | 50.000 / 50.000 | 0.0% / 0.0% |
| DINO-cosine soft | 100.0% / 100.0% | 0.641 / 0.676 | 0.0% / 0.0% |
| DINO-cosine hard | 10.2% / 12.5% | 0.102 / 0.125 | 94.8% / 90.6% |
| SAM-region median | 27.0% / 27.7% | 0.270 / 0.277 | 46.7% / 87.6% |

채택 픽셀 비율은 각 방법의 모든 학습 카메라의 유효 영역에서 계산했다. 지면 ROI 무지지 비율은 S>10⁻⁶인 관측이 한 번도 투영되지 않은 평가 지면의 비율이다. **무지지=실제로 배경이 존재하지 않음**을 뜻하지 않는다. 마스크가 잘못 제외했을 수도 있다. Soft S에서는 양수라는 이유만으로 의미상 올바른 배경 관측이라는 보장도 없다.

### 해석

- 이번 단순 이식에서는 기존 UV + shadow guard의 고점유 PSNR을 넘지 못했다. 이 결과는 원 논문의 NeRF/3DGS 순위나 전체 모델 성능을 뜻하지 않는다.
- SSIM-soft는 두 장면 모두 σ floor 0.1에 도달해 S≈50으로 일정해졌다. 차량 선택 능력을 얻지 못한 실패이며, L2와의 작은 수치 차이는 데이터/색정규화 상대 배율도 달라졌으므로 제거 이득으로 해석하면 안 된다.
- DINO-cosine hard는 유효 픽셀의 약 10–12%만 채택하고, 지면 ROI의 약 91–95%가 최종 관측 지지를 잃었다. 일부 PSNR·LPIPS가 개선되어도 넓은 영역이 평균 초기값에 남는 현상과 분리해 보아야 한다. 실제 차량 제거 성공을 입증한 결과가 아니다.
- SAM 영역 투표도 지면 ROI를 과도하게 제외했다. 영역 경계를 보존하는 것만으로 반복 차량과 정적 배경을 식별할 수 없고, 어떤 영역을 제외할지 결정하는 증거가 중요하다.
- DINO의 입력 리사이즈·patch granularity는 작은 항공영상 차량에 불리할 가능성이 있다. 고정 warm-up과 온라인 갱신의 차이, SSIM target 스케일, 특징·마스크 해상도, S와 색 정규화의 상대 배율을 후속으로 분리 검증해야 한다. 이는 현재 진단에 근거한 가설이며 별도 실험으로 확정하지 않았다.
- 이번에 uncertainty predictor는 장면마다 seed42로 한 번 학습하고 고정했다. 3-seed 표준편차는 **baking 랜덤성만** 반영하며 마스크 학습의 불확실성까지 추정하지 않는다.
- 새 결과를 보고 가중치나 임계값을 조정하지 않았다. 다음 실험은 온라인 uncertainty 갱신과 지면 관측 지지 제약을 미리 정해 검증하는 방향이 적합하다.

## 피겨와 파일

- [B 장면 S 비교](assets/industrial_b_new_masks.png) / [C 장면 S 비교](assets/industrial_c_new_masks.png): 같은 2025년 20° 입력. Soft 색상은 log10 S를 [−1,1] 구간으로 표시한 공통 범위이고 표시 밖은 clip한다. 학습 가중치는 clip하지 않았다. Hard/SAM은 흰색이 채택 영역, 회색은 무효 투영이다.
- [동일 OBB 확대 비교](assets/new_qualitative.png): B/C의 기존 평가 주석 첫 OBB를 그대로 선택, seed42. 좋은 결과만 골라 crop을 선택하지 않았다. 학습 입력 / 실제 2026 배경 / 기존 L2 / 새 cosine soft / 새 cosine hard / 새 SAM 투표 / 기존 UV + shadow guard 순서다.
- [새 24행 지표](results/metrics.csv), [기존 결과 포함 비교](results/combined_metrics.csv), [위치별 지표](results/object_metrics.csv), [집계 JSON](results/summary.json).
- 재현 코드: `baking_pilot/reference_study_v3/prepare_dino.py`, `run_experiments.py`, `evaluate.py`, `protocol.json`. 실행 환경은 기존 `baking_pilot/.venv`를 재사용했다.
- 실행 순서: `prepare_dino.py` → `run_experiments.py` → `evaluate.py` → `make_results.py`. 결과 보존을 위해 학습 스크립트는 완료된 비-SAM baking을 재실행하지 않는다. 새 조건으로 다시 실행하려면 별도 결과 폴더에서 진행한다.
- 논문 원문·일부 피겨 crop은 `papers/`, `assets/*_paper.png`, 출처 및 crop 좌표는 `assets/paper_figure_manifest.json`에 기록했다. 원 논문 피겨는 설명 목적으로 인용했으며 저작권은 각 저자에게 있다.
- 항공영상: © Beeldmateriaal Nederland / PDOK, CC BY 4.0. 정합·색정규화·투영·베이킹·마스크 피겨로 가공했다.

# 기준선별 수학적 요약

실제 베이킹 코드의 수식이다. 원 논문 전체 NeRF를 재현한 수식과 구별한다. 자세한 예외 처리·평가·결과는 `PROJECT_INTEGRATED.tex`의 Method에 있다.

## 공통 모델

연도별 정사사진을 $J_t(u)$, 카메라 픽셀에서 UV로의 대응을 $\phi_c(r)$, 512² 파라미터를 $\theta$, 그 1024² bilinear 보간을 $T_\theta=U(\theta)$라 둔다. 입력과 예측은

$$I_{t,c}(r)=J_t(\phi_c(r)),\qquad P_i(r)=A_iT_\theta(\phi_c(r))+b_i.$$

픽셀 RGB MSE와 RMS 잔차는

$$e_i(r)=\tfrac13\|P_i(r)-I_i(r)\|_2^2,\qquad\varepsilon_i(r)=\sqrt{e_i(r)}.$$

valid 마스크 $V_i$와 관측 선택 $S_i$를 적용한 공통 손실은

$$L=\frac{\sum_{i\in\mathcal B,r}V_i(r)S_i(r)e_i(r)}{\max(1,\sum_{i\in\mathcal B,r}V_i(r))}+10^{-3}\big[\operatorname{mean}((A_i-I_3)^2)+\operatorname{mean}(b_i^2)\big].$$

분모는 선택된 픽셀 수가 아닌 valid 픽셀 수다. S를 적게 쓰면 데이터항의 규모도 줄어든다. Adam600업데이트, batch4, 텍스처lr0.025, 색lr0.001. 2023년 nadir의 색변환을 identity로 고정한다.

## 1. L2 / no-removal

$$S_i(r)=1.$$

L2는 정확히는 squared L2의 RGB 평균인 MSE다. 색보정과 재표본화를 무시한 단일 texel 문제에서는 최소화 해가 관측 산술평균이다. 따라서 차량도 평균에 들어가며 다수인 경우 더 강하게 남는다.

## 2. 시간 중앙값

$$T_k(u)=\operatorname{median}_{t=1,\ldots,5}J_{t,k}(u),\quad k\in\{R,G,B\}.$$

채널별 절대오차 합의 최소화 해다. 별도 베이킹을 하지 않는다. 차량이 다수면 배경 대신 차량색을 택할 수 있다. 원 중앙값 기준선은 512² 기저를 거치지 않는다.

## 3. Block-NeRF형

차량 OBB의 합집합을 5×5 팽창한 이진 마스크가 $D_i$이면

$$S_i^{\mathrm{Block}}=1-D_i.$$

각 사선 입력에 DOTA YOLO11s-OBB를 독립 적용한다. confidence0.25, vehicle class9/10. 원 논문의 Panoptic-DeepLab을 대체하고 포즈가 알려져 COLMAP 마스킹은 생략했다. 차량 그림자와 검출 누락은 남을 수 있다.

## 4. NeRF-HuGS형

20장 모든 쌍의 SIFT 매칭과 homography 검증 후, 자기 영상 포함5개 이상 영상에서 지지된 특징점을 SAM 양성 prompt로 준다. 얻은 영역을 $F_i$라 한다. 실제 3D SfM track 대신 2D 지지 횟수를 사용한다.

L2 warm-up600회의 잔차 평균을 $\mu_i$,95분위수를 $q_i$라 하면

$$C_i=\{\varepsilon_i\le\mu_i\},\quad Q_i=\{\varepsilon_i\le q_i\}\cup C_i,$$

$$H_i^0=(F_i\cup C_i)\cap Q_i,\quad H_i=H_i^0\cup\{B_7(1_{H_i^0})\ge0.5\}.$$

SAM으로 겹침을 해소한 분리 영역 $m_{i\ell}$ 중

$$\frac{|m_{i\ell}\cap H_i|}{|m_{i\ell}|}\ge0.5$$

인 영역의 합집합을 5×5 erosion해 정적 S를 얻는다. $B_k$는 k×k box 평균이다. 최종 베이킹은 warm-up을 이어받지 않고 공통 평균 초기값에서 다시600회 한다. 추가 마스크 생성 비용이 있다.

## 5. RobustNeRF형과 변형

각 반복, 각 영상의 valid RMS 잔차 중앙값으로

$$\tau_i=\operatorname{median}_{V_i=1}\varepsilon_i,\quad a_i=V_i1[\varepsilon_i\le\tau_i],\quad W_i=1[B_3(a_i)\ge0.5].$$

$p_h(r)$가 h×h 타일 중심이면 기본 선택은

$$S_i(r)=\operatorname{stopgrad}\big(1[B_{16}(W_i)(p_8(r))\ge0.6]\big).$$

- **OR**: $W_i\leftarrow W_i\lor a_i$ 후, 최종 타일 판정에도 $W_i$를 OR한다.
- **차량 크기**: $k_c=\lceil\operatorname{median}_{V_c=1}(4.5\times480/D_c)\rceil$, $h_c=\max(4,\lfloor k_c/2\rfloor)$. 문맥16과 타일8을 k와 h로 바꾼다.
- **깊이 적응**: 픽셀별 $k_c(r)=\operatorname{clip}(\lceil4.5\times480/D_c(r)\rceil,8,64)$의 가변 box를 사용하고 타일은8로 유지한다.

S의 gradient는 끊고 매 반복 다시 계산한다. 차량이 반복돼 잔차가 작으면 차량을 inlier로 선택할 수 있다.

## 6. 비차량 UV fusion

원본 UV의 차량 마스크 $D_t(u)$에서 $C_t=1-D_t$, $n(u)=\sum_tC_t(u)$로 둔다.

$$F_k(u)=\operatorname{median}_{t:C_t(u)=1}J_{t,k}(u),\qquad n(u)>0.$$

차량을 뺀 관측만 쓰므로 빈 배경이 소수여도 쓸 수 있다. $n=0$ 위치는 가장 가까운 관측값으로 초기화한 뒤256회

$$F^{s+1}(u)=\tfrac14\sum_{v\in\mathcal N_4(u)}F^s(v)\quad(n(u)=0)$$

로 갱신하고 $n>0$의 값은 고정한다. 최종 출력은 $U_{512\to1024}(D_{1024\to512}(F^{256}))$다. 보간은 관측하지 못한 배경의 추정이지 정답 복원이 아니다. 학습형 affine 색보정은 없고 공통 전처리만 사용한다.

## 7. UV fusion + shadow guard

밝기 $\ell_t$와 비차량 주변 평균을

$$\bar\ell_t=\frac{B_{61}(\ell_tC_t)}{\max(B_{61}(C_t),0.05)}$$

로 두고, 차량 마스크를 반경18texel의 ellipse로 팽창한 근처에서

$$G_t=\operatorname{dilate}_{37\times37}(D_t)\,1[\ell_t<0.72\bar\ell_t],\quad D'_t=D_t\lor G_t$$

를 만든다. $D'$로 같은 fusion과 보간을 수행한다. 진짜 그림자 라벨이 아니므로 어두운 지면·표식도 제외할 수 있다.

## 8. Geometry mask

$$S_{t,c}^{\mathrm{geo}}(r)=1[D_t(\phi_c(r))\le0.05].$$

원본 UV의 마스크를 bilinear 투영해 네 합성 시점이 같은 검출을 공유하게 한다. 공통 평균 초기값과600회 베이킹을 유지한다.

## 9. Geometry + init

위 $S^{\mathrm{geo}}$에 UV fusion의512² 결과 $\theta_F$를 초기값으로 쓴다. 손실은 공통 베이킹과 같다.

## 10. Geometry + prior

512격자로 area 리샘플링한 관측비율 $q=D_{\mathrm{area}}(n/5)$에서

$$w=\frac1{1+5q},\qquad a=1[q<0.01]$$

로 두고 인접 위치의 w 평균을 차분 가중치로 쓴다.

$$L_{\mathrm{TV}}=\operatorname{mean}(w^x|\Delta_x\theta|)+\operatorname{mean}(w^y|\Delta_y\theta|),$$

$$L_{\mathrm{anchor}}=\operatorname{mean}\big(a(\theta-\theta_F)^2\big),\qquad L_{\mathrm{total}}=L+0.01L_{\mathrm{TV}}+0.05L_{\mathrm{anchor}}.$$

관측이 적으면 평활화를 더 강하게 하고, 거의 미관측이면 보간 초기값에 묶는다. anchor는 선택된 위치 수가 아닌 전체 격자 평균이다. 세 Geometry 변형에는 shadow guard를 추가하지 않았다.

원 아이디어 확인: [Block-NeRF](https://arxiv.org/abs/2202.05263), [HuGS 공식 마스크 구현](https://github.com/cnhaox/NeRF-HuGS/blob/master/HuGS/generate_static_mask.py), [RobustNeRF 공식 코드](https://github.com/google-research/multinerf/blob/main/internal/robustnerf.py). 위 수식의 실제 실행 기준은 이 프로젝트의 `pilot.py`, `train.py`, `shadow_guard.py`다.

## 11. Gradient harmonization: S를 유지하고 경계 연결 개선

동일한 shadow-guard S로 만든 UV 합성값을 F, 배경 관측 수를 n이라 한다. 양 끝이 같은 연도에 사용된 인접 관측의 색 차이 중앙값을 g로 둔다. 해당 관측이 없는 edge는 g=0이다.

$$T^*=\arg\min_T\left[\sum_u\frac{n(u)}5\|T(u)-F(u)\|_2^2+2\sum_{(u,v)}\|T(v)-T(u)-g_{uv}\|_2^2\right].$$

1024격자의 선형계를 풀고 공통512기저를 거쳐 출력한다. 관측이 많은 곳의 색을 유지하면서 주변 변화량을 연결한다. S나 차량 검출을 바꾼 방법은 아니다.

## 12. Patch transfer: S를 유지하고 미관측부에 주변 질감 복사

완전 미관측부만 수정한다. 21×21 target patch의 알려진 문맥 K와 가장 가까운 donor Q를 고른다.

$$Q^*=\arg\min_Q\operatorname{mean}_{u\in K}\|F_P(u)-(F_Q(u)+\delta_Q)\|_2^2.$$

Donor는 전 픽셀에 적어도 한 번의 배경 관측이 있는 패치다. δ는 문맥의 채널별 median 차이를±0.12로 제한한 값이다. 중앙9×9의 미관측 픽셀만 복사하고 경계3픽셀은 원래 보간과 혼합한다. 관측된 texel은 고정한다. 최대2500회이며 남은 곳은 원래 조화 보간값을 사용한다. 복사된 질감이 진짜 배경과 같다는 보장은 없다.

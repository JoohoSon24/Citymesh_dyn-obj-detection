from pathlib import Path
import json,base64,re,html,zipfile,shutil
import markdown
P=Path(__file__).parent;BASE=P.parent.parent;V=P.parent/'presentation_v5';ST=P.parent/'reference_study_v3';summary=json.loads((ST/'results/summary.json').read_text());meta=json.loads((V/'assets/manifest.json').read_text())
def get(scene,m):return next(x for x in summary if x['scene']==scene and x['method']==m)
D=[]
def add(key,name,one,example,steps,meaning,limit,scope):D.append(dict(key=key,name=name,one=one,example=example,steps=steps,meaning=meaning,limit=limit,scope=scope))
add('l2','L2 — 모든 관측을 그대로 맞추기',
 '차량과 바닥을 구분하지 않고, 모든 사진의 색을 하나의 Texture로 최대한 설명한다.',
 '같은 주차면이 다섯 장 중 두 장에서는 빈 바닥, 세 장에서는 흰 차로 보인다고 하자. 바닥색과 차 색을 모두 맞추려다 중간색이나 흐릿한 차량 흔적이 남을 수 있다.',
 ['Texture를 현재 카메라에서 보았을 때의 예측 이미지를 만든다.','예측과 입력 사진의 RGB 색 차이를 제곱해 손실로 계산한다. 모든 유효 픽셀의 S는 1이다.','손실이 작아지도록 Texture와 카메라별 색보정을 600회 업데이트한다.'],
 '차량 제거 기능을 넣지 않았을 때의 출발점이다. 다른 방법이 이보다 좋아지는지 확인하는 기준이며, 비교 대상에서 빼면 개선 효과를 판단하기 어렵다.',
 '관측의 다수가 차량이어도 그대로 학습한다. 단일 위치의 단순화된 문제에서는 평균색이 해이지만, 실제 결과에는 투영·보간·색보정도 영향을 준다.',
 '공통 평면 투영, 평균 초기값, 512² Texture 파라미터와 1024² 출력을 사용한 L2 baking. 별도의 차량 마스크가 없다.')
add('median','Temporal median — 중간값으로 합치기',
 '같은 위치를 찍은 여러 사진에서, 색의 평균 대신 채널별 중간값을 고른다.',
 '다섯 사진에서 바닥이 네 번, 차가 한 번 보이면 특이한 차량 색의 영향을 줄이기 쉽다. 반대로 차가 세 번 이상 보이면 중간값도 차 쪽 색을 선택할 수 있다.',
 ['정합된 다섯 학습 연도에서 같은 지면 위치의 RGB 값을 모은다.','R·G·B 각각에서 중앙값을 계산해 Texture를 직접 만든다.','카메라로 다시 렌더링하며 Texture를 최적화하는 600회 학습은 하지 않는다.'],
 '복잡한 모델 없이도 다중 시기 영상만으로 어느 정도 차량 흔적을 줄일 수 있는지 보는 직접 합성 기준선이다.',
 '차량을 이해하고 제외하는 방법이 아니다. 빈 배경이 소수인 고점유 주차장에서는 다수 차량색을 배경처럼 선택할 수 있다.',
 '학습 2021–2025의 채널별 중앙값을 사용한 결정적 1회 계산. 기존 중앙값 기준선은 학습형 방법의 512² 파라미터 제약을 거치지 않으므로 표현 조건도 완전히 같지는 않다.')
add('block','Block-NeRF형 — 차량으로 검출된 픽셀은 제외하기',
 '먼저 “여기는 차량”이라는 마스크를 만들고, 그 부분은 바닥 Texture를 학습시키지 않는다.',
 '2021년에는 차가 있고 2022년에는 바닥이 보인다면, 2021년 차량 픽셀을 제외하고 2022년 바닥 관측을 사용한다. 몇 년 동안 차가 자주 보였는지보다 검출이 맞았는지가 중요하다.',
 ['각 합성 카메라 입력에서 차량 검출기로 차량의 회전 사각형 영역을 찾는다.','영역을 조금 넓혀 차량 경계까지 제외한다. 제외 마스크를 D라 하면 S=1−D이다.','남겨진 관측으로 동일한 600회 baking을 수행한다.'],
 '두 장면의 차량 위치 PSNR이 L2보다 높았다. 차량이라는 종류를 직접 찾는 정보가 반복 차량의 영향도 줄일 수 있음을 보여준다.',
 '검출하지 못한 차량은 남는다. 사각형 밖의 그림자는 놓칠 수 있고, 사각형 안에 섞인 바닥은 함께 제외한다. 검출기 기반 평가 위치는 같은 검출기를 쓰는 방법에 유리할 가능성도 있다.',
 '원 Block-NeRF의 semantic 제외 아이디어만 차용했다. 원래 분할기 대신 DOTA YOLO11s-OBB를 사용하고 차량 영역에 5×5 팽창을 적용했다. 전체 Block-NeRF 재현이 아니다.')
add('hugs','NeRF-HuGS형 — 정적 근거를 모아 영역별로 투표하기',
 '여러 영상에서 반복 확인되거나 예측과 잘 맞는 픽셀을 정적 후보로 보고, SAM 영역마다 후보가 절반 이상이면 채택한다.',
 'SAM이 나눈 한 바닥 영역에서 16개 픽셀 중 12개가 정적 후보라면 75%이므로 영역을 채택한다. 다른 영역은 4개만 후보라면 25%이므로 제외한다. 이 숫자는 이해용 예시이며 실제 실험 통계가 아니다.',
 ['여러 영상에 걸쳐 같은 위치로 대응되는 특징점을 찾고, 그 점 주변을 SAM으로 확장해 정적 근거를 만든다.','L2 warm-up의 예측과 색이 잘 맞는 픽셀도 정적 근거로 추가한다. 두 근거를 합치되 큰 오차와 주변 일관성으로 보정한다.','SAM으로 나눈 각 영역에서 정적 후보 비율을 계산한다. 50% 이상인 영역을 채택하고, 마지막에 경계를 조금 축소한다.'],
 '픽셀 하나씩 들쭉날쭉 판정하는 대신 영역 단위로 판정을 정리한다. 다만 B에서는 L2보다 낮고 C에서는 높아, 영역 단위 처리가 모든 장면에서 개선을 보장하지 않았다.',
 'SAM은 영역을 나누는 도구이지 자동으로 “차량은 제외”라고 판단하는 정답 장치가 아니다. 반복 차량이나 잘못된 정적 후보도 영역 전체의 채택으로 이어질 수 있다.',
 '실제 3D SfM track 대신 평면 SIFT 매칭의 영상별 지지 횟수를 사용했다. 자기 영상을 포함해 5개 이상 영상에서 지지된 점을 사용한다. 공통 L2 600회 warm-up으로 마스크를 만든 뒤, 평균 초기값에서 다시 600회 baking했다.')
add('robust','RobustNeRF형 — 예측과 안 맞는 관측을 이상치로 보기',
 '차량 검출기 대신, 현재 Texture로 설명하기 어려운 관측의 영향을 줄인다.',
 '대부분 바닥인 위치에 한 번만 빨간 차가 나타나면, 바닥 예측과 차 사진의 차이가 커져 차를 제외하기 쉽다. 반대로 계속 주차된 차는 예측에 흡수되어 오차가 작아질 수 있다.',
 ['현재 예측 이미지와 입력 이미지의 잔차, 즉 색 차이의 크기를 계산한다.','영상의 잔차 중앙값과 주변 픽셀·패치의 일관성으로 채택 마스크를 만든다.','이 마스크를 사용해 Texture를 갱신하고, 다음 반복에서 잔차와 마스크를 다시 계산한다.'],
 '이번 고점유 B/C에서는 차량 위치 PSNR이 L2보다 낮았다. 반복 차량이 많은 조건에서 “오차가 작다”는 근거만으로 빈 배경을 고르기 어려웠다.',
 '큰 오차는 차량뿐 아니라 정합오차·조명·그림자·아직 학습되지 않은 바닥 무늬에서도 발생한다. 작은 오차도 반드시 배경이라는 뜻은 아니다.',
 '공식 RobustNeRF의 잔차·공간 문맥 기반 선택을 평면 baking에 옮긴 변형이다. 이 계열은 매 반복 S를 갱신하지만, 새 DINO 실험은 고정 warm-up에서 만든 S를 사용한다는 차이가 있다.')
add('geo_mask','Geometry mask — 같은 차량 마스크를 여러 시점이 공유하기',
 '시점마다 차량을 다시 검출하지 않고, 원본 사진의 차량 마스크를 카메라에 맞춰 옮겨 쓴다.',
 '같은 2021년 사진에서 만든 네 가상 시점은 동일한 차량을 보고 있다. 각 시점에서 따로 검출해 하나는 찾고 하나는 놓치기보다, 원본의 한 검출 결과를 네 시점이 공유하게 한다.',
 ['원본 정사영상에서 연도별 차량 마스크를 만든다.','알려진 평면 좌표 대응으로 마스크를 각 가상 카메라에 투영한다.','투영된 차량 영역을 제외하고, 평균 초기값에서 600회 baking한다.'],
 'B/C 점수는 Block형과 비슷한 범위였다. 마스크를 시점 간 일관되게 만드는 것만으로 큰 추가 개선이 생기지는 않았다.',
 '원본 검출이 틀리면 여러 시점이 같은 오류를 공유한다. 실제 3D 표면의 가림과 높이 차이는 평면 homography만으로 처리할 수 없다.',
 '투영된 마스크 보간값이 0.05보다 크면 제외했다. 실제 3D mesh의 visibility를 계산한 방법이 아니라, 알려진 평면 투영을 이용한 마스크 공유다.')
add('uv_fusion','UV fusion — 차량 없는 사진의 바닥만 직접 모으기',
 '같은 바닥 위치에서 차량이 없다고 검출된 연도만 모아, 최종 Texture를 직접 합성한다.',
 '다섯 해 중 네 해는 차가 있고 한 해만 바닥이 보여도, 차량 마스크가 정확하면 그 한 번의 바닥 관측을 사용할 수 있다. 다수 관측을 따르는 중앙값과 다른 점이다.',
 ['같은 지면 위치에 대응하는 다섯 연도의 색과 차량 마스크를 모은다.','차량으로 검출된 연도를 빼고, 남은 관측의 채널별 중앙값을 사용한다.','한 번도 바닥이 관측되지 않은 위치만 주변 값으로 보간하고, 공통 512² 표현을 거쳐 1024²로 출력한다.'],
 'B/C 모두 L2·Block·HuGS·Robust형보다 높은 차량 위치 PSNR을 얻었다. 공통 지면 좌표를 알고 있는 이번 조건에서는 배경 관측을 직접 모으는 방식이 강했다.',
 '한 번도 바닥을 보지 못했다면 실제 바닥색을 알 수 없다. 이때 보간한 결과는 추정이며, 차선이나 표식을 틀리게 채울 수 있다. 그림자가 차량 마스크 밖에 있으면 포함될 수 있다.',
 '관측이 없는 위치는 최근접값으로 초기화한 후 4이웃 조화 보간 256회로 채웠다. 관측된 값은 보간 중 고정했다. 별도의 600회 baking 최적화는 하지 않는다.')
add('uv_shadow_guard','UV fusion + Shadow guard — 차량 주변의 어두운 그림자도 제외하기',
 '차량 차체를 제외한 뒤에도 남는 어두운 흔적을 줄이려고, 차량 근처의 유난히 어두운 관측까지 빼고 합성한다.',
 '흰 차의 차체는 검출 마스크 안에 들어가지만, 옆으로 드리운 그림자는 밖에 남을 수 있다. 차체만 제외하면 그 그림자가 바닥 Texture에 섞이므로 추가로 걸러낸다.',
 ['차량 주변 반경 18 texel을 살펴본다.','비차량 주변 평균 밝기의 0.72배보다 어두운 픽셀을 추가 제외한다.','남은 배경 관측으로 UV fusion을 수행한다. 사진 아래 가운데 결과는 마스크가 아니라 이 과정을 마친 최종 Texture다.'],
 '발표에 포함한 실행 방법 중 B/C 차량 위치 PSNR과 LPIPS가 가장 좋았다. 다만 C의 정적 PSNR은 UV fusion보다 조금 낮아, 제거와 바닥 보존이 항상 함께 좋아지지는 않았다.',
 '진짜 그림자를 이해한 분할 모델이 아니라 위치와 밝기의 휴리스틱이다. 원래 어두운 아스팔트나 표식도 제외할 수 있으며, 모든 그림자를 지운다는 보장은 없다.',
 '61×61 비차량 주변 평균 밝기와 차량 주변 18 texel 조건을 사용했다. 이후 합성·미관측부 보간·출력 표현은 UV fusion과 동일하다.')
add('geo_fusion','Geometry + Fusion init — 좋은 Texture에서 학습을 시작하기',
 'Geometry mask baking의 출발점을 평균색 대신 UV fusion 결과로 바꾼다.',
 '그림을 처음부터 그리는 대신 어느 정도 깨끗한 밑그림에서 시작하는 방식이다. 다만 이후 잘못된 관측을 계속 맞추면 밑그림의 장점이 사라질 수 있다.',
 ['UV fusion으로 초기 Texture를 만든다.','Geometry mask로 차량 관측을 제외한다.','같은 손실과 600회 baking으로 초기 Texture를 다시 최적화한다.'],
 'Geometry mask보다 C의 차량 위치 PSNR이 개선됐지만 B는 거의 같았다. 직접 UV fusion보다는 두 장면 모두 낮아, 좋은 초기값을 넣는 것만으로 합성 결과의 이득을 보존하지는 못했다.',
 '초기값은 정답을 고정하는 장치가 아니다. 학습 손실과 관측 선택이 같은 한, 600회 업데이트 후에는 다른 색으로 이동할 수 있다.',
 'S와 최적화 설정은 Geometry mask와 같고 Texture 초기값만 fusion 결과로 바꾼 분리 실험이다.')
add('geo_smooth','Geometry + Spatial prior — 관측이 부족하면 주변과 자연스럽게 연결하기',
 'Fusion 초기값에서 baking하되, 잘 안 보인 위치는 주변과 매끄럽게 연결하고 처음 추정값에서도 너무 멀어지지 않게 한다.',
 '관측이 충분한 바닥은 사진을 믿고, 계속 차에 가려진 구멍은 주변 바닥을 참고한다. 다만 매끄러운 바닥을 선호하다가 실제 차선도 흐려질 수 있다.',
 ['Geometry mask와 UV fusion 초기값을 사용한다.','배경 관측이 적은 위치에서 인접 Texture 값의 큰 차이를 더 강하게 억제한다. 이것이 가중 TV이다.','거의 미관측인 위치는 초기 fusion 보간값에서 멀어지지 않게 anchor를 추가한다.'],
 'Geometry + init보다 두 장면의 차량 위치 PSNR이 높았다. 하지만 직접 UV fusion에는 미치지 못해, 이번 설정에서는 공간 규제만으로 직접 합성을 대체하지 못했다.',
 '주변처럼 매끈하다는 가정은 규칙적인 아스팔트에는 맞을 수 있지만, 가려진 차선·표식·작은 구조를 정확히 복원하는 근거는 아니다.',
 '공통 손실에 0.01×가중 TV와 0.05×anchor를 추가했다. 이 Geometry 변형에는 shadow guard를 추가하지 않았다.')
add('dino_ssim_soft','DINO-SSIM soft — 픽셀을 얼마나 믿을지 학습하기',
 '단번에 사용/제외하지 않고, 영상의 특징에서 각 위치의 불확실성을 예측해 관측의 기여도를 조절한다.',
 '예측과 입력의 밝기·대비·구조가 많이 다르면 신뢰하기 어려운 관측으로 볼 수 있다. 이 차이를 DINO 특징과 연결해 비슷한 특징에도 비슷한 가중치를 주려는 방향이다.',
 ['학습 사진과 공통 L2 warm-up의 예측을 준비한다.','변형 SSIM의 밝기·대비·구조 오차 곱을 목표로, DINO 특징을 입력받는 작은 MLP가 불확실성을 학습한다.','불확실성이 클수록 작은 inverse-variance 가중치 S를 주고, 그 S를 고정한 채 baking한다.'],
 '두 장면 모두 불확실성이 하한 0.1에 붙어 S가 약 50으로 같아졌다. 서로 다른 위치를 다르게 선택하는 능력을 얻지 못한 실패다. L2와 비슷한 점수만으로 제거 성공이라고 해석할 수 없다.',
 'S는 0–1 확률이 아니므로 1보다 클 수 있다. 모든 S가 50으로 같아도 색 정규화에 대한 데이터 손실의 상대 크기는 달라져, L2와 수치가 완전히 같지는 않을 수 있다.',
 'NeRF On-the-go의 일부 아이디어만 옮겼다. 원 논문은 uncertainty와 NeRF를 분리된 gradient로 함께 학습하고 유사 특징 규제·dilated sampling을 사용하지만, 이번에는 고정 warm-up과 box-window target을 사용했다.')
add('dino_cosine_soft','DINO-cosine soft — RGB 색보다 특징 차이로 신뢰도 정하기',
 '입력과 예측의 색이 아니라 DINO 특징의 차이로 불확실성을 학습하고, 관측을 연속 가중치로 반영한다.',
 '같은 바닥이 조명 때문에 밝아진 경우와 바닥 위에 차량이 생긴 경우를 단순 RGB 차이만으로 나누기는 어렵다. 특징 공간의 차이를 사용해 이를 구분하려는 시도다.',
 ['입력과 L2 warm-up 예측에서 DINO 특징을 추출한다.','두 특징의 cosine 유사도로 불일치 정도를 계산하고, 특징→불확실성 변환을 학습한다.','S=1/(2σ²)의 연속 가중치를 고정해 baking한다. 다음 hard 방법과 같은 σ를 사용한다.'],
 '이번 조건에서는 B/C 차량 위치 PSNR이 L2보다 낮았다. 의미 특징을 사용한다는 사실만으로 차량 제거가 개선되지는 않았다.',
 'DINO 특징도 차량 정답 마스크가 아니다. 리사이즈와 14픽셀 patch 단위 특징이 작은 항공 차량에 불리할 가능성이 있지만, 그 원인을 별도로 분리 검증한 결과는 아니다.',
 'WildGaussians의 DINO 기반 uncertainty 학습을 참고한 soft 대조군이다. 원 논문의 최종 binary 선택, Gaussian appearance와 density control을 모두 포함한 모델이 아니다.')
add('dino_cosine_hard','DINO-cosine hard — 같은 불확실성으로 사용/제외를 결정하기',
 '앞 방법과 똑같이 학습한 불확실성을 사용하되, 일정 기준을 통과한 관측만 사용한다.',
 '믿기 어려운 관측을 조금씩 남기는 대신 아예 빼는 방식이다. 잘 작동하면 차량의 기여를 강하게 억제하지만, 기준이 너무 엄격하면 바닥 사진까지 거의 모두 잃는다.',
 ['DINO-cosine soft와 동일한 σ를 사용한다.','1/(2σ²)>1인 픽셀은 S=1, 나머지는 S=0으로 만든다. 이 기준은 “정적 확률 50%” 같은 의미가 아니다.','선택된 관측으로 평균 초기값에서 600회 baking한다.'],
 '일부 차량 위치 PSNR과 LPIPS는 L2보다 좋아졌지만, B/C 지면 ROI의 약 94.8%/90.6%가 채택 관측의 지지를 잃었다. 넓은 영역이 평균 초기값에 남을 수 있으므로 점수만으로 배경 복원을 주장할 수 없다.',
 '제외를 많이 했다는 것과 차량만 정확히 제외했다는 것은 다르다. 원래 배경 관측이 있었더라도 마스크가 이를 지우면 새 데이터로 Texture를 고칠 수 없다.',
 'WildGaussians는 큰 gradient가 Gaussian densification에 영향을 주는 문제 때문에 binary mask를 사용한다. 우리 고정 Texture에는 그 과정이 없으며, 고정 S로만 규칙을 옮겼다. 원 논문의 우열을 평가한 결과가 아니다.')
add('sam_region_median','SAM-region vote — 큰 오차를 영역 단위로 정리하기',
 '오차가 큰 픽셀을 찾은 뒤, 그 픽셀이 많은 SAM 영역을 통째로 제외한다.',
 '한 영역의 픽셀 100개 중 45개가 큰 오차라면 40%를 넘으므로 영역을 제외한다. 영역 경계는 깔끔해질 수 있지만, 처음 오차 판정이 틀리면 바닥 전체를 지울 수도 있다.',
 ['L2 warm-up의 잔차가 영상의 중앙값보다 큰 위치를 제외 후보로 만든다.','3×3 주변 다수결로 조각난 후보를 정리한다.','저장된 SAM 분리 영역에서 제외 후보가 40%를 넘으면 영역 전체를 제외하고, 나머지는 사용한다.'],
 'B의 차량 위치 PSNR은 L2보다 낮고 C는 높았다. 하지만 지면의 무지지 비율이 B 약 46.7%, C 약 87.6%로 커서 과도한 제외가 주요 진단이었다.',
 '영역을 나누는 SAM과 정적/동적 판단은 별개의 역할이다. 차량으로 잘못 선택한 바닥을 SAM이 자동으로 되살려 주는 것은 아니다.',
 'Robust 3D Gaussian Splatting의 object-aware masking 취지만 단순 차용했다. 원 논문의 logistic 결정 경계와 SAM IoU 절차는 구현하지 않았다. HuGS와도 후보 생성 및 영역 투표 기준이 다르다.')

# Original-paper studies without pilot execution.
L=[
 dict(key='spotless',name='SpotLessSplats — 의미가 비슷한 픽셀을 함께 판단하기',one='RGB 오차만 보지 않고, Stable Diffusion 특징에서 의미적으로 비슷한 영역을 찾아 distractor 영향을 줄인다.',example='차량 가운데 일부가 바닥과 비슷한 색이면 그 부분만 오차가 작아 마스크에 구멍이 날 수 있다. 의미 특징으로 연결된 영역을 같이 판단해 물체의 범위를 더 잘 다루려는 접근이다.',steps=['Stable Diffusion을 이미지를 새로 그리는 용도가 아니라 특징 추출기로 사용한다.','공간 clustering의 투표 또는 특징을 입력받는 inlier MLP로 사용/제외를 판단한다. MLP에는 잔차로 확실한 부분만 약한 supervision을 준다.','점진적 masking, 여러 batch의 잔차 분포 추적, Gaussian 활용도 기반 pruning을 함께 사용해 3DGS 학습과 마스크의 충돌을 줄인다.'],relation='고정 Texture에도 특징 기반 영역 선택은 옮길 수 있다. 그러나 Gaussian을 생성·제거하는 부분은 그대로 적용할 대상이 없다. 이번 DINO 실험은 SD 특징이나 이 supervision을 쓰지 않았으므로 SpotLessSplats의 재현으로 부르면 안 된다.',paper='https://arxiv.org/abs/2406.20055',figure='spotless'),
 dict(key='gs_w',name='GS-W — 외관 변화와 가려짐을 따로 처리하기',one='조명·날씨에 따른 외관 변화는 appearance 모델로 설명하고, 가려진 관측은 학습된 visibility map으로 덜 사용한다.',example='같은 벽이 햇빛 때문에 밝아진 것은 벽이 사라진 것이 아니다. 반대로 벽 앞을 사람이 가린 관측은 벽 색을 학습하는 데 쓰지 않는 편이 좋다. 두 현상을 다른 장치로 다룬다.',steps=['각 Gaussian의 고유한 appearance와 촬영 조건에 따른 dynamic appearance를 나누어 모델링한다. 여기서 dynamic appearance는 이동 물체 자체를 뜻하지 않는다.','별도 U-Net이 2D visibility map을 만들고, 낮은 visibility 위치의 영상 손실 기여를 줄인다.','모든 픽셀을 안 보인다고 하면 학습을 피할 수 있으므로, visibility를 1로 유도하는 정규화로 퇴화를 막는다.'],relation='우리 S도 학습형 visibility로 만들 수 있다는 근거다. 다만 이번에는 이 U-Net이나 Gaussian별 appearance를 실행하지 않았다. 우리 전역 affine 색보정은 훨씬 제한적인 보정이다.',paper='https://arxiv.org/abs/2403.15704',figure='gs_w'),
 dict(key='robust_gs',name='Robust Gaussian Splatting — 차량보다 촬영 오차를 보정하는 연구',one='카메라 포즈, motion/defocus blur, 사진 간 색 차이 때문에 재구성이 망가지는 문제를 줄인다.',example='차량이 하나도 없어도 사진이 흔들리거나 위치가 조금 어긋나면 입력과 예측의 차이가 커진다. 이 차이를 모두 차량 탓으로 보면 멀쩡한 바닥까지 제외할 수 있다.',steps=['카메라 포즈를 Gaussian 분포로 모델링해 motion blur와 포즈 보정을 함께 다룬다.','투영된 Gaussian의 covariance에 보정을 더해 defocus blur를 다룬다.','이미지별 affine 색변환으로 색 불일치를 보정한다.'],relation='정적 배경에서도 잔차가 커질 수 있으므로 정합·색보정이 중요하다는 보조 레퍼런스다. 이름이 비슷한 Robust 3D Gaussian Splatting for Novel View Synthesis in Presence of Distractors와 다르며, 후자가 SAM-region 실험과 연결되는 동적 제거 연구다.',paper='https://arxiv.org/abs/2404.04211',figure='robust_gs')]
refs=json.loads((ST/'references.json').read_text());paper_map={x['key']:x for x in refs}
sourcekeys={'block':None,'hugs':None,'robust':'robustnerf','dino_ssim_soft':'on_the_go','dino_cosine_soft':'wildgaussians','dino_cosine_hard':'wildgaussians','sam_region_median':'robust_3dgs'}
intro='''# 차량 제거 베이킹: 방법별 쉬운 설명 보고서

PPT `dynamic_object_baking_v5.pptx`에 대응하는 설명이다. 실행한 14개 방법과 문헌만 검토한 3개 연구를 구분했다. 수치와 그림은 이미 저장된 실험 결과를 사용했으며 새 실험을 추가하지 않았다.

## 먼저 이해할 공통 상황

같은 주차장을 2021–2025년에 찍은 사진 다섯 장이 있다. 어떤 주차면은 차가 있고, 어떤 해에는 빈 바닥이 보인다. 목표는 이 사진들을 이용해 **차량 흔적은 줄이고 차선·바닥 무늬는 보존한 한 장의 Texture**를 만드는 것이다.

이 파일럿에서는 실제 mesh를 사진에서 복원하지 않았다. 지면을 120m×120m의 평면으로 가정하고, 각 연도 사진을 같은 바닥 좌표에 맞췄다. 이 대응에서 합성 카메라 네 개를 만들어 베이킹을 모사했다. 실제 사선 촬영의 parallax나 비평면 건물 형상을 검증한 실험은 아니다.

차량 제거는 대체로 **차량 픽셀이 바닥 Texture를 학습시키지 않게 하는 것**이다. 모든 사진에서 계속 가려진 바닥을 실제로 관측한 것처럼 복원하는 기능은 아니다.

### 용어를 짧게 풀면

| 용어 | 쉬운 뜻 |
|---|---|
| Texture | 바닥 표면에 입힐 색과 무늬를 담은 이미지 |
| UV | 표면의 어느 위치를 Texture의 어느 위치에 대응시킬지 정한 좌표 |
| Baking | 여러 사진의 색을 표면의 Texture로 모으거나, 관측 사진을 잘 설명하도록 Texture를 최적화하는 과정 |
| S | 관측 픽셀을 얼마나 사용할지 정하는 가중치. 이진이면 0=제외, 1=사용. 연속 가중치는 1보다 클 수도 있음 |
| 잔차(residual) | 현재 예측과 실제 입력의 색 차이. 차량의 정답 라벨은 아님 |
| Warm-up | 마스크를 만들기 전에 먼저 학습한 Texture와 그 예측. 이번에는 공통 L2 600회 결과를 활용 |
| SAM | 이미지를 여러 영역으로 나누는 모델. 영역이 차량인지 정적 배경인지를 자동으로 정답 판정해 주는 장치는 아님 |
| DINO 특징 | 사진의 색을 그대로 쓰는 대신 사전학습 모델이 추출한 표현. 차량 검출 정답과는 다름 |
| Soft / Hard | 조금씩 기여도를 조절 / 기준에 따라 사용하거나 제외 |
| 미관측 / 무지지 | 원래 빈 배경을 한 번도 못 봄 / 현재 S가 사용할 관측을 하나도 남기지 않음. 둘은 다름 |

### 사진과 숫자를 읽는 방법

각 실행 방법의 사진은 **차량 입력 → 최종 Texture → 2026년 실제 빈 배경** 순서다. C 장면의 같은 고정 평가 위치를 사용했다. 가운데 사진이 결론적인 산출물이며 마스크 그림이 아니다. 사진은 seed42 결과다.

차량 위치 PSNR은 2026년 빈 배경과의 색 오차로, 높을수록 좋다. LPIPS는 지각적인 이미지 차이로 낮을수록 좋다. 정적 PSNR은 차량이 없는 대조영역의 보존을 확인한다. 차량 위치 PSNR은 고점유 OBB 21개, LPIPS는 전체 OBB 33개의 고정 crop을 사용하므로 평가 범위도 다르다. 2026년 배경에도 연도 변화·그림자·정합오차가 있어 완벽한 정답 영상은 아니다.

학습형 PSNR은 3개 baking seed의 평균±표본표준편차, 직접 합성은 1회다. DINO 실험은 장면마다 마스크 학습을 한 번만 하고 고정했으므로 이 표준편차가 마스크 학습의 불확실성까지 나타내지는 않는다.

'''
main=intro;individual=P/'individual';individual.mkdir(exist_ok=True)
for i,a in enumerate(D,1):
 b=get('industrial_b',a['key']);c=get('industrial_c',a['key'])
 def ps(z):return f"{z['PSNR_occupied3_4']:.2f}"+(f" ± {z['PSNR_occupied3_4_std']:.2f}" if z['PSNR_occupied3_4_std'] is not None else '')
 section=f"## {i}. {a['name']}\n\n**한 문장으로:** {a['one']}\n\n### 예를 들면\n\n{a['example']}\n\n### 어떻게 작동하나\n\n"
 for j,st in enumerate(a['steps'],1):section+=f'{j}. {st}\n'
 section+=f"\n### 이번 파일럿에서 실제로 한 일\n\n{a['scope']}\n\n### 결과와 해석\n\n차량 위치 PSNR: **B {ps(b)} dB / C {ps(c)} dB**.\n\n{a['meaning']}\n\n### 무엇을 조심해야 하나\n\n{a['limit']}\n\n"
 section+='| 차량이 있는 입력 | 최종 Texture | 실제 빈 배경(2026) |\n|---|---|---|\n'
 paths=['industrial_c_input.png','industrial_c_'+a['key']+'.png','industrial_c_reference.png']
 section+='| '+' | '.join('!['+lab+'](assets/'+path+')' for lab,path in zip(['입력','최종 결과','빈 배경'],paths))+' |\n\n'
 section+='사진은 C의 첫 고정 평가 OBB 주변을 동일 범위로 자른 결과다. 인접 차량·그림자는 평가 폴리곤 밖일 수 있으며, 화면 전체가 해당 PSNR의 평가 영역이라는 뜻은 아니다.\n\n'
 sk=sourcekeys.get(a['key'])
 if sk:section+='원 논문 연결: ['+paper_map[sk]['name']+']('+paper_map[sk]['paper']+'). 원 논문 전체 성능과 위 파일럿 수치를 구분해야 한다.\n\n'
 if a['key']=='hugs':section+='**SAM을 쓰는 다른 방법과의 차이:** HuGS형은 정적 후보가 **50% 이상이면 채택**, SAM-region vote는 큰 잔차 후보가 **40% 초과면 제외**한다. 후보를 만드는 근거도 다르므로 같은 방법으로 볼 수 없다.\n\n'
 if a['key']=='uv_fusion':
  detail=(P/'uv_fusion_detail.md').read_text()
  section=section.replace('### 결과와 해석',detail+'\n### 결과와 해석',1)
 main+=section
 (individual/(f'{i:02d}_'+a['key']+'.md')).write_text(section.replace('(assets/','(../assets/'))
for j,a in enumerate(L,15):
 section=f"## {j}. {a['name']}\n\n**실행 상태: 문헌 검토만 수행. 아래 원 논문 이미지를 우리 파일럿의 실행 결과로 해석하면 안 된다.**\n\n**한 문장으로:** {a['one']}\n\n### 예를 들면\n\n{a['example']}\n\n### 핵심 동작\n\n"
 for k,st in enumerate(a['steps'],1):section+=f'{k}. {st}\n'
 section+='\n### 우리 실험과의 관계\n\n'+a['relation']+'\n\n![원 논문의 정성 결과](assets/'+a['figure']+'_paper_result.png)\n\n위 그림은 원 논문의 결과 인용이다. 우리 B/C 영상의 결과나 정량 비교 대상이 아니다. [원문]('+a['paper']+').\n\n'
 main+=section;(individual/(f'{j:02d}_'+a['key']+'.md')).write_text(section.replace('(assets/','(../assets/'))
main+='''## 실행한 방법의 전체 비교

| 방법 | B 차량 PSNR ↑ | C 차량 PSNR ↑ | B / C LPIPS ↓ | B / C 정적 PSNR ↑ |
|---|---:|---:|---:|---:|
'''
labels=json.loads((V/'methods.json').read_text())['labels']
for a in D:
 b=get('industrial_b',a['key']);c=get('industrial_c',a['key']);main+=f"| {labels[a['key']]} | {ps(b)} | {ps(c)} | {b['LPIPS_object_crop_mean']:.3f} / {c['LPIPS_object_crop_mean']:.3f} | {b['PSNR_static_control_proxy']:.2f} / {c['PSNR_static_control_proxy']:.2f} |\n"
main+='''
## 방법들의 차이를 기억하는 틀

- **L2 / 중앙값:** 차량을 이해하지 않고 모든 관측을 통계적으로 합친다.
- **Block / Geometry mask:** 차량 검출을 믿고 차량 관측을 제외한다. Geometry는 같은 검출을 여러 시점이 공유한다.
- **HuGS / SAM 투표:** 픽셀 수준의 근거를 영역 단위 판정으로 정리한다. 어떤 후보를 썼는지가 중요하다.
- **Robust:** 현재 예측과 맞지 않는 관측을 반복적으로 제외한다. 반복 차량이 예측에 흡수되면 실패할 수 있다.
- **UV fusion / Shadow guard:** 차량이나 그림자를 뺀 실제 배경 관측을 공통 좌표에서 직접 모은다.
- **Init / Prior:** 시작 Texture와 미관측부에 대한 가정을 바꾼다. 새로운 배경 관측을 만들어 내는 것은 아니다.
- **DINO soft / hard:** 특징에서 관측의 불확실성을 학습한다. 좋은 특징을 써도 가중치가 같아지거나 과도하게 제외하는 실패가 가능하다.

이번 평면 조건에서의 판단은 **UV fusion을 강한 기준선으로 보고, 그림자 문제가 있으면 고정 shadow guard의 이득과 정적 영역 손상을 함께 확인한다**는 것이다. 다른 실제 3D 장면이나 원 논문의 전체 모델에 대한 순위로 일반화할 수 없다.

## 출처와 재현 자료

- PPT: `dynamic_object_baking_v5.pptx`.
- 실제 수치: `baking_pilot/reference_study_v3/results/summary.json`, 원시 지표 및 학습 조건은 같은 프로젝트의 `metrics.csv`, `protocol.json`.
- 기본 방법의 수식: `baking_pilot/latex/BASELINE_MATH.md`. 구현: `pilot.py`, `harder_dataset/improved_baselines/train.py`, `shadow_guard.py`.
- DINO / SAM 이식: `baking_pilot/reference_study_v3/run_experiments.py`와 `protocol.json`.
- 문헌의 자세한 3줄 요약·범위 구분: `dynamic_object_baking_references_v3.md`.
- Block-NeRF 원문: [Block-NeRF](https://arxiv.org/abs/2202.05263). HuGS의 마스크 원 구현: [NeRF-HuGS](https://github.com/cnhaox/NeRF-HuGS/blob/master/HuGS/generate_static_mask.py).
- 그림은 실제 저장된 Texture 및 입력에서 만들었다. 항공영상 © Beeldmateriaal Nederland / PDOK, CC BY 4.0. 논문 결과 그림의 저작권은 각 저자에게 있다.
'''
assets=P/'assets';assets.mkdir(exist_ok=True)
for path in (V/'assets').glob('industrial_c_*.png'):shutil.copy2(path,assets/path.name)
for key in ['spotless','gs_w','robust_gs']:shutil.copy2(V/'assets'/(key+'_paper_result.png'),assets/(key+'_paper_result.png'))
(P/'method_explanations.md').write_text(main)
# Embed pictures to make HTML a self-contained report, including offline viewing.
body=markdown.markdown(main,extensions=['tables','fenced_code','toc'])
def embed(m):
 rel=m.group(1);q=P/rel
 return 'src="data:image/png;base64,'+base64.b64encode(q.read_bytes()).decode()+'"' if q.exists() else m.group(0)
body=re.sub(r'src="(assets/[^\"]+)"',embed,body)
css='''body{font-family:Arial,"Noto Sans CJK KR","Malgun Gothic",sans-serif;line-height:1.8;color:#1d2938;max-width:1060px;margin:40px auto;padding:0 24px}h1{color:#254b95}h2{margin-top:60px;padding-top:22px;border-top:2px solid #254b95;color:#254b95}h3{margin-top:26px}table{width:100%;border-collapse:collapse;margin:20px 0}th,td{border:1px solid #d8dee8;padding:9px 12px;text-align:left}th{background:#edf2f9}td img{width:100%;height:auto;display:block}img{max-width:100%;max-height:620px;object-fit:contain}a{color:#254b95}code{background:#f0f3f7;padding:2px 4px;overflow-wrap:anywhere}p,li{overflow-wrap:anywhere}@media print{h2{break-before:page}body{margin:0;max-width:none;font-size:10pt}table,figure{break-inside:avoid}}'''
page='<!DOCTYPE html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>차량 제거 베이킹 — 방법별 쉬운 설명</title><style>'+css+'</style><body>'+body+'</body></html>'
(P/'method_explanations.html').write_text(page)
shutil.copy2(P/'method_explanations.html','/mnt/data/dynamic_object_baking_method_reports.html')
# Standalone root Markdown uses root-relative project links; ZIP contains portable relative copies.
rootmd=main.replace('(assets/','(baking_pilot/method_reports/assets/')
(BASE/'dynamic_object_baking_method_reports.md').write_text(rootmd)
# Downloadable MD stays adjacent to downloadable asset folder when extracted from the ZIP.
shutil.copy2(P/'method_explanations.md','/mnt/data/dynamic_object_baking_method_reports.md')
with zipfile.ZipFile('/mnt/data/dynamic_object_baking_method_reports.zip','w',zipfile.ZIP_DEFLATED) as z:
 for q in [P/'method_explanations.md',P/'method_explanations.html']+list(individual.glob('*.md'))+list(assets.glob('*.png')):z.write(q,q.relative_to(P))
manifest={'executed_method_reports':len(D),'literature_only_reports':len(L),'individual_reports':len(list(individual.glob('*.md'))),'metric_source':str(ST/'results/summary.json'),'new_experiments':False,'images':'actual same fixed C audit crop; literature images labelled separately'}
(P/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(manifest)

from pathlib import Path
import numpy as np,json,csv,hashlib
from PIL import Image,ImageDraw,ImageFont
P=Path(__file__).parent;BASE=P.parent;A=P/'assets';R=P/'results';font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
names={'dino_ssim_soft':'DINO-SSIM soft','dino_cosine_soft':'DINO-cosine soft','dino_cosine_hard':'DINO-cosine hard','sam_region_median':'SAM-region median','l2':'L2 (existing)','uv_shadow_guard':'UV + shadow guard (existing)'}
for scene in ['industrial_b','industrial_c']:
 old=BASE/'harder_dataset/scenes'/scene/'results';v=np.load(old/'views.npz');im=v['images'][17];valid=v['valid'][1].astype(bool);pan=[('Input (2025 / 20 deg)',im)]
 for method in ['dino_ssim_soft','dino_cosine_soft','dino_cosine_hard','sam_region_median']:
  m=np.load(R/scene/(method+'_mask.npz'))['static'][17]
  if 'soft' in method:
   z=np.clip((np.log10(np.maximum(m,.1))+1)/2,0,1);pic=np.stack([z,.2+z*.6,1-z],-1)
  else:pic=np.repeat(m[...,None],3,-1)
  pic[~valid]=.83;pan.append((names[method],pic))
 c=Image.new('RGB',(2300,390),'white');d=ImageDraw.Draw(c)
 for j,(label,pic) in enumerate(pan):
  x=j*460;d.text((x+5,6),label,font=font,fill='black');im2=Image.fromarray(np.uint8(np.clip(pic,0,1)*255)).resize((448,336));c.paste(im2,(x+5,42))
 c.save(A/(scene+'_new_masks.png'))
# Reuse first fixed audit OBB for each scene. No selection by output quality.
cols=['input','reference','l2','dino_cosine_soft','dino_cosine_hard','sam_region_median','uv_shadow_guard'];labels=['Training input','2026 background','L2','DINO-cos soft','DINO-cos hard','SAM-region','UV + shadow']
c=Image.new('RGB',(1960,690),'white');d=ImageDraw.Draw(c);meta=[]
for row,scene in enumerate(['industrial_b','industrial_c']):
 old=BASE/'harder_dataset/scenes'/scene/'results';ann=json.loads((old/'verified_annotations.json').read_text())['accepted'][0];source=np.load(old/'aligned.npz')['images'];x,y=np.mean(ann['poly'],axis=0).astype(int);crop=(max(0,x-64),max(0,y-64),min(1024,x+64),min(1024,y+64));meta.append({'scene':scene,'annotation':ann,'crop':list(map(int,crop))});yy=row*345
 d.text((5,yy+5),scene+' / fixed OBB #'+str(ann['id']),font=font,fill='black')
 for j,m in enumerate(cols):
  if m=='input':z=source[ann['year']-2021]
  elif m=='reference':z=source[5]
  elif m=='l2':z=np.asarray(Image.open(old/'textures/l2.png'))/255
  elif m=='uv_shadow_guard':z=np.asarray(Image.open(BASE/'harder_dataset/improved_baselines/scenes'/scene/'results/textures/uv_shadow_guard.png'))/255
  else:z=np.asarray(Image.open(R/scene/'42/textures'/(m+'.png')))/255
  z=Image.fromarray(np.uint8(z.clip(0,1)*255));x0,y0,x1,y1=crop;z=z.crop(crop).resize((268,268));c.paste(z,(j*280+5,yy+65));d.text((j*280+5,yy+36),labels[j],font=font,fill='black')
c.save(A/'new_qualitative.png');(A/'qualitative_manifest.json').write_text(json.dumps(meta,indent=2))
# Write report: results and limitations are backed by actual output files.
s=json.loads((R/'summary.json').read_text());cfg=json.loads((P/'protocol.json').read_text())
def get(scene,m):return next(z for z in s if z['scene']==scene and z['method']==m)
md=(P/'reference_notes_base.md').read_text();md+='## 실제 실행 결과\n\n24회 baking을 완료했다. 아래 새 방법은 **고정 S 이식 실험**이며 원 논문 이름의 모델 점수가 아니다. 학습형 PSNR은 3 seeds의 평균 ± 표본표준편차, LPIPS·정적 PSNR은 평균이다. 기존 직접 fusion은 결정적 1회다.\n\n'
md+='| 방법 | B 고점유 PSNR ↑ | C 고점유 PSNR ↑ | B / C LPIPS ↓ | B / C 정적 PSNR ↑ |\n|---|---:|---:|---:|---:|\n'
for m in ['l2','robust','uv_shadow_guard']+list(cfg['methods']):
 b=get('industrial_b',m);cc=get('industrial_c',m)
 def val(z):return f"{z['PSNR_occupied3_4']:.2f}"+(f" ± {z['PSNR_occupied3_4_std']:.2f}" if z['PSNR_occupied3_4_std'] is not None else '')
 md+=f"| {names.get(m,m)} | {val(b)} | {val(cc)} | {b['LPIPS_object_crop_mean']:.3f} / {cc['LPIPS_object_crop_mean']:.3f} | {b['PSNR_static_control_proxy']:.2f} / {cc['PSNR_static_control_proxy']:.2f} |\n"
md+='\n### 마스크·관측 지지 진단\n\n| 규칙 | B / C 채택 픽셀 비율 | B / C 평균 S | B / C 지면 ROI 무지지 비율 |\n|---|---:|---:|---:|\n'
for m in cfg['methods']:
 b=json.loads((R/'industrial_b/mask_diagnostics.json').read_text())['stats'][m];cc=json.loads((R/'industrial_c/mask_diagnostics.json').read_text())['stats'][m];bm=get('industrial_b',m);cm=get('industrial_c',m)
 md+=f"| {names[m]} | {100*b['positive_fraction']:.1f}% / {100*cc['positive_fraction']:.1f}% | {b['mean_weight']:.3f} / {cc['mean_weight']:.3f} | {100*bm['unsupported_roi']:.1f}% / {100*cm['unsupported_roi']:.1f}% |\n"
md+='''
채택 픽셀 비율在各方法的所有训练相机有效区域内计算；지면 ROI 무지지 비율은 S>10⁻⁶인 관측이 한 번도 투영되지 않은 평가 지면의 비율이다. **무지지=실제로 배경이 존재하지 않음**을 뜻하지 않는다. 마스크가 잘못 제외했을 수도 있다. Soft S에서는 양수라는 이유만으로 의미상 올바른 배경 관측이라는 보장도 없다.

### 해석

- 이번 단순 이식에서는 기존 UV + shadow guard의 고점유 PSNR을 넘지 못했다. 이 결과는 원 논문의 NeRF/3DGS 순위나 전체 모델 성능을 뜻하지 않는다.
- SSIM-soft는 두 장면 모두 σ floor 0.1에 도달해 S≈50으로 일정해졌다. 차량 선택 능력을 얻지 못한 실패이며, L2와의 작은 수치 차이는 데이터/색정규화 상대 배율도 달라졌으므로 제거 이득으로 해석하면 안 된다.
- DINO-cosine hard는 유효 픽셀의 약 10–12%만 채택하고, 지면 ROI의 약 91–95%가 최종 관측 지지를 잃었다. 일부 PSNR·LPIPS가 개선되어도 넓은 영역이 평균 초기값에 남는 현상과 분리해 보아야 한다. 실제 차량 제거 성공을 입증한 결과가 아니다.
- SAM 영역 투표도 지면 ROI를 과도하게 제외했다. 영역 경계를 보존하는 것만으로 반복 차량과 정적 배경을 식별할 수 없고, 어떤 영역을 제외할지 결정하는 증거가 중요하다.
- DINO의 입력 리사이즈·patch granularity는 작은 항공영상 차량에 불리할 가능성이 있다. 고정 warm-up과 온라인 갱신의 차이, SSIM target 스케일, 특징·마스크 해상도, S와 색 정규화의 상대 배율을 후속으로 분리 검증해야 한다. 이는 현재 진단에 근거한 가설이며 별도 실험으로 확정하지 않았다.
- 이번에 uncertainty predictor는 장면마다 seed42로 한 번 학습하고 고정했다. 3-seed 표준편차는 **baking 랜덤성만** 반영하며 마스크 학습의 불확실성까지 추정하지 않는다.
- 새 결과를 보고 가중치나 임계값을 조정하지 않았다. 다음 실험은 온라인 uncertainty 갱신과 지면 관측 지지 제약을 미리 정해 검증하는 방향이 적합하다.

## 피겨와 파일

- [B 장면 S 비교](baking_pilot/reference_study_v3/assets/industrial_b_new_masks.png) / [C 장면 S 비교](baking_pilot/reference_study_v3/assets/industrial_c_new_masks.png): 같은 2025년 20° 입력. Soft 색상은 log10 S를 [−1,1] 구간으로 표시한 공통 범위이고 표시 밖은 clip한다. 학습 가중치는 clip하지 않았다. Hard/SAM은 흰색이 채택 영역, 회색은 무효 투영이다.
- [동일 OBB 확대 비교](baking_pilot/reference_study_v3/assets/new_qualitative.png): B/C의 기존 평가 주석 첫 OBB를 그대로 선택, seed42. 좋은 결과만 골라 crop을 선택하지 않았다. 학습 입력 / 실제 2026 배경 / 기존 L2 / 새 cosine soft / 새 cosine hard / 새 SAM 투표 / 기존 UV + shadow guard 순서다.
- [새 24행 지표](baking_pilot/reference_study_v3/results/metrics.csv), [기존 결과 포함 비교](baking_pilot/reference_study_v3/results/combined_metrics.csv), [위치별 지표](baking_pilot/reference_study_v3/results/object_metrics.csv), [집계 JSON](baking_pilot/reference_study_v3/results/summary.json).
- 재현 코드: `baking_pilot/reference_study_v3/prepare_dino.py`, `run_experiments.py`, `evaluate.py`, `protocol.json`. 실행 환경은 기존 `baking_pilot/.venv`를 재사용했다.
- 실행 순서: `prepare_dino.py` → `run_experiments.py` → `evaluate.py` → `make_results.py`. 결과 보존을 위해 학습 스크립트는 완료된 비-SAM baking을 재실행하지 않는다. 새 조건으로 다시 실행하려면 별도 결과 폴더에서 진행한다.
- 논문 원문·일부 피겨 crop은 `papers/`, `assets/*_paper.png`, 출처 및 crop 좌표는 `assets/paper_figure_manifest.json`에 기록했다. 원 논문 피겨는 설명 목적으로 인용했으며 저작권은 각 저자에게 있다.
- 항공영상: © Beeldmateriaal Nederland / PDOK, CC BY 4.0. 정합·색정규화·투영·베이킹·마스크 피겨로 가공했다.
'''
md=md.replace('채택 픽셀 비율在各方法的所有训练相机有效区域内计算；','채택 픽셀 비율은 각 방법의 모든 학습 카메라의 유효 영역에서 계산했다. ')
(BASE.parent/'dynamic_object_baking_references_v3.md').write_text(md);(P/'REPORT.md').write_text(md)
# Verify independent output trace counts, finite masks, model dimensions, metrics count.
checks={'baking_runs':0,'all_600_updates':True,'all_training_masks_finite':True,'metric_rows':len(json.loads((R/'metrics.json').read_text())),'expected_runs':24,'no_training_holdout_indices':True}
for scene in cfg['scenes']:
 for seed in cfg['seeds']:
  for method in cfg['methods']:
   tr=json.loads((R/scene/str(seed)/(method+'_trace.json')).read_text());checks['baking_runs']+=1;checks['all_600_updates']&=tr['updates']==600
 for method in cfg['methods']:
  z=np.load(R/scene/(method+'_mask.npz'))['static'];checks['all_training_masks_finite']&=bool(np.isfinite(z).all() and z.shape==(20,576,768) and z.min()>=0)
 prov=json.loads((R/scene/'feature_provenance.json').read_text());checks['no_training_holdout_indices']&=max(prov['training_indices'])<20 and not prov['uses_holdout']
checks['passed']=checks['baking_runs']==24 and checks['metric_rows']==24 and all(checks[k] for k in ['all_600_updates','all_training_masks_finite','no_training_holdout_indices'])
(R/'validation.json').write_text(json.dumps(checks,indent=2));print(checks)

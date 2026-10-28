"""Regenerate report tables and copied figures from frozen project artifacts."""
from pathlib import Path
import csv,json,hashlib,shutil
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parent;P=R.parent
read=lambda p:list(csv.DictReader(p.open()))
easy=read(P/'results/metrics.csv');manual={r['method']:r for r in read(P/'results/manual_metrics.csv')}
hard=read(P/'harder_dataset/results/metrics.csv');allrows=read(P/'harder_dataset/improved_baselines/results/combined_metrics.csv')
names={'l2':'L2 (no-removal)','median':'Temporal median','block':'Block-type','hugs':'HuGS-type','robust':'Robust (base)','robust_or':'Robust (OR)','robust_car':'Robust (car)','robust_depth':'Robust (depth)','uv_fusion':'UV fusion','uv_shadow_guard':'UV + shadow guard','geo_mask':'Geometry mask','geo_fusion':'Geometry + init','geo_smooth':'Geometry + prior'}
order=['l2','median','block','hugs','robust','robust_or','robust_car','robust_depth']
def f(v,n=2):return f'{float(v):.{n}f}'
lines=[]
for m in order:
 r=next(a for a in easy if a['method']==m)
 lines.append(' & '.join([names[m],f(manual[m]['manual_core_PSNR']),f(r['PSNR_roi']),f(r['SSIM_roi'],3),f(r['LPIPS_roi'],3),f(r['holdout_static_raw_PSNR']),r['vehicle_detections_roi']])+r' \\')
(R/'easy_table.tex').write_text(r'''\begin{table}[htbp]\centering\scriptsize
\caption{Easy의 저장 결과(seed42). 차량 표본은16개 작은 원, 지면 기준은6년 consensus다. 검출 수는 같은 검출기를 사용한 보조 지표다.}
\begin{tabular}{lrrrrrr}\toprule
방법 & 차량 표본 & 지면 PSNR & 지면 SSIM & 지면 LPIPS & 2026 정적 & 검출 수\\\midrule
'''+ '\n'.join(lines)+r''' \\[-6pt]\bottomrule\end{tabular}
\end{table}
''')
def values(rows,s,m,key):return np.array([float(r[key]) for r in rows if r['scene']==s and r['method']==m])
def avg(rows,s,m,key,sd=False,n=2):
 v=values(rows,s,m,key);return f(v.mean(),n)+(r' $\pm$ '+f(v.std(ddof=1),n) if sd and len(v)>1 else '')
lines=[]
for s in ['industrial_b','industrial_c']:
 for m in ['l2','block','hugs','robust']:
  lines.append(' & '.join([s[-1].upper(),names[m]]+[avg(hard,s,m,k,True) for k in ['PSNR_occupied3_4','PSNR_recoverable_hard_pixels','PSNR_static_control_proxy']])+r' \\')
(R/'hard_table.tex').write_text(r'''\begin{table}[htbp]\centering\small
\caption{Hard의 기존 주요 기준선. 평균$\pm$표본표준편차,3seed. 각 열은 PSNR(dB). B/C의3--4회 점유 OBB는 각각4/17개다.}
\begin{tabular}{llrrr}\toprule 장면 & 방법 & 3--4회 점유 OBB & 관측 가능한 hard 픽셀 & 정적 대조\\\midrule
'''+ '\n'.join(lines)+r'''\bottomrule\end{tabular}\end{table}
''')
lines=[]
for m in order:
 vals=[]
 for s in ['industrial_b','industrial_c']:
  r=next(r for r in hard if r['scene']==s and r['method']==m and int(r['seed'])==42)
  vals.extend([f(r['PSNR_occupied3_4']),f(r['LPIPS_object_crop_mean'],3)])
 lines.append(' & '.join([names[m]]+vals)+r' \\')
(R/'robust_table.tex').write_text(r'''\begin{table}[htbp]\centering\small
\caption{Hard的 추가 변형(seed42). PSNR은3--4회 점유 OBB, LPIPS는전체 채택 OBB crop 평균이다. 주요 표의3seed 평균과 구분한다.}
\begin{tabular}{lrrrr}\toprule & \multicolumn{2}{c}{B} & \multicolumn{2}{c}{C}\\ 방법 & PSNR & LPIPS & PSNR & LPIPS\\\midrule
'''.replace('Hard的','Hard의')+'\n'.join(lines)+r'''\bottomrule\end{tabular}\end{table}
''')
lines=[]
for m in ['l2','block','hugs','robust','uv_fusion','uv_shadow_guard','geo_mask','geo_fusion','geo_smooth']:
 lines.append(' & '.join([names[m],avg(allrows,'industrial_b',m,'PSNR_occupied3_4',True),avg(allrows,'industrial_c',m,'PSNR_occupied3_4',True),avg(allrows,'industrial_b',m,'LPIPS_object_crop_mean',n=3)+' / '+avg(allrows,'industrial_c',m,'LPIPS_object_crop_mean',n=3),avg(allrows,'industrial_b',m,'PSNR_static_control_proxy')+' / '+avg(allrows,'industrial_c',m,'PSNR_static_control_proxy')])+r' \\')
(R/'improved_table.tex').write_text(r'''\begin{table}[htbp]\centering\scriptsize
\caption{동일 Hard 평가에서의 개선안. 학습형 PSNR은3seed 평균$\pm$표본표준편차, 다른 열은 평균. UV 직접 합성 두 방법은 결정적 계산1회로 표준편차가 없다.}
\begin{tabular}{lrrrr}\toprule 방법 & B 차량 PSNR & C 차량 PSNR & B / C LPIPS & B / C 정적 PSNR\\\midrule
'''+ '\n'.join(lines)+r'''\bottomrule\end{tabular}\end{table}
''')
assets={'assets/easy_crops.jpg':'results/qualitative_no_removal_crops.png','assets/hard_hero.jpg':'harder_dataset/results/harder_hero.jpg','assets/improved_crops.jpg':'harder_dataset/improved_baselines/results/qualitative_comparison.jpg','assets/improved_full.jpg':'harder_dataset/improved_baselines/results/full_scene_comparison.jpg'}
for dest,src in assets.items():
 if src.endswith('.png'):Image.open(P/src).convert('RGB').save(R/dest,quality=94)
 else:shutil.copy2(P/src,R/dest)
# Same fixed hard crops, with readable headers and scoring polygons from improved figure.
im=Image.open(R/'assets/improved_crops.jpg');im.crop((0,0,im.width*6//9,im.height)).save(R/'assets/hard_hero.jpg',quality=94)
files=['PROJECT_EASY_PILOT.md','PROJECT_HARD_BASELINES.md','PROJECT.md','pilot.py','harder_dataset/improved_baselines/train.py','harder_dataset/improved_baselines/shadow_guard.py','harder_dataset/improved_baselines/protocol.json','harder_dataset/improved_baselines/shadow_protocol.json','results/metrics.csv','results/manual_metrics.csv','harder_dataset/results/metrics.csv','harder_dataset/improved_baselines/results/combined_metrics.csv']
(R/'source_manifest.json').write_text(json.dumps({str(p):hashlib.sha256((P/p).read_bytes()).hexdigest() for p in files},indent=2))
(R/'PROJECT_INTEGRATED.tex').write_text((R/'PROJECT_INTEGRATED.tex').read_text().replace('合치는','합치는'))
print('4 tables and 4 figures generated from saved artifacts.')

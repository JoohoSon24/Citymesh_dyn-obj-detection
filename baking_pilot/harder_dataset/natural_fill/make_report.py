from pathlib import Path
import json,numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parent;H=R.parent;P=H.parent;L=P/'latex';I=H/'improved_baselines'
rows=json.loads((R/'results/combined_metrics.json').read_text());diag=json.loads((R/'results/fill_diagnostics.json').read_text());font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
methods=[('l2','No removal (L2)'),('block','Block'),('uv_fusion','UV fusion'),('uv_shadow_guard','UV + shadow guard'),('gradient_harmonize','Gradient harmonize'),('patch_transfer','Patch transfer')]
def olddir(scene):return H/'scenes'/scene/'results'
def newdir(scene):return R/'scenes'/scene/'results'
def folder(scene,m):return olddir(scene) if m in ['l2','block'] else (I/'scenes'/scene/'results' if m in ['uv_fusion','uv_shadow_guard'] else newdir(scene))
def get(scene,m,key):return np.array([r[key] for r in rows if r['scene']==scene and r['method']==m],float)
def fmt(scene,m,key,sd=False,n=2):
 a=get(scene,m,key);return f'{a.mean():.{n}f}'+(r' $\pm$ '+f'{a.std(ddof=1):.{n}f}' if sd and len(a)>1 else '')
lines=[];md='| 방법 | B 차량 PSNR | C 차량 PSNR | B/C LPIPS | B/C 정적 PSNR |\n|---|---:|---:|---:|---:|\n'
for m,label in methods:
 row=[label,fmt('industrial_b',m,'PSNR_occupied3_4',True),fmt('industrial_c',m,'PSNR_occupied3_4',True),fmt('industrial_b',m,'LPIPS_object_crop_mean',n=3)+' / '+fmt('industrial_c',m,'LPIPS_object_crop_mean',n=3),fmt('industrial_b',m,'PSNR_static_control_proxy')+' / '+fmt('industrial_c',m,'PSNR_static_control_proxy')];lines.append(' & '.join(row)+r' \\');md+='| '+' | '.join(row).replace('$','').replace('\\pm','±')+' |\n'
(L/'natural_fill_table.tex').write_text(r'''\begin{table}[htbp]\centering\scriptsize
\caption{추가 채움 방법의 동일 평가 비교. 기존 학습형은3seed 평균$\pm$표본표준편차, 직접 재구성은 결정적1회다. PSNR은dB, LPIPS는전체 채택OBB의 객체 평균이다.}
\begin{tabular}{lrrrr}\toprule 방법 & B 차량 PSNR & C 차량 PSNR & B / C LPIPS & B / C 정적 PSNR\\\midrule
'''+ '\n'.join(lines)+r'''\bottomrule\end{tabular}\end{table}''')
cols=[('source','Car observation'),('target','2026 background'),('l2','No removal (L2)'),('block','Block'),('uv_shadow_guard','UV + shadow'),('gradient_harmonize','Gradient harmonize'),('patch_transfer','Patch transfer')];pw,ph=220,258;canvas=Image.new('RGB',(pw*len(cols),ph*4),'white');draw=ImageDraw.Draw(canvas)
for j,(scene,idx) in enumerate([('industrial_b',0),('industrial_b',3),('industrial_c',0),('industrial_c',10)]):
 old=olddir(scene);source=np.load(old/'aligned.npz')['images'];a=next(a for a in json.loads((old/'verified_annotations.json').read_text())['accepted'] if a['id']==idx);cx,cy=np.mean(a['poly'],0).astype(int);box=(cx-40,cy-40,cx+40,cy+40)
 for k,(m,label) in enumerate(cols):
  im=Image.fromarray(np.uint8(np.clip(source[a['year']-2021 if m=='source' else 5],0,1)*255)) if m in ['source','target'] else Image.open(folder(scene,m)/'textures'/f'{m}.png').convert('RGB');im=im.crop(box).resize((pw,pw));d=ImageDraw.Draw(im);poly=[((x-box[0])*2.75,(y-box[1])*2.75) for x,y in a['poly']];d.line(poly+[poly[0]],fill=(255,150,0),width=1);canvas.paste(im,(k*pw,j*ph+38));draw.text((k*pw+4,j*ph+2),label,font=font,fill='black');draw.text((k*pw+4,j*ph+20),f'{scene[-1].upper()} #{idx} | occupied {a["occupancy"]}/5',font=font,fill='black')
canvas.save(R/'results/qualitative_comparison.jpg',quality=95)
full=Image.new('RGB',(360*len(methods),390*2),'white');d=ImageDraw.Draw(full)
for j,scene in enumerate(['industrial_b','industrial_c']):
 for k,(m,label) in enumerate(methods):
  im=Image.open(folder(scene,m)/'textures'/f'{m}.png').convert('RGB').resize((360,360));full.paste(im,(360*k,390*j+30));d.text((360*k+4,390*j+5),scene[-1].upper()+': '+label,font=font,fill='black')
full.save(R/'results/full_scene_comparison.jpg',quality=94)
# Same S for both additional methods, explicitly show identical red exclusion.
maskfig=Image.new('RGB',(384*3,320*2),'white');d=ImageDraw.Draw(maskfig)
for j,scene in enumerate(['industrial_b','industrial_c']):
 v=np.load(olddir(scene)/'views.npz');valid=v['valid'][1].astype(bool);src=v['images'][17]
 for k,(m,label) in enumerate([('input','Input 2025 / tilt20'),('gradient_harmonize','Gradient: red = S=0'),('patch_transfer','Patch: red = S=0')]):
  a=src.copy();a[~valid]=.85
  if m!='input':
   ex=(np.load(newdir(scene)/f'{m}_fit.npz')['static'][17]<.5)&valid;a[ex]=.45*a[ex]+.55*np.array([1,0,0])
  im=Image.fromarray(np.uint8(np.clip(a,0,1)*255)).resize((384,288));maskfig.paste(im,(k*384,j*320+32));d.text((k*384+5,j*320+6),scene[-1].upper()+': '+label,font=font,fill='black')
maskfig.save(R/'results/exclusion_masks.jpg',quality=95)
# Show the largest connected hole inside fixed accepted OBBs, selected without outputs.
holesheet=Image.new('RGB',(240*4,276*2),'white');d=ImageDraw.Draw(holesheet);selections=[]
for j,scene in enumerate(['industrial_b','industrial_c']):
 old=olddir(scene);count=np.load(I/'scenes'/scene/'results/shadow_fusion_init.npz')['count'];mask=np.zeros((1024,1024),np.uint8)
 for a in json.loads((old/'verified_annotations.json').read_text())['accepted']:cv2.fillPoly(mask,[np.int32(a['poly'])],1)
 hole=((count==0)&(mask>0)&np.load(old/'ground_roi.npy')).astype(np.uint8);nc,lab,stats,cents=cv2.connectedComponentsWithStats(hole);idx=1+stats[1:,4].argmax();cx,cy=np.rint(cents[idx]).astype(int);box=(cx-40,cy-40,cx+40,cy+40);selections.append(dict(scene=scene,center=[int(cx),int(cy)],component_pixels=int(stats[idx,4]),selection='largest zero-support component inside frozen accepted OBB, independent of output'))
 for k,(m,label) in enumerate([('target','2026 background'),('uv_shadow_guard','UV + shadow'),('gradient_harmonize','Gradient harmonize'),('patch_transfer','Patch transfer')]):
  im=Image.fromarray(np.uint8(np.clip(np.load(old/'aligned.npz')['images'][5],0,1)*255)) if m=='target' else Image.open(folder(scene,m)/'textures'/f'{m}.png').convert('RGB');im=im.crop(box).resize((240,240));holesheet.paste(im,(240*k,276*j+36));d.text((240*k+4,276*j+3),label,font=font,fill='black');d.text((240*k+4,276*j+20),f'{scene[-1].upper()} hole ({cx},{cy})',font=font,fill='black')
holesheet.save(R/'results/hole_crops.jpg',quality=95);(R/'results/hole_crop_selection.json').write_text(json.dumps(selections,indent=2))
for fn in ['qualitative_comparison.jpg','full_scene_comparison.jpg','exclusion_masks.jpg','hole_crops.jpg']:
 import shutil
 shutil.copy2(R/'results'/fn,L/'assets'/('natural_'+fn))
lines=[]
for scene in ['industrial_b','industrial_c']:
 for m in ['uv_shadow_guard','gradient_harmonize','patch_transfer']:
  a=next(a for a in diag if a['scene']==scene and a['method']==m);lines.append(' & '.join([scene[-1].upper(),dict(methods)[m],f'{a["gradient_L1_OBB"]:.4f}',f'{a["PSNR_hole_OBB"]:.2f}',f'{a["PSNR_hole_boundary_OBB"]:.2f}',str(a['hole_OBB_pixels'])])+r' \\')
(L/'natural_diagnostics_table.tex').write_text(r'''\begin{table}[htbp]\centering\scriptsize
\caption{추가 진단(seed42). Gradient L1은2026 기준과의 인접 RGB 차분 오차(낮을수록 좋음), hole은모든 학습 관측이 제외된 픽셀, 경계는hole의3픽셀 이웃이다. 작은 부분집합이며 자연스러움의 사람 평가가 아니다.}
\begin{tabular}{llrrrr}\toprule 장면 & 방법 & Gradient L1 & hole PSNR & 경계 PSNR & hole 픽셀\\\midrule
'''+ '\n'.join(lines)+r'''\bottomrule\end{tabular}\end{table}''')
(R/'PROJECT.md').write_text('# 부자연스러운 채움 개선 추가 기준선\n\n같은 S=shadow guard를 유지한 gradient-domain 조화 및 주변 패치 전사. 학습2021–2025만 사용, 평가2026. 고정33OBB 및주요21OBB 재사용.\n\n'+md+'\n![동일 위치 비교](results/qualitative_comparison.jpg)\n\n![hole 비교](results/hole_crops.jpg)\n\n![제외 마스크](results/exclusion_masks.jpg)\n\n고정 계수는protocol.json, 실제미충전/수렴/시간은scenes/*/results/*_trace.json, 전체CSV는results/combined_metrics.csv. 관측되지 않은 배경을 정확히복원한 것은 아니며 일부지표가 악화될수있다.\n')
print(md)

from pathlib import Path
import json,numpy as np
from PIL import Image,ImageDraw,ImageFont
P=Path(__file__).parent;BASE=P.parent;H=BASE/'harder_dataset';S=BASE/'reference_study_v3';A=P/'assets'
methods=['l2','median','block','hugs','robust','geo_mask','uv_fusion','uv_shadow_guard','geo_fusion','geo_smooth','dino_ssim_soft','dino_cosine_soft','dino_cosine_hard','sam_region_median']
labels={'l2':'L2','median':'Temporal median','block':'Block-type','hugs':'HuGS-type','robust':'Robust-type','geo_mask':'Geometry mask','uv_fusion':'UV fusion','uv_shadow_guard':'UV + shadow guard','geo_fusion':'Geometry + init','geo_smooth':'Geometry + prior','dino_ssim_soft':'DINO-SSIM soft','dino_cosine_soft':'DINO-cosine soft','dino_cosine_hard':'DINO-cosine hard','sam_region_median':'SAM-region vote'}
cache={};manifest={}
for scene in ['industrial_b','industrial_c']:
 old=H/'scenes'/scene/'results';new=H/'improved_baselines/scenes'/scene/'results';study=S/'results'/scene/'42';ann=json.loads((old/'verified_annotations.json').read_text())['accepted'][0];source=np.load(old/'aligned.npz')['images'];x,y=np.mean(ann['poly'],axis=0).astype(int);rect=(x-112,y-64,x+112,y+64);manifest[scene]={'annotation':ann,'crop':list(map(int,rect))};cache[scene]={}
 for m in ['input','reference']+methods:
  if m=='input':im=Image.fromarray(np.uint8(np.clip(source[ann['year']-2021],0,1)*255))
  elif m=='reference':im=Image.fromarray(np.uint8(np.clip(source[5],0,1)*255))
  else:
   paths=[old/'textures'/(m+'.png'),new/'textures'/(m+'.png'),study/'textures'/(m+'.png')];p=next(p for p in paths if p.exists());im=Image.open(p).convert('RGB')
   manifest[scene].setdefault('textures',{})[m]=str(p)
  im.crop(rect).save(A/(scene+'_'+m+'.png'));cache[scene][m]=im
# Each final overview panel shows the same B/C audit locations, seed42.
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',34)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
c=Image.new('RGB',(2400,1192),'white');d=ImageDraw.Draw(c)
for k,m in enumerate(['input','reference']+methods):
 x=k%4*600;y=k//4*298;label={'input':'Input (vehicle present)','reference':'Observed background (2026)'}.get(m,labels.get(m,m));d.text((x+6,y+5),label,font=font,fill='black')
 for j,scene in enumerate(['industrial_b','industrial_c']):
  ann=manifest[scene]['annotation'];cx,cy=np.mean(ann['poly'],axis=0).astype(int);rect=(cx-64,cy-52,cx+64,cy+52);im=cache[scene][m].crop(rect).resize((288,234),Image.Resampling.LANCZOS);c.paste(im,(x+6+j*296,y+48));d.text((x+12+j*296,y+50),'B' if j==0 else 'C',font=small,fill='white',stroke_width=2,stroke_fill='black')
c.save(A/'all_methods_qualitative.png')
# Paper-produced final renderings, labelled as such in the deck, never mixed into our metrics.
import subprocess
for name,idx,rect in [('spotless',2,(316,77,562,288)),('gs_w',10,(133,131,483,583)),('robust_gs',0,(33,151,381,265))]:
 code='import pymupdf as f; p=f.open('+repr(str(S/'papers'/(name+'.pdf')))+')['+str(idx)+']; p.get_pixmap(matrix=f.Matrix(3,3),clip=f.Rect('+repr(rect)+')).save('+repr(str(A/(name+'_paper_result.png')))+')'
 subprocess.run(['python3','-c',code],check=True)
(A/'manifest.json').write_text(json.dumps(manifest,indent=2))
(P/'methods.json').write_text(json.dumps({'methods':methods,'labels':labels},indent=2))
print('Saved aligned final-result crops, 16-panel qualitative overview, and 3 cited-paper result figures')

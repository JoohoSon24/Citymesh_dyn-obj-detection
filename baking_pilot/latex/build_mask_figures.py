"""Overlay saved S=0 masks on identical inputs; no new inference or training."""
from pathlib import Path
import numpy as np,json,hashlib
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parent;P=R.parent
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
entries=[('easy',P/'results',None),('industrial_b',P/'harder_dataset/scenes/industrial_b/results',P/'harder_dataset/improved_baselines/scenes/industrial_b/results'),('industrial_c',P/'harder_dataset/scenes/industrial_c/results',P/'harder_dataset/improved_baselines/scenes/industrial_c/results')]
meta={}
for name,old,new in entries:
 views=np.load(old/'views.npz');idx=17 # 2025, camera1 (20deg) across every method and scene
 im=views['images'][idx];valid=views['valid'][idx%4].astype(bool);h,w=valid.shape
 panels=[('Input (2025, tilt 20deg)',None,None)]
 for method,label in [('l2','No removal (L2)'),('block','Block'),('hugs','HuGS'),('robust','Robust (final S)')]:
  f=old/f'{method}_fit.npz';s=np.load(f)['static'][idx];panels.append((label,s,f))
 if new:
  for method,label in [('geo_mask','Geometry / UV fusion'),('uv_shadow_guard','UV + shadow guard')]:
   f=new/f'{method}_fit.npz';panels.append((label,np.load(f)['static'][idx],f))
 else:
  for method,label in [('robust_or','Robust OR (final S)'),('robust_car','Robust car (final S)'),('robust_depth','Robust depth (final S)')]:
   f=old/f'{method}_fit.npz';panels.append((label,np.load(f)['static'][idx],f))
 pw,ph=384,320;canvas=Image.new('RGB',(pw*4,ph*2),'white');draw=ImageDraw.Draw(canvas);masks={}
 for j,(label,s,f) in enumerate(panels):
  pic=im.copy();pic[~valid]=.85
  if s is not None:
   excluded=(s<.5)&valid;pic[excluded]=.45*pic[excluded]+.55*np.array([1,0,0]);frac=float(excluded.sum()/valid.sum());label+=f' | excluded {100*frac:.1f}%';masks[label]={'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'excluded_valid_fraction':frac}
  tile=Image.fromarray(np.uint8(np.clip(pic,0,1)*255)).resize((pw,288),Image.Resampling.LANCZOS);x=j%4*pw;y=j//4*ph;canvas.paste(tile,(x,y+32));draw.text((x+5,y+7),label,font=font,fill='black')
 if len(panels)<8:
  for k,line in enumerate(['RED = excluded observation (S=0)','GRAY = outside valid ground projection','Same input / camera / color scale','Robust: final-mask snapshot','Geometry init/prior share Geometry S']):draw.text((3*pw+10,ph+44+k*33),line,font=font,fill=(130,0,0) if k==0 else 'black')
 dest=R/'assets'/f'{name}_exclusion_masks.jpg';canvas.save(dest,quality=95);meta[name]={'view_index':idx,'year':2025,'camera_tilt':20,'red_alpha':.55,'masks':masks,'figure':str(dest)}
(R/'mask_figure_manifest.json').write_text(json.dumps(meta,indent=2));print('Saved 3 matched-input red exclusion overlay figures.')

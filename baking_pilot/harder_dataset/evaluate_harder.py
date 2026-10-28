"""Held-out observed background evaluation; no majority-vote/median GT."""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import json,csv,math,numpy as np,cv2,torch
from PIL import Image,ImageDraw,ImageFont
from skimage.metrics import structural_similarity
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent;METHODS=['l2','median','block','hugs','robust','robust_or','robust_car','robust_depth']
fonts=list(Path('/usr/share/fonts').rglob('DejaVuSans.ttf'))+list((BASE/'.venv').rglob('DejaVuSans.ttf'));font=ImageFont.truetype(str(fonts[0]),16) if fonts else ImageFont.load_default()
def load(p):return np.asarray(Image.open(p).convert('RGB')).astype(float)/255
def maskpoly(poly):
 a=np.zeros((1024,1024),np.uint8);cv2.fillPoly(a,[np.int32(poly)],1);return a.astype(bool)
def psnr(a,b,m):return float(-10*np.log10(max(np.mean((a[m]-b[m])**2),1e-12))) if m.any() else None
def saveim(p,a):Image.fromarray(np.uint8(np.clip(a,0,1)*255)).save(p)
def panel(items,path,cols,size=280):
 out=Image.new('RGB',(cols*(size+8)+8,math.ceil(len(items)/cols)*(size+35)),'white');d=ImageDraw.Draw(out)
 for j,(label,a) in enumerate(items):
  im=Image.fromarray(np.uint8(np.clip(a,0,1)*255)).resize((size,size),Image.Resampling.BILINEAR);x=8+j%cols*(size+8);y=j//cols*(size+35);out.paste(im,(x,y+30));d.text((x,y+4),label,fill='#b45309' if 'No removal' in label else '#111827',font=font)
 out.save(path)
def scene_eval(name):
 scene=ROOT/'scenes'/name;out=scene/'results';source=np.load(out/'aligned.npz')['images'];reference=source[5];roi=np.load(out/'ground_roi.npy');det=np.load(out/'texture_detections.npz')['dynamic'].astype(bool);occ=det[:5].sum(0);v=np.load(out/'views.npz');cams=json.loads((out/'cameras.json').read_text());ann=json.loads((out/'verified_annotations.json').read_text());
 masks={n:np.zeros((1024,1024),bool) for n in ['all','occupied1_2','occupied3_4','occupied5']}
 for item in ann['accepted']:
  m=maskpoly(item['poly'])&roi;masks['all']|=m;key='occupied1_2' if item['occupancy']<=2 else 'occupied3_4' if item['occupancy']<=4 else 'occupied5';masks[key]|=m
 masks['recoverable_hard_pixels']=masks['occupied3_4']&(occ>=3)&(occ<=4)
 masks['always_occluded_pixels']=masks['all']&(occ==5)
 for key,m in masks.items():saveim(out/f'eval_{key}.png',m.astype(float))
 clear=(cv2.dilate(det[5].astype(np.uint8),np.ones((9,9),np.uint8))==0)&roi
 static=(occ==0)&clear&(np.std(source[:5],axis=0).max(-1)<.08)
 saveim(out/'heldout_detector_clear_proxy.png',clear);saveim(out/'static_control_proxy.png',static)
 saveim(out/'textures/median.png',np.median(source[:5],axis=0));rows=[];perobjects=[]
 for seed in [42,43,44]:
  fitdir=out if seed==42 else scene/'seeds'/str(seed)
  for method in (METHODS if seed==42 else ['l2','block','hugs','robust']):
   texpath=fitdir/'textures'/f'{method}.png'
   if not texpath.exists():continue
   tex=load(texpath);row=dict(scene=name,seed=seed,method=method,PSNR_heldout_clear_proxy=psnr(tex,reference,clear),PSNR_static_control_proxy=psnr(tex,reference,static))
   for key,m in masks.items():row['PSNR_'+key]=psnr(tex,reference,m);row['pixels_'+key]=int(m.sum())
   _,ss=structural_similarity(tex,reference,data_range=1,channel_axis=-1,full=True);row['SSIM_audit']=float(ss[masks['all']].mean()) if masks['all'].any() else None
   ghostnum=ghostden=0
   for item in ann['accepted']:
    m=maskpoly(item['poly'])&roi;err=np.mean((tex-reference)**2,-1);ev=np.mean((tex-source[item['year']-2021])**2,-1);g=int((ev[m]<err[m]).sum());ghostnum+=g;ghostden+=int(m.sum())
    perobjects.append(dict(scene=name,seed=seed,method=method,id=item['id'],occupancy=item['occupancy'],PSNR=psnr(tex,reference,m),ghost=float(g/m.sum())))
   row['ghost_audit']=ghostnum/max(1,ghostden)
   # LPIPS on fixed 80x80 object crops; outside the footprint is replaced by reference.
   predicted_crops=[];reference_crops=[]
   for item in ann['accepted']:
    mm=maskpoly(item['poly']);x,y=np.int32(np.mean(item['poly'],axis=0));x0,x1=x-40,x+40;y0,y1=y-40,y+40;g=reference[y0:y1,x0:x1];z=tex[y0:y1,x0:x1];patch=np.where(mm[y0:y1,x0:x1,None],z,g)
    predicted_crops.append(patch);reference_crops.append(g)
   with torch.no_grad():
    aa=torch.from_numpy(np.stack(predicted_crops).transpose(0,3,1,2)).float().cuda()*2-1;bb=torch.from_numpy(np.stack(reference_crops).transpose(0,3,1,2)).float().cuda()*2-1;row['LPIPS_object_crop_mean']=float(LPIPS(aa,bb).mean())
   row['unsupported_roi']=None;row['unsupported_audit']=None
   if method!='median':
    fit=np.load(fitdir/f'{method}_fit.npz');support=np.zeros((1024,1024),np.float32)
    for i,s in enumerate(fit['static']):support+=cv2.warpPerspective(s,np.linalg.inv(np.array(cams[i%4]['homography'])),(1024,1024),flags=cv2.INTER_NEAREST)
    row['unsupported_roi']=float((support[roi]==0).mean());row['unsupported_audit']=float((support[masks['all']]==0).mean())
    saveim(fitdir/f'{method}_unsupported.png',support==0)
   rows.append(row)
  
 # Crop coordinates chosen before comparing final method quality, in scene.json.
 cfg=json.loads((scene/'scene.json').read_text());texs={k:load(out/'textures'/f'{k}.png') for k in METHODS}
 for crop,box in cfg['crops'].items():
  x0,y0,x1,y1=box;items=[('2023 training input',source[2,y0:y1,x0:x1]),('2026 held-out observation',reference[y0:y1,x0:x1])]+[(('No removal (L2)' if k=='l2' else k),texs[k][y0:y1,x0:x1]) for k in ['l2','block','hugs','robust']]
  panel(items,out/f'qual_{crop}.jpg',6,300)
 panel([('No removal (L2)',texs['l2'])]+[(k,texs[k]) for k in ['block','hugs','robust','robust_or','median']],out/'qual_full.jpg',3,420)
 # Full-vehicle annotations only, with independent held-out reference and predictions.
 items=[]
 for item in ann['accepted']:
  poly=np.array(item['poly']);center=poly.mean(0);x,y=np.int32(center);half=40;box=[max(0,x-half),max(0,y-half),min(1024,x+half),min(1024,y+half)];x0,y0,x1,y1=box;yr=item['year']-2021
  row=[(f"ID{item['id']} input {item['year']}",source[yr,y0:y1,x0:x1]),(f"2026 clear; occ{item['occupancy']}/5",reference[y0:y1,x0:x1])]+[(('No removal (L2)' if k=='l2' else k),texs[k][y0:y1,x0:x1]) for k in ['l2','block','hugs','robust']]
  items+=row
  if len(items)>=6*10:break
 panel(items,out/'qual_verified_vehicles.jpg',6,180)
 # Display training S for a crowded epoch rather than a hand-selected method outcome.
 inputs=v['images'];example=8;items=[]
 for k in ['l2','block','hugs','robust','robust_or']:
  s=np.load(out/f'{k}_fit.npz')['static'][example];im=inputs[example].copy();ex=(s<.5)&v['valid'][example%4];im[ex]=.45*im[ex]+.55*np.array([1,0,0]);# letterbox keeps camera aspect ratio
  im=cv2.copyMakeBorder(im,96,96,0,0,cv2.BORDER_CONSTANT);items.append((('No removal (L2)' if k=='l2' else k),im))
 panel(items,out/'qual_masks.jpg',5,300)
 (out/'evaluation_protocol.json').write_text(json.dumps(dict(reference='Aligned, common exposure-normalized 2026 actual image; excluded from baking initialization/loss/SfM/SAM/warmup',annotations=ann,main_metric='PSNR on verified whole-vehicle OBB footprints, stratified by detected training occupancy. No temporal median used as GT.',bias='Proposals and occupancy use same DOTA detector as Block; assistant visually verifies 2026 background and cars. Not independent human labels. 2026 illumination/scene change may affect error.',seeds=[42,43,44],repeat_scope='Masks/SfM/SAM from seed42 shared; only baking batch order changes in repeated runs.',static_control_pixels=int(static.sum()),clear_proxy_pixels=int(clear.sum())),indent=2))
 return rows,perobjects
if __name__=='__main__':
 import lpips
 LPIPS=lpips.LPIPS(net='alex').cuda().eval()
 rows=[];objects=[]
 for name in ['industrial_b','industrial_c']:
  r,o=scene_eval(name);rows+=r;objects+=o
 for name,data in [('metrics',rows),('object_metrics',objects)]:
  (ROOT/'results'/f'{name}.json').write_text(json.dumps(data,indent=2))
  with (ROOT/'results'/f'{name}.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=data[0].keys());w.writeheader();w.writerows(data)
 print('Evaluated',len(rows),'scene/seed/method runs against observed 2026 background')

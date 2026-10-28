"""Same metric calculation as the frozen hard-dataset evaluator; new methods only."""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import sys,json,csv,numpy as np,cv2,torch
from skimage.metrics import structural_similarity
ROOT=Path(__file__).resolve().parent;HARD=ROOT.parent
sys.path.insert(0,str(HARD))
from evaluate_harder import maskpoly,psnr,saveim,load
METHODS=['uv_fusion','uv_shadow_guard','geo_mask','geo_fusion','geo_smooth']
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
  for method in (METHODS if seed==42 else ['geo_mask','geo_fusion','geo_smooth']):
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
  
 return rows,perobjects

if __name__=='__main__':
 import lpips
 LPIPS=lpips.LPIPS(net='alex').cuda().eval()
 rows=[];objects=[]
 for name in ['industrial_b','industrial_c']:
  scene=ROOT/'scenes'/name;out=scene/'results';old=HARD/'scenes'/name/'results'
  for fn in ['aligned.npz','ground_roi.npy','texture_detections.npz','views.npz','cameras.json','verified_annotations.json']:
   dest=out/fn
   if not dest.exists():dest.symlink_to(old/fn)
  r,o=scene_eval(name);rows+=r;objects+=o
 for name,data in [('metrics',rows),('object_metrics',objects)]:
  (ROOT/'results'/f'{name}.json').write_text(json.dumps(data,indent=2))
  with (ROOT/'results'/f'{name}.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=data[0].keys());w.writeheader();w.writerows(data)
 old=json.loads((HARD/'results/metrics.json').read_text());combined=old+rows
 (ROOT/'results/combined_metrics.json').write_text(json.dumps(combined,indent=2))
 with (ROOT/'results/combined_metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=combined[0].keys());w.writeheader();w.writerows(combined)
 print('Evaluated',len(rows),'new rows; combined with',len(old),'frozen baseline rows')

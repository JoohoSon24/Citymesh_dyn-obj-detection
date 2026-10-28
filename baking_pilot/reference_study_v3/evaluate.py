"""Read heldout imagery only here, after fixed masks/training; reuse original evaluator."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import sys,json,csv,numpy as np,torch,cv2,time
from PIL import Image
P=Path(__file__).parent;BASE=P.parent;H=BASE/'harder_dataset';sys.path.insert(0,str(H))
from evaluate_harder import maskpoly,psnr,load
from skimage.metrics import structural_similarity
CFG=json.loads((P/'protocol.json').read_text());cv2.setNumThreads(4);torch.set_num_threads(4)
import lpips
LP=lpips.LPIPS(net='alex').cuda().eval();rows=[];objects=[]
for name in CFG['scenes']:
 old=H/'scenes'/name/'results';out=P/'results'/name;source=np.load(old/'aligned.npz')['images'];gt=source[5];roi=np.load(old/'ground_roi.npy').astype(bool);det=np.load(old/'texture_detections.npz')['dynamic'].astype(bool);occ=det[:5].sum(0);ann=json.loads((old/'verified_annotations.json').read_text());cams=json.loads((old/'cameras.json').read_text());masks={n:np.zeros((1024,1024),bool) for n in ['all','occupied1_2','occupied3_4','occupied5']}
 for o in ann['accepted']:
  mm=maskpoly(o['poly'])&roi;masks['all']|=mm;key='occupied1_2' if o['occupancy']<=2 else 'occupied3_4' if o['occupancy']<=4 else 'occupied5';masks[key]|=mm
 clear=(cv2.dilate(det[5].astype(np.uint8),np.ones((9,9),np.uint8))==0)&roi;static=(occ==0)&clear&(np.std(source[:5],axis=0).max(-1)<.08)
 supportstats={}
 for method in CFG['methods']:
  sm=np.load(out/(method+'_mask.npz'))['static'];support=np.zeros((1024,1024),np.float32)
  for i,s in enumerate(sm):support+=cv2.warpPerspective((s>1e-6).astype(np.float32),np.linalg.inv(np.array(cams[i%4]['homography'])),(1024,1024),flags=cv2.INTER_NEAREST)
  supportstats[method]={'unsupported_roi':float((support[roi]==0).mean()),'unsupported_audit':float((support[masks['all']]==0).mean())}
  Image.fromarray(np.uint8(support==0)*255).save(out/(method+'_unsupported.png'))
 for seed in CFG['seeds']:
  for method in CFG['methods']:
   tex=load(out/str(seed)/'textures'/(method+'.png'));row={'scene':name,'seed':seed,'method':method,'PSNR_occupied3_4':psnr(tex,gt,masks['occupied3_4']),'PSNR_all':psnr(tex,gt,masks['all']),'PSNR_static_control_proxy':psnr(tex,gt,static),'PSNR_heldout_clear_proxy':psnr(tex,gt,clear),**supportstats[method]};aa=[];bb=[]
   for o in ann['accepted']:
    mm=maskpoly(o['poly']);x,y=np.int32(np.mean(o['poly'],axis=0));x0,x1=x-40,x+40;y0,y1=y-40,y+40;g=gt[y0:y1,x0:x1];z=tex[y0:y1,x0:x1];pp=np.where(mm[y0:y1,x0:x1,None],z,g);aa.append(pp);bb.append(g);objects.append({'scene':name,'seed':seed,'method':method,'id':o['id'],'occupancy':o['occupancy'],'PSNR':psnr(tex,gt,mm&roi)})
   with torch.no_grad():row['LPIPS_object_crop_mean']=float(LP(torch.from_numpy(np.stack(aa).transpose(0,3,1,2)).float().cuda()*2-1,torch.from_numpy(np.stack(bb).transpose(0,3,1,2)).float().cuda()*2-1).mean())
   _,ss=structural_similarity(tex,gt,data_range=1,channel_axis=-1,full=True);row['SSIM_audit']=float(ss[masks['all']].mean());rows.append(row);print(name,seed,method,round(row['PSNR_occupied3_4'],3),flush=True)
for n,rr in [('metrics',rows),('object_metrics',objects)]:
 (P/'results'/(n+'.json')).write_text(json.dumps(rr,indent=2))
 with (P/'results'/(n+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=rr[0].keys());w.writeheader();w.writerows(rr)
# Old metrics remain untouched and are clearly sourced separately.
old=json.loads((H/'improved_baselines/results/combined_metrics.json').read_text());keys=list(rows[0].keys());combined=[]
for rr in old:combined.append({k:rr.get(k) for k in keys})
combined+=rows
(P/'results/combined_metrics.json').write_text(json.dumps(combined,indent=2))
with (P/'results/combined_metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(combined)
summary=[]
for scene in CFG['scenes']:
 for method in dict.fromkeys(x['method'] for x in combined):
  rr=[x for x in combined if x['scene']==scene and x['method']==method]
  if not rr:continue
  z={'scene':scene,'method':method,'n':len(rr)}
  for k in ['PSNR_occupied3_4','PSNR_all','PSNR_static_control_proxy','LPIPS_object_crop_mean','unsupported_roi','unsupported_audit']:
   vals=[x[k] for x in rr if x[k] is not None];z[k]=float(np.mean(vals)) if vals else None;z[k+'_std']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
  summary.append(z)
(P/'results/summary.json').write_text(json.dumps(summary,indent=2));print('Evaluated',len(rows),'runs;',len(objects),'object rows',flush=True)

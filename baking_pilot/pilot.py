"""Planar multi-epoch baking pilot. See REPORT.md for deviations and proxy metrics."""
import os, json, time, math, csv, hashlib, argparse
from pathlib import Path
os.environ['MPLBACKEND']='Agg'
import numpy as np, cv2, torch
import torch.nn.functional as F
from PIL import Image,ImageDraw
from scipy.ndimage import uniform_filter, binary_dilation
from skimage.metrics import structural_similarity
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results'; DATA=ROOT/'data'/'selected'
DEVICE='cuda'; N=1024; W,H=768,576; SEED=42
np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(6)
def log(*x): print(time.strftime('%H:%M:%S'),*x,flush=True)
def save(p,a):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(np.uint8(np.clip(a,0,1)*255)).save(p)
def tensor(a):return torch.from_numpy(np.ascontiguousarray(a)).float().to(DEVICE)
def rgb(t):return t.detach().cpu().numpy().transpose(1,2,0)
def sheet(items,p,cols=3,size=400):
 rows=math.ceil(len(items)/cols);o=Image.new('RGB',(cols*size,rows*(size+30)), 'white');d=ImageDraw.Draw(o)
 for j,(name,a) in enumerate(items):
  im=Image.fromarray(np.uint8(np.clip(a,0,1)*255)); im.thumbnail((size,size));x=j%cols*size;y=j//cols*(size+30);o.paste(im,(x,y+30));d.text((x+8,y+8),name,fill='black')
 o.save(p)
def prep():
 imgs=[np.asarray(Image.open(DATA/f'{y}_orthoHR.png').convert('RGB')).astype(np.float32)/255 for y in range(2021,2027)]
 # SIFT robust affine alignment using stationary textured surfaces. RANSAC excludes cars.
 sift=cv2.SIFT_create(nfeatures=8000);ref=np.uint8(imgs[2]*255);kr,dr=sift.detectAndCompute(cv2.cvtColor(ref,cv2.COLOR_RGB2GRAY),None)
 aligned=[];stats=[];bf=cv2.BFMatcher()
 for y,im in zip(range(2021,2027),imgs):
  k,d=sift.detectAndCompute(cv2.cvtColor(np.uint8(im*255),cv2.COLOR_RGB2GRAY),None)
  pairs=bf.knnMatch(d,dr,k=2);good=[a for a,b in pairs if a.distance<.7*b.distance]
  src=np.float32([k[a.queryIdx].pt for a in good]);dst=np.float32([kr[a.trainIdx].pt for a in good])
  A,ok=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=5000)
  if A is None:raise RuntimeError('Alignment failed')
  err=np.linalg.norm(src@A[:,:2].T+A[:,2]-dst,axis=1)
  stats.append(dict(year=y,matches=len(good),inliers=int(ok.sum()),median_error_px=float(np.median(err[ok[:,0]>0])),affine=A.tolist()))
  aligned.append(cv2.warpAffine(im,A,(1500,1500),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT))
 # Robust per-channel exposure normalization to training reference 2023.
 # Fit on ground only, trim residual outliers (cars/shadows), ridge keeps color contrast.
 cfg=json.loads((ROOT/'scene.json').read_text());fitroi=np.zeros((1500,1500),np.uint8)
 for poly in cfg['ground_polygons']:cv2.fillPoly(fitroi,[np.int32(poly)],1)
 for poly in cfg.get('exclude_polygons',[]):cv2.fillPoly(fitroi,[np.int32(poly)],0)
 yy,xx=np.where(fitroi>0);sel=np.arange(len(yy))[::max(1,len(yy)//50000)];yy,xx=yy[sel],xx[sel]
 reference=aligned[2];norm=[];photo=[]
 for im in aligned:
  corrected=np.empty_like(im);b=np.zeros((4,3))
  for ch in range(3):
   x=np.c_[im[yy,xx,ch],np.ones(len(yy))];target=reference[yy,xx,ch];coef=np.array([1.,0.]);weights=np.ones(len(yy))
   ridge=np.diag([len(yy)*.01,len(yy)*.0001]);prior=np.array([len(yy)*.01,0.])
   for it in range(12):
    coef=np.linalg.solve(x.T@(x*weights[:,None])+ridge,x.T@(target*weights)+prior)
    coef[0]=np.clip(coef[0],.65,1.8);coef[1]=np.clip(coef[1],-.25,.25)
    residual=np.abs(x@coef-target);threshold=np.quantile(residual,.6);weights=np.minimum(1.,threshold/np.maximum(residual,1e-6))**2
   corrected[...,ch]=im[...,ch]*coef[0]+coef[1];b[ch,ch]=coef[0];b[3,ch]=coef[1]
  norm.append(np.clip(corrected,0,1).astype(np.float32));photo.append(b.tolist())
 norm=np.stack([cv2.resize(a,(N,N),interpolation=cv2.INTER_AREA) for a in norm])
 np.savez_compressed(OUT/'aligned.npz',images=norm)
 (OUT/'alignment.json').write_text(json.dumps(dict(alignment=stats,photometric=photo,stable_fit_pixels=len(yy)),indent=2))
 sheet([(str(y),im) for y,im in zip(range(2021,2027),norm)],OUT/'inputs.jpg')
 # Ground-only ROI polygon supplied in 1500 px source coordinates.
 roi=np.zeros((N,N),np.uint8)
 cfg=json.loads((ROOT/'scene.json').read_text())
 for poly in cfg.get('ground_polygons',[[[20,20],[1480,20],[1480,1480],[20,1480]]]):cv2.fillPoly(roi,[np.int32(np.array(poly)*N/1500)],1)
 for poly in cfg.get('exclude_polygons',[]):cv2.fillPoly(roi,[np.int32(np.array(poly)*N/1500)],0)
 save(OUT/'ground_roi.png',roi);np.save(OUT/'ground_roi.npy',roi.astype(bool))
 # Camera looks at ground origin; ground square 120m. Each camera's focal chosen to cover square.
 cams=[];grids=[];depths=[];valid=[];views=[]
 uv=np.stack(np.meshgrid(np.arange(W),np.arange(H)),axis=-1).astype(np.float32)
 for angle,az in [(0,0),(20,0),(35,100),(45,220)]:
  th,ph=np.deg2rad([angle,az]);C=np.array([100*np.sin(th)*np.cos(ph),100*np.sin(th)*np.sin(ph),100*np.cos(th)])
  forward=-C/np.linalg.norm(C);right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right);down=np.cross(forward,right)
  rot=np.stack([right,down,forward]);f=480.;K=np.array([[f,0,(W-1)/2],[0,f,(H-1)/2],[0,0,1.]])
  # Pixel texture coords -> XY plane -> camera projection.
  plane=np.array([[120/(N-1),0,-60],[0,-120/(N-1),60],[0,0,0.]])
  extr=rot@plane;extr[:,2]-=rot@C;hom=K@extr
  q=np.concatenate([uv,np.ones((H,W,1))],-1)@np.linalg.inv(hom).T;q=q[...,:2]/q[...,2:]
  grid=q/(N-1)*2-1;val=(np.abs(grid).max(-1)<=.98)
  XY=np.stack([q[...,0]/(N-1)*120-60,60-q[...,1]/(N-1)*120,np.zeros((H,W))],-1)
  depth=(XY-C)@forward
  grids.append(grid);depths.append(depth);valid.append(val);cams.append(dict(tilt=angle,azimuth=az,f=f,C=C.tolist(),homography=hom.tolist(),depth_min=float(depth[val].min()),depth_max=float(depth[val].max())))
 for epoch in range(6):
  for c in range(4):views.append(cv2.remap(norm[epoch],np.float32((grids[c][...,0]+1)*(N-1)/2),np.float32((grids[c][...,1]+1)*(N-1)/2),cv2.INTER_LINEAR))
 np.savez_compressed(OUT/'views.npz',images=np.stack(views),grids=np.stack(grids).astype(np.float32),depths=np.stack(depths).astype(np.float32),valid=np.stack(valid))
 (OUT/'cameras.json').write_text(json.dumps(cams,indent=2));
 for i,im in enumerate(views):save(OUT/'views'/f'{2021+i//4}_{i%4}.png',im)
 sheet([(f'{2021+i//4} / {cams[i%4]["tilt"]}deg',views[i]) for i in [0,1,2,3,16,17,18,19]],OUT/'cameras.jpg',4)
 log('prepared',stats)
def detect():
 from ultralytics import YOLO
 model=YOLO(str(ROOT/'weights/yolo11s-obb.pt'))
 a=np.load(OUT/'views.npz');imgs=a['images'];masks=[];counts=[]
 for i,im in enumerate(imgs):
  pred=model.predict(np.uint8(im[...,::-1]*255),imgsz=1024,conf=.25,max_det=1000,classes=[9,10],device=0,verbose=False)[0]
  mask=np.zeros((H,W),np.uint8)
  polys=pred.obb.xyxyxyxy.cpu().numpy() if pred.obb is not None else []
  for p in polys:cv2.fillPoly(mask,[np.int32(p)],1)
  mask=cv2.dilate(mask,np.ones((5,5),np.uint8));masks.append(mask);counts.append(len(polys));log('detector view',i,len(polys))
 np.savez_compressed(OUT/'semantic_masks.npz',dynamic=np.stack(masks),counts=counts)
 # Texture detector also used as explicitly biased auxiliary metric and for occupancy diagnostics.
 sources=np.load(OUT/'aligned.npz')['images'];dets=[];texture_masks=[]
 for im in sources:
  p=model.predict(np.uint8(im[...,::-1]*255),imgsz=1536,conf=.25,max_det=1000,classes=[9,10],device=0,verbose=False)[0];polys=p.obb.xyxyxyxy.cpu().numpy();dets.append(polys.tolist());m=np.zeros((N,N),np.uint8)
  for poly in polys:cv2.fillPoly(m,[np.int32(poly)],1)
  texture_masks.append(cv2.dilate(m,np.ones((5,5),np.uint8)))
 np.savez_compressed(OUT/'texture_detections.npz',dynamic=np.stack(texture_masks));(OUT/'detections.json').write_text(json.dumps(dets))
 log('detector source counts',[len(p) for p in dets])
def resolved(tex):return F.interpolate(tex,size=(N,N),mode='bilinear',align_corners=True) if tex.shape[-1]!=N else tex
def render(tex,grids):return F.grid_sample(resolved(tex).expand(len(grids),-1,-1,-1),grids,align_corners=True,padding_mode='zeros')
def box(a,k):
 # TensorFlow SAME convention for even kernels.
 lo=(k-1)//2;hi=k-1-lo
 return F.avg_pool2d(F.pad(a,(lo,hi,lo,hi)),k,stride=1)
def variable_box(a,kmap):
 b,_,h,w=a.shape;ii=F.pad(a.cumsum(-1).cumsum(-2),(1,0,1,0));y,x=torch.meshgrid(torch.arange(h,device=a.device),torch.arange(w,device=a.device),indexing='ij')
 k=kmap.long();lo=(k-1)//2;hi=k-lo;y0=(y-lo).clamp(0,h);y1=(y+hi).clamp(0,h);x0=(x-lo).clamp(0,w);x1=(x+hi).clamp(0,w);z=torch.arange(b,device=a.device)[:,None,None]
 return ((ii[z,0,y1,x1]-ii[z,0,y0,x1]-ii[z,0,y1,x0]+ii[z,0,y0,x0])/((y1-y0)*(x1-x0)).clamp(min=1))[:,None]
def robust(error,valid,depth,mode):
 if mode=='robust_car' and len(error)>1:return torch.cat([robust(e[None],v[None],d[None],mode) for e,v,d in zip(error,valid,depth)],0)
 thresholds=torch.stack([error[i,0][valid[i,0]>0].median() for i in range(len(error))])[:,None,None,None]
 low=(error<=thresholds).float()*valid;sm=(box(low,3)>=.5).float()
 if mode=='robust_or':sm=torch.maximum(sm,low)
 if mode=='robust_depth':
  k=(4.5*480/depth).ceil().clamp(8,64).long();ctx=variable_box(sm,k); # 8px cells preserve piecewise-constant IRLS patches, with depth-adaptive context.
  tile=8
 else:
  k=16 if mode in ['robust','robust_or'] else int(torch.median((4.5*480/depth)[valid[:,0]>0]).ceil().item())
  ctx=box(sm,k);tile=8 if mode in ['robust','robust_or'] else max(4,k//2)
 # Sample context at tile centers, assign one decision per inner patch.
 ys=torch.arange(H,device=error.device)//tile*tile+tile//2;xs=torch.arange(W,device=error.device)//tile*tile+tile//2
 out=(ctx[:,:,ys.clamp(max=H-1)[:,None],xs.clamp(max=W-1)[None,:]]>=.6).float()
 if mode=='robust_or':out=torch.maximum(out,sm)
 return out*valid

def bake(method,iterations=600):
 a=np.load(OUT/'views.npz');images=tensor(a['images'][:20].transpose(0,3,1,2));grids=tensor(a['grids']);valid=tensor(a['valid'][:,None]);depth=tensor(a['depths']);source=np.load(OUT/'aligned.npz')['images']
 tex=torch.nn.Parameter(F.interpolate(tensor(source[:5].mean(0).transpose(2,0,1))[None],size=(512,512),mode='bilinear',align_corners=True));aff=torch.nn.Parameter(torch.eye(3,device=DEVICE)[None].repeat(20,1,1));bias=torch.nn.Parameter(torch.zeros(20,3,device=DEVICE))
 opt=torch.optim.Adam([{'params':[tex],'lr':.025},{'params':[aff,bias],'lr':.001}]);fixed=None
 if method=='block':fixed=tensor(1-np.load(OUT/'semantic_masks.npz')['dynamic'][:20,None])
 if method=='hugs':fixed=tensor(np.load(OUT/'hugs_masks.npz')['static'][:20,None])
 trace=[];start=time.time();rng=np.random.default_rng(SEED)
 for step in range(iterations):
  ids=rng.choice(20,4,replace=False);ii=torch.tensor(ids,device=DEVICE);c=ii%4;v=valid[c];g=grids[c];pred=render(tex,g);p=torch.einsum('bij,bjhw->bihw',aff[ii],pred)+bias[ii,:,None,None];err=(p-images[ii]).square().mean(1,keepdim=True)
  with torch.no_grad():
   if method.startswith('robust'):s=robust(err.sqrt(),v,depth[c],method)
   elif fixed is not None:s=fixed[ii]*v
   else:s=v
  # Same denominator and regularizer in every run; anchor reference view 2023/nadir.
  loss=(err*s).sum()/v.sum().clamp(min=1)+.001*((aff-torch.eye(3,device=DEVICE)).square().mean()+bias.square().mean())
  opt.zero_grad();loss.backward();aff.grad[8]=0;bias.grad[8]=0;opt.step()
  with torch.no_grad():tex.clamp_(0,1)
  if step%100==0 or step==iterations-1:
   row=dict(step=step,loss=float(loss),active=float(s.sum()/v.sum()));trace.append(row);log(method,row)
 masks=[];res=[];preds=[]
 with torch.no_grad():
  for i in range(20):
   c=i%4;p=torch.einsum('bij,bjhw->bihw',aff[i:i+1],render(tex,grids[c:c+1]))+bias[i:i+1,:,None,None];e=(p-images[i:i+1]).square().mean(1,keepdim=True).sqrt()
   s=robust(e,valid[c:c+1],depth[c:c+1],method) if method.startswith('robust') else valid[c:c+1]*(fixed[i:i+1] if fixed is not None else 1)
   masks.append(s[0,0].cpu().numpy());res.append(e[0,0].cpu().numpy());preds.append(rgb(p[0]))
 save(OUT/'textures'/f'{method}.png',rgb(resolved(tex)[0]));np.savez_compressed(OUT/f'{method}_fit.npz',static=np.stack(masks),residual=np.stack(res),affine=aff.detach().cpu().numpy(),bias=bias.detach().cpu().numpy())
 (OUT/f'{method}_trace.json').write_text(json.dumps(dict(iterations=iterations,seconds=time.time()-start,trace=trace),indent=2))
 if method=='l2':np.savez_compressed(OUT/'warmup.npz',predictions=np.stack(preds),residual=np.stack(res))
 log('finished',method,round(time.time()-start,1))

def hugs():
 from segment_anything import sam_model_registry,SamAutomaticMaskGenerator,SamPredictor
 from segment_anything.utils.amg import calculate_stability_score
 images=np.load(OUT/'views.npz')['images'][:20];valid=np.load(OUT/'views.npz')['valid'];warm=np.load(OUT/'warmup.npz')['residual'];
 # Full all-pairs SIFT and homography RANSAC, counting distinct supporting images including self.
 sift=cv2.SIFT_create(nfeatures=2500);features=[sift.detectAndCompute(cv2.cvtColor(np.uint8(im*255),cv2.COLOR_RGB2GRAY),None) for im in images];counts=[np.ones(len(k),np.int32) for k,d in features];bf=cv2.BFMatcher();matchstats=[]
 for i in range(20):
  for j in range(i+1,20):
   ki,di=features[i];kj,dj=features[j];pairs=bf.knnMatch(di,dj,k=2);good=[a for a,b in pairs if a.distance<.75*b.distance]
   if len(good)<8:continue
   p=np.float32([ki[a.queryIdx].pt for a in good]);q=np.float32([kj[a.trainIdx].pt for a in good]);_,ok=cv2.findHomography(p,q,cv2.RANSAC,2.)
   if ok is None:continue
   good=[a for a,flag in zip(good,ok.ravel()) if flag];idsi=set(a.queryIdx for a in good);idsj=set(a.trainIdx for a in good)
   counts[i][list(idsi)]+=1;counts[j][list(idsj)]+=1;matchstats.append([i,j,len(good)])
 sam=sam_model_registry['vit_b'](checkpoint=str(ROOT/'weights/sam_vit_b_01ec64.pth')).to(DEVICE);sam.eval();predictor=SamPredictor(sam)
 gen=SamAutomaticMaskGenerator(sam,points_per_side=64,points_per_batch=64,pred_iou_thresh=.8,stability_score_thresh=.9,crop_n_layers=1,crop_n_points_downscale_factor=2,min_mask_region_area=50)
 outputs=[];stats=[]
 for i,im in enumerate(images):
  cache=OUT/'sam_cache'/f'{i}.npz';cache.parent.mkdir(exist_ok=True)
  if cache.exists():
   c=np.load(cache);outputs.append(c['static']);stats.append(json.loads(str(c['stats'])));continue
  t=time.time();u8=np.uint8(im*255);am=sorted(gen.generate(u8),key=lambda x:x['area'],reverse=True)
  # Official code turns overlapping SAM masks into disjoint regions, smallest mask wins.
  labels=np.full((H,W),-1,np.int32)
  for idx,m in enumerate(am):labels[m['segmentation']]=idx
  _,cc=cv2.connectedComponents(np.uint8(labels<0));labels[labels<0]=cc[labels<0]+len(am)
  points=np.float32([k.pt for k,n in zip(features[i][0],counts[i]) if n>=5]);sfm=np.zeros((H,W),bool)
  if len(points):
   predictor.set_image(u8);pts=predictor.transform.apply_coords_torch(tensor(points),(H,W))
   with torch.no_grad():
    for start in range(0,len(pts),128):
     p=pts[start:start+128,None];ms,scores,_=predictor.predict_torch(p,torch.ones(p.shape[:2],device=DEVICE),multimask_output=True,return_logits=True)
     idx=scores.argmax(1);ms=ms[torch.arange(len(idx),device=DEVICE),idx];sc=scores[torch.arange(len(idx),device=DEVICE),idx];keep=(sc>.8)&(calculate_stability_score(ms,sam.mask_threshold,1.)>=.92)
     if keep.any():sfm|=(ms[keep]>sam.mask_threshold).any(0).cpu().numpy()
  sfm=np.logical_or(sfm,uniform_filter(sfm.astype(float),size=5,mode='constant')>=.5)
  r=warm[i];v=valid[i%4];base=(r<=r[v].mean());upper=(r<=np.quantile(r[v],.95))|base
  heuristic=(sfm|base)&upper;heuristic|=uniform_filter(heuristic.astype(float),size=7,mode='constant')>=.5
  out=np.zeros((H,W),np.uint8)
  for lab in np.unique(labels):
   region=labels==lab
   if heuristic[region].mean()>=.5:out[region]=1
  out=cv2.erode(out,np.ones((5,5),np.uint8))*v
  st=dict(view=i,sift_features=len(counts[i]),static_points=len(points),sam_masks=len(am),static_fraction=float(out[v].mean()),seconds=time.time()-t)
  np.savez_compressed(cache,static=out,labels=labels,sfm=sfm,heuristic=heuristic,stats=json.dumps(st));outputs.append(out);stats.append(st);log('HuGS',st)
 np.savez_compressed(OUT/'hugs_masks.npz',static=np.stack(outputs));(OUT/'hugs_stats.json').write_text(json.dumps(dict(views=stats,pair_matches=matchstats,threshold_count=5,threshold_normalized=5/20,residual_quantile=.95,mask_fraction=.5,sam='vit_b',points_per_side=64,crop_layers=1),indent=2))
 del predictor,gen,sam;torch.cuda.empty_cache()
def refresh_hugs():
 warm=np.load(OUT/'warmup.npz')['residual'];valid=np.load(OUT/'views.npz')['valid'];outputs=[];stats=json.loads((OUT/'hugs_stats.json').read_text())
 for i in range(20):
  p=OUT/'sam_cache'/f'{i}.npz';c=np.load(p);sfm=c['sfm'];labels=c['labels'];r=warm[i];v=valid[i%4];base=r<=r[v].mean();upper=(r<=np.quantile(r[v],.95))|base;he=(sfm|base)&upper;he|=uniform_filter(he.astype(float),size=7,mode='constant')>=.5;out=np.zeros((H,W),np.uint8)
  for lab in np.unique(labels):
   region=labels==lab
   if he[region].mean()>=.5:out[region]=1
  out=cv2.erode(out,np.ones((5,5),np.uint8))*v;st=json.loads(str(c['stats']));st['static_fraction']=float(out[v].mean());np.savez_compressed(p,static=out,labels=labels,sfm=sfm,heuristic=he,stats=json.dumps(st));outputs.append(out);stats['views'][i]=st
 np.savez_compressed(OUT/'hugs_masks.npz',static=np.stack(outputs));stats['residual_refresh']='Final 512-basis L2 warmup, identical cached SIFT/SAM geometry';(OUT/'hugs_stats.json').write_text(json.dumps(stats,indent=2));log('refreshed HuGS residual decisions')

def evaluate():
 from ultralytics import YOLO
 import lpips
 source=np.load(OUT/'aligned.npz')['images'];train=source[:5];gt=np.median(source,0);median=np.median(train,0);save(OUT/'consensus.png',gt);save(OUT/'textures'/'median.png',median)
 roi=np.load(OUT/'ground_roi.npy');
 # Proxy residual changes are not semantic GT. Threshold fixed before inspecting results.
 delta=np.sqrt(np.mean((train-gt)**2,-1));dynamic=delta>.12;dyn=dynamic.any(0)&roi
 holdstatic=(np.sqrt(np.mean((source[5]-gt)**2,-1))<.06)&roi
 save(OUT/'dynamic_proxy.png',dyn);save(OUT/'holdout_static.png',holdstatic)
 views=np.load(OUT/'views.npz');cams=json.loads((OUT/'cameras.json').read_text());v=views['valid'];imgs=views['images'];lp=lpips.LPIPS(net='alex').to(DEVICE).eval();detector=YOLO(str(ROOT/'weights/yolo11s-obb.pt'))
 methods=['l2','median','block','hugs','robust','robust_or','robust_car','robust_depth'];results=[];visual=[];maskvisual=[];resvisual=[]
 def psnr(a,b,m):
  mse=np.mean((a[m]-b[m])**2);return float(-10*np.log10(max(mse,1e-12))) if m.any() else None
 for method in methods:
  path=OUT/'textures'/f'{method}.png'
  if not path.exists():continue
  tex=np.asarray(Image.open(path)).astype(np.float32)/255;visual.append((method,tex));
  # ROI-masked LPIPS image outside ROI replaced with target to avoid roof influence; padding receptive fields remain.
  with torch.no_grad():
   masked=np.where(roi[...,None],tex,gt);lpv=float(lp(tensor(masked.transpose(2,0,1))[None]*2-1,tensor(gt.transpose(2,0,1))[None]*2-1));lpfull=float(lp(tensor(tex.transpose(2,0,1))[None]*2-1,tensor(gt.transpose(2,0,1))[None]*2-1))
  _,ssimmap=structural_similarity(tex,gt,channel_axis=-1,data_range=1,full=True)
  egt=np.mean((tex-gt)**2,-1);esrc=np.mean((tex[None]-train)**2,-1)
  ghost=float(((esrc<egt[None])&dynamic&roi).sum()/max(1,(dynamic&roi).sum()))
  pred=detector.predict(np.uint8(tex[...,::-1]*255),imgsz=1536,conf=.25,max_det=1000,classes=[9,10],device=0,verbose=False)[0]
  polys=pred.obb.xyxyxyxy.cpu().numpy();ndet=sum(bool(roi[int(np.clip(p[:,1].mean(),0,N-1)),int(np.clip(p[:,0].mean(),0,N-1))]) for p in polys)
  hs=[];hsraw=[]
  for c in range(4):
   grid=views['grids'][c];mx=np.float32((grid[...,0]+1)*(N-1)/2);my=np.float32((grid[...,1]+1)*(N-1)/2)
   im=cv2.remap(tex,mx,my,cv2.INTER_LINEAR);target=imgs[20+c];sm=cv2.remap(holdstatic.astype(np.uint8),mx,my,cv2.INTER_NEAREST).astype(bool)&v[c]
   hsraw.append(psnr(im,target,sm))
   # Held-out color affine estimated from training-defined temporally stable support, then scored on disjoint pixels.
   stable=(np.std(train,axis=0).max(-1)<.04)&roi;support=cv2.remap(stable.astype(np.uint8),mx,my,cv2.INTER_NEAREST).astype(bool)&v[c];checker=np.indices((H,W)).sum(0)%2==0
   fit=support&checker;mat=np.linalg.lstsq(np.c_[im[fit],np.ones(fit.sum())],target[fit],rcond=None)[0];corrected=np.clip(np.concatenate([im,np.ones((H,W,1))],-1)@mat,0,1);hs.append(psnr(corrected,target,sm&~checker))
  prec=rec=None;active=None
  if method!='median':
   fit=np.load(OUT/f'{method}_fit.npz');s=fit['static'];tp=fp=fn=0;active=[]
   for i in range(20):
    c=i%4;grid=views['grids'][c];mx=np.float32((grid[...,0]+1)*(N-1)/2);my=np.float32((grid[...,1]+1)*(N-1)/2)
    dm=cv2.remap(dynamic[i//4].astype(np.uint8),mx,my,cv2.INTER_NEAREST).astype(bool);rm=cv2.remap(roi.astype(np.uint8),mx,my,cv2.INTER_NEAREST).astype(bool)&v[c];exclude=s[i]<.5
    tp+=int((exclude&dm&rm).sum());fp+=int((exclude&~dm&rm).sum());fn+=int((~exclude&dm&rm).sum());active.append(float(s[i][rm].mean()))
   prec=tp/max(tp+fp,1);rec=tp/max(tp+fn,1);active=float(np.mean(active))
   i=16;im=imgs[i].copy();excluded=(s[i]<.5)&v[i%4];overlay=im.copy();overlay[excluded]=.45*im[excluded]+.55*np.array([1,0,0]);maskvisual.append((method+' red=excluded',overlay));
   heat=cv2.applyColorMap(np.uint8(np.clip(fit['residual'][i]/.3,0,1)*255),cv2.COLORMAP_INFERNO)[...,::-1]/255;resvisual.append((method,heat))
  row=dict(method=method,PSNR_full=psnr(tex,gt,np.ones((N,N),bool)),SSIM_full=float(ssimmap.mean()),LPIPS_full=lpfull,holdout_static_raw_PSNR=float(np.mean(hsraw)),PSNR_roi=psnr(tex,gt,roi),SSIM_roi=float(ssimmap[roi].mean()),LPIPS_roi=lpv,PSNR_dynamic=psnr(tex,gt,dyn),ghost_score=ghost,holdout_static_PSNR=float(np.mean(hs)),exclude_precision_proxy=prec,exclude_recall_proxy=rec,active_fraction=active,vehicle_detections_roi=int(ndet))
  results.append(row);log('metrics',row)
 (OUT/'metrics.json').write_text(json.dumps(results,indent=2));
 with open(OUT/'metrics.csv','w') as f:w=csv.DictWriter(f,fieldnames=results[0].keys());w.writeheader();w.writerows(results)
 sheet(visual+[('consensus proxy (all 6 years)',gt)],OUT/'comparison.jpg',3,480);sheet(maskvisual,OUT/'masks.jpg',3,480);sheet(resvisual,OUT/'residuals.jpg',3,480)
 # Equal, predeclared image-coordinate crops for cars, markings, shadows.
 cropcfg=json.loads((ROOT/'scene.json').read_text()).get('crops',{'cars':[300,300,600,600],'markings':[500,600,800,900],'shadows':[50,50,350,350]})
 for name,(x0,y0,x1,y1) in cropcfg.items():sheet([(m,a[y0:y1,x0:x1]) for m,a in visual]+[('consensus proxy',gt[y0:y1,x0:x1])],OUT/f'crop_{name}.jpg',3,360)
 # Per-texel absence/coverage diagnostics independent of the learned result.
 occupancy=np.load(OUT/'texture_detections.npz')['dynamic']
 occ=occupancy[:5].sum(0);save(OUT/'occupancy.png',np.stack([occ/5,1-occ/5,np.zeros_like(occ)],-1))
 diagnostics=dict(roi_pixels=int(roi.sum()),roi_fraction=float(roi.mean()),dynamic_proxy_fraction=float(dyn.sum()/roi.sum()),holdout_static_fraction=float(holdstatic.sum()/roi.sum()),all_train_years_vehicle_fraction=float(((occ==5)&roi).sum()/roi.sum()),source_vehicle_counts_roi=[])
 for year in range(6):
  ps=json.loads((OUT/'detections.json').read_text())[year];diagnostics['source_vehicle_counts_roi'].append(sum(bool(roi[int(np.clip(np.mean(p,axis=0)[1],0,N-1)),int(np.clip(np.mean(p,axis=0)[0],0,N-1))]) for p in ps))
 # Mask support, inverse-warp camera masks and union epochs; zero support retains mean initialization.
 for method in methods:
  if method=='median' or not (OUT/f'{method}_fit.npz').exists():continue
  ms=np.load(OUT/f'{method}_fit.npz')['static'];support=np.zeros((N,N),np.float32)
  for i,m in enumerate(ms):support+=cv2.warpPerspective(m,np.linalg.inv(np.array(cams[i%4]['homography'])),(N,N),flags=cv2.INTER_NEAREST)
  save(OUT/f'{method}_unsupported.png',support==0);diagnostics[method+'_unsupported_fraction']=float(((support==0)&roi).sum()/roi.sum())
 (OUT/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2));log('diagnostics',diagnostics)

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prep','detect','hugs','refresh_hugs','bake','evaluate']);p.add_argument('--method',default='l2');p.add_argument('--iterations',type=int,default=600);args=p.parse_args();OUT.mkdir(exist_ok=True)
 if args.stage=='bake':bake(args.method,args.iterations)
 else:globals()[args.stage]()
if __name__=='__main__':main()

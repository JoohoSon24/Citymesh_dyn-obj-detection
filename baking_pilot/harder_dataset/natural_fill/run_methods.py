from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import sys,json,time,warnings
import numpy as np,cv2
from scipy.sparse.linalg import LinearOperator,cg
from scipy.ndimage import distance_transform_edt
from PIL import Image
R=Path(__file__).resolve().parent;H=R.parent;I=H/'improved_baselines';cfg=json.loads((R/'protocol.json').read_text());cv2.setNumThreads(4)
def save(p,a):Image.fromarray(np.uint8(np.clip(a,0,1)*255)).save(p)
def basis(a):
 import torch,torch.nn.functional as F
 x=torch.from_numpy(a.transpose(2,0,1).copy())[None];return F.interpolate(F.interpolate(x,size=(512,512),mode='bilinear',align_corners=True),size=(1024,1024),mode='bilinear',align_corners=True)[0].numpy().transpose(1,2,0)
def harmonize(source,clear,F,count):
 w=count.astype(np.float32)/5;h,ww=w.shape;lam=2.
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning)
  gx=np.nanmedian(np.where((clear[:,:,1:]&clear[:,:,:-1])[...,None],source[:,:,1:]-source[:,:,:-1],np.nan),0)
  gy=np.nanmedian(np.where((clear[:,1:]&clear[:,:-1])[...,None],source[:,1:]-source[:,:-1],np.nan),0)
 gx=np.nan_to_num(gx);gy=np.nan_to_num(gy)
 degree=np.full((h,ww),4,np.float32);degree[[0,-1]]-=1;degree[:,[0,-1]]-=1
 def mat(x):
  a=x.reshape(h,ww);y=(w+lam*degree)*a;y[:,:-1]-=lam*a[:,1:];y[:,1:]-=lam*a[:,:-1];y[:-1]-=lam*a[1:];y[1:]-=lam*a[:-1];return y.ravel()
 op=LinearOperator((h*ww,h*ww),matvec=mat,dtype=np.float32);pre=LinearOperator(op.shape,matvec=lambda x:x/(w+lam*degree).ravel(),dtype=np.float32);out=np.zeros_like(F);info=[]
 for c in range(3):
  b=w*F[...,c];b[:,:-1]-=lam*gx[...,c];b[:,1:]+=lam*gx[...,c];b[:-1]-=lam*gy[...,c];b[1:]+=lam*gy[...,c]
  sol,code=cg(op,b.ravel(),x0=F[...,c].ravel(),M=pre,rtol=1e-5,maxiter=160);out[...,c]=sol.reshape(h,ww);info.append(int(code))
 return np.clip(out,0,1),dict(cg_info=info)
def patchfill(F,count):
 h,w=count.shape;hole=count==0;original=F.copy();filled=F.copy();unknown=hole.copy();known=~hole;ps=21;rad=10
 # Donor patches contain observed backgrounds only, never other synthesized holes.
 safe=cv2.erode(known.astype(np.uint8),np.ones((ps,ps),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
 yy,xx=np.where(safe & ((np.indices((h,w))[0]%5)==0)&((np.indices((h,w))[1]%5)==0));steps=0;remaining=[]
 while unknown.any() and steps<2500:
  boundary=unknown&(cv2.dilate((~unknown).astype(np.uint8),np.ones((3,3),np.uint8))>0)
  by,bx=np.where(boundary);inside=(by>=rad)&(by<h-rad)&(bx>=rad)&(bx<w-rad);by,bx=by[inside],bx[inside]
  if len(by)==0:break
  coverage=cv2.boxFilter((~unknown).astype(np.float32),-1,(ps,ps),normalize=False)
  j=np.argmax(coverage[by,bx]);y,x=int(by[j]),int(bx[j]);context=~unknown[y-rad:y+rad+1,x-rad:x+rad+1];target=filled[y-rad:y+rad+1,x-rad:x+rad+1]
  if context.mean()<.15:break
  cand=(abs(yy-y)<=160)&(abs(xx-x)<=160)
  cy,cx=yy[cand],xx[cand]
  if len(cy)==0:cy,cx=yy,xx
  if len(cy)==0:break
  # Fixed sparse context samples speed exact local appearance matching.
  sy,sx=np.where(context);sub=np.arange(0,len(sy),max(1,len(sy)//80));sy,sx=sy[sub],sx[sub]
  samples=original[cy[:,None]+sy[None,:]-rad,cx[:,None]+sx[None,:]-rad]
  delta=np.clip(np.median(target[sy,sx][None]-samples,axis=1),-.12,.12)
  err=np.mean((samples+delta[:,None]-target[sy,sx][None])**2,axis=(1,2));j=err.argmin();donor=original[cy[j]-rad:cy[j]+rad+1,cx[j]-rad:cx[j]+rad+1]+delta[j]
  # Copy only a compact central region to advance the front without long seams.
  dst=filled[y-4:y+5,x-4:x+5];m=unknown[y-4:y+5,x-4:x+5];dst[m]=donor[rad-4:rad+5,rad-4:rad+5][m];unknown[y-4:y+5,x-4:x+5]=False;steps+=1
  if steps%250==0:print('patch step',steps,'unfilled',int(unknown.sum()),flush=True)
 # Three-pixel interior feather merges copied textures with the baseline boundary.
 alpha=np.minimum(distance_transform_edt(hole)/3,1).astype(np.float32);out=filled*alpha[...,None]+F*(1-alpha[...,None]);out[~hole]=F[~hole]
 return np.clip(out,0,1),dict(steps=steps,unfilled_pixels=int(unknown.sum()),hole_pixels=int(hole.sum()),donor_centers=len(yy))
for scene in ['industrial_b','industrial_c']:
 old=H/'scenes'/scene/'results';base=I/'scenes'/scene/'results';out=R/'scenes'/scene/'results';(out/'textures').mkdir(parents=True,exist_ok=True)
 source=np.load(old/'aligned.npz')['images'][:5].copy();z=np.load(base/'shadow_fusion_init.npz');F=z['fused'];count=z['count'];clear=~z['dynamic'].astype(bool)
 for name,fn in [('gradient_harmonize',lambda:harmonize(source,clear,F,count)),('patch_transfer',lambda:patchfill(F,count))]:
  t=time.time();a,meta=fn();assert np.isfinite(a).all();save(out/'textures'/f'{name}.png',basis(a));np.savez_compressed(out/f'{name}_fit.npz',static=np.load(base/'uv_shadow_guard_fit.npz')['static']);np.savez_compressed(out/f'{name}_raw.npz',image=a)
  meta.update(seconds=time.time()-t,protocol=cfg,train_slice='[:5]',heldout_used=False);(out/f'{name}_trace.json').write_text(json.dumps(meta,indent=2));print(scene,name,meta,flush=True)

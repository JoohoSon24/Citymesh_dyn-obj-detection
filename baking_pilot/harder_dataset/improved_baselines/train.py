"""Training-only improvements. No loading of 2026 imagery or evaluation annotations."""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
import sys,json,time,hashlib,warnings,argparse
import numpy as np,cv2,torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
ROOT=Path(__file__).resolve().parent;HARD=ROOT.parent;BASE=HARD.parent
sys.path.insert(0,str(HARD));import run_scene as core
p=core.p;cv2.setNumThreads(4);torch.set_num_threads(6)
CFG=json.loads((ROOT/'protocol.json').read_text())
def log(*x):print(time.strftime('%H:%M:%S'),*x,flush=True)
def paths(name):
 scene=ROOT/'scenes'/name;out=scene/'results';out.mkdir(parents=True,exist_ok=True)
 return scene,out,HARD/'scenes'/name/'results'
def save(path,a):Image=__import__('PIL.Image',fromlist=['Image']);Image.fromarray(np.uint8(np.clip(a,0,1)*255)).save(path)
def prepare(name):
 scene,out,old=paths(name)
 # Only first five epochs enter these operations. No held-out array is retained.
 source=np.load(old/'aligned.npz')['images'][:5].copy();dynamic=np.load(old/'texture_detections.npz')['dynamic'][:5].astype(bool);a=np.load(old/'views.npz');grids=p.tensor(a['grids']);valid=a['valid']
 with torch.no_grad():
  mapped=[]
  for epoch in range(5):
   m=p.tensor(dynamic[epoch][None,None].astype(np.float32));projections=p.render(m,grids);mapped.append((projections[:,0]>.05).cpu().numpy())
 static=(~np.concatenate(mapped))*np.tile(valid,(5,1,1));np.savez_compressed(out/'geometry_masks.npz',static=static.astype(np.uint8))
 count=(~dynamic).sum(0).astype(np.float32)
 masked=np.where(dynamic[...,None],np.nan,source)
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',category=RuntimeWarning);fused=np.nanmedian(masked,axis=0).astype(np.float32)
 missing=count==0
 if missing.any():
  idx=distance_transform_edt(missing,return_distances=False,return_indices=True);fused[missing]=fused[idx[0][missing],idx[1][missing]]
 # Dirichlet harmonic fill: observed texels never change. This is a prior, not observed GT.
 kernel=np.array([[0,.25,0],[.25,0,.25],[0,.25,0]],np.float32)
 for _ in range(CFG['harmonic_iterations']):
  sm=cv2.filter2D(fused,-1,kernel,borderType=cv2.BORDER_REFLECT);fused[missing]=sm[missing]
 fused=np.clip(fused,0,1)
 basis=F.interpolate(p.tensor(fused.transpose(2,0,1))[None],size=(512,512),mode='bilinear',align_corners=True)
 resolved=p.rgb(p.resolved(basis)[0]);(out/'textures').mkdir(exist_ok=True);save(out/'textures/uv_fusion.png',resolved)
 np.savez_compressed(out/'fusion_init.npz',fused=fused,count=count,basis=basis.cpu().numpy())
 # A direct-fusion support map is described by these projected training source masks.
 np.savez_compressed(out/'uv_fusion_fit.npz',static=static.astype(np.uint8))
 save(out/'background_observation_count.png',np.stack([1-count/5,count/5,np.zeros_like(count)],-1));save(out/'unobserved_uv.png',missing)
 (out/'fusion_preparation.json').write_text(json.dumps(dict(train_epochs=5,source_slice='[:5]',mask_slice='[:5]',unobserved_fraction_full=float(missing.mean()),observed_fraction_full=float((count>0).mean()),source_sha256=hashlib.sha256((old/'aligned.npz').read_bytes()).hexdigest(),harmonic_iterations=CFG['harmonic_iterations'],hole_fill='training-neighbor spatial interpolation; uncertain, not measured background'),indent=2))
 log(name,'prepared UV masks and fusion',float(missing.mean()))
def bake(name,method,seed):
 scene,primary,old=paths(name);out=primary if seed==42 else scene/'seeds'/str(seed);out.mkdir(parents=True,exist_ok=True);(out/'textures').mkdir(exist_ok=True)
 torch.manual_seed(seed);np.random.seed(seed)
 a=np.load(old/'views.npz');images=p.tensor(a['images'][:20].transpose(0,3,1,2));grids=p.tensor(a['grids']);valid=p.tensor(a['valid'][:,None]);source=np.load(old/'aligned.npz')['images'][:5]
 init=np.load(primary/'fusion_init.npz');fusedbasis=p.tensor(init['basis']);confidence=p.tensor(init['count'][None,None]/5);confidence=F.interpolate(confidence,size=(512,512),mode='area');weights=1/(1+5*confidence);unknown=(confidence<.01).float()
 if method=='geo_mask':initial=F.interpolate(p.tensor(source.mean(0).transpose(2,0,1))[None],size=(512,512),mode='bilinear',align_corners=True)
 else:initial=fusedbasis.clone()
 tex=torch.nn.Parameter(initial.clone());aff=torch.nn.Parameter(torch.eye(3,device='cuda')[None].repeat(20,1,1));bias=torch.nn.Parameter(torch.zeros(20,3,device='cuda'));fixed=p.tensor(np.load(primary/'geometry_masks.npz')['static'][:,None]);opt=torch.optim.Adam([{'params':[tex],'lr':.025},{'params':[aff,bias],'lr':.001}]);rng=np.random.default_rng(seed);trace=[];start=time.time()
 for step in range(600):
  ids=rng.choice(20,4,replace=False);ii=torch.tensor(ids,device='cuda');c=ii%4;v=valid[c];prediction=torch.einsum('bij,bjhw->bihw',aff[ii],p.render(tex,grids[c]))+bias[ii,:,None,None];err=(prediction-images[ii]).square().mean(1,keepdim=True);s=fixed[ii]*v
  data=(err*s).sum()/v.sum().clamp(min=1);color=.001*((aff-torch.eye(3,device='cuda')).square().mean()+bias.square().mean());tv=tex.sum()*0;anchor=tex.sum()*0
  if method=='geo_smooth':
   dx=(tex[:,:,:,1:]-tex[:,:,:,:-1]).abs();dy=(tex[:,:,1:,:]-tex[:,:,:-1,:]).abs();wx=.5*(weights[:,:,:,1:]+weights[:,:,:,:-1]);wy=.5*(weights[:,:,1:,:]+weights[:,:,:-1,:]);tv=.01*((dx*wx).mean()+(dy*wy).mean());anchor=.05*((tex-fusedbasis).square()*unknown).mean()
  loss=data+color+tv+anchor;opt.zero_grad();loss.backward();aff.grad[8]=0;bias.grad[8]=0;opt.step()
  with torch.no_grad():tex.clamp_(0,1)
  if step%100==0 or step==599:
   row=dict(step=step,data=float(data),color=float(color),tv=float(tv),anchor=float(anchor),active=float(s.sum()/v.sum()));trace.append(row);log(name,method,seed,row)
 save(out/'textures'/f'{method}.png',p.rgb(p.resolved(tex)[0]));np.savez_compressed(out/f'{method}_fit.npz',static=np.load(primary/'geometry_masks.npz')['static'],affine=aff.detach().cpu().numpy(),bias=bias.detach().cpu().numpy())
 (out/f'{method}_trace.json').write_text(json.dumps(dict(seed=seed,iterations=600,seconds=time.time()-start,trace=trace,initialization='mean' if method=='geo_mask' else 'training-only UV fusion',protocol=CFG),indent=2));log('finished',name,method,seed,round(time.time()-start,1))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--scene',choices=['industrial_b','industrial_c','all'],default='all');args=ap.parse_args()
 for name in (['industrial_b','industrial_c'] if args.scene=='all' else [args.scene]):
  prepare(name)
  for seed in CFG['seeds']:
   for method in ['geo_mask','geo_fusion','geo_smooth']:bake(name,method,seed)

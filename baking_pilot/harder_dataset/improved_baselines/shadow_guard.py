"""One fixed training-only photometric guard ablation; does not overwrite prior variants."""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
import json,warnings,numpy as np,cv2,torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from PIL import Image
import train as t
ROOT=Path(__file__).resolve().parent;cfg=json.loads((ROOT/'shadow_protocol.json').read_text());cv2.setNumThreads(4)
for name in ['industrial_b','industrial_c']:
 scene,out,old=t.paths(name);source=np.load(old/'aligned.npz')['images'][:5].copy();original=np.load(old/'texture_detections.npz')['dynamic'][:5].astype(bool);dynamic=[];extra=[]
 for im,mask in zip(source,original):
  lum=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY);clear=(~mask).astype(np.float32);den=cv2.blur(clear,(61,61));ref=cv2.blur(lum*clear,(61,61))/np.maximum(den,.05);near=cv2.dilate(mask.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(37,37)))>0;guard=near&(lum<.72*ref);dynamic.append(mask|guard);extra.append(guard&~mask)
 dynamic=np.stack(dynamic);count=(~dynamic).sum(0).astype(np.float32)
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning);fused=np.nanmedian(np.where(dynamic[...,None],np.nan,source),axis=0).astype(np.float32)
 missing=count==0
 if missing.any():
  idx=distance_transform_edt(missing,return_distances=False,return_indices=True);fused[missing]=fused[idx[0][missing],idx[1][missing]]
 kernel=np.array([[0,.25,0],[.25,0,.25],[0,.25,0]],np.float32)
 for _ in range(256):
  smooth=cv2.filter2D(fused,-1,kernel,borderType=cv2.BORDER_REFLECT);fused[missing]=smooth[missing]
 basis=F.interpolate(t.p.tensor(fused.transpose(2,0,1))[None],size=(512,512),mode='bilinear',align_corners=True);t.save(out/'textures/uv_shadow_guard.png',t.p.rgb(t.p.resolved(basis)[0]));a=np.load(old/'views.npz');grids=t.p.tensor(a['grids']);valid=a['valid'];mapped=[]
 with torch.no_grad():
  for epoch in range(5):mapped.append((t.p.render(t.p.tensor(dynamic[epoch][None,None].astype(np.float32)),grids)[:,0]>.05).cpu().numpy())
 static=(~np.concatenate(mapped))*np.tile(valid,(5,1,1));np.savez_compressed(out/'uv_shadow_guard_fit.npz',static=static.astype(np.uint8));np.savez_compressed(out/'shadow_fusion_init.npz',fused=fused,count=count,dynamic=dynamic)
 t.save(out/'unobserved_uv_shadow_guard.png',missing);overlay=source[2].copy();e=extra[2];overlay[e]=.4*overlay[e]+.6*np.array([1,0,0]);t.save(out/'shadow_guard_overlay.png',overlay)
 (out/'shadow_preparation.json').write_text(json.dumps(dict(protocol=cfg,extra_exclusion_fraction=[float(e.mean()) for e in extra],unobserved_fraction_full=float(missing.mean()),train_epochs=5),indent=2));t.log(name,'shadow guard prepared',float(missing.mean()))

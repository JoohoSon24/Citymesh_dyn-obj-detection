from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import sys,json,csv,numpy as np,cv2,torch
R=Path(__file__).resolve().parent;H=R.parent;I=H/'improved_baselines';sys.path.insert(0,str(I))
# Load the existing evaluator by explicit file path to avoid this file's same name.
import importlib.util
spec=importlib.util.spec_from_file_location('frozen_improved_evaluator',I/'evaluate.py');e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
import lpips
e.ROOT=R;e.HARD=H;e.METHODS=['gradient_harmonize','patch_transfer'];e.LPIPS=lpips.LPIPS(net='alex').cuda().eval()
rows=[];objects=[]
for scene in ['industrial_b','industrial_c']:
 out=R/'scenes'/scene/'results';old=H/'scenes'/scene/'results'
 for fn in ['aligned.npz','ground_roi.npy','texture_detections.npz','views.npz','cameras.json','verified_annotations.json']:
  dest=out/fn
  if not dest.exists():dest.symlink_to(old/fn)
 rr,oo=e.scene_eval(scene);rows+=rr;objects+=oo
# These are additional diagnostics on the same frozen OBBs; no new label selection.
diagnostics=[]
for scene in ['industrial_b','industrial_c']:
 old=H/'scenes'/scene/'results';base=I/'scenes'/scene/'results';new=R/'scenes'/scene/'results';ref=np.load(old/'aligned.npz')['images'][5];ann=json.loads((old/'verified_annotations.json').read_text());mask=np.zeros((1024,1024),bool)
 for a in ann['accepted']:mask|=e.maskpoly(a['poly'])
 mask&=np.load(old/'ground_roi.npy');count=np.load(base/'shadow_fusion_init.npz')['count'];hole=(count==0);edge=(cv2.dilate(hole.astype(np.uint8),np.ones((7,7),np.uint8))!=cv2.erode(hole.astype(np.uint8),np.ones((7,7),np.uint8)))&mask
 for method,folder in [('l2',old),('block',old),('uv_fusion',base),('uv_shadow_guard',base),('gradient_harmonize',new),('patch_transfer',new)]:
  tex=e.load(folder/'textures'/f'{method}.png');mx=mask[:,1:]&mask[:,:-1];my=mask[1:]&mask[:-1];gx=np.abs(np.diff(tex,axis=1)-np.diff(ref,axis=1));gy=np.abs(np.diff(tex,axis=0)-np.diff(ref,axis=0));grad=float((gx[mx].sum()+gy[my].sum())/(3*(mx.sum()+my.sum())))
  diagnostics.append(dict(scene=scene,method=method,gradient_L1_OBB=grad,PSNR_hole_OBB=e.psnr(tex,ref,mask&hole),hole_OBB_pixels=int((mask&hole).sum()),PSNR_hole_boundary_OBB=e.psnr(tex,ref,edge),boundary_OBB_pixels=int(edge.sum())))
for name,data in [('metrics',rows),('object_metrics',objects),('fill_diagnostics',diagnostics)]:
 (R/'results'/f'{name}.json').write_text(json.dumps(data,indent=2))
 with (R/'results'/f'{name}.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=data[0].keys());w.writeheader();w.writerows(data)
combined=json.loads((I/'results/combined_metrics.json').read_text())+rows
(R/'results/combined_metrics.json').write_text(json.dumps(combined,indent=2))
with (R/'results/combined_metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=combined[0].keys());w.writeheader();w.writerows(combined)
print('New method rows:',len(rows),'combined rows:',len(combined))

from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import sys,json,hashlib,warnings,numpy as np,cv2,torch
from PIL import Image
ROOT=Path(__file__).resolve().parent;HARD=ROOT.parent
sys.path.insert(0,str(HARD));from evaluate_harder import psnr,load,maskpoly
checks={'scenes':{}};cfg=json.loads((ROOT/'protocol.json').read_text());assert cfg['seeds']==[42,43,44]
rows=json.loads((ROOT/'results/metrics.json').read_text());assert len(rows)==22
for scene in ['industrial_b','industrial_c']:
 out=ROOT/'scenes'/scene/'results';old=HARD/'scenes'/scene/'results';z=np.load(out/'fusion_init.npz');source=np.load(old/'aligned.npz')['images'][:5];dyn=np.load(old/'texture_detections.npz')['dynamic'][:5].astype(bool);count=(~dyn).sum(0);assert np.array_equal(count,z['count']);assert np.isfinite(z['fused']).all();assert z['basis'].shape==(1,3,512,512)
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning);expected=np.nanmedian(np.where(dyn[...,None],np.nan,source),axis=0)
 error=float(np.max(np.abs(expected[count>0]-z['fused'][count>0])));assert error<1e-6
 masks=np.load(out/'geometry_masks.npz')['static'];valid=np.load(old/'views.npz')['valid'];assert masks.shape==(20,576,768) and np.isin(masks,[0,1]).all();assert not masks[np.tile(~valid,(5,1,1))].any()
 for seed in [42,43,44]:
  f=out if seed==42 else ROOT/'scenes'/scene/'seeds'/str(seed)
  for method in ['geo_mask','geo_fusion','geo_smooth']:
   assert Image.open(f/'textures'/f'{method}.png').size==(1024,1024);trace=json.loads((f/f'{method}_trace.json').read_text());assert trace['iterations']==600 and trace['seed']==seed
   fit=np.load(f/f'{method}_fit.npz');assert np.array_equal(fit['static'],masks);assert np.isfinite(fit['affine']).all() and np.isfinite(fit['bias']).all()
 assert Image.open(out/'textures/uv_fusion.png').size==(1024,1024)
 shadow=np.load(out/'shadow_fusion_init.npz');sd=shadow['dynamic'].astype(bool);sc=(~sd).sum(0);assert np.array_equal(sc,shadow['count']);assert ((sd | dyn)==sd).all()
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning);se=np.nanmedian(np.where(sd[...,None],np.nan,source),axis=0)
 assert np.max(np.abs(se[sc>0]-shadow['fused'][sc>0]))<1e-6
 assert np.isfinite(shadow['fused']).all();assert Image.open(out/'textures/uv_shadow_guard.png').size==(1024,1024)
 ann=json.loads((old/'verified_annotations.json').read_text());mask=np.zeros((1024,1024),bool)
 for item in ann['accepted']:
  if 3<=item['occupancy']<=4:mask|=maskpoly(item['poly'])
 mask&=np.load(old/'ground_roi.npy');reference=np.load(old/'aligned.npz')['images'][5];baseline=psnr(load(old/'textures/l2.png'),reference,mask);oldrow=next(r for r in json.loads((HARD/'results/metrics.json').read_text()) if r['scene']==scene and r['seed']==42 and r['method']=='l2');assert abs(baseline-oldrow['PSNR_occupied3_4'])<1e-10
 checks['scenes'][scene]=dict(known_fusion_max_error=error,baseline_metric_recalculation_error=abs(baseline-oldrow['PSNR_occupied3_4']),annotation_sha256=hashlib.sha256((old/'verified_annotations.json').read_bytes()).hexdigest(),no_observation_fraction=float((count==0).mean()),learned_runs=9,deterministic_fusion_runs=2)
for r in rows:
 for k,v in r.items():
  if isinstance(v,(int,float)):assert np.isfinite(v),(r['scene'],r['method'],k)
checks['evaluation']='22 new metric rows; unchanged frozen evaluation footprints and metric formula; 18 matched 600-update bakes + 4 deterministic fusions'
checks['protocol_sha256']=hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest();(ROOT/'results/validation.json').write_text(json.dumps(checks,indent=2));print('Validation passed: observed fusion preserved, masks valid, matched training budget, frozen metric reproduction and finite results')

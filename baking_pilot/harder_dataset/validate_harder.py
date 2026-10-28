from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import json,hashlib,sys,numpy as np,torch,cv2
from PIL import Image
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
from gpu_matcher import validate
checks={'gpu_sift_knn':validate(),'scenes':{}}
records=json.loads((ROOT/'candidates/source_manifest.json').read_text())
for r in records:assert hashlib.sha256((ROOT/r['file']).read_bytes()).hexdigest()==r['sha256']
checks['source_checksums']='18 images match recorded source responses'
for name in ['industrial_b','industrial_c']:
 out=ROOT/'scenes'/name/'results';views=np.load(out/'views.npz');assert views['images'].shape==(24,576,768,3);assert np.isfinite(views['images']).all();cams=json.loads((out/'cameras.json').read_text());g=views['grids'];errors=[]
 for c in range(4):
  yy,xx=np.meshgrid(np.arange(576),np.arange(768),indexing='ij');q=np.stack([(g[c,...,0]+1)*1023/2,(g[c,...,1]+1)*1023/2,np.ones((576,768))],-1);px=q@np.array(cams[c]['homography']).T;px=px[...,:2]/px[...,2:];errors.append(float(np.max(np.abs(px-np.stack([xx,yy],-1)))))
 assert max(errors)<.001
 tex=np.load(out/'aligned.npz')['images'][0];tt=torch.from_numpy(tex.transpose(2,0,1).copy()).float().cuda()[None].repeat(4,1,1,1);pred=torch.nn.functional.grid_sample(tt,torch.from_numpy(g).cuda(),align_corners=True).cpu().numpy().transpose(0,2,3,1);rmse=float(np.sqrt(np.mean((pred[views['valid']]-views['images'][:4][views['valid']])**2)));assert rmse<.006
 count=0
 for fitdir in [out,ROOT/'scenes'/name/'seeds/43',ROOT/'scenes'/name/'seeds/44']:
  for f in (fitdir/'textures').glob('*.png'):
   assert Image.open(f).size==(1024,1024);count+=1
   if f.stem=='median':continue
   z=np.load(fitdir/f'{f.stem}_fit.npz');m=z['static'];assert m.shape==(20,576,768);assert np.isfinite(m).all() and np.isin(m,[0,1]).all();assert not m[np.tile(~views['valid'],(5,1,1))].any()
 assert count==16,(name,count)
 ann=json.loads((out/'verified_annotations.json').read_text());ids=[a['id'] for a in ann['accepted']];assert len(ids)==len(set(ids));assert not set(ids)&{a['id'] for a in ann['rejected']}
 checks['scenes'][name]=dict(projection_max_error_px=max(errors),renderer_rmse=rmse,final_textures=count,accepted_footprints=len(ids),registration=json.loads((out/'alignment.json').read_text())['alignment'])
rows=json.loads((ROOT/'results/metrics.json').read_text());assert len(rows)==32,len(rows)
for r in rows:
 for k,v in r.items():
  if isinstance(v,(float,int)):assert np.isfinite(v),(r['scene'],r['method'],k,v)
checks['evaluation']='32 scene/seed/method rows, finite non-null metrics; None reserved for empty strata'
(ROOT/'results/validation.json').write_text(json.dumps(checks,indent=2));print('Validation passed: sources, projection, rendering, 32 textures, masks and metric rows')

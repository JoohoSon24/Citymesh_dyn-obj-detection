from pathlib import Path
import json,numpy as np,hashlib
from PIL import Image
R=Path(__file__).resolve().parent;H=R.parent;I=H/'improved_baselines';checks={}
rows=json.loads((R/'results/metrics.json').read_text());assert len(rows)==4;combined=json.loads((R/'results/combined_metrics.json').read_text());assert len(combined)==58;old=json.loads((I/'results/combined_metrics.json').read_text());assert combined[:54]==old
for scene in ['industrial_b','industrial_c']:
 base=I/'scenes'/scene/'results';new=R/'scenes'/scene/'results';z=np.load(base/'shadow_fusion_init.npz');known=z['count']>0;F=z['fused'];static=np.load(base/'uv_shadow_guard_fit.npz')['static'];sc={}
 for m in ['gradient_harmonize','patch_transfer']:
  raw=np.load(new/f'{m}_raw.npz')['image'];assert raw.shape==(1024,1024,3) and np.isfinite(raw).all() and raw.min()>=0 and raw.max()<=1;assert Image.open(new/'textures'/f'{m}.png').size==(1024,1024);assert np.array_equal(np.load(new/f'{m}_fit.npz')['static'],static)
  t=json.loads((new/f'{m}_trace.json').read_text());assert t['heldout_used']==False
  if m=='patch_transfer':assert np.array_equal(raw[known],F[known]);sc['patch_known_pixels_unchanged']=True;sc['patch_hole_filled_fraction']=1-t['unfilled_pixels']/t['hole_pixels']
  else:assert t['cg_info']==[0,0,0];sc['cg_converged_all_channels']=True
 checks[scene]=sc
for row in rows:
 assert all(np.isfinite(v) for v in row.values() if isinstance(v,(int,float)))
checks.update(original_54_rows_preserved=True,new_metric_rows=4,identical_S_to_shadow_guard=True,protocol_sha256=hashlib.sha256((R/'protocol.json').read_bytes()).hexdigest());(R/'results/validation.json').write_text(json.dumps(checks,indent=2));print('Validation passed',checks)

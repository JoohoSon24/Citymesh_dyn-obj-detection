from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import numpy as np,json,cv2
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
fonts=list(Path('/usr/share/fonts').rglob('DejaVuSans.ttf'))+list((BASE/'.venv').rglob('DejaVuSans.ttf'));font=ImageFont.truetype(str(fonts[0]),16) if fonts else ImageFont.load_default()
def study(name,path):
 data=np.load(path/'texture_detections.npz')['dynamic'].astype(bool);roi=np.load(path/'ground_roi.npy');occ=data[:5].sum(0);ever=(occ>0)&roi
 polys=json.loads((path/'detections.json').read_text());counts=[]
 for ps in polys:
  counts.append(sum(roi[int(np.clip(np.mean(p,axis=0)[1],0,1023)),int(np.clip(np.mean(p,axis=0)[0],0,1023))] for p in ps))
 diag=dict(scene=name,source_vehicle_counts_roi=list(map(int,counts)),vehicle_area_fraction_per_train_epoch=[float((im&roi).sum()/roi.sum()) for im in data[:5]],train_occupancy_histogram_among_ever_vehicle={str(n):float(((occ==n)&ever).sum()/max(ever.sum(),1)) for n in range(1,6)},ever_vehicle_fraction_roi=float(ever.mean()/roi.mean()),all_train_occluded_fraction_roi=float(((occ==5)&roi).sum()/roi.sum()))
 (path/'occupancy_diagnostics.json').write_text(json.dumps(diag,indent=2))
 if name=='old_easy':return diag
 # Full detected vehicle footprints, not tiny centers. Use training masks for occupancy; 2026 for observed clean reference.
 src=np.load(path/'aligned.npz')['images'];candidates=[]
 for year in range(5):
  for j,poly in enumerate(polys[year]):
   poly=np.float32(poly);mask=np.zeros((1024,1024),np.uint8);cv2.fillPoly(mask,[np.int32(poly)],1);area=mask.sum()
   if area<80:continue
   mask=mask.astype(bool)
   if roi[mask].mean()<.95:continue
   # Reject a near-neighbor held-out vehicle using dilation, independent of baked outputs.
   hm=cv2.dilate(data[5].astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
   if hm[mask].mean()>.02:continue
   fractions=[float(im[mask].mean()) for im in data[:5]];n=sum(f>=.55 for f in fractions)
   if n<1:continue
   center=poly.mean(0)
   candidates.append(dict(year=2021+year,poly=poly.tolist(),center=center.tolist(),occupancy=n,fractions=fractions,area=int(area),heldout_vehicle_fraction=float(data[5][mask].mean())))
 candidates.sort(key=lambda c:(-int(3<=c['occupancy']<=4),-c['occupancy'],-c['area'],c['center'][1],c['center'][0]))
 selected=[]
 for c in candidates:
  center=np.array(c['center'])
  if any(np.linalg.norm(center-np.array(d['center']))<34 for d in selected):continue
  c['id']=len(selected);selected.append(c)
  if len(selected)>=18:break
 (path/'audit_candidates.json').write_text(json.dumps(selected,indent=2))
 # Independent input-only contact sheets, to visually reject missed hold-out cars or scene changes.
 for page in range((len(selected)+8)//9):
  subset=selected[page*9:(page+1)*9];w=6*160;rh=184;out=Image.new('RGB',(w,len(subset)*rh+28),'white');d=ImageDraw.Draw(out)
  for i,y in enumerate(range(2021,2027)):d.text((i*160+4,4),str(y),fill='black',font=font)
  for row,c in enumerate(subset):
   x,y=c['center'];box=(int(x-38),int(y-38),int(x+38),int(y+38));poly=np.array(c['poly']);
   for epoch in range(6):
    tile=Image.fromarray(np.uint8(src[epoch]*255)).crop(box).resize((156,156));draw=ImageDraw.Draw(tile);q=(poly-np.array(box[:2]))*156/76;draw.polygon([tuple(p) for p in q],outline='#ff5500',width=2)
    xx=epoch*160;yy=row*rh+28;out.paste(tile,(xx,yy));d.text((xx+3,yy+157),f"ID{c['id']} occ{c['occupancy']}/5",fill='black',font=font)
  out.save(path/f'audit_candidates_page{page}.jpg')
 overlay=np.uint8(src[5]*255).copy()
 for c in selected:cv2.polylines(overlay,[np.int32(c['poly'])],True,(255,0,0),2);cv2.putText(overlay,str(c['id']),tuple(np.int32(c['center'])),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,0,0),1)
 Image.fromarray(overlay).save(path/'audit_candidates_overview.jpg')
 return diag
if __name__=='__main__':
 ds=[study('old_easy',BASE/'results')]
 for n in ['industrial_b','industrial_c']:ds.append(study(n,ROOT/'scenes'/n/'results'))
 (ROOT/'results/dataset_diagnostics.json').write_text(json.dumps(ds,indent=2));print(json.dumps(ds,indent=2))

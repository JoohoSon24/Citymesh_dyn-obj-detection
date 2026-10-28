from pilot import *
source=np.load(OUT/'aligned.npz')['images'];mask=np.load(OUT/'manual_vehicle_cores.npy');ann=json.loads((OUT/'manual_annotations.json').read_text());pts=np.array(ann['texture_points']);views=np.load(OUT/'views.npz');target=source[1];vehicle=source[4];rows=[];items=[('2025 vehicles (audit input)',vehicle[180:580,200:600]),('2022 empty observed background',target[180:580,200:600])]
methods=['l2','median','block','hugs','robust','robust_or','robust_car','robust_depth']
for method in methods:
 tex=np.array(Image.open(OUT/'textures'/f'{method}.png')).astype(float)/255;eg=np.mean((tex-target)**2,-1);ev=np.mean((tex-vehicle)**2,-1);ghost=float((ev[mask]<eg[mask]).mean());per=[]
 for x,y in pts:
  sub=np.zeros((N,N),np.uint8);cv2.circle(sub,(round(x),round(y)),5,1,-1);m=sub>0;per.append(float(eg[m].mean())<float(ev[m].mean()))
 recalls=[]
 if method!='median':
  sm=np.load(OUT/f'{method}_fit.npz')['static']
  for c in range(4):
   grid=views['grids'][c];mx=np.float32((grid[...,0]+1)*(N-1)/2);my=np.float32((grid[...,1]+1)*(N-1)/2);cm=cv2.remap(mask.astype(np.uint8),mx,my,cv2.INTER_NEAREST).astype(bool)&views['valid'][c];recalls.append(float((sm[16+c][cm]<.5).mean()))
 rows.append(dict(method=method,manual_core_PSNR=float(-10*np.log10(eg[mask].mean())),manual_core_ghost_fraction=ghost,background_closer_objects=int(sum(per)),objects=len(pts),manual_core_exclusion_recall_2025=float(np.mean(recalls)) if recalls else None))
 items.append((method,tex[180:580,200:600]))
(OUT/'manual_metrics.json').write_text(json.dumps(rows,indent=2));
with open(OUT/'manual_metrics.csv','w') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
sheet(items,OUT/'vehicle_audit_comparison.jpg',5,440);print(json.dumps(rows,indent=2))

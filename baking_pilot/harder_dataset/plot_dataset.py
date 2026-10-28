from pathlib import Path
import os
os.environ['MPLBACKEND']='Agg';os.environ['OPENBLAS_NUM_THREADS']='4'
import numpy as np,json
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
rows=json.loads((ROOT/'results/dataset_diagnostics.json').read_text());names=['Old easy pilot','New industrial B','New industrial C'];fig,ax=plt.subplots(figsize=(10,4));x=np.arange(5)
for j,(row,label) in enumerate(zip(rows,names)):ax.bar(x+(j-1)*.25,np.array(row['vehicle_area_fraction_per_train_epoch'])*100,.24,label=label)
ax.set_xticks(x,range(2021,2026));ax.set_ylabel('Vehicle-mask area / ground ROI (%)');ax.set_ylim(0,70);ax.grid(axis='y',alpha=.2);ax.legend();ax.set_title('Source occupancy: identical detector + mask dilation; no baking results used');fig.tight_layout();fig.savefig(ROOT/'results/occupancy_comparison.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(12,3.5));colors=np.array([[240,240,240],[219,234,254],[147,197,253],[250,204,21],[249,115,22],[185,28,28]],dtype=np.uint8)
for ax,row,label in zip(axes,rows,names):
 p=BASE/'results' if row['scene']=='old_easy' else ROOT/'scenes'/row['scene']/'results';mask=np.load(p/'texture_detections.npz')['dynamic'][:5];roi=np.load(p/'ground_roi.npy');occ=mask.astype(bool).sum(0);rgb=colors[occ];rgb[~roi]=255;ax.imshow(rgb);ax.axis('off');ax.set_title(label)
fig.suptitle('Vehicle-mask coverage across 5 epochs: grey=0, blue=1-2, yellow/orange=3-4, red=5');fig.tight_layout();fig.savefig(ROOT/'results/occupancy_maps.png',dpi=180);plt.close(fig)
for name in ['industrial_b','industrial_c']:
 out=ROOT/'scenes'/name/'results';src=np.load(out/'aligned.npz')['images'];roi=np.load(out/'ground_roi.npy');im=src[5].copy();im[~roi]*=.3;Image.fromarray(np.uint8(np.clip(im,0,1)*255)).save(out/'ground_roi_overlay.png')
print('Saved occupancy comparison plots and ROI overlays')

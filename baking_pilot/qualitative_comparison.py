"""Compose saved final textures into matched qualitative panels; no rebaking."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
source=np.load(OUT/'aligned.npz')['images']
images={k:Image.open(OUT/'textures'/f'{k}.png').convert('RGB') for k in ['l2','block','hugs','robust']}
columns=[('input','2025 input / cars',Image.fromarray(np.uint8(np.clip(source[4],0,1)*255))),('reference','2022 observed background',Image.fromarray(np.uint8(np.clip(source[1],0,1)*255)))]+[(k,label,images[k]) for k,label in [('l2','No removal (L2; S=1)'),('block','Block-style S'),('hugs','HuGS-style S'),('robust','Robust-style S')]]
fonts=list(Path('/usr/share/fonts').rglob('DejaVuSans.ttf'))+list((ROOT/'.venv').rglob('DejaVuSans.ttf'))
font=ImageFont.truetype(str(fonts[0]),17) if fonts else ImageFont.load_default()
small=ImageFont.truetype(str(fonts[0]),15) if fonts else font
crops=json.loads((ROOT/'scene.json').read_text())['crops'];size=300;left=16;header=62;rowheight=size+45
canvas=Image.new('RGB',(left+len(columns)*(size+12)+12,header+3*rowheight),'white');d=ImageDraw.Draw(canvas)
for col,(key,label,im) in enumerate(columns):
 x=left+col*(size+12);d.text((x,12),label,fill='#b45309' if key=='l2' else '#0f172a',font=small)
for row,(name,box) in enumerate(crops.items()):
 y=header+row*rowheight;d.text((left,y),f'{chr(65+row)}. {name} | texture crop {box}',fill='#334155',font=font)
 for col,(key,label,im) in enumerate(columns):
  x=left+col*(size+12);tile=im.crop(tuple(box)).resize((size,size),Image.Resampling.BILINEAR);canvas.paste(tile,(x,y+30))
  if key=='l2':d.rectangle((x,y+30,x+size-1,y+30+size-1),outline='#d97706',width=3)
canvas.save(OUT/'qualitative_no_removal_crops.png')
size=380;full=Image.new('RGB',(4*(size+12)+12,size+55),'white');d=ImageDraw.Draw(full)
for col,(k,label) in enumerate([('l2','No removal (L2; S=1)'),('block','Block-style S'),('hugs','HuGS-style S'),('robust','Robust-style S')]):
 x=12+col*(size+12);d.text((x,10),label,fill='#b45309' if k=='l2' else '#0f172a',font=font);full.paste(images[k].resize((size,size),Image.Resampling.BILINEAR),(x,40))
 if k=='l2':d.rectangle((x,40,x+size-1,40+size-1),outline='#d97706',width=3)
full.save(OUT/'qualitative_no_removal_full.png')
(OUT/'qualitative_comparison.json').write_text(json.dumps(dict(columns=[dict(key=k,label=label) for k,label,im in columns],crops=crops,baseline='Saved final l2.png; no dynamic removal mask (S=1 on valid). Same baking optimizer, initialization, color model and basis as other methods.',display='Identical crop coordinates and scale per row. No per-method exposure adjustment or retouching. Orange border marks no-removal baseline.'),indent=2))
print('Created full and 3-crop qualitative comparisons, explicitly labeled no-removal L2 baseline.')

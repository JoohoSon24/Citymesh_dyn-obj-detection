from pathlib import Path
import math,urllib.request,concurrent.futures,json,time,hashlib,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
SITES={'industrial_a':(52.0758,5.0470),'industrial_b':(52.0740,5.0460),'industrial_c':(52.0773,5.04895)}
def fetch(item):
 name,y=item;lat,lon=SITES[name];d=60/111320;dl=d/math.cos(math.radians(lat));p=ROOT/'candidates'/name/f'{y}_orthoHR.png';p.parent.mkdir(exist_ok=True)
 if name=='industrial_a':
  old=BASE/'data/ikea'/p.name
  if old.exists() and not p.exists():p.symlink_to(old)
 url=f'https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0?service=WMS&version=1.3.0&request=GetMap&layers={y}_orthoHR&styles=&crs=EPSG:4326&bbox={lat-d},{lon-dl},{lat+d},{lon+dl}&width=1500&height=1500&format=image/png'
 for retry in range(3):
  try:
   if not p.exists():
    with urllib.request.urlopen(url,timeout=90) as r:body=r.read()
    Image.open(__import__('io').BytesIO(body)).verify();p.write_bytes(body)
   return dict(scene=name,year=y,lat=lat,lon=lon,url=url,file=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
  except Exception as e:
   if retry==2:raise
   time.sleep(1)
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:records=list(pool.map(fetch,[(n,y) for n in SITES for y in range(2021,2027)]))
 (ROOT/'candidates/source_manifest.json').write_text(json.dumps(records,indent=2))
 for n in SITES:
  out=Image.new('RGB',(1800,330),'white');d=ImageDraw.Draw(out)
  for i,y in enumerate(range(2021,2027)):
   im=Image.open(ROOT/'candidates'/n/f'{y}_orthoHR.png').convert('RGB').resize((300,300));out.paste(im,(i*300,30));d.text((i*300+10,8),f'{n} {y}',fill='black')
  out.save(ROOT/'results'/f'{n}_years.jpg')
 print('Downloaded and validated',len(records),'candidate year images',flush=True)

import urllib.request, math, concurrent.futures, pathlib, time
root=pathlib.Path(__file__).parent
lat,lon,half=52.08718,5.10243,60
d=half/111320; dl=d/math.cos(math.radians(lat))
def get(y):
 layer=f'{y}_orthoHR' if isinstance(y,int) else y
 url=f'https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0?service=WMS&version=1.3.0&request=GetMap&layers={layer}&styles=&crs=EPSG:4326&bbox={lat-d},{lon-dl},{lat+d},{lon+dl}&width=1500&height=1500&format=image/png'
 try:
  p=root/'data'/f'{layer}.png'
  if not p.exists():
   with urllib.request.urlopen(url,timeout=120) as r: p.write_bytes(r.read())
  print(layer,p.stat().st_size,flush=True)
 except Exception as e: print(layer,str(e),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=7) as ex:list(ex.map(get,[2021,2022,2023,2024,2025,2026,'2026_quickorthoHR']))

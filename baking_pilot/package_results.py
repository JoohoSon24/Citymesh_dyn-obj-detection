from pathlib import Path
import zipfile,json,hashlib
root=Path(__file__).resolve().parent
files=[]
for p in root.iterdir():
 if p.is_file() and p.suffix in ['.md','.html','.py','.json','.lock','.sh','.txt']:files.append(p)
for p in (root/'data'/'selected').glob('*.png'):files.append(p)
for p in (root/'results').iterdir():
 if p.is_file() and p.suffix in ['.jpg','.png','.pdf','.json','.csv','.log']:files.append(p)
for d in ['textures','renders']:
 files.extend(p for p in (root/'results'/d).glob('*') if p.is_file())
manifest=[dict(file=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(set(files))]
(root/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(root/'baking_pilot_results.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(set(files+[root/'DELIVERY_MANIFEST.json'])):z.write(p,'baking_pilot/'+str(p.relative_to(root)))
print('Packaged',len(files),'files;',round((root/'baking_pilot_results.zip').stat().st_size/1e6,1),'MB')
# Verify complete archive CRCs.
with zipfile.ZipFile(root/'baking_pilot_results.zip') as z:assert z.testzip() is None

"""Restore SHA-256-verified large snapshot files without replacing existing files."""
from pathlib import Path
import argparse, hashlib, json, os, tarfile, urllib.request
ROOT=Path(__file__).resolve().parent.parent
p=argparse.ArgumentParser();p.add_argument('--verify-only',action='store_true');args=p.parse_args()
meta=json.loads((ROOT/'RELEASE_ASSETS.json').read_text())
cache=ROOT/'.release-downloads';cache.mkdir(exist_ok=True)
for item in meta['assets']:
 dest=cache/item['name'];url=f"https://github.com/{meta['repository']}/releases/download/{meta['tag']}/{item['name']}"
 def digest(path):
  h=hashlib.sha256()
  with path.open('rb') as f:
   for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
  return h.hexdigest()
 if not dest.exists() or digest(dest)!=item['sha256']:
  print('Downloading',item['name'],flush=True);urllib.request.urlretrieve(url,dest)
 if digest(dest)!=item['sha256']:raise RuntimeError('Checksum mismatch: '+item['name'])
 if not args.verify_only:
  with tarfile.open(dest,'r:gz') as archive:
   for member in archive:
    target=ROOT/member.name
    try:target.resolve().relative_to(ROOT)
    except ValueError:raise RuntimeError('Unsafe archive path: '+member.name)
    if not member.isfile():raise RuntimeError('Unexpected archive entry: '+member.name)
    if target.exists() or target.is_symlink():continue
    target.parent.mkdir(parents=True,exist_ok=True)
    source=archive.extractfile(member)
    with target.open('xb') as out:
     for block in iter(lambda:source.read(4*1024**2),b''):out.write(block)
 print('Verified',item['name'],flush=True)
print('Release assets verified' if args.verify_only else 'Release data restored')

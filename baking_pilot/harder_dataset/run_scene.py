"""Reuse the original baking implementation, changing scene data only."""
from pathlib import Path
import os
os.environ["OPENBLAS_NUM_THREADS"]="4"
os.environ["OMP_NUM_THREADS"]="4"
import importlib.util,sys,json,numpy as np,torch,shutil,argparse
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
spec=importlib.util.spec_from_file_location('baking_core',BASE/'pilot.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p);p.cv2.setNumThreads(4)
from gpu_matcher import CudaSIFTMatcher,validate
p.cv2.BFMatcher=CudaSIFTMatcher
# Alignment is common to all methods; local flow uses the known WMS coordinates.
from registration import align_images
import inspect
prep_source=inspect.getsource(p.prep)
start=prep_source.index(' # SIFT robust affine alignment')
end=prep_source.index(' # Robust per-channel exposure normalization')
prep_source=prep_source[:start]+' aligned,stats=align_images(imgs)\n'+prep_source[end:]
p.__dict__['align_images']=align_images
exec(prep_source,p.__dict__)
def configure(name):
 scene=ROOT/'scenes'/name;p.ROOT=scene;p.OUT=scene/'results';p.DATA=scene/'data';p.OUT.mkdir(parents=True,exist_ok=True);return scene
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('scene');a.add_argument('stage',choices=['prep','detect','l2','hugs','bake','repeat']);args=a.parse_args();scene=configure(args.scene)
 if args.stage=='prep':p.prep()
 elif args.stage=='detect':p.detect()
 elif args.stage=='l2':p.bake('l2')
 elif args.stage=='hugs':p.hugs()
 elif args.stage=='bake':
  for method in ['block','hugs','robust','robust_or','robust_car','robust_depth']:p.bake(method)
 else:
  primary=p.OUT
  for seed in [43,44]:
   p.OUT=scene/'seeds'/str(seed);p.OUT.mkdir(parents=True,exist_ok=True)
   for name in ['aligned.npz','views.npz','semantic_masks.npz','hugs_masks.npz','cameras.json','ground_roi.npy']:
    dest=p.OUT/name
    if not dest.exists():dest.symlink_to(primary/name)
   p.SEED=seed;np.random.seed(seed);torch.manual_seed(seed)
   for method in ['l2','block','hugs','robust']:p.bake(method)

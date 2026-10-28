"""Exhaustive float32 Euclidean SIFT KNN on CUDA (same distance as BFMatcher)."""
import cv2,numpy as np,torch
class CudaSIFTMatcher:
 def knnMatch(self,query,train,k=2):
  with torch.no_grad():
   a=torch.as_tensor(np.ascontiguousarray(query),device='cuda',dtype=torch.float32);b=torch.as_tensor(np.ascontiguousarray(train),device='cuda',dtype=torch.float32)
   dist=torch.cdist(a,b,p=2,compute_mode='use_mm_for_euclid_dist');values,indices=dist.topk(min(k,len(train)),dim=1,largest=False,sorted=True);vv=values.cpu().numpy();ii=indices.cpu().numpy()
  return [[cv2.DMatch(j,int(t),float(d)) for t,d in zip(inds,ds)] for j,(inds,ds) in enumerate(zip(ii,vv))]
def validate():
 rng=np.random.default_rng(42);q=rng.random((32,128),dtype=np.float32)*255;t=rng.random((256,128),dtype=np.float32)*255
 expected=np.linalg.norm(q[:,None]-t[None],axis=-1);got=CudaSIFTMatcher().knnMatch(q,t,2);err=0.
 for i,ms in enumerate(got):
  ref=np.argsort(expected[i])[:2];assert np.array_equal(ref,[m.trainIdx for m in ms]);err=max(err,max(abs(m.distance-float(expected[i,m.trainIdx])) for m in ms))
 return dict(validation='Exhaustive CUDA KNN compared to direct NumPy Euclidean distances, 32x256 SIFT-like vectors, top2 indices identical',max_distance_error=err)
_original_estimate=cv2.estimateAffinePartial2D
def georeferenced_alignment(src,dst,**kwargs):
 # All epochs requested the identical WMS bbox; true registration is already close.
 # Prevent repeated-car/parking-row matches from collapsing the affine scale to zero.
 spatial=np.linalg.norm(src-dst,axis=1)<20
 if spatial.sum()<8:raise RuntimeError(f'Insufficient georeferenced matches: {spatial.sum()}')
 A,local=_original_estimate(src[spatial],dst[spatial],**kwargs)
 if A is None:raise RuntimeError('RANSAC failed after georeferenced spatial gating')
 scale=float(np.sqrt(abs(np.linalg.det(A[:,:2]))));translation=float(np.linalg.norm(A[:,2]))
 if not (.98<scale<1.02 and translation<20):raise RuntimeError(f'Implausible georeferenced alignment: scale {scale}, translation {translation}, A {A}')
 full=np.zeros((len(src),1),np.uint8);full[spatial]=local
 print('alignment accepted',int(spatial.sum()),'spatial matches',int(local.sum()),'inliers','scale',scale,'translation',translation,flush=True)
 return A,full

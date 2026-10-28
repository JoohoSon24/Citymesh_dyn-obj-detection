"""Register already georeferenced epochs using local optical flow and RANSAC.
Dense repeated cars defeat unconstrained global SIFT matches. LK starts at the
same WMS image coordinates; forward-backward validation rejects moving content.
"""
import cv2,numpy as np

def align_images(images):
 cv2.setNumThreads(4)
 def gray(im):
  g=cv2.cvtColor(np.uint8(im*255),cv2.COLOR_RGB2GRAY);return cv2.createCLAHE(2.,(16,16)).apply(g)
 ref=gray(images[2]);p0=cv2.goodFeaturesToTrack(ref,maxCorners=3000,qualityLevel=.015,minDistance=12,blockSize=7)
 result=[];stats=[]
 for y,im in zip(range(2021,2027),images):
  cur=gray(im)
  if y==2023:A=np.array([[1.,0.,0.],[0.,1.,0.]]);info=dict(year=y,method='identity reference',inliers=len(p0),median_error_px=0.,affine=A.tolist())
  else:
   q,st,e=cv2.calcOpticalFlowPyrLK(ref,cur,p0,None,winSize=(31,31),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01))
   back,st2,e2=cv2.calcOpticalFlowPyrLK(cur,ref,q,None,winSize=(31,31),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01))
   pp=p0[:,0];qq=q[:,0];fb=np.linalg.norm(back[:,0]-pp,axis=1);disp=np.linalg.norm(qq-pp,axis=1);keep=(st[:,0]>0)&(st2[:,0]>0)&(fb<.7)&(disp<12)
   if keep.sum()<20:raise RuntimeError(f'{y}: too few georeferenced optical-flow tracks {keep.sum()}')
   A,ok=cv2.estimateAffinePartial2D(qq[keep],pp[keep],method=cv2.RANSAC,ransacReprojThreshold=1.5,maxIters=5000)
   scale=np.sqrt(abs(np.linalg.det(A[:,:2])))
   if not (.98<scale<1.02 and np.linalg.norm(A[:,2])<15):raise RuntimeError(f'implausible registration {y}: {A}')
   err=np.linalg.norm(qq[keep]@A[:,:2].T+A[:,2]-pp[keep],axis=1);inlier=ok[:,0]>0
   info=dict(year=y,method='WMS-initialized CLAHE LK forward-backward + RANSAC similarity',tracks=int(keep.sum()),inliers=int(inlier.sum()),median_error_px=float(np.median(err[inlier])),affine=A.tolist(),scale=float(scale),median_forward_back_error_px=float(np.median(fb[keep])))
  result.append(cv2.warpAffine(im,A,(1500,1500),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT));stats.append(info);print('registration',info,flush=True)
 return result,stats

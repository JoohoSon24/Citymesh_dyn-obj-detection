import os
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import torch,torch.nn.functional as F,numpy as np,time,json,hashlib
P=Path(__file__).parent;BASE=P.parent;torch.set_num_threads(4)
print('Loading official DINOv2 ViT-S/14',flush=True)
m=torch.hub.load('facebookresearch/dinov2','dinov2_vits14',pretrained=True,trust_repo=True).cuda().eval()
mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
for name in ['industrial_b','industrial_c']:
 out=P/'results'/name;out.mkdir(exist_ok=True,parents=True);src=BASE/'harder_dataset/scenes'/name/'results';v=np.load(src/'views.npz');images=v['images'][:20];warm=np.load(src/'warmup.npz')['predictions'];print(name,'warmup shape',warm.shape,flush=True)
 if warm.shape[1]==3:warm=warm.transpose(0,2,3,1)
 fs=[];fp=[]
 for i in range(20):
  inp=np.stack([images[i],warm[i]])
  x=torch.from_numpy(inp.transpose(0,3,1,2)).float().cuda();x=F.interpolate(x,size=(392,518),mode='bilinear',align_corners=False);x=(x-mean)/std
  with torch.no_grad():features=m.forward_features(x)['x_norm_patchtokens'].reshape(2,28,37,384).permute(0,3,1,2).cpu().numpy()
  fs.append(features[0]);fp.append(features[1])
  if i%5==0:print(name,i,flush=True)
 f=np.stack(fs);pred=np.stack(fp);cos=(f*pred).sum(1)/np.maximum(np.linalg.norm(f,axis=1)*np.linalg.norm(pred,axis=1),1e-8);err=np.minimum(1,2-2*cos).clip(0,1)
 np.savez_compressed(out/'dino_features.npz',features=f.astype(np.float16),warmup_features=pred.astype(np.float16),feature_error=err.astype(np.float32))
 (out/'feature_provenance.json').write_text(json.dumps({'model':'official facebookresearch/dinov2 dinov2_vits14','input_shape':[392,518],'feature_shape':list(f.shape),'training_indices':list(range(20)),'uses_holdout':False,'prediction':'saved L2 warmup, training images only','source_sha256':hashlib.sha256((src/'views.npz').read_bytes()).hexdigest()},indent=2))
 print('DONE',name,flush=True)

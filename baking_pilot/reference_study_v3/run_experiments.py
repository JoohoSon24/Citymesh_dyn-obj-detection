"""Training-only frozen-S mechanism transfers; protocol.json predates evaluation."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F,cv2,json,time,hashlib
from PIL import Image
P=Path(__file__).parent;BASE=P.parent;CFG=json.loads((P/'protocol.json').read_text());torch.set_num_threads(4);cv2.setNumThreads(4)
D='cuda'
def log(*a):print(time.strftime('%H:%M:%S'),*a,flush=True)
def t(a):return torch.as_tensor(a,dtype=torch.float32,device=D)
def saveim(p,a):Image.fromarray(np.uint8(np.clip(a,0,1)*255)).save(p)
def fit_uncert(features,target,nonlinear,valid):
 torch.manual_seed(42);rng=torch.Generator(device=D).manual_seed(42)
 model=torch.nn.Sequential(torch.nn.Linear(384,64),torch.nn.ReLU(),torch.nn.Linear(64,1)) if nonlinear else torch.nn.Linear(384,1)
 model=model.to(D);opt=torch.optim.Adam(model.parameters(),lr=.001)
 x=features.permute(0,2,3,1).reshape(-1,384);y=target.reshape(-1);v=valid.reshape(-1)>.95;x=x[v];y=y[v];trace=[]
 for step in range(1000):
  ids=torch.randint(len(x),(4096,),device=D,generator=rng);sigma=F.softplus(model(x[ids])[:,0]).clamp(min=.1);loss=(y[ids]/(2*sigma.square())+.5*sigma.log()).mean()
  opt.zero_grad();loss.backward();opt.step()
  if step in [0,499,999]:trace.append({'step':step,'loss':float(loss)})
 with torch.no_grad():sigma=F.softplus(model(features.permute(0,2,3,1))).permute(0,3,1,2).clamp(min=.1)
 return sigma,trace,model.state_dict()
def prep(name):
 old=BASE/'harder_dataset/scenes'/name/'results';out=P/'results'/name;out.mkdir(parents=True,exist_ok=True)
 a=np.load(old/'views.npz');images=t(a['images'][:20].transpose(0,3,1,2));valid=t(np.tile(a['valid'],(5,1,1))[:,None]);d=np.load(out/'dino_features.npz');features=t(d['features']);warm=t(np.load(old/'warmup.npz')['predictions'].transpose(0,3,1,2));patchvalid=F.interpolate(valid,size=(28,37),mode='area')
 def pool(z):return F.avg_pool2d(z,11,1,5)
 with torch.no_grad():
  ux=pool(images);uy=pool(warm);vx=(pool(images.square())-ux.square()).clamp(min=0);vy=(pool(warm.square())-uy.square()).clamp(min=0);cov=pool(images*warm)-ux*uy;sd=(vx*vy).sqrt()
  L=(2*ux*uy+.01**2)/(ux.square()+uy.square()+.01**2);C=(2*sd+.03**2)/(vx+vy+.03**2);S=(cov+.03**2/2)/(sd+.03**2/2)
  target=((1-L)*(1-C)*(1-S)).clamp(min=0).mean(1,keepdim=True);target=F.adaptive_avg_pool2d(target,(28,37))
 sigma_ss,tr_ss,ms=fit_uncert(features,target,True,patchvalid);sigma_co,tr_co,mc=fit_uncert(features,t(d['feature_error'][:,None]),False,patchvalid)
 torch.save({'ssim':ms,'cosine':mc},out/'uncertainty_weights.pt')
 sigss=F.interpolate(sigma_ss,size=(576,768),mode='bilinear',align_corners=False);sigco=F.interpolate(sigma_co,size=(576,768),mode='bilinear',align_corners=False)
 masks={'dino_ssim_soft':(1/(2*sigss.square())*valid)[:,0].cpu().numpy(),'dino_cosine_soft':(1/(2*sigco.square())*valid)[:,0].cpu().numpy(),'dino_cosine_hard':((1/(2*sigco.square())>1)*valid)[:,0].cpu().numpy()}
 rms=np.load(old/'warmup.npz')['residual'];sams=[]
 for i in range(20):
  v=a['valid'][i%4].astype(bool);raw=(rms[i]>np.median(rms[i][v])).astype(np.float32);raw=cv2.boxFilter(raw,-1,(3,3))>.5;labels=np.load(old/'sam_cache'/f'{i}.npz')['labels'];excluded=raw.copy()
  for lab in np.unique(labels):
   reg=(labels==lab)&v
   if reg.any():excluded[reg]=raw[reg].mean()>.4
  sams.append((~excluded)&v)
 masks['sam_region_median']=np.stack(sams).astype(np.float32)
 stats={}
 for k,m in masks.items():
  np.savez_compressed(out/(k+'_mask.npz'),static=m.astype(np.float32));vv=np.tile(a['valid'],(5,1,1));stats[k]={'positive_fraction':float((m[vv]>.000001).mean()),'mean_weight':float(m[vv].mean()),'min':float(m[vv].min()),'max':float(m[vv].max()),'fraction_weight_gt_1':float((m[vv]>1).mean())}
 np.savez_compressed(out/'uncertainty_maps.npz',ssim_sigma=sigma_ss.cpu().numpy(),cosine_sigma=sigma_co.cpu().numpy(),ssim_target=target.cpu().numpy(),cosine_target=d['feature_error'])
 (out/'mask_diagnostics.json').write_text(json.dumps({'stats':stats,'ssim_trace':tr_ss,'cosine_trace':tr_co,'training_only':True},indent=2));log('MASKS',name,stats)
 del images,features,warm;torch.cuda.empty_cache()
 return masks

def bake(name,method,seed,masks):
 out=P/'results'/name/str(seed);(out/'textures').mkdir(parents=True,exist_ok=True)
 if (out/(method+'_trace.json')).exists() and method!='sam_region_median':
  log('SKIP completed',name,method,seed);return
 old=BASE/'harder_dataset/scenes'/name/'results';torch.manual_seed(seed);rng=np.random.default_rng(seed);start=time.time()
 a=np.load(old/'views.npz');images=t(a['images'][:20].transpose(0,3,1,2));grids=t(a['grids']);valid=t(a['valid'][:,None]);src=np.load(old/'aligned.npz')['images'][:5]
 tex=torch.nn.Parameter(F.interpolate(t(src.mean(0).transpose(2,0,1))[None],size=(512,512),mode='bilinear',align_corners=True));aff=torch.nn.Parameter(torch.eye(3,device=D)[None].repeat(20,1,1));bias=torch.nn.Parameter(torch.zeros(20,3,device=D));mask=t(masks[method][:,None]);opt=torch.optim.Adam([{'params':[tex],'lr':.025},{'params':[aff,bias],'lr':.001}]);trace=[]
 for step in range(600):
  ii=torch.tensor(rng.choice(20,4,replace=False),device=D);c=ii%4;v=valid[c];resolved=F.interpolate(tex,size=(1024,1024),mode='bilinear',align_corners=True);render=F.grid_sample(resolved.expand(4,-1,-1,-1),grids[c],align_corners=True,padding_mode='zeros');pred=torch.einsum('bij,bjhw->bihw',aff[ii],render)+bias[ii,:,None,None];err=(pred-images[ii]).square().mean(1,keepdim=True);data=(err*mask[ii]*v).sum()/v.sum().clamp(min=1);color=.001*((aff-torch.eye(3,device=D)).square().mean()+bias.square().mean());loss=data+color;opt.zero_grad();loss.backward();aff.grad[8]=0;bias.grad[8]=0;opt.step()
  with torch.no_grad():tex.clamp_(0,1)
  if step in [0,199,399,599]:trace.append({'step':step,'data':float(data),'color':float(color)})
 with torch.no_grad():z=F.interpolate(tex,size=(1024,1024),mode='bilinear',align_corners=True)[0].permute(1,2,0).cpu().numpy()
 saveim(out/'textures'/(method+'.png'),z);np.savez_compressed(out/(method+'_fit.npz'),affine=aff.detach().cpu().numpy(),bias=bias.detach().cpu().numpy())
 (out/(method+'_trace.json')).write_text(json.dumps({'seed':seed,'updates':600,'seconds':time.time()-start,'trace':trace,'mask_is_fixed':True,'mask_path':str(P/'results'/name/(method+'_mask.npz'))},indent=2));log('DONE',name,method,seed,round(time.time()-start,1))
 del images,tex,mask,grids;torch.cuda.empty_cache()
if __name__=='__main__':
 for name in CFG['scenes']:
  masks=prep(name)
  for seed in CFG['seeds']:
   for method in CFG['methods']:bake(name,method,seed,masks)
 log('ALL TRAINING COMPLETE')

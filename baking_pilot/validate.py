from pilot import *
a=np.load(OUT/'views.npz');g=a['grids'];cams=json.loads((OUT/'cameras.json').read_text());checks={}
# Projection inverse check for analytic plane homographies.
for c in range(4):
 y,x=np.meshgrid(np.arange(H),np.arange(W),indexing='ij');q=np.stack([(g[c,...,0]+1)*(N-1)/2,(g[c,...,1]+1)*(N-1)/2,np.ones((H,W))],-1);p=q@np.array(cams[c]['homography']).T;p=p[...,:2]/p[...,2:];error=np.max(np.abs(p-np.stack([x,y],-1)));assert error<.001;checks[f'camera_{c}_inverse_max_px']=float(error)
# Differentiable renderer must agree with OpenCV synthesis to interpolation quantization accuracy.
tex=np.load(OUT/'aligned.npz')['images'][0];r=render(tensor(tex.transpose(2,0,1))[None],tensor(g)).detach().cpu().numpy().transpose(0,2,3,1);ref=a['images'][:4];valid=a['valid'];rmse=np.sqrt(np.mean((r[valid]-ref[valid])**2));assert rmse<.006;checks['renderer_cv2_rmse']=float(rmse)
# Variable integral image means compared against direct, clipped spatial windows.
x=np.random.default_rng(42).random((2,1,H,W),dtype=np.float32);k=np.random.default_rng(43).integers(8,40,(2,H,W));out=variable_box(tensor(x),tensor(k)).cpu().numpy();errors=[]
for b,y,z in [(0,0,0),(1,575,767),(0,120,200),(1,230,310)]:
 n=k[b,y,z];lo=(n-1)//2;hi=n-lo;target=x[b,0,max(0,y-lo):min(H,y+hi),max(0,z-lo):min(W,z+hi)].mean();errors.append(abs(float(out[b,0,y,z])-float(target)))
assert max(errors)<.001;checks['variable_box_max_error']=max(errors)
for method in ['l2','block','hugs','robust','robust_or','robust_car','robust_depth']:
 p=OUT/f'{method}_fit.npz'
 if not p.exists():continue
 s=np.load(p)['static'];assert s.shape==(20,H,W);assert np.isfinite(s).all();assert np.isin(s,[0,1]).all();assert not s[np.tile(~valid,(5,1,1))].any()
checks['mask_checks']='binary, finite, correct shapes, no invalid pixels';(OUT/'validation.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))

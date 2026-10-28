from pilot import *
# Visually marked vehicle BODY INTERIOR points in manual_reference.jpg (800x800).
# Coordinates converted into aligned 1024 texture space. This is an audit subset, not full semantic GT.
pts=[[398,223],[447,163],[478,287],[581,270],[619,304],[685,328],[207,401],[278,495],[446,512],[454,620],[329,524],[370,565],[529,347],[555,369],[339,391],[385,442]]
points=np.array(pts)/2+np.array([200,180]);source=np.load(OUT/'aligned.npz')['images'];mask=np.zeros((N,N),np.uint8)
for x,y in points:cv2.circle(mask,(round(x),round(y)),5,1,-1)
(OUT/'manual_annotations.json').write_text(json.dumps(dict(reference='manual_reference.jpg',reference_crop=[200,180,600,580],reference_resize=[800,800],reference_points=pts,texture_points=points.tolist(),radius_px=5,vehicle_epoch=2025,clean_epoch=2022,annotation='Vehicle body interiors visually selected by assistant; audit subset, not human-verified full masks'),indent=2))
np.save(OUT/'manual_vehicle_cores.npy',mask.astype(bool));items=[]
for idx in [1,2,4,5]:
 im=np.uint8(source[idx]*255)
 for j,(x,y) in enumerate(points):cv2.circle(im,(round(x),round(y)),7,(255,0,0),1);cv2.putText(im,str(j+1),(round(x)+5,round(y)),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,0,0),1)
 items.append((str(2021+idx),im[180:580,200:600]/255.))
sheet(items,OUT/'manual_audit.jpg',2,650)

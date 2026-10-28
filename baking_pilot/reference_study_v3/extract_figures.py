from pathlib import Path
import pymupdf as f,json
P=Path(__file__).parent;A=P/'assets';A.mkdir(exist_ok=True)
crops={
 'on_the_go':(2,(49,61,287,204),'Fig. 2: DINO uncertainty pipeline'),
 'robustnerf':(3,(49,66,288,204),'Fig. 3: residual spatial structure'),
 'spotless':(1,(316,77,562,141),'Fig. 2: semantic clustering and masking'),
 'wildgaussians':(2,(107,72,504,218),'Fig. 2: appearance + DINO uncertainty'),
 'gs_w':(4,(133,114,483,315),'Fig. 2: architecture'),
 'robust_gs':(0,(33,151,381,265),'Fig. 1: noisy captures and robust reconstruction'),
 'robust_3dgs':(4,(133,114,483,210),'Fig. 2: residual mask + SAM objects')}
manifest={}
for n,(idx,b,title) in crops.items():
 p=f.open(P/'papers'/(n+'.pdf'))[idx];pix=p.get_pixmap(matrix=f.Matrix(3,3),clip=f.Rect(b));pix.save(A/(n+'_paper.png'));manifest[n]={'pdf_page':idx+1,'crop_pt':b,'caption':title,'source':'papers/'+n+'.pdf'}
(A/'paper_figure_manifest.json').write_text(json.dumps(manifest,indent=2))
print('Cropped',len(crops),'paper figures')

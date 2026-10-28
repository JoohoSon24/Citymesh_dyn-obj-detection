from pathlib import Path
import subprocess
A=Path(__file__).parent/'assets'
E={'sam_vote':r'S(m)=\mathbf{1}\!\left[\frac{N_{\mathrm{outlier}}(m)}{N_{\mathrm{all}}(m)}\le0.4\right]', 'shadow_simple':r'D^{\prime}=D\lor G,\quad S=1-D^{\prime},\quad G=\operatorname{nearVehicle}(D)\,\mathbf{1}[\ell<0.72\bar\ell]'}
for n,e in E.items():
 t=r'\documentclass[border=3pt]{standalone}\usepackage{amsmath,amssymb}\begin{document}$\displaystyle '+e+r'$\end{document}';p=A/(n+'.tex');p.write_text(t);q=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory',str(A),str(p)],stdout=subprocess.PIPE);assert q.returncode==0
 subprocess.run(['pdftocairo','-png','-singlefile','-transp','-r','300',str(A/(n+'.pdf')),str(A/(n+'_eq'))],check=True)

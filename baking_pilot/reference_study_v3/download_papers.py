from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests,subprocess
P=Path(__file__).parent/'papers'
urls={'on_the_go':'https://openaccess.thecvf.com/content/CVPR2024/papers/Ren_NeRF_On-the-go_Exploiting_Uncertainty_for_Distractor-free_NeRFs_in_the_Wild_CVPR_2024_paper.pdf','spotless':'https://arxiv.org/pdf/2406.20055','wildgaussians':'https://arxiv.org/pdf/2407.08447','gs_w':'https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09802.pdf','robust_gs':'https://arxiv.org/pdf/2404.04211','robust_3dgs':'https://arxiv.org/pdf/2408.11697','robustnerf':'https://openaccess.thecvf.com/content/CVPR2023/papers/Sabour_RobustNeRF_Ignoring_Distractors_With_Robust_Losses_CVPR_2023_paper.pdf'}
def work(kv):
 n,u=kv
 try:
  rr=requests.get(u,timeout=90);rr.raise_for_status();assert rr.content.startswith(b'%PDF');(P/(n+'.pdf')).write_bytes(rr.content);subprocess.run(['pdftotext','-layout',str(P/(n+'.pdf')),str(P/(n+'.txt'))],check=True);return n,len(rr.content)
 except Exception as e:return n,str(e)
with ThreadPoolExecutor(max_workers=4) as ex:
 for r in ex.map(work,urls.items()):print(r,flush=True)

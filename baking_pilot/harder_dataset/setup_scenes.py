from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
configs={
'industrial_b':dict(name='Utrecht industrial B - dense paved lots',lat=52.0740,lon=5.0460,ground_polygons=[[[25,600],[1450,925],[1370,1260],[25,945]],[[1140,230],[1480,300],[1450,850],[1030,750]]],exclude_polygons=[[[850,855],[1220,955],[1195,1040],[825,935]],[[1160,430],[1310,460],[1290,550],[1150,515]]],crops={'cars':[350,480,650,780],'background':[450,600,750,900],'markings':[40,450,340,750]}),
'industrial_c':dict(name='Utrecht industrial C - variable occupancy lot',lat=52.0773,lon=5.04895,ground_polygons=[[[480,470],[1450,750],[1280,1360],[600,1180],[350,980]]],exclude_polygons=[],crops={'cars':[310,350,610,650],'background':[380,520,680,820],'markings':[600,540,900,840]})}
for name,cfg in configs.items():
 scene=ROOT/'scenes'/name;scene.mkdir(exist_ok=True);(scene/'results').mkdir(exist_ok=True)
 cfg.update(size_m=120,train_years=list(range(2021,2026)),holdout_year=2026,selection='Selected from source-year previews and occupancy diagnostics before method outputs. Ground ROI excludes visible roofs; results not used in selection.')
 (scene/'scene.json').write_text(json.dumps(cfg,indent=2))
 for dest,src in [(scene/'data',ROOT/'candidates'/name),(scene/'weights',ROOT.parent/'weights')]:
  if not dest.exists():dest.symlink_to(src,target_is_directory=True)
print('Set up two new scenes')

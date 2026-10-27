# GPA-GUI-Detector + EasyOCR elements (the detector settings of GUI-Lens) for every ScreenSpot-Pro screenshot,
# saved as $DET_DIR/<id>.json for the Set-of-Mark baseline.
# Usage: SSP_DIR=data/ScreenSpot-Pro DET_DIR=runs/det python baselines/detect_elements.py [shard num_shards]; resumable.
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gui_lens.perception.detector import detect_all
DS = os.environ.get('SSP_DIR', 'data/ScreenSpot-Pro'); DET = os.environ.get('DET_DIR', 'runs/det'); os.makedirs(DET, exist_ok=True)
k, n = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (0, 1)
ids = [a['id'] for a in json.load(open(f'{DS}/annotations/all.json'))][k::n]
for i, sid in enumerate(ids):
    out = f'{DET}/{sid}.json'
    if os.path.exists(out): continue
    _, _, merged, w, h = detect_all(f'{DS}/images/{sid}.png')
    els = [{k2: e[k2] for k2 in ('type', 'x', 'y', 'w', 'h', 'cx', 'cy', 'label')} for e in merged]
    json.dump({'w': w, 'h': h, 'elements': els}, open(out + '.tmp', 'w'), ensure_ascii=False); os.replace(out + '.tmp', out)
    if i % 50 == 0: print(k, i, len(ids), flush=True)
print('DET_DONE', k, flush=True)

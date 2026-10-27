"""Lay out the official ScreenSpot-Pro release (https://huggingface.co/datasets/likaixin/ScreenSpot-Pro) as
<dest>/annotations/all.json and <dest>/images/<id>.png, the layout every runner in this repository reads.
Image bytes are copied unchanged.

    python eval/prepare_screenspot_pro.py --source ScreenSpot-Pro --dest data/ScreenSpot-Pro
"""
import argparse, json, shutil
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--source', type=Path, required=True, help='official release with annotations/*.json and images/')
ap.add_argument('--dest', type=Path, required=True)
args = ap.parse_args()

rows = []
for f in sorted((args.source / 'annotations').glob('*.json')):
    rows += json.loads(f.read_text(encoding='utf-8-sig'))
ids = [r['id'] for r in rows]
if len(set(ids)) != len(ids):
    raise SystemExit('duplicate ids in the annotations')
(args.dest / 'images').mkdir(parents=True, exist_ok=True)
(args.dest / 'annotations').mkdir(exist_ok=True)
for r in rows:
    shutil.copyfile(args.source / 'images' / r['img_filename'], args.dest / 'images' / f"{r['id']}.png")
(args.dest / 'annotations/all.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
print(f'{len(rows)} questions written to {args.dest}')

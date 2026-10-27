"""ScreenSpot-Pro accuracy by category (Text/Icon), as in the paper table.

Reads GUI-Lens results.jsonl files (sample_id, correctness), merged into one row, and baseline rows
(id, method, hit), one row per method. A question without a result counts as wrong; a later file overrides an earlier one for the same question.

    python eval/score.py --data-dir data/ScreenSpot-Pro runs/ds_guilens/results.jsonl runs/ds_baselines.jsonl
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

GROUPS = ['CAD', 'Dev', 'Creative', 'Scientific', 'Office', 'OS']

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--data-dir', type=Path, required=True)
ap.add_argument('--name', default='GUI-Lens', help='row name for GUI-Lens results.jsonl files')
ap.add_argument('results', nargs='+', type=Path)
args = ap.parse_args()
ann = {a['id']: a for a in json.loads((args.data_dir / 'annotations/all.json').read_text(encoding='utf-8-sig'))}

hits = defaultdict(dict)                       # row name -> question id -> hit
for path in args.results:
    for line in path.open():
        try: r = json.loads(line)
        except ValueError: continue
        if 'sample_id' in r:   # GUI-Lens files (shards, reruns) merge into one row
            hits[args.name][r['sample_id']] = r.get('correctness') == 'correct'
        elif 'error' not in r:
            hits[f"{path.parent.name}/{path.stem}:{r['method']}"][r['id']] = bool(r['hit'])

def acc(h, keep):
    ids = [i for i, a in ann.items() if keep(a)]
    return 100 * sum(h.get(i, False) for i in ids) / len(ids)

cols = [f'{g[:3]}-{t[0].upper()}' for g in GROUPS for t in ('text', 'icon')] + ['Text', 'Icon', 'Avg', 'Done']
print('\t'.join(['Run'] + cols))
for name, h in hits.items():
    vals = [acc(h, lambda a, g=g, t=t: a['group'] == g and a['ui_type'] == t) for g in GROUPS for t in ('text', 'icon')]
    vals += [acc(h, lambda a: a['ui_type'] == 'text'), acc(h, lambda a: a['ui_type'] == 'icon'), acc(h, lambda a: True)]
    print('\t'.join([name] + [f'{v:.1f}' for v in vals] + [f'{len(h)}/{len(ann)}']))

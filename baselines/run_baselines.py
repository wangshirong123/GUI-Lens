# Training-free grounding baselines on ScreenSpot-Pro with an API base model.
# Usage: SSP_DIR=data/ScreenSpot-Pro FMT=toolcall TOOL_MP=1.81 DEEPSEEK_API_KEY=... \
#        python baselines/run_baselines.py --ids ids.json --methods direct,zoomclick --out runs/ds.jsonl
# The base model is DeepSeek (DEEPSEEK_API_KEY) unless OC_BASE/OC_MODEL/OPENCODE_API_KEY name another
# OpenAI-compatible endpoint; EFFORT sets reasoning_effort (default low). Rows are appended and a rerun resumes.
# Every localization call uses the base model's own direct-click prompt (FMT: toolcall = [0,1000] tool call,
# pixel = pixels of the sent image), so each method differs from Direct only in its search procedure.
#   ZoomClick  (arXiv 2512.05941, official repo Princeton-AI2-Lab/ZoomClick, run_zoomclick_qwen3.slurm defaults):
#              pre-zoom over 2x2 quadrants with a 50 px agreement threshold, depth 3, shrink 0.5, min crop 768, clip.
#   BAMI       (arXiv 2605.06664, Alg. 1 and App. C): N=2 crop iterations, M=2 masked candidates per iteration
#              (click expanded by 25 px and blacked out), selector prompt of App. C.1 answered by the same base model,
#              crop ratio 0.5 around the selected candidate; output is the last selected point.
#   AdaZoom    (arXiv 2603.17441, Sec. 2 and 3.1): refinement prompt and grounding prompt (click + box) of the paper,
#              zoom when (w<=200 and h<=1000) or (h<=200 and w<=1000) px, zoom ratio 2 around the click,
#              zoomed crop resized back to the screenshot size; output is the second-pass click.
#   SoM        Set-of-Mark with the GUI-Lens detector (GPA-GUI-Detector + EasyOCR, det/<id>.json): numbered boxes on the
#              screenshot, the base model names one id, the click is that element's center.
#   Vote       call-matched direct clicking: k=5 direct clicks (about the mean calls of GUI-Lens), the click with the most
#              other clicks within 1% of the screenshot diagonal wins.
#   UI-Zoomer  (official repo ZJU-REAL/UI-Zoomer, uizoomer.py and its ScreenSpot-Pro script): 8 boxes at temperature 0.9 with
#              the paper's box prompt, confidence = exp(mean answer-token logprob), S = mean pairwise IoU + mean confidence;
#              S > 1.5: IoU>0.5 vote and click the box center; else density-filtered Gaussian square crop (sigma scale 2.5,
#              min 512 px) resized to 1-4 MP and one box at temperature 0. Needs logprobs (DeepSeek, OpenCode Go).
import json, os, re, io, sys, time, math, base64, argparse, threading
from concurrent.futures import ThreadPoolExecutor
import requests, uuid
OC_SESSION = str(uuid.uuid4())
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.environ.get('SSP_DIR', 'data/ScreenSpot-Pro')       # annotations/all.json and images/<id>.png
TOOL_SYS = open(os.path.join(HERE, 'prompts', 'computer_use_tool.txt')).read()

ANN = {a['id']: a for a in json.load(open(f'{DS}/annotations/all.json'))}
FMT = os.environ.get('FMT', 'toolcall')
FIT = float(os.environ.get('FIT_MP') or 0)
USAGE = threading.local()

def call(msgs, lp=False, **extra):
    """One chat call to the configured endpoint; returns the answer text (and answer-token logprobs if lp)."""
    if lp: extra['logprobs'] = True
    err = None
    for attempt in range(12):
        try:
            if os.environ.get('OC_MODEL'):
                r = requests.post(os.environ['OC_BASE'] + '/chat/completions', timeout=900,
                                  headers={'Authorization': 'Bearer ' + os.environ['OPENCODE_API_KEY'], 'User-Agent': 'OpenProgram', 'x-opencode-session': OC_SESSION},
                                  json={'model': os.environ['OC_MODEL'], 'messages': msgs, 'max_tokens': 16384,
                                        'reasoning_effort': os.environ.get('EFFORT', 'low'), **extra})
            else:
                r = requests.post('https://api.deepseek.com/chat/completions', timeout=900,
                                  headers={'Authorization': 'Bearer ' + os.environ['DEEPSEEK_API_KEY']},
                                  json={'model': 'deepseek-flash', 'messages': msgs, 'max_tokens': 32768,
                                        'reasoning_effort': os.environ.get('EFFORT', 'low'), **extra})
            j = r.json()
            if 'choices' not in j: raise RuntimeError(f'{r.status_code} ' + json.dumps(j)[:200])
            m = j['choices'][0]['message']; out = m.get('content') or ''
            if m.get('tool_calls'): out = (out + '\n' + json.dumps(m['tool_calls'], ensure_ascii=False)).strip()
            u = j.get('usage') or {}
            USAGE.calls = getattr(USAGE, 'calls', 0) + 1
            USAGE.tok = getattr(USAGE, 'tok', 0) + (u.get('total_tokens') or 0)
            if not lp: return out
            toks = ((j['choices'][0].get('logprobs') or {}).get('content') or [])
            cut = max([i + 1 for i, t in enumerate(toks) if '</think>' in (t.get('token') or '')], default=0)
            return out, [t['logprob'] for t in toks[cut:] if t.get('logprob') is not None]
        except Exception as e:   # rate limits and gateway errors back off longer than other failures
            err = str(e); limited = any(k in err.lower() for k in ('429', 'rate', 'limit', 'quota', 'overload', '503', '502'))
            time.sleep(min(300, (30 if limited else 5) * (attempt + 1)))
    raise RuntimeError(err)

def enc(img):
    """PNG data URL of img and the size of its coordinate frame in img pixels.
    pixel: downscaled to the image budget; the frame is the sent image.
    toolcall: padded at the bottom or right to at most 16:9 and scaled to the image budget (TOOL_MP), because
    DeepSeek mixes pixel and [0,1000] answers on very wide or small images; the frame is the padded image."""
    W, H = img.size
    if FMT == 'pixel':
        k = min(1.0, (FIT * 1e6 / (W * H)) ** 0.5) if FIT else 1.0
        if k < 1: img = img.resize((round(W * k), round(H * k)), Image.LANCZOS)
        fw, fh = img.size
    else:
        fw, fh = max(W, round(H * 9 / 16)), max(H, round(W * 9 / 16))
        canvas = Image.new('RGB', (fw, fh), (0, 0, 0)); canvas.paste(img, (0, 0))
        mp = float(os.environ.get('TOOL_MP') or 0)
        k = (mp * 1e6 / (fw * fh)) ** 0.5 if mp else 1.0
        img = canvas.resize((round(fw * k), round(fh * k)), Image.LANCZOS) if abs(k - 1) > 1e-3 else canvas
    b = io.BytesIO(); img.convert('RGB').save(b, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode(), fw, fh

def nums(pat, text):
    m = re.findall(pat, text); return [float(v) for v in m[-1]] if m else None

def locate(img, instr):
    """Direct click of the base model on img; point in img pixels or None."""
    url, w, h = enc(img); W, H = img.size
    if FMT == 'pixel':
        msgs = [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': url}}, {'type': 'text', 'text':
                f'The screenshot is {w}x{h} pixels. Find the UI element to click for this instruction: {instr}\n'
                'Reply with ONLY JSON: {"x": <int>, "y": <int>} in pixels of the screenshot.'}]}]
        v = nums(r'"x"\s*:\s*(-?[\d.]+)\s*,\s*"y"\s*:\s*(-?[\d.]+)', call(msgs))
        return v and [v[0] * W / w, v[1] * H / h]
    msgs = [{'role': 'system', 'content': TOOL_SYS}, {'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': url}}, {'type': 'text', 'text': instr}]}]
    out = call(msgs)
    v = (nums(r'"coordinate"\s*:\s*\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)', out) or nums(r'coordinate"[^>]*>\s*\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)', out)
         or nums(r'coordinate[^\[\d]{0,20}\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)', out) or nums(r'coordinate"[^>]*>\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)', out))
    return v and [v[0] / 1000 * w, v[1] / 1000 * h]

def crop_at(img, cx, cy, tw, th, mode):
    """Window of tw x th around (cx, cy): 'clip' cuts it at the border, 'shift' moves it inside."""
    W, H = img.size; tw = min(max(1, round(tw)), W); th = min(max(1, round(th)), H)
    x1, y1 = round(cx - tw / 2), round(cy - th / 2)
    if mode == 'clip':
        box = (max(0, x1), max(0, y1), min(W, x1 + tw), min(H, y1 + th))
        if box[2] <= box[0] or box[3] <= box[1]: box = (0, 0, W, H)
    else:
        x1 = min(max(0, x1), W - tw); y1 = min(max(0, y1), H - th); box = (x1, y1, x1 + tw, y1 + th)
    return img.crop(box), box

# ---------------- ZoomClick ----------------
def zoomclick(img, instr, depth=3, ratio=0.5, min_crop=768, thresh=50.0):
    W0, H0 = img.size; cur, ox, oy, last = img, 0, 0, None
    for it in range(depth):
        if it == 0:
            d = locate(cur, instr); pts = []
            for (a, b, c, e) in [(0, 0, .5, .5), (.5, 0, 1, .5), (0, .5, .5, 1), (.5, .5, 1, 1)]:
                box = (int(a * W0), int(b * H0), int(c * W0), int(e * H0)); p = locate(img.crop(box), instr)
                pts.append([box[0] + p[0], box[1] + p[1]] if p else [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            if d is not None:
                dist = [((q[0] - d[0]) ** 2 + (q[1] - d[1]) ** 2) ** .5 for q in pts]
                p = pts[dist.index(min(dist))] if min(dist) < thresh else d
            else:
                p = pts[0]
        else:
            p = locate(cur, instr)
            if p is None: break
        last = [ox + p[0], oy + p[1]]
        if it == depth - 1: break
        cw, ch = cur.size
        tw, th = min(max(round(cw * ratio), min_crop), cw), min(max(round(ch * ratio), min_crop), ch)
        if (tw, th) == (cw, ch): break
        cur, box = crop_at(cur, p[0], p[1], tw, th, 'clip'); ox += box[0]; oy += box[1]
    return last

# ---------------- BAMI ----------------
BAMI_SEL = open(os.path.join(HERE, 'prompts', 'bami_selector.txt')).read()

def mark(img, pt, color, label):
    im = img.copy().convert('RGB'); d = ImageDraw.Draw(im); r = 25; lw = max(3, round(max(im.size) / 500))
    x, y = pt; d.rectangle([x - r, y - r, x + r, y + r], outline=color, width=lw)
    fs = max(20, round(max(im.size) / 60))
    try: font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', fs)
    except Exception: font = ImageFont.load_default()
    ty = y - r - fs - 4 if y - r - fs - 4 >= 0 else y + r + 4
    d.rectangle([x - r, ty, x - r + fs, ty + fs + 2], fill=color); d.text((x - r + fs * .25, ty), label, fill='white', font=font)
    return im

def bami(img, instr, N=2, M=2, lam=0.5, r=25):
    cur, ox, oy, sel = img, 0, 0, None
    for t in range(N):
        cands = []
        for i in range(M):
            masked = cur.copy().convert('RGB'); d = ImageDraw.Draw(masked)
            for (x, y) in cands: d.rectangle([x - r, y - r, x + r, y + r], fill='black')
            p = locate(masked, instr)
            if p is not None: cands.append(p)
        if not cands: break
        sel = cands[0]
        if len(cands) >= 2:
            u1, _, _ = enc(mark(cur, cands[0], (0, 200, 0), '1')); u2, _, _ = enc(mark(cur, cands[1], (230, 0, 0), '2'))
            out = call([{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': u1}}, {'type': 'image_url', 'image_url': {'url': u2}},
                                                      {'type': 'text', 'text': BAMI_SEL.replace('{user_query}', instr)}]}])
            m = re.findall(r'<answer>\s*([12])', out)
            if m and m[-1] == '2': sel = cands[1]
        final = [ox + sel[0], oy + sel[1]]
        if t < N - 1:
            cw, ch = cur.size; cur, box = crop_at(cur, sel[0], sel[1], cw * lam, ch * lam, 'shift'); ox += box[0]; oy += box[1]
    return final if sel is not None else None

# ---------------- AdaZoom-GUI ----------------
ADA_REFINE = ('You are given a task description and a screenshot of a GUI. The task can be completed with only one click.\n'
              'You need to find out the target to click, and then refine the task description to let user easily locate the target on the screen.\n'
              'Possible refinements include adding location information, describing visual features (color, size, text, icon shape, ...), clarifying ambiguous terms, etc.\n\n'
              'Only reply with the refined description. Do not add explanations.\n\nTask: {instruction}')
ADA_SYS = ('You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform pyautogui click/moveTo action to complete the task, '
           'and then provide the bouding box of the target object. The answer format is `pyautogui.click(x=?, y=?), <|box_start|>(x1,y1),(x2,y2)<|box_end|>`. '
           'If the task is infeasible (e.g., the task is already completed, the target does not exist in the image, or the instruction is unrelated to the screenshot), '
           'output a null action exactly as follows: `pyautogui.click(x=0, y=0), <|box_start|>(0,0),(0,0)<|box_end|>`.')

def ada_ground(img, task):
    url, w, h = enc(img); W, H = img.size
    space = (f'Coordinates are in pixels of the screenshot, which is {w}x{h} pixels.' if FMT == 'pixel'
             else 'Coordinates are relative values in [0, 1000] of the screenshot width and height.')
    out = call([{'role': 'system', 'content': ADA_SYS}, {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': url}}, {'type': 'text', 'text':
                'Please complete the following tasks by clicking using `pyautogui.click` and returning the bounding box:\n' + space + f'\nTask: {task}'}]}])
    sx, sy = (W / w, H / h) if FMT == 'pixel' else (w / 1000, h / 1000)
    c = nums(r'click\(\s*x\s*=\s*(-?[\d.]+)\s*,\s*y\s*=\s*(-?[\d.]+)', out)
    b = nums(r'\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)\s*,\s*\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)', out)
    c = c and [c[0] * sx, c[1] * sy]; b = b and [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
    if c is None and b is not None: c = [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]
    return c, b

def adazoom(img, instr, alpha=200, beta=1000, ratio=2):
    url, _, _ = enc(img)
    task = call([{'role': 'system', 'content': 'You are a helpful GUI assistant.'}, {'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': url}}, {'type': 'text', 'text': ADA_REFINE.replace('{instruction}', instr)}]}]).strip() or instr
    c, b = ada_ground(img, task)
    if c is None: return None, task
    if b is not None:
        bw, bh = abs(b[2] - b[0]), abs(b[3] - b[1])
        if (bw <= alpha and bh <= beta) or (bh <= alpha and bw <= beta):
            W, H = img.size; crop, box = crop_at(img, c[0], c[1], W / ratio, H / ratio, 'shift')
            z = crop.resize((W, H), Image.LANCZOS); c2, _ = ada_ground(z, task)
            if c2 is not None:
                return [box[0] + c2[0] * crop.size[0] / W, box[1] + c2[1] * crop.size[1] / H], task
    return c, task

def direct(img, instr):
    return locate(img, instr)

# ---------------- Set-of-Mark ----------------
DET = os.environ.get('DET_DIR', 'runs/det')                   # detect_elements.py output, for SoM
COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (0, 150, 150), (240, 50, 230), (128, 128, 0)]

def som(img, instr, sid):
    els = json.load(open(f'{DET}/{sid}.json'))['elements']
    if not els: return None
    im = img.copy(); d = ImageDraw.Draw(im); W, H = im.size
    fs = max(14, round(max(W, H) / 120)); lw = max(2, round(max(W, H) / 1500))
    try: font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', fs)
    except Exception: font = ImageFont.load_default()
    for i, e in enumerate(els):
        c = COLORS[i % len(COLORS)]; d.rectangle([e['x'], e['y'], e['x'] + e['w'], e['y'] + e['h']], outline=c, width=lw)
        tw = d.textlength(str(i), font=font); ty = e['y'] - fs - 2 if e['y'] - fs - 2 >= 0 else e['y']
        d.rectangle([e['x'], ty, e['x'] + tw + 4, ty + fs + 2], fill=c); d.text((e['x'] + 2, ty), str(i), fill='white', font=font)
    url, _, _ = enc(im)
    out = call([{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': url}}, {'type': 'text', 'text':
                'The UI elements in the screenshot are marked with numbered boxes. Which marked element should be clicked for this instruction: '
                f'{instr}\nReply with ONLY JSON: {{"id": <int>}}'}]}])
    m = re.findall(r'"id"\s*:\s*(\d+)', out)
    if not m or int(m[-1]) >= len(els): return None
    e = els[int(m[-1])]; return [e['cx'], e['cy']]

# ---------------- call-matched voting ----------------
def vote(img, instr, k=5, frac=0.01):
    pts = [p for p in (locate(img, instr) for _ in range(k)) if p is not None]
    if not pts: return None
    r = frac * math.hypot(*img.size)
    n = [sum(math.hypot(p[0] - q[0], p[1] - q[1]) <= r for q in pts) for p in pts]
    return pts[n.index(max(n))]

# ---------------- UI-Zoomer ----------------
UIZ_PROMPT = 'Outline the position corresponding to the instruction: {}. The output should be only [x1,y1,x2,y2].'

def uiz_box(img, instr, temperature):
    """One box in img pixels and its confidence exp(mean answer-token logprob)."""
    url, w, h = enc(img); W, H = img.size
    space = (f'The screenshot is {w}x{h} pixels; give the box [x1,y1,x2,y2] (top-left and bottom-right corners) in pixels of the screenshot.' if FMT == 'pixel'
             else 'Give the box [x1,y1,x2,y2] (top-left and bottom-right corners) in relative coordinates in [0, 1000] of the screenshot width and height.')
    out, lps = call([{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': url}}, {'type': 'text', 'text':
                     UIZ_PROMPT.format(instr) + '\n' + space}]}], lp=True, temperature=temperature)
    b = nums(r'\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]', out)
    if b is None: return None, 0.0
    sx, sy = (W / w, H / h) if FMT == 'pixel' else (w / 1000, h / 1000)
    x1, x2 = sorted([b[0] * sx, b[2] * sx]); y1, y2 = sorted([b[1] * sy, b[3] * sy])
    box = [min(max(x1, 0), W), min(max(y1, 0), H), min(max(x2, 0), W), min(max(y2, 0), H)]
    return box, (math.exp(sum(lps) / len(lps)) if lps else 0.0)

def iou(a, b):
    i = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 1e-6 else 0.0

def resize_mp(img, lo, hi):
    W, H = img.size; k = (lo / (W * H)) ** .5 if W * H < lo else (hi / (W * H)) ** .5 if W * H > hi else 1.0
    return img.resize((max(1, round(W * k)), max(1, round(H * k))), Image.BICUBIC) if k != 1.0 else img

def uizoomer(img, instr, n=8, thresh=1.5, sigma=2.5, min_crop=512):
    cands = [c for c in (uiz_box(img, instr, 0.9) for _ in range(n)) if c[0] is not None]
    if not cands: return None, {'branch': 'no_box'}
    ctr = lambda b: [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]
    boxes = [b for b, _ in cands]; conf = [c for _, c in cands]
    pair = [iou(boxes[i], boxes[j]) for i in range(len(boxes)) for j in range(len(boxes)) if i != j]
    Sc = (sum(pair) / len(pair) if pair else 1.0) + sum(conf) / len(conf)
    if Sc > thresh:
        votes = [sum(iou(boxes[i], boxes[j]) > 0.5 for j in range(len(boxes)) if j != i) for i in range(len(boxes))]
        best = max((i for i in range(len(boxes)) if votes[i] == max(votes)), key=lambda i: conf[i])
        return ctr(boxes[best]), {'branch': 'vote', 'S': Sc}
    W, H = img.size
    cs = [ctr(b) for b in boxes]; med = [sorted(v)[len(v) // 2] if len(v) % 2 else sum(sorted(v)[len(v) // 2 - 1:len(v) // 2 + 1]) / 2 for v in zip(*cs)]
    keep = sorted(range(len(cs)), key=lambda i: math.hypot(cs[i][0] - med[0], cs[i][1] - med[1]))[:max(1, int(len(cs) * 0.75))]
    mu, sd = [], []
    for a in (0, 1):
        v = [cs[i][a] for i in keep]; m = sum(v) / len(v)
        intra = sum(((boxes[i][a + 2] - boxes[i][a]) / 4) ** 2 for i in keep) / len(keep)
        mu.append(m); sd.append((sum((x - m) ** 2 for x in v) / len(v) + intra) ** .5)
    s = max(2 * sigma * sd[0], 2 * sigma * sd[1], min_crop); half = s / 2
    x1, y1, x2, y2 = mu[0] - half, mu[1] - half, mu[0] + half, mu[1] + half
    if x1 < 0: x2 -= x1; x1 = 0
    if y1 < 0: y2 -= y1; y1 = 0
    if x2 > W: x1 -= x2 - W; x2 = W
    if y2 > H: y1 -= y2 - H; y2 = H
    box = (int(max(0, x1)), int(max(0, y1)), int(min(W, x2)), int(min(H, y2)))
    crop = img.crop(box); z = resize_mp(crop, 1e6, 4e6)
    b, _ = uiz_box(z, instr, 0.0)
    if b is None: return ctr(boxes[conf.index(max(conf))]), {'branch': 'crop_fail', 'S': Sc}
    kx, ky = crop.size[0] / z.size[0], crop.size[1] / z.size[1]
    return [box[0] + (b[0] + b[2]) / 2 * kx, box[1] + (b[1] + b[3]) / 2 * ky], {'branch': 'crop', 'S': Sc, 'crop': list(box)}

METHODS = {'direct': direct, 'zoomclick': zoomclick, 'bami': bami, 'adazoom': adazoom, 'som': som, 'vote': vote, 'uizoomer': uizoomer}

def run(job):
    sid, meth = job; a = ANN[sid]; x1, y1, x2, y2 = a['bbox']; USAGE.calls = 0; USAGE.tok = 0
    try:
        img = Image.open(f'{DS}/images/{sid}.png').convert('RGB')
        out = som(img, a['instruction'], sid) if meth == 'som' else METHODS[meth](img, a['instruction']); extra = {}
        if meth == 'adazoom': out, extra = out[0], {'refined': out[1]}
        if meth == 'uizoomer': out, extra = out
        hit = out is not None and x1 <= out[0] <= x2 and y1 <= out[1] <= y2
        return {'id': sid, 'method': meth, 'hit': hit, 'point': out, 'calls': USAGE.calls, 'tokens': USAGE.tok, **extra}
    except Exception as e:
        return {'id': sid, 'method': meth, 'error': str(e)[:300], 'calls': USAGE.calls}

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--ids', help='JSON {"ids": [...]}; default all questions'); ap.add_argument('--methods', default='zoomclick,bami,adazoom')
    ap.add_argument('--out', required=True); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args(); ids = json.load(open(a.ids))['ids'] if a.ids else list(ANN)
    done = set()
    if os.path.exists(a.out):            # resume: keep finished rows, drop a line cut off by an interrupt
        good = []
        for l in open(a.out):
            try: r = json.loads(l)
            except ValueError: continue
            good.append(l if l.endswith('\n') else l + '\n')
            if 'error' not in r: done.add((r['id'], r['method']))
        open(a.out, 'w').writelines(good)
    jobs = [(i, m) for m in a.methods.split(',') for i in ids if (i, m) not in done]
    print('todo', len(jobs), flush=True)
    with open(a.out, 'a') as f, ThreadPoolExecutor(a.workers) as ex:
        for k, r in enumerate(ex.map(run, jobs)):
            f.write(json.dumps(r, ensure_ascii=False) + '\n'); f.flush()
            if (k + 1) % 25 == 0: print(k + 1, flush=True)

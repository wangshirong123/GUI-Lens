"""Run GUI-Lens (or the single-shot baseline) on ScreenSpot-Pro.

    python run_guilens.py --model deepseek-v4.1-flash --config configs/guilens.yaml \
        --data-dir data/ScreenSpot-Pro --output-dir runs/ds_guilens --all --thinking low

Results go to <output-dir>/results.jsonl, one line per question; --resume skips questions already there.
API keys are read from the environment (DEEPSEEK_API_KEY, ARK_API_KEY, OPENCODE_API_KEY, OPENROUTER_API_KEY);
openai-codex and anthropic use the OpenProgram login (`openprogram providers login`).
"""
import argparse
import os
import runpy
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Shortcut -> (provider, model id, base URL for models missing from the OpenProgram catalog).
MODELS = {
    'deepseek-v4.1-flash': ('deepseek', 'deepseek-flash', 'https://api.deepseek.com/v1'),
    'glm-5.3-flash-ark': ('ark', 'glm-5-3-flash-260828', 'https://ark.cn-beijing.volces.com/api/coding/v3'),
    'glm-5.3-flash-go': ('opencode-go', 'glm-5.3-flash', 'https://opencode.ai/zen/go/v1'),
    'minimax-m3-go': ('opencode-go', 'minimax-m3', 'https://opencode.ai/zen/go/v1'),
}
KEY_ENV = {'deepseek': 'DEEPSEEK_API_KEY', 'ark': 'ARK_API_KEY', 'opencode-go': 'OPENCODE_API_KEY',
           'openrouter': 'OPENROUTER_API_KEY'}


def parse_indexes(spec, n):
    out = set()
    for part in spec.split(','):
        lo, _, hi = part.strip().partition('-')
        out.update(range(int(lo), int(hi or lo) + 1))
    if not out or min(out) < 0 or max(out) >= n:
        raise SystemExit(f'indexes must lie in 0-{n - 1}')
    return ','.join(map(str, sorted(out)))


def register_models(provider, model, base_url):
    """Add an OpenAI-compatible vision model that the bundled OpenProgram catalog lacks."""
    from openprogram.providers.models_generated import MODELS as catalog
    from openprogram.providers.types import Model
    catalog.setdefault(f'{provider}/{model}', Model.model_validate({
        'id': model, 'name': model, 'api': 'openai-completions', 'provider': provider, 'base_url': base_url,
        'reasoning': True, 'thinking_levels': ['low', 'medium', 'high'], 'default_thinking_level': 'low',
        'thinking_variant': None, 'input': ['text', 'image'],
        'cost': {'input': 0.0, 'output': 0.0, 'cache_read': 0.0, 'cache_write': 0.0},
        'context_window': 1048576, 'max_tokens': 32768, 'headers': None, 'compat': None}))


def use_env_keys():
    """Serve API keys from the environment instead of the OpenProgram auth store."""
    from openprogram.auth import store, types
    orig = store.AuthStore.get_pool

    def get_pool(self, provider_id, profile_id):
        key = os.environ.get(KEY_ENV.get(provider_id, ''), '')
        if not key:
            return orig(self, provider_id, profile_id)
        cred = types.Credential(provider_id=provider_id, profile_id=profile_id, kind='api_key',
                                payload=types.ApiKeyPayload(api_key=key), status='valid', source='env',
                                read_only=True, credential_id='env')
        return types.CredentialPool(provider_id=provider_id, profile_id=profile_id, credentials=[cred])
    store.AuthStore.get_pool = get_pool


def opencode_go_headers():
    """OpenCode Go requires a session id and a client user agent on every request."""
    import openai.resources.chat.completions as occ
    headers = {'x-opencode-session': 'guilens-' + uuid.uuid4().hex, 'User-Agent': 'guilens-eval/1.0'}
    orig = occ.AsyncCompletions.create

    async def create(self, *args, **kwargs):
        if '/zen/go/' in str(self._client.base_url):
            kwargs['extra_headers'] = {**(kwargs.get('extra_headers') or {}), **headers}
        return await orig(self, *args, **kwargs)
    occ.AsyncCompletions.create = create


def set_thinking(level):
    from openprogram.agentic_programming import runtime
    orig = runtime.Runtime.__init__

    def init(self, *a, **kw):
        orig(self, *a, **kw)
        self.thinking_level = level
    runtime.Runtime.__init__ = init


def set_ocr(backend):
    from gui_lens.perception import ocr
    ocr.detect_text = (lambda p, return_logical=False: ocr._detect_text_easyocr(p)) if backend == 'easyocr' \
        else (lambda p, return_logical=False: ocr._detect_text_apple_vision_tiled(p))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help=f'one of {sorted(MODELS)}, or a model id with --provider')
    ap.add_argument('--provider', help='OpenProgram provider for a model id not in the shortcut list')
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, required=True, help='contains annotations/all.json and images/<id>.png')
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--indexes', default='0-4', help='0-based question indexes, e.g. 0-99 or 1,5,9')
    ap.add_argument('--all', action='store_true', help='all 1,581 questions')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--thinking', default='', help='low|medium|high; empty sends no reasoning setting')
    ap.add_argument('--ocr', default='easyocr', choices=['easyocr', 'apple'], help='apple = tiled Apple Vision (macOS)')
    ap.add_argument('--gpa-model', type=Path, help='local GPA-GUI-Detector weight; downloaded from Hugging Face if omitted')
    args = ap.parse_args()

    if args.model in MODELS:
        provider, model, base_url = MODELS[args.model]
    elif args.provider:
        provider, model, base_url = args.provider, args.model, None
    else:
        raise SystemExit('unknown --model; pass --provider as well')
    if provider in KEY_ENV and not os.environ.get(KEY_ENV[provider]):
        raise SystemExit(f'{KEY_ENV[provider]} is not set')

    data, output = args.data_dir.resolve(), args.output_dir.resolve()
    import json
    n = len(json.loads((data / 'annotations/all.json').read_text(encoding='utf-8-sig')))
    indexes = parse_indexes(f'0-{n - 1}' if args.all else args.indexes, n)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise SystemExit('output directory is not empty; use a new one or --resume')
    output.mkdir(parents=True, exist_ok=True)
    for env, rel in {'TORCH_HOME': 'cache/torch', 'HF_HOME': 'cache/huggingface', 'YOLO_CONFIG_DIR': 'cache/ultralytics'}.items():
        os.environ.setdefault(env, str(output / rel))
        Path(os.environ[env]).mkdir(parents=True, exist_ok=True)
    if args.gpa_model:
        os.environ['GPA_MODEL_PATH'] = str(args.gpa_model.resolve())
    os.environ['OPENPROGRAM_EXEC_TIMEOUT_S'] = '180'

    sys.path.insert(0, str(ROOT))
    if base_url:
        register_models(provider, model, base_url)
    use_env_keys()
    opencode_go_headers()
    if args.thinking:
        set_thinking(args.thinking)
    set_ocr(args.ocr)

    runner = ROOT / 'eval/screenspot_pro.py'
    sys.argv = [str(runner), '--data-dir', str(data), '--annotation', 'all.json', '--indexes', indexes,
                '--provider', provider, '--model', model, '--config', str(args.config.resolve()),
                '--output', str(output / 'results.jsonl'), '--work-dir', str(output / 'work'),
                '--exec-timeout-s', '180', '--runtime-retries', '1', '--retry-provider-errors', '1'] \
        + (['--skip-existing'] if args.resume else [])
    runpy.run_path(str(runner), run_name='__main__')


if __name__ == '__main__':
    main()

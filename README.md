<div align="center">
<img src="assets/images/title.png?raw=true" width="90%" alt="GUI-Lens: Enabling Precise GUI Grounding for Off-the-Shelf VLMs via Coarse-to-Fine Cropping">
<hr>

Zichuan Fu<sup>*1</sup>&nbsp;
Shirong Wang<sup>*1</sup>&nbsp;
Wenlin Zhang<sup>1</sup>&nbsp;
Guojing Li<sup>1</sup>&nbsp;
Yimin Deng<sup>1</sup>&nbsp;
Jingtong Gao<sup>1</sup>&nbsp;
Junjia Qi<sup>1</sup><br>
Hanyu Yan<sup>1</sup>&nbsp;
Yefeng Zheng<sup>3</sup>&nbsp;
Xiaopeng Li<sup>1</sup>&nbsp;
Wanyu Wang<sup>1</sup>&nbsp;
Xian Wu<sup>2</sup>&nbsp;
Xiangyu Zhao<sup>1†</sup>

<sup>1</sup> City University of Hong Kong&nbsp;&nbsp;<sup>2</sup> Tencent Jarvis Lab&nbsp;&nbsp;<sup>3</sup> Westlake University<br>
<sup>*</sup> Equal Contribution&nbsp;&nbsp;&nbsp;&nbsp;<sup>†</sup> Corresponding Author

<h4>
<a href="https://arxiv.org/abs/2608.03270">📄 arXiv Paper</a> &nbsp;
<a href="https://wangshirong123.github.io/GUI-Lens/">🌐 Project Page</a>
</h4>

</div>

<div align="center">
<img src="assets/images/framework.png?raw=true" width="100%">
</div>

Figure 1. Overview of GUI-Lens. Coordinate Priming turns OCR text and detected UI components into coordinate
references. Using the task, the current crop and the visible references, the VLM either selects a finer crop or
proposes a click. Accepted crops advance the search, rejected proposals restart it, and the accepted click is mapped
back to the original screen.

## :sparkles: Highlights

🤔 **A VLM often recognizes the requested control but cannot click it precisely** on high-resolution, densely
populated screens. Methods that still end with one direct click let an inaccurate first estimate reach the result.

🔍 **GUI-Lens lets the VLM look closer before it clicks.** It selects the region and scale of the next view, which is
cropped and enlarged, over successively focused views until the target is determined.

📍 **Coordinate Priming** lists OCR text and detected UI components with their positions as coordinate references,
so the model can point at a listed element instead of estimating raw coordinates.

✅ **Crops and clicks are checked against the instruction**, and the final local point is mapped back to screen
coordinates. No training is needed; GUI-Lens works with any API VLM.

🎯 **GUI-Lens improves every evaluated base model on ScreenSpot-Pro**, reaching **87.9** with GPT-5.5 and
**82.3** with Claude Opus 4.7.

## :bar_chart: Main Results

Table 1. Accuracy (%) on all 1,581 ScreenSpot-Pro questions. The baselines use the same base model for every call.

| Base model | Direct | ZoomClick | BAMI | AdaZoom-GUI | UI-Zoomer | **GUI-Lens** |
|---|---|---|---|---|---|---|
| GPT-5.5 | 74.8 | - | - | - | - | **87.9** |
| Claude Opus 4.7 | 57.4 | - | - | - | - | **82.3** |
| DeepSeek-V4.1-Flash | 26.6 | 58.6 | 55.0 | 27.5 | 55.6 | **72.9** |
| GLM-5.3-Flash | 18.3 | 38.0 | 49.9 | 52.1 | - | **70.0** |
| MiniMax-M3 | 26.4 | - | - | - | - | **47.4** |

## :rescue_worker_helmet: Installation

1. Clone this repo:
```bash
git clone https://github.com/wangshirong123/GUI-Lens.git
cd GUI-Lens
```
2. Create an environment and install the dependencies (install the PyTorch build for your machine first if needed):
```bash
conda create -n gui_lens python=3.11
conda activate gui_lens
pip install -e .
```
3. Set the API keys of the base models you use (see `.env.example`): `DEEPSEEK_API_KEY`, `ARK_API_KEY`
(Volcengine Ark) or `OPENCODE_API_KEY` (OpenCode Go). GPT and Claude run through an OpenProgram login
(`openprogram providers login`).

The GPA-GUI-Detector weight is downloaded from Hugging Face on first use, and EasyOCR downloads its English and
Simplified Chinese models.

## :minidisc: Data Preparation

Download [ScreenSpot-Pro](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro) and lay it out as
`annotations/all.json` and `images/<id>.png`:
```bash
python eval/prepare_screenspot_pro.py --source ScreenSpot-Pro --dest data/ScreenSpot-Pro
```

## :checkered_flag: Evaluation on ScreenSpot-Pro

Run GUI-Lens with a base model of the paper and score it:
```bash
bash scripts/run_guilens_ssp.sh deepseek      # or glm, minimax
```

Or call the runner directly:
```bash
python run_guilens.py --model deepseek-v4.1-flash --config configs/guilens.yaml --thinking low \
    --data-dir data/ScreenSpot-Pro --output-dir runs/guilens_deepseek --all
python eval/score.py --data-dir data/ScreenSpot-Pro runs/guilens_deepseek/results.jsonl
```

- `--model`: `deepseek-v4.1-flash`, `glm-5.3-flash-ark`, `glm-5.3-flash-go`, `minimax-m3-go`, or any OpenProgram
  model together with `--provider` (e.g. `--provider openai-codex --model gpt-5.5`).
- `--config`: `configs/guilens.yaml` asks for pixel coordinates and `configs/guilens_norm.yaml` for relative
  [0, 1000] coordinates. `configs/direct*.yaml` is the single-shot baseline, and `configs/ablations/` holds the
  ablations of the paper.
- `--indexes 0-99` runs a subset, and `--resume` continues a stopped run in the same output directory.

## :balance_scale: Baselines

`baselines/run_baselines.py` reimplements training-free grounding methods for an API base model. Every localization
step uses the base model's own direct-click call, so the methods differ only in their search procedure.
```bash
bash scripts/run_baselines_ssp.sh deepseek    # or glm
```

| Method | Setting |
|---|---|
| `direct` | one click on the full screenshot |
| `zoomclick` | [ZoomClick](https://github.com/Princeton-AI2-Lab/ZoomClick): 2×2 pre-zoom (50 px agreement), depth 3, shrink 0.5, min crop 768 |
| `bami` | [BAMI](https://arxiv.org/abs/2605.06664): 2 rounds × 2 masked candidates, the paper's selector prompt, crop ratio 0.5 |
| `adazoom` | [AdaZoom-GUI](https://arxiv.org/abs/2603.17441): instruction refinement, click + box, zoom ×2 for small boxes |
| `som` | Set-of-Mark over the GUI-Lens detector (GPA-GUI-Detector + EasyOCR) elements |
| `vote` | 5 direct clicks (about the calls of GUI-Lens); the click with the most others within 1% of the diagonal wins |
| `uizoomer` | [UI-Zoomer](https://github.com/ZJU-REAL/UI-Zoomer): 8 sampled boxes, logprob confidence gate, Gaussian crop (σ×2.5) and refinement; needs logprobs |

## :file_folder: Repository Structure

```
run_guilens.py         GUI-Lens / single-shot runner
configs/               GUI-Lens, single-shot and ablation configurations
gui_lens/              method: planning/screenspot_locator.py (crop-and-click loop), perception/ (detector, OCR)
eval/                  ScreenSpot-Pro evaluator, dataset preparation and scoring
baselines/             baseline reimplementations and their prompts
scripts/               commands for the paper's ScreenSpot-Pro runs
```

## :+1: Acknowledgements

GUI-Lens uses [GPA-GUI-Detector](https://huggingface.co/Salesforce/GPA-GUI-Detector) for UI components,
[EasyOCR](https://github.com/JaidedAI/EasyOCR) for text and [OpenProgram](https://github.com/Fzkuji/OpenProgram) as the model runtime. We thank the authors of
[ScreenSpot-Pro](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding),
[ZoomClick](https://github.com/Princeton-AI2-Lab/ZoomClick), [BAMI](https://arxiv.org/abs/2605.06664),
[AdaZoom-GUI](https://arxiv.org/abs/2603.17441) and [UI-Zoomer](https://github.com/ZJU-REAL/UI-Zoomer) for their work.

## :memo: Citation

If you find this work useful in your research, please consider citing:
```bibtex
@article{fu2026guilens,
  title={GUI-Lens: Enabling Precise GUI Grounding for Off-the-Shelf VLMs via Coarse-to-Fine Cropping},
  author={Fu, Zichuan and Wang, Shirong and Zhang, Wenlin and Li, Guojing and Deng, Yimin and Gao, Jingtong and Qi, Junjia and Yan, Hanyu and Zheng, Yefeng and Li, Xiaopeng and Wang, Wanyu and Wu, Xian and Zhao, Xiangyu},
  journal={arXiv preprint arXiv:2608.03270},
  year={2026}
}
```

## License

MIT (see `LICENSE`). ScreenSpot-Pro, GPA-GUI-Detector and EasyOCR are external and follow their own licenses.

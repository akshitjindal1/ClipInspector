# CLIP-Inspector

[![CVPR 2026 Findings paper](docs/badges/cvpr-findings.svg)](https://openaccess.thecvf.com/content/CVPR2026F/html/Jindal_CLIP-Inspector_Model-Level_Backdoor_Detection_for_Prompt-Tuned_CLIP_via_OOD_Trigger_CVPRF_2026_paper.html) [![arXiv paper 2604.09101](docs/badges/arxiv.svg)](https://arxiv.org/abs/2604.09101)

CLIP-Inspector detects backdoors in prompt-tuned CLIP models and uses the reconstructed trigger to repair the prompt learner. This repository supports clean training and `badclip` checkpoints with the `ViT-B/16` backbone.

## Setup

The five-checkpoint evaluation used Python 3.12.7, PyTorch 2.9.1, torchvision 0.24.1, and CUDA 12.8. Run the commands below from the repository root. For another CUDA version, select the corresponding [PyTorch build](https://pytorch.org/get-started/previous-versions/).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install fiftyone==1.7.0 numpy==2.0.2 Pillow==12.1.0 ftfy==6.3.1 regex==2024.11.6 tqdm==4.67.1 PyYAML==6.0.2 matplotlib==3.10.0
```

The CLIP loader caches pretrained weights in `~/.cache/clip/`. Detection uses FiftyOne to load the Open Images V7 training split with up to 50,000 images, then selects up to 1,000 images after excluding images labeled with the candidate classes. Allow for the initial download if the data and weights are missing from your caches.

## Downstream datasets

Set `data_root` to the directory containing the datasets. The table gives the expected image paths relative to `data_root`. Put the split file in the dataset directory, alongside the image subdirectory. Use the published splits to reproduce the benchmark. [Dataset preparation and split files](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md)

| `dataset_name` | Image path | Split file |
|---|---|---|
| `caltech101` | `caltech-101/101_ObjectCategories/` | `split_zhou_Caltech101.json` |
| `dtd` | `dtd/images/` | `split_zhou_DescribableTextures.json` |
| `ucf101` | `ucf101/UCF-101-midframes/` | `split_zhou_UCF101.json` |
| `oxford_pets` | `oxford_pets/images/` | `split_zhou_OxfordPets.json` |
| `food101` | `food-101/images/` | `split_zhou_Food101.json` |

## Run detection and removal

**Use the five saved checkpoints.** Download [clipinspector-badclip-checkpoints.zip](https://github.com/akshitjindal1/ClipInspector/releases/download/badclip-checkpoints-v1/clipinspector-badclip-checkpoints.zip) from the GitHub release into the repository root, then extract it:

```bash
unzip -n clipinspector-badclip-checkpoints.zip
```

The archive contains `config.yaml`, `model_checkpoint.pth`, and `trigger_info.pth` for each of the five datasets listed above. They extract to `outputs/<dataset_name>/<run_name>/`. Each config uses `data_root: ./datasets`. If you keep the images elsewhere, edit `data_root` in the extracted run's `config.yaml` before detection. The Caltech101 command below uses the exact path from the archive.

The bundled files support detection, removal, clean accuracy evaluation, and attack success rate (ASR) evaluation with each original trigger. `trigger_info.pth` contains serialized Python objects. Only load checkpoints and trigger files from sources you trust.

To generate a clean baseline locally, set `dataset_name` and `data_root` in [the clean-training config](configs/clean_training.yaml) and run:

```bash
python main.py --config configs/clean_training.yaml --gpu_id 0
```

To train a `badclip` model, set `dataset_name` and `data_root` in [configs/attacks/badclip.yaml](configs/attacks/badclip.yaml), then run:

```bash
python main.py --config configs/attacks/badclip.yaml --gpu_id 0
```

Training writes `config.yaml`, `model_checkpoint.pth`, and `trigger_info.pth` to a timestamped run directory under `outputs/<dataset_name>/`.

**Detection.** Replace the example run path with your own. The detector performs one margin reconstruction per candidate class. With the default config it evaluates up to 50 classes. If the benchmark samples classes and the run config specifies the original attack target, it includes that target in the candidate set.

```bash
run_dir=outputs/caltech101/badclip_20250719_150149
python main_defense.py --output_dir "$run_dir" --gpu_id 0
```

Read `backdoor_detection_summary.txt` in the new `$run_dir/clip_inspector/<timestamp>/` directory. It records the model decision, highest-scoring class, and per-class ASR and mean margin loss. The class score is `z(ASR) - z(mean margin loss)`, with both quantities standardized across candidate classes. This release flags a model when its highest score exceeds `mean(scores) + 3 * std(scores)`. The paper reports a two-standard-deviation default. [Paper, Section 3.3](https://arxiv.org/html/2604.09101v1#S3.SS3)

**Removal.** Replace the timestamp below with the one from your detection run. Invoking removal selects the highest-scoring class even when the model score falls below the detection cutoff. The remover fine-tunes the prompt learner on clean images and copies with that class's reconstructed trigger, using their original labels.

```bash
detection_dir="$run_dir/clip_inspector/<timestamp>"
python main_removal.py --recon_dir "$detection_dir" --gpu_id 0 \
  --summary_out "$detection_dir/removal_summary.json"
```

The summary gives the repaired checkpoint directory (`removal_output_dir`), the selected class, and the before-and-after metrics. The checkpoint and `results.json` are under `removal_logs_cleanonly/`.

## Evaluation

In the five `badclip` runs on the datasets above, CLIP-Inspector flagged all five checkpoints at the three-standard-deviation cutoff. Mean original-trigger ASR fell from 98.73% to 6.14% after removal. Mean clean accuracy changed from 88.12% to 87.86%. Each dataset contributes equally to these means. Measuring false positives requires a separate evaluation on clean checkpoints.

## License

The project uses the [MIT License](LICENSE). The bundled OpenAI CLIP implementation and the CoCoOp code adapted in `classes.py` retain their upstream notices. [OpenAI CLIP license](clip/LICENSE) and [CoCoOp license](licenses/CoCoOp-LICENSE)

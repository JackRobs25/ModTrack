# ModTrack

ModTrack is a research-oriented codebase for multi-camera BEV tracking evaluation, focused on:
- datasets: `wildtrack`, `multiviewx`, `radarscenes`
- modes: `spatial`, `semantic`, `joint`
- evaluation command with only `--dataset` and `--mode`
- optional finetuning workflows for detector (YOLO), Lift depth, and OSNet semantics.

## Paper Summary

Multi-View Multi-Object Tracking (MV-MOT) aims to localize and maintain consistent identities of objects observed by multiple sensors. This task is challenging, as viewpoint changes and occlusion disrupt identity consistency across views and time. Recent end-to-end approaches address this by jointly learning 2D Bird's Eye View (BEV) representations and identity associations, achieving high tracking accuracy. However, these methods offer no principled uncertainty accounting and remain tightly coupled to their training configuration, limiting generalization across sensor layouts, modalities, or datasets without retraining. We propose ModTrack, a modular MV-MOT system that matches end-to-end performance while providing cross-modal, sensor-agnostic generalization and traceable uncertainty. ModTrack confines learning methods to just the \textit{Detection and Feature Extraction} stage of the MV-MOT pipeline, performing all fusion, association, and tracking with closed-form analytical methods. Our design reduces each sensor's output to calibrated position-covariance pairs $(\mathbf{z}, R)$; cross-view clustering and precision-weighted fusion then yield unified estimates $(\hat{\mathbf{z}}, \hat{R})$ for identity assignment and temporal tracking. A feedback-coupled, identity-informed Gaussian Mixture Probability Hypothesis Density (GM-PHD) filter with HMM motion modes uses these fused estimates to maintain identities under missed detections and heavy occlusion. ModTrack achieves 95.5 IDF1 and 91.4 MOTA on \textit{WildTrack}, surpassing all prior modular methods by over 21 points and rivaling the state-of-the-art end-to-end methods while providing deployment flexibility they cannot. Specifically, the same tracker core transfers unchanged to \textit{MultiviewX} and \textit{RadarScenes}, with only perception-module replacement required to extend to new domains and sensor modalities.

## Installation (uv)
```bash
export MODTRACK_ENV_ROOT=/path/to/modtrack-env
export UV_CACHE_DIR="${MODTRACK_ENV_ROOT}/.uv-cache"
export UV_LINK_MODE=copy
uv python install 3.9.21
mkdir -p "${MODTRACK_ENV_ROOT}"
uv venv --python 3.9.21 "${MODTRACK_ENV_ROOT}/.venv_uv"
source "${MODTRACK_ENV_ROOT}/.venv_uv/bin/activate"
uv pip install -r requirements.txt
uv pip install -e .
```

Paper reproduction environment:
- Python: `3.9.21`
- PyTorch: `2.8.0`
- TorchVision: `0.23.0`
- Ultralytics: `8.3.218`
- OpenCV: `4.11.0`
- TrackEval: `12c8791b303e0a0b50f753af204249e622d0281a`
- torchreid: `0.2.5`

General code compatibility is `3.9`-`3.13`, but use Python `3.9.21` for reproducing the paper tables. PyTorch wheels are not available for `3.14` in this setup.

Optional finetuning dependencies:
```bash
uv pip install -r requirements-train.txt
uv pip install -e .
```

For `--mode semantic` and `--mode joint`, install `requirements-train.txt` as well (OSNet uses `torchreid`).
If `torchreid` build isolation fails on your system, run:
```bash
uv pip install numpy scipy Cython gdown tensorboard torch torchvision Pillow opencv-python
uv pip install --no-build-isolation torchreid==0.2.5
```

```bash
export MODTRACK_ENV_ROOT=/path/to/runtime/storage
export UV_CACHE_DIR="${MODTRACK_ENV_ROOT}/.uv-cache"
export UV_LINK_MODE=copy
uv python install 3.9.21
mkdir -p "${MODTRACK_ENV_ROOT}"
uv venv --python 3.9.21 "${MODTRACK_ENV_ROOT}/.venv_uv"
source "${MODTRACK_ENV_ROOT}/.venv_uv/bin/activate"
uv pip install -r requirements-train.txt
uv pip install -e .
```

## Components
- Core evaluator pipeline centered on `evaluate_modtrack.py`
- Tracker core (`graph clustering`, `fusion`, `GM-PHD-HMM`)
- Dataset adapters for WildTrack, MultiviewX, RadarScenes
- Metrics backends:
  - `TrackEval`
  - `py-motmetrics`
  - `GOSPA`
  - `RadarScenes LSTQ`
- Optional finetuning entrypoints for YOLO / Lift / OSNet and semantics crop preparation

## Supported Datasets and Modes
- Datasets: `wildtrack`, `multiviewx`, `radarscenes`
- Modes: `spatial`, `semantic`, `joint`

Notes:
- `radarscenes` runs spatial tracking internally. If you pass `semantic` or `joint`, ModTrack warns and falls back to spatial behavior.

## Default Directory Layout

### Datasets
Place datasets in:
- `datasets/wildtrack/`
- `datasets/multiviewx/`
- `datasets/radarscenes/`

### Weights
Place checkpoints in:
- `weights/wildtrack/yolo.pt`
- `weights/wildtrack/lift.pt`
- `weights/wildtrack/osnet.pt`
- `weights/multiviewx/yolo.pt`
- `weights/multiviewx/lift.pt`
- `weights/multiviewx/osnet.pt`

ModTrack auto-discovers these paths. No weight path CLI flags are required.

### Optional Environment Overrides
- `MODTRACK_DATA_ROOT`: parent folder containing dataset subfolders
- `MODTRACK_WEIGHTS_ROOT`: parent folder containing weight subfolders

## Pretrained Weights
Download pretrained weights:
N/A (to maintain anonymity)

## Evaluation
Run evaluation with only dataset and mode:
```bash
modtrack eval --dataset wildtrack --mode spatial
modtrack eval --dataset wildtrack --mode semantic
modtrack eval --dataset wildtrack --mode joint

modtrack eval --dataset multiviewx --mode spatial
modtrack eval --dataset multiviewx --mode semantic
modtrack eval --dataset multiviewx --mode joint

modtrack eval --dataset radarscenes --mode spatial
```

Outputs are written under `results/<dataset>/<mode>/`.

Default behavior:
- `wildtrack` and `multiviewx` evaluate the default frame window `[360, 400)` by default.
- `radarscenes` keeps its benchmark path (spatial/LSTQ evaluation).

Metrics included:
- TrackEval MOT metrics
- py-motmetrics metrics
- GOSPA metrics
- RadarScenes LSTQ metrics
- Inference timing and calibration statistics

## Dataset Adapter Integration Points
- Adapter registry: `modtrack/data/adapters/registry.py`
- Dataset adapters: `modtrack/data/adapters/wildtrack.py`, `modtrack/data/adapters/multiviewx.py`, `modtrack/data/adapters/radarscenes.py`

To add a future dataset, add a new adapter and register it in the registry.

## Optional Finetuning
Finetuning is separate from evaluation and opt-in.

### 1) YOLO finetune
Recommended baseline checkpoint: `yolo11x.pt`.
YOLO finetuning always starts from that baseline.

```bash
modtrack finetune yolo --dataset wildtrack --data-root /path/to/wildtrack --output-dir /path/to/workdir
modtrack finetune yolo --dataset multiviewx --data-root /path/to/multiviewx --output-dir /path/to/workdir
```
Result is copied to:
- `weights/<dataset>/yolo.pt`

### 2) Lift finetune
Use an LSS pretrained checkpoint as the initialization for finetuning.
Reference repo: https://github.com/nv-tlabs/lift-splat-shoot
Lift finetuning always starts from:
- `weights/lss/model525000.pt`

If that file is missing, Lift finetune fails with an explicit error.

```bash
modtrack finetune lift --dataset wildtrack --data-root /path/to/wildtrack --output-dir /path/to/workdir
modtrack finetune lift --dataset multiviewx --data-root /path/to/multiviewx --output-dir /path/to/workdir
```
Result is copied to:
- `weights/<dataset>/lift.pt`

### 3) Semantics data preparation (person crops)
```bash
modtrack finetune semantics-prepare --dataset wildtrack --data-root /path/to/wildtrack --output-dir /path/to/workdir
modtrack finetune semantics-prepare --dataset multiviewx --data-root /path/to/multiviewx --output-dir /path/to/workdir
```
This writes crops into `<output-dir>/person_crops/` (or directly into `<output-dir>` if it already ends with `person_crops`).

### 4) OSNet semantics finetune
This command expects prepared crops under `<output-dir>/person_crops` (or directly at `<output-dir>` if it points to the `person_crops` folder).
```bash
modtrack finetune semantics --dataset wildtrack --data-root /path/to/wildtrack --output-dir /path/to/workdir
modtrack finetune semantics --dataset multiviewx --data-root /path/to/multiviewx --output-dir /path/to/workdir
```
Result is copied to:
- `weights/<dataset>/osnet.pt`

## One-Command Re-evaluation after Finetuning
After finetuning copies new checkpoints into `weights/`, evaluation automatically uses them:
```bash
modtrack eval --dataset wildtrack --mode joint
```

## Notes
- Large pretrained weights are intentionally excluded from git.
- If required assets are missing, CLI reports explicit actionable errors.

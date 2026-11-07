<div align="center">

# 🌿 Transfer Learning for Plant Disease Detection

**A comparative study of EfficientNetV2, MobileNetV4, ConvNeXtV2 and ViT-Base**
**under three adaptation regimes: Feature Extraction · Partial Fine-Tuning · Full Fine-Tuning**

![Course](https://img.shields.io/badge/Course-Deep%20Learning-blue)
![Program](https://img.shields.io/badge/Master-ISI%20S3-green)
![Year](https://img.shields.io/badge/Academic%20Year-2026--2027-green)
![Framework](https://img.shields.io/badge/TensorFlow%20%2B%20Keras%203-orange)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![Quality Gate](https://img.shields.io/badge/Quality%20Gate-SonarQube-lightgrey)

Département d'Informatique · Faculté des Sciences Semlalia, Université Cadi Ayyad, Marrakech
Supervised by **Pr. Mustapha AATILA**

</div>

---

## 📌 Overview

Plant diseases cause major agricultural losses, and manual diagnosis is slow, subjective and
hard to scale. CNNs need large labelled datasets, which agricultural data rarely provides.
This project asks one question:

> **Can Transfer Learning improve plant disease classification, and which model and
> adaptation strategy gives the best accuracy / efficiency / robustness trade-off?**

We compare modern pre-trained architectures against a **CNN trained from scratch**,
under identical training conditions, and explain predictions with **Grad-CAM**.

## 🎯 Contributions

- Comparison of several pre-trained architectures (CNNs and a Transformer)
- Study of the impact of fine-tuning depth (3 strategies)
- Performance analysis across multiple plant diseases
- Interpretability of predictions via Grad-CAM

## 🧪 Experimental Design

| Axis | Values |
|---|---|
| **Models** | EfficientNetV2 · MobileNetV4 · ConvNeXtV2 · ViT-Base (Transformer) · CNN from scratch (baseline) |
| **Strategies** | Feature extraction (frozen backbone) · Partial fine-tuning (last blocks) · Full fine-tuning |
| **Dataset** | [Plant Leaf Diseases (Kaggle)](https://www.kaggle.com/datasets/moazawadali/plant-leaf-diseases) |
| **Metrics** | Accuracy · Precision · Recall · F1-score · ROC / AUC · Confusion matrix · #Params · Training time |
| **Evaluation** | Train / validation / test split + cross-validation |
| **Explainability** | Grad-CAM (diseased regions, detected symptoms) |

> All models are trained under **the same conditions** (optimizer, learning rate,
> batch size, epochs, augmentation, seed) so the comparison is fair.

## 🛠️ Technical Stack

| Layer | Choice | Why it fits this project |
|---|---|---|
| **Deep learning** | **TensorFlow 2 + Keras 3**, with **KerasHub** (`keras-hub`) for pretrained backbones | Keras `fit()` / callbacks give a short, readable training loop; `tf.data` gives fast input pipelines. Pretrained-model availability per architecture is detailed below. |
| **Training speed** | `tf.data` (`cache`, `prefetch(AUTOTUNE)`), Keras mixed precision, XLA (`jit_compile=True`, optional) | Keeps the free Colab/Kaggle GPU busy, which makes 12+ runs realistic. |
| **Data & augmentation** | `tf.data` + Keras preprocessing layers (`RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomContrast`), scikit-learn (`StratifiedKFold`) | Augmentation runs on the GPU inside the model/pipeline. Stratified splits and k-fold cross-validation as required by the brief. |
| **Explainability** | Grad-CAM implemented with `tf.GradientTape` (optionally `tf-keras-vis`) | Short custom implementation for the CNNs. For ViT-Base, use the last transformer block's patch tokens reshaped to a 14x14 grid (or attention rollout). |
| **Run logging** | Plain files per run in `results/<run>/` (`config.yaml`, `history.csv`, `metrics.json`, figures) | No server or account needed. Files are downloaded from the Kaggle/Colab output and compared with a small pandas script. |
| **Data versioning** | **DVC** (or Kaggle as the source of truth) | Reproducible dataset + split manifest without committing images to Git. |
| **Config** | YAML + OmegaConf/Hydra | One shared `base.yaml` guarantees "same training conditions" across models. |
| **Demo app** | **Gradio** on Hugging Face Spaces | Upload a leaf, get top-k prediction + Grad-CAM overlay. Free hosting, shareable link for the jury. |
| **Model export** | **TensorFlow Lite** (+ post-training quantization) | Native to TensorFlow. Backs the "smartphone deployment" perspective with real latency and file-size numbers. |
| **Model hosting** | Hugging Face Hub | Versioned Keras weights for the demo app. |
| **Code audit** | **SonarQube** (see dedicated section) | The only automated quality/security tool; regular commits give a clean Git history, a graded deliverable. |
| **Packaging** | `uv` or `pip-tools` lockfile, optional Docker | Same environment on WSL, Kaggle and Colab. |

### Pretrained model availability in TensorFlow / Keras

| Model | Source in the TF/Keras ecosystem | Status |
|---|---|---|
| **EfficientNetV2** | `keras.applications` (`EfficientNetV2B0`...`L`) or KerasHub `EfficientNetImageClassifier` presets | ✅ Official |
| **ViT-Base** | KerasHub `ViTImageClassifier`, preset `vit_base_patch16_224_imagenet` (ImageNet-21k variant also available) | ✅ Official |
| **ConvNeXtV2** | `kerasformers` (Keras 3 conversion of the timm weights, runs on the TensorFlow backend). `keras.applications` only ships ConvNeXt **V1**. | ⚠️ Third-party, validate before relying on it |
| **MobileNetV4** | No official TensorFlow weights found at the time of writing (the public MobileNetV4 weights are in timm / PyTorch). | ❌ Verify `kerasformers` coverage, otherwise see fallback |

> **Fallback for MobileNetV4:** use **MobileNetV3-Large** (`keras.applications` / KerasHub) as the
> lightweight mobile-class model, and state this substitution explicitly in the paper and to the
> supervisor, since the brief's results table lists MobileNetV4. Decide this **before** the first
> training run so all comparisons stay consistent.

### Compute strategy

| Stage | Environment | Notes |
|---|---|---|
| Development and debugging | **WSL2 (Ubuntu) or native Linux** + VS Code | Fast iteration on a small data subset. TensorFlow has no native-Windows GPU support (versions 2.11+), so use WSL2/Linux for a local GPU. |
| Main training (12+ runs) | **Kaggle Notebooks** (preferred) or **Google Colab** | The dataset already lives on Kaggle (no download). Free GPU, weekly quota. Checkpoint every epoch (sessions can disconnect). |
| Run logs | `results/<run>/` files saved to the Kaggle/Colab output | Download them after each session; nothing to host. |
| Demo | Hugging Face Spaces (Gradio) | CPU is enough for inference. |
| Databricks / cloud ML platforms | **Optional, not recommended for training here** | Free tiers have limited GPU access. Overkill for a dataset of this size. Not needed for this project. |

```text
# requirements.txt (pin exact versions with a lockfile before the final run)
tensorflow                  # TF 2.x runtime (use the GPU build on WSL2/Linux)
keras>=3                    # Keras 3 API
keras-hub                   # EfficientNetV2, ViT-Base presets
kerasformers                # ConvNeXtV2 (and possibly MobileNetV4), verify availability
scikit-learn                # metrics, StratifiedKFold
pandas                      # run logs (CSV) and results tables
omegaconf                   # config management
gradio                      # demo app
imagehash                   # near-duplicate detection (data leakage check)
nbstripout                  # strip notebook outputs before commit (git filter)
```

### Rigor safeguards (what makes the comparison trustworthy)

- **Leakage check:** perceptual-hash deduplication *before* splitting, so augmented copies of the same leaf never cross train/val/test.
- **Fair comparison:** same input size (224), seed, epochs, augmentation and budget for all models. Learning rate chosen per model from a small, equal-size sweep (ViT needs a lower LR than CNNs).
- **Clear strategy definitions:** feature extraction = frozen backbone + BatchNorm layers in inference mode (`training=False`); partial FT = last N stages unfrozen (same fraction for every model); full FT = everything trainable.
- **Statistics:** several seeds, report mean ± std, McNemar test for the top models.
- **Imbalance:** report macro-F1 and per-class recall, not only accuracy.
- **Efficiency beyond parameters:** FLOPs (TensorFlow profiler), CPU latency, TFLite file size, peak memory.
- **Robustness:** accuracy under blur, noise, brightness/contrast shifts, plus an out-of-distribution test on field-condition images.

## 🔒 Code Audit with SonarQube

**SonarQube is the single audit tool of this project.** The code is analysed on SonarQube Cloud
(automatic analysis of the GitHub repo, if available on your plan) or with `sonar-scanner` run locally.
The quality gate must be green before the final submission.

| Item | Choice |
|---|---|
| **Tool** | **SonarQube Cloud** (hosted) or self-hosted SonarQube Community Build |
| **Analysed** | `src/`, `app/`, `scripts/` (Python) |
| **Reports** | Bugs, vulnerabilities, security hotspots, code smells, duplication, cognitive complexity (no coverage: the project has no automated tests) |
| **Config** | `sonar-project.properties` at the repo root |
| **Token** | `SONAR_TOKEN` kept in an environment variable, never in code |
| **Dashboard** | Link to be added in the badge above and in the paper |

> SonarQube analyses **source code only**. It does not scan dependency CVEs, secrets in Git
> history or model files, so the practices below are followed manually.

### Quality gate (suggested thresholds)

```text
Sonar quality gate on NEW code:
  - 0 blocker / critical issues          # hard fail
  - 0 security hotspots left unreviewed  # review each one: fix, or mark safe with a reason
  - duplication  < 3%
Coverage condition: disabled          # no automated tests in this project
Submission rule: gate green on the SonarQube dashboard before the final push
```

### Manual security practices (outside SonarQube's scope)

- **Secrets:** Kaggle (`kaggle.json`) and Hugging Face tokens live only in environment variables, never in notebooks, configs or Git history.
- **Notebooks:** strip outputs before commit (`nbstripout`). Outputs often leak paths, tokens and data samples.
- **Model files:** save as `.keras` / `.weights.h5`, load with `safe_mode=True` (Keras default), pin the source and revision of every pretrained checkpoint. Never load models from untrusted sources (legacy formats and custom layers can execute code).
- **Dependencies:** pin versions with a lockfile and review upgrades before the final run.
- **Dataset provenance:** record the Kaggle dataset version, licence and the checksum of the downloaded archive.
- **Gradio demo:** limit upload size and file types (images only), re-encode images with Pillow before inference, keep `share=False` and `debug=False` in production, pin the `gradio` version.
- **Repo hygiene:** branch protection on `main`, regular small commits, `SECURITY.md` describing how to report a vulnerability.

### Run the analysis locally

```bash
# Install nbstripout as a git filter once (strips notebook outputs on commit, no pre-commit needed)
nbstripout --install

# Analyse with SonarScanner (needs the CLI and a token from your SonarQube Cloud account)
export SONAR_TOKEN=<your-token>              # PowerShell: $env:SONAR_TOKEN = "<your-token>"
sonar-scanner                                # reads sonar-project.properties
```

## 🔁 Workflow

```text
Data ─▶ Pre-processing ─▶ Split (train/val/test) ─▶ Augmentation
     (resize, normalize)                              │
                                                      ▼
   Predictions ◀─ Evaluation ◀─ Training (+ cross-validation) ◀─ Model + strategy
                      │
                      └─▶ Metrics · ROC/AUC · Confusion matrix · Grad-CAM
```

## 📂 Repository Layout

```text
.
├── README.md                 # Project description, dependencies, how to run
├── requirements.txt          # Dependencies (+ lockfile)
├── sonar-project.properties  # SonarQube project key and analysed sources
├── SECURITY.md               # How to report a vulnerability
├── Dockerfile                # Optional: identical env everywhere
├── dvc.yaml                  # Optional: data + split versioning
├── configs/                  # One YAML per experiment (model x strategy)
│   └── base.yaml             # Shared hyper-parameters (same for all models!)
├── data/                     # Not tracked by git (see "Quick Start")
├── src/
│   ├── datasets.py           # Loading, dedup, stratified split, augmentation
│   ├── models.py             # Keras backbones (KerasHub / kerasformers) + CNN from scratch
│   ├── strategies.py         # freeze / partial unfreeze / full fine-tune
│   ├── train.py              # Keras training (mixed precision, checkpoints, CSV/JSON run logs)
│   ├── evaluate.py           # Metrics, confusion matrix, ROC/AUC
│   ├── robustness.py         # Corruption + out-of-distribution tests
│   ├── benchmark.py          # FLOPs, latency, TFLite size
│   └── gradcam.py            # Grad-CAM visualisations
├── app/
│   └── app.py                # Gradio demo (prediction + Grad-CAM overlay)
├── scripts/
│   └── export_tflite.py      # TFLite export + post-training quantization
├── notebooks/                # EDA and Kaggle/Colab entry points
├── results/                  # One folder per run: config, history.csv, metrics.json, figures
├── paper/                    # English paper (LaTeX / Word) + figures
└── CONTRIBUTIONS.md          # Who did what (required deliverable)
```

## 🚀 Quick Start

```bash
# 1. Clone and enter the project repo
git clone https://github.com/<ORG_NAME>/<REPO_NAME>.git
cd <REPO_NAME>

# 2. Create an isolated environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset (needs a Kaggle API token in ~/.kaggle/kaggle.json)
kaggle datasets download -d moazawadali/plant-leaf-diseases -p data/ --unzip

# 5. Train one experiment (model x strategy)
python -m src.train --config configs/base.yaml \
    --model efficientnetv2 \
    --strategy partial_ft      # feature_extraction | partial_ft | full_ft

# 6. Evaluate and generate figures (metrics, ROC, confusion matrix)
python -m src.evaluate --run results/efficientnetv2_partial_ft

# 7. Grad-CAM explanations
python -m src.gradcam --run results/efficientnetv2_partial_ft --n 12

# 8. Efficiency + robustness benchmarks (FLOPs, latency, corruptions)
python -m src.benchmark --run results/efficientnetv2_partial_ft
python -m src.robustness --run results/efficientnetv2_partial_ft

# 9. Launch the demo locally (same app is deployed on Hugging Face Spaces)
python app/app.py

# 10. Strip notebook outputs on commit, then run the SonarQube analysis (see "Code Audit with SonarQube")
nbstripout --install
sonar-scanner
```

## 📊 Results

> ⏳ To be filled once experiments are finished. Same table is reported per strategy.

| Model | Accuracy | Precision | Recall | F1-score | AUC | #Params | Training time |
|---|---|---|---|---|---|---|---|
| CNN from scratch | – | – | – | – | – | – | – |
| EfficientNetV2 | – | – | – | – | – | – | – |
| MobileNetV4 | – | – | – | – | – | – | – |
| ConvNeXtV2 | – | – | – | – | – | – | – |
| ViT-Base | – | – | – | – | – | – | – |

**Efficiency & robustness (extra table, supports "lightest" and "most robust"):**

| Model | FLOPs | CPU latency (ms) | TFLite size (MB) | Acc. under blur / noise / brightness | OOD accuracy |
|---|---|---|---|---|---|
| EfficientNetV2 | – | – | – | – | – |
| MobileNetV4 | – | – | – | – | – |
| ConvNeXtV2 | – | – | – | – | – |
| ViT-Base | – | – | – | – | – |

**Questions answered in the paper:** which model is the most accurate? the lightest?
the most robust?

## 🔍 Interpretability

Grad-CAM heatmaps highlight the leaf regions driving each prediction (lesions, spots,
discoloration), to check that models look at actual symptoms and not background artefacts.

## 🖥️ Live Demo

A Gradio app (Hugging Face Spaces) lets anyone upload a leaf image and get the predicted disease, top-k confidences and the Grad-CAM overlay. Link: _to be added_.

## ⚠️ Limitations & Perspectives

- **Limits:** controlled-condition dataset, limited generalisation to real field images, possible class imbalance
- **Next steps:** multimodal learning, smartphone deployment, real-time detection with drones

## ✅ Deliverables Checklist

- [ ] Original work
- [ ] Git repo: clean commented code, `README.md` (description, dependencies, run procedure), regular commit history
- [ ] SonarQube quality gate green, no secrets in code or Git history
- [ ] Paper **in English** following the prescribed structure (Abstract → Conclusion, with GitHub link inside)
- [ ] Team contributions table at the end of the paper
- [ ] Final submission (code + paper) **no later than the DL exam day** (no changes accepted afterwards)

## 👥 Team

| Student | Contribution |
|---|---|
| Student 1 | _tasks_ |
| Student 2 | _tasks_ |
| Student 3 | _tasks_ |
| Student 4 | _tasks_ |

## 📚 References

_Maintained in `paper/references.bib`._

---

<div align="center">
<sub>Master ISI S3 · Deep Learning (Transfer Learning) · FSSM, Université Cadi Ayyad · 2025–2026</sub>
</div>

<div align="center">

# 🌿 Transfer Learning for Plant Disease Detection

**A comparative study of EfficientNetV2, MobileNetV4, ConvNeXtV2 and ViT-Base**
**under three adaptation regimes: Feature Extraction · Partial Fine-Tuning · Full Fine-Tuning**

![Course](https://img.shields.io/badge/Course-Deep%20Learning-blue)
![Program](https://img.shields.io/badge/Master-ISI%20S3-green)
![Year](https://img.shields.io/badge/Academic%20Year-2025--2026-orange)
![Framework](https://img.shields.io/badge/PyTorch%20%2B%20timm-red)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-informational)
![Security](https://img.shields.io/badge/Security-CodeQL%20%7C%20Bandit%20%7C%20pip--audit-success)
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
| **Deep learning** | **PyTorch + [timm](https://github.com/huggingface/pytorch-image-models)** | All 4 required models (EfficientNetV2, MobileNetV4, ConvNeXtV2, ViT-Base) ship with pretrained weights behind one API: `timm.create_model(name, pretrained=True)`. TensorFlow/Keras has no equivalent for MobileNetV4 / ConvNeXtV2. |
| **Training speed** | Mixed precision (AMP), `channels_last`, `torch.compile` (optional) | Roughly 2x faster on a free Colab/Kaggle GPU, which makes 12+ runs realistic. |
| **Data & augmentation** | torchvision `transforms.v2` or Albumentations, scikit-learn (`StratifiedKFold`) | Stratified splits and k-fold cross-validation as required by the brief. |
| **Explainability** | `pytorch-grad-cam` | Supports CNNs and ViT (ViT needs a `reshape_transform`). |
| **Experiment tracking** | **MLflow** (local, free) | Logs params, curves, metrics, artifacts per run. Same API as managed MLflow on Databricks if you migrate later. |
| **Data versioning** | **DVC** (or Kaggle as the source of truth) | Reproducible dataset + split manifest without committing images to Git. |
| **Config** | YAML + OmegaConf/Hydra | One shared `base.yaml` guarantees "same training conditions" across models. |
| **Demo app** | **Gradio** on Hugging Face Spaces | Upload a leaf, get top-k prediction + Grad-CAM overlay. Free hosting, shareable link for the jury. |
| **Model export** | ONNX / ONNX Runtime (+ INT8 quantization) | Backs the "smartphone deployment" perspective with real latency and size numbers. |
| **Model hosting** | Hugging Face Hub | Versioned weights for the demo app. |
| **Quality & CI** | Ruff, pre-commit, pytest (smoke tests), GitHub Actions | Clean commit history and a reproducible repo, both graded deliverables. |
| **Packaging** | `uv` or `pip-tools` lockfile, optional Docker | Same environment on WSL, Colab and CI. |

### Compute strategy

| Stage | Environment | Notes |
|---|---|---|
| Development, debugging, unit tests | **WSL2 (Ubuntu) or native Linux** + VS Code | Fast iteration on a small data subset. |
| Main training (12+ runs) | **Kaggle Notebooks** (preferred) or **Google Colab** | The dataset already lives on Kaggle (no download). Free GPU, weekly quota. Checkpoint every epoch (sessions can disconnect). |
| Tracking server | MLflow local file store, synced as an artifact | No cloud account needed. |
| Demo | Hugging Face Spaces (Gradio) | CPU is enough for inference. |
| Databricks / cloud ML platforms | **Optional, not recommended for training here** | Free tiers have limited GPU access. Overkill for a dataset of this size. Worth it only if you want managed MLflow for the report. |

```text
# requirements.txt (pin exact versions with a lockfile before the final run)
torch, torchvision          # core framework
timm>=1.0                   # EfficientNetV2, MobileNetV4, ConvNeXtV2, ViT-Base
scikit-learn                # metrics, StratifiedKFold
pytorch-grad-cam            # Grad-CAM (CNN + ViT)
mlflow                      # experiment tracking
omegaconf                   # config management
albumentations              # augmentation (or torchvision.transforms.v2)
gradio                      # demo app
onnx, onnxruntime           # export + latency benchmark
fvcore                      # FLOPs counting
imagehash                   # near-duplicate detection (data leakage check)
ruff, pytest, pre-commit    # quality
safetensors                 # safe weight serialization (no pickle)
```

```text
# requirements-dev.txt  (audit + test tooling, not needed at inference time)
ruff, mypy                  # lint, format, types
bandit, semgrep             # SAST
pip-audit                   # dependency CVEs
pytest, pytest-cov          # tests + coverage.xml for SonarQube
nbstripout                  # strip notebook outputs before commit
modelscan                   # unsafe model-file detection
```

### Rigor safeguards (what makes the comparison trustworthy)

- **Leakage check:** perceptual-hash deduplication *before* splitting, so augmented copies of the same leaf never cross train/val/test.
- **Fair comparison:** same input size (224), seed, epochs, augmentation and budget for all models. Learning rate chosen per model from a small, equal-size sweep (ViT needs a lower LR than CNNs).
- **Clear strategy definitions:** feature extraction = frozen backbone + BN in eval mode; partial FT = last N stages unfrozen (same fraction for every model); full FT = everything trainable.
- **Statistics:** several seeds, report mean ± std, McNemar test for the top models.
- **Imbalance:** report macro-F1 and per-class recall, not only accuracy.
- **Efficiency beyond parameters:** FLOPs, CPU latency, ONNX file size, peak memory.
- **Robustness:** accuracy under blur, noise, brightness/contrast shifts, plus an out-of-distribution test on field-condition images.

## 🔒 Code Audit & Security

Quality and security are checked **automatically on every push and pull request**
(GitHub Actions), and the same checks run locally before each commit (pre-commit).

| Concern | Tool | Scope in this project |
|---|---|---|
| **Quality gate + dashboard** | **SonarQube Cloud** (or self-hosted SonarQube Community Build) | Bugs, code smells, duplication, complexity, coverage, security hotspots. One public dashboard to show the jury. |
| **SAST (primary)** | **GitHub CodeQL** | Deep semantic analysis of Python, results appear in the repo's *Security* tab. |
| **SAST (Python-specific)** | **Bandit** + **Semgrep** | `pickle`/`eval`/`subprocess` misuse, hard-coded secrets, unsafe deserialization, insecure file handling. |
| **Dependency vulnerabilities** | **pip-audit** + **Dependabot** | CVEs in `torch`, `gradio`, `pillow`, etc. and automatic update PRs. |
| **Secret scanning** | **Gitleaks** (pre-commit) + GitHub secret scanning / push protection | Kaggle token, Hugging Face token, MLflow credentials. |
| **Lint, format, types** | **Ruff** + **mypy** (or pyright) | Style, dead imports, type errors, enforced by pre-commit. |
| **Tests & coverage** | **pytest** + **pytest-cov** | Smoke tests per model, unit tests for splits/metrics, coverage sent to Sonar. |
| **Model file safety** | **safetensors**, `torch.load(..., weights_only=True)`, **ModelScan** | Pickle-based checkpoints can execute code on load. Never load untrusted `.pt`/`.pth` files. |
| **Container (if Docker used)** | **Hadolint** + **Trivy** | Dockerfile best practices and image CVE scan. |
| **Supply chain (optional)** | **OpenSSF Scorecard**, SBOM via `cyclonedx-py` | Repo hygiene score and a bill of materials for the paper's reproducibility section. |

> **Why not SonarQube alone?** SonarQube is the best *dashboard* and quality gate, but its
> free tiers have limited security rules for Python. CodeQL + Bandit/Semgrep cover the
> security side, `pip-audit` covers dependencies, Gitleaks covers secrets. Use them together.

### ML-specific security checklist

- **Secrets:** Kaggle (`kaggle.json`), Hugging Face and MLflow tokens live only in environment variables / CI secrets. Never in notebooks, configs or Git history.
- **Notebooks:** strip outputs before commit (`nbstripout`). Outputs often leak paths, tokens and data samples.
- **Model files:** publish weights as `safetensors`, load with `weights_only=True`, pin the exact source and revision of every pretrained checkpoint.
- **Dataset provenance:** record the Kaggle dataset version and licence, and the checksum of the downloaded archive.
- **Gradio demo:** limit upload size and file types (images only), re-encode images with Pillow before inference, add a request queue / rate limit, keep `share=False` and `debug=False` in production, pin the `gradio` version (the library has had security advisories).
- **Reproducibility = integrity:** locked dependencies, fixed seeds, versioned split manifest.
- **Repo hygiene:** branch protection on `main`, required status checks, signed commits (optional), `SECURITY.md` describing how to report a vulnerability.

### Quality gate (suggested thresholds)

```text
Sonar quality gate on NEW code:
  - 0 blocker / critical issues         # hard fail
  - 0 security hotspots left unreviewed # review each one: fix or mark safe with a reason
  - duplication  < 3%
  - coverage     >= 60% on src/         # training loops are hard to cover; target data, metrics, strategies
CI must pass before merge:
  - ruff, mypy, bandit, pip-audit, gitleaks, pytest, CodeQL
```

### Run the audit locally

```bash
# Lint, format and type-check
ruff check . && ruff format --check .      # style + common bugs
mypy src/                                   # type errors

# Security (SAST + dependencies + secrets)
bandit -r src/ app/ -ll                     # -ll = medium severity and above
semgrep --config auto src/ app/             # community Python/security rules
pip-audit -r requirements.txt               # known CVEs in dependencies
gitleaks detect --source . --no-banner      # secrets in files and git history

# Tests + coverage (feeds SonarQube)
pytest --cov=src --cov-report=xml           # writes coverage.xml

# Optional: scan the model file and the container
modelscan -p results/best_model.pt          # detects unsafe pickle payloads
trivy image <ORG_NAME>/<REPO_NAME>:latest   # image vulnerabilities
```

```yaml
# .github/workflows/ci.yml  (excerpt: pin actions to the current major version or a commit SHA)
name: CI
on: [push, pull_request]
permissions:
  contents: read              # least privilege for the default token
jobs:
  quality-security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }              # full history: needed by Sonar and Gitleaks
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: ruff check . && mypy src/        # lint + types
      - run: bandit -r src/ app/ -ll          # SAST
      - run: pip-audit -r requirements.txt    # dependency CVEs
      - run: pytest --cov=src --cov-report=xml
      - uses: SonarSource/sonarqube-scan-action@v5   # check the latest major version
        env:
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}     # stored as a repo secret, never in code
  codeql:                                             # separate workflow file in practice
    uses: ./.github/workflows/codeql.yml
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
├── requirements-dev.txt      # Audit/test tools (ruff, mypy, bandit, pip-audit, pytest-cov)
├── pyproject.toml            # Ruff / mypy / pytest config
├── sonar-project.properties  # SonarQube project key, sources, coverage report path
├── SECURITY.md               # How to report a vulnerability
├── .pre-commit-config.yaml   # Ruff, gitleaks, nbstripout on every commit
├── .gitleaks.toml            # Secret-scanning rules / allowlist
├── .github/
│   ├── dependabot.yml        # Automatic dependency update PRs
│   └── workflows/
│       ├── ci.yml            # Lint, types, SAST, dependency audit, tests, Sonar
│       └── codeql.yml        # CodeQL analysis (push, PR, weekly schedule)
├── Dockerfile                # Optional: identical env everywhere
├── dvc.yaml                  # Optional: data + split versioning
├── configs/                  # One YAML per experiment (model x strategy)
│   └── base.yaml             # Shared hyper-parameters (same for all models!)
├── data/                     # Not tracked by git (see "Quick Start")
├── src/
│   ├── datasets.py           # Loading, dedup, stratified split, augmentation
│   ├── models.py             # timm backbones + CNN from scratch
│   ├── strategies.py         # freeze / partial unfreeze / full fine-tune
│   ├── train.py              # Training loop (AMP, checkpoints, MLflow logging)
│   ├── evaluate.py           # Metrics, confusion matrix, ROC/AUC
│   ├── robustness.py         # Corruption + out-of-distribution tests
│   ├── benchmark.py          # FLOPs, latency, ONNX size
│   └── gradcam.py            # Grad-CAM visualisations
├── app/
│   └── app.py                # Gradio demo (prediction + Grad-CAM overlay)
├── scripts/
│   └── export_onnx.py        # ONNX export + INT8 quantization
├── tests/                    # Smoke tests (one forward/backward pass per model)
├── notebooks/                # EDA and Kaggle/Colab entry points
├── results/                  # Curves, tables, confusion matrices, Grad-CAM figures
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

# 9. Browse all experiments (curves, metrics, artifacts)
mlflow ui --port 5000

# 10. Launch the demo locally (same app is deployed on Hugging Face Spaces)
python app/app.py

# 11. Install hooks once, then audit before pushing (see "Code Audit & Security")
pip install -r requirements-dev.txt && pre-commit install
pre-commit run --all-files
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

| Model | FLOPs | CPU latency (ms) | ONNX size (MB) | Acc. under blur / noise / brightness | OOD accuracy |
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
- [ ] Quality gate green (CI, CodeQL, SonarQube), no secrets in code or Git history, dependencies audited
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
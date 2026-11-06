<#
.SYNOPSIS
    Creates the repository layout for the Plant Disease Transfer Learning project.

.DESCRIPTION
    Idempotent: existing files are NEVER overwritten, so it is safe to re-run.
    Works on Windows PowerShell 5.1 and PowerShell 7+ (files written as UTF-8 without BOM, LF endings).

.EXAMPLE
    .\New-ProjectLayout.ps1                                  # creates .\plant-disease-transfer-learning
    .\New-ProjectLayout.ps1 -Root D:\work\plant-disease -GitInit
    .\New-ProjectLayout.ps1 -WhatIf                          # dry run, shows what would be created
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$Root = ".\plant-disease-transfer-learning",  # target folder (created if missing)
    [switch]$GitInit                                      # also run "git init -b main"
)

$ErrorActionPreference = "Stop"
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

# ---------------------------------------------------------------------------
# 1. Directories (empty ones get a .gitkeep so Git tracks them)
# ---------------------------------------------------------------------------
$Directories = @(
    ".github/workflows",   # CI + CodeQL
    "configs",             # one YAML per experiment (model x strategy)
    "data",                # NOT tracked (dataset downloaded here)
    "src",                 # training / evaluation code
    "app",                 # Gradio demo
    "scripts",             # export + utilities
    "tests",               # smoke tests
    "notebooks",           # EDA, Kaggle/Colab entry points
    "results",             # curves, tables, figures
    "paper"                # English paper + figures
)
$KeepDirs = @("data", "notebooks", "results", "paper")

# ---------------------------------------------------------------------------
# 2. Files -> one-line purpose (used to generate a starter header per file)
# ---------------------------------------------------------------------------
$Files = [ordered]@{
    "README.md"                         = "Project description, dependencies, how to run"
    "requirements.txt"                  = $null   # seeded below
    "requirements-dev.txt"              = $null   # seeded below
    "pyproject.toml"                    = "Ruff / mypy / pytest configuration"
    "sonar-project.properties"          = "SonarQube project key, sources, coverage report path"
    "SECURITY.md"                       = $null   # seeded below
    ".pre-commit-config.yaml"           = "Ruff, gitleaks, nbstripout hooks"
    ".gitleaks.toml"                    = "Secret-scanning rules / allowlist"
    ".gitignore"                        = $null   # seeded below
    "Dockerfile"                        = "Optional: identical environment everywhere"
    "dvc.yaml"                          = "Optional: data + split versioning"
    "CONTRIBUTIONS.md"                  = "Who did what (required deliverable)"
    ".github/dependabot.yml"            = $null   # seeded below
    ".github/workflows/ci.yml"          = "Lint, types, SAST, dependency audit, tests, Sonar"
    ".github/workflows/codeql.yml"      = "CodeQL analysis (push, PR, weekly schedule)"
    "configs/base.yaml"                 = "Shared hyper-parameters (same for ALL models)"
    "src/__init__.py"                   = "Package marker (enables: python -m src.train)"
    "src/datasets.py"                   = "Loading, dedup, stratified split, augmentation"
    "src/models.py"                     = "timm backbones + CNN from scratch"
    "src/strategies.py"                 = "Freeze / partial unfreeze / full fine-tune"
    "src/train.py"                      = "Training loop (AMP, checkpoints, MLflow logging)"
    "src/evaluate.py"                   = "Metrics, confusion matrix, ROC/AUC"
    "src/robustness.py"                 = "Corruption + out-of-distribution tests"
    "src/benchmark.py"                  = "FLOPs, latency, ONNX size"
    "src/gradcam.py"                    = "Grad-CAM visualisations"
    "app/app.py"                        = "Gradio demo (prediction + Grad-CAM overlay)"
    "scripts/export_onnx.py"            = "ONNX export + INT8 quantization"
    "tests/test_smoke.py"               = "One forward/backward pass per model"
}

# ---------------------------------------------------------------------------
# 3. Seeded content for files that benefit from real starter content
#    (single-quoted here-strings: nothing is expanded, closing '@ must start the line)
# ---------------------------------------------------------------------------
$Seed = @{}

$Seed["requirements.txt"] = @'
# Runtime dependencies (pin exact versions with a lockfile before the final run)
torch                       # core framework
torchvision
timm>=1.0                   # EfficientNetV2, MobileNetV4, ConvNeXtV2, ViT-Base
scikit-learn                # metrics, StratifiedKFold
pytorch-grad-cam            # Grad-CAM (CNN + ViT)
mlflow                      # experiment tracking
omegaconf                   # config management
albumentations              # augmentation
gradio                      # demo app
onnx
onnxruntime                 # export + latency benchmark
fvcore                      # FLOPs counting
imagehash                   # near-duplicate detection (data leakage check)
safetensors                 # safe weight serialization (no pickle)
'@

$Seed["requirements-dev.txt"] = @'
# Audit + test tooling (not needed at inference time)
ruff
mypy                        # lint, format, types
bandit
semgrep                     # SAST
pip-audit                   # dependency CVEs
pytest
pytest-cov                  # tests + coverage.xml for SonarQube
pre-commit
nbstripout                  # strip notebook outputs before commit
modelscan                   # unsafe model-file detection
'@

$Seed[".gitignore"] = @'
# Environments
.venv/
__pycache__/
*.pyc

# Data and experiment artifacts (versioned with DVC / MLflow, never in Git)
data/*
!data/.gitkeep
mlruns/
results/*
!results/.gitkeep
*.pt
*.pth
*.onnx

# Secrets (NEVER commit)
.env
kaggle.json

# Notebooks / tooling
.ipynb_checkpoints/
.coverage
coverage.xml
.mypy_cache/
.ruff_cache/
.pytest_cache/
.scannerwork/
'@

$Seed["SECURITY.md"] = @'
# Security Policy

## Reporting a vulnerability
Please do not open a public issue. Contact the maintainers privately
(see the organization profile) and include steps to reproduce.

## Scope
Source code, the Gradio demo, CI workflows and published model weights.
'@

$Seed[".github/dependabot.yml"] = @'
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
'@

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Get-StarterHeader {
    param([string]$RelPath, [string]$Purpose)
    $name = Split-Path $RelPath -Leaf
    if ($name -like "*.py") {                       # Python: module docstring
        $q = '"' * 3
        return $q + $Purpose + $q + "`n"
    }
    if ($name -like "*.md") {                       # Markdown: title + purpose
        $title = [IO.Path]::GetFileNameWithoutExtension($name)
        return "# $title`n`n$Purpose`n"
    }
    return "# $Purpose`n"                           # YAML, TOML, properties, Dockerfile...
}

function Write-NewFile {
    param([string]$Path, [string]$Content)
    if (Test-Path -LiteralPath $Path) { Write-Host "  skip    $Path (exists)" -ForegroundColor DarkGray; return }
    if ($PSCmdlet.ShouldProcess($Path, "Create file")) {
        $text = $Content -replace "`r`n", "`n"          # force LF endings
        [System.IO.File]::WriteAllText($Path, $text, $Utf8NoBom)
        Write-Host "  create  $Path" -ForegroundColor Green
    }
}

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
if ($PSCmdlet.ShouldProcess($Root, "Create project root")) {
    New-Item -ItemType Directory -Force -Path $Root | Out-Null
}
$RootFull = if (Test-Path $Root) { (Resolve-Path $Root).Path } else { $Root }
Write-Host "`nProject root: $RootFull`n" -ForegroundColor Cyan

Write-Host "Directories" -ForegroundColor Cyan
foreach ($dir in $Directories) {
    $target = Join-Path $RootFull $dir
    if ($PSCmdlet.ShouldProcess($target, "Create directory")) {
        New-Item -ItemType Directory -Force -Path $target | Out-Null
        Write-Host "  dir     $dir" -ForegroundColor Green
    }
}
foreach ($dir in $KeepDirs) {
    Write-NewFile -Path (Join-Path $RootFull "$dir/.gitkeep") -Content ""
}

Write-Host "`nFiles" -ForegroundColor Cyan
foreach ($rel in $Files.Keys) {
    $content = if ($Seed.ContainsKey($rel)) { $Seed[$rel] } else { Get-StarterHeader -RelPath $rel -Purpose $Files[$rel] }
    Write-NewFile -Path (Join-Path $RootFull $rel) -Content $content
}

if ($GitInit) {
    if (Get-Command git -ErrorAction SilentlyContinue) {
        if ($PSCmdlet.ShouldProcess($RootFull, "git init -b main")) {
            git -C $RootFull init -b main | Out-Null
            Write-Host "`nGit repository initialised (branch: main)" -ForegroundColor Green
        }
    } else {
        Write-Warning "git not found in PATH, skipping -GitInit."
    }
}

if (-not $WhatIfPreference) {
    Write-Host "`nResulting layout" -ForegroundColor Cyan
    Get-ChildItem -Path $RootFull -Recurse -Force |
        Where-Object { $_.FullName -notmatch '[\\/]\.git([\\/]|$)' } |
        ForEach-Object { "  " + $_.FullName.Substring($RootFull.Length + 1) } |
        Sort-Object
    Write-Host "`nNext: copy the project README into README.md, then:" -ForegroundColor Cyan
    Write-Host "  python -m venv .venv ; .\.venv\Scripts\Activate.ps1"
    Write-Host "  pip install -r requirements.txt -r requirements-dev.txt"
}
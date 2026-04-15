# Layer-Aware Quantization Pipeline for Efficient DistilBERT Deployment

A modular PyTorch-based system for analyzing quantization tradeoffs (FP32, FP16, INT8 PTQ, INT8 QAT) on DistilBERT using the SST-2 sentiment classification dataset.

## Purpose

This pipeline answers three deployment questions:
1. **PTQ vs QAT**: When is simple post-training quantization sufficient vs. costly retraining?
2. **Layer Sensitivity**: Which transformer layers tolerate INT8 quantization and which need higher precision?
3. **Robustness**: Does INT8 accuracy degrade more than FP32 when inputs are noisy?

## Repository Structure

```
├── quant_pipeline/                  # Main package
│   ├── __init__.py
│   ├── core/                        # Pipeline, benchmark, metrics
│   │   ├── __init__.py
│   │   ├── pipeline.py              #   End-to-end inference pipeline
│   │   ├── benchmark.py             #   Accuracy + latency measurement
│   │   └── metrics.py               #   Accuracy computation
│   ├── models/nlp/                  # Model loaders
│   │   ├── __init__.py
│   │   └── distilbert.py            #   DistilBERT + tokenizer loading
│   ├── data/                        # Data loading
│   │   ├── __init__.py
│   │   └── loaders.py               #   SST-2 dataset from HuggingFace
│   ├── quantization/                # Quantization logic
│   │   ├── __init__.py
│   │   ├── utils.py                 #   FP16/INT8 PTQ/QAT + fallback
│   │   └── qat_trainer.py           #   QAT fine-tuning loop
│   ├── analysis/                    # Analysis modules
│   │   ├── __init__.py
│   │   ├── sensitivity.py           #   Per-layer quantization analysis
│   │   ├── robustness.py            #   Text perturbations + evaluation
│   │   └── visualization.py         #   Charts and plots
│   └── utils/                       # Utilities
│       ├── __init__.py
│       ├── memory.py                #   Model size profiling
│       └── export.py                #   CSV export
├── scripts/                         # Executable scripts
│   ├── run_benchmark.py             #   Full benchmark + robustness eval
│   └── run_sensitivity.py           #   Layer sensitivity analysis
├── tests/                           # Unit tests
│   ├── __init__.py
│   └── test_all.py                  #   24 tests across all modules
├── app_streamlit.py                 # Streamlit dashboard
├── setup.py                         # Package installation
├── requirements.txt                 # Dependencies
├── .env                             # HuggingFace token (not tracked)
└── outputs/                         # Generated results (not tracked)
    ├── results.csv
    ├── robustness.csv
    ├── robustness.png
    └── sensitivity.png
```

## Prerequisites

- Python 3.9+
- A free HuggingFace account and access token

## Creating a HuggingFace Token

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **Create new token**
3. Name it anything (e.g., "quantization-project")
4. Select **Read** permission
5. Click **Create token**
6. Copy the token (starts with `hf_`)

## Setup & Run — Local (Mac / Windows / Linux)

### 1. Clone the repository

```bash
git clone https://github.com/saketh-kuppili/Layer-Aware-Quantization-Pipeline-for-Efficient-DistilBERT-Deployment.git
```

```bash
cd Layer-Aware-Quantization-Pipeline-for-Efficient-DistilBERT-Deployment
```

### 2. Create virtual environment

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install --upgrade pip
```
```bash
pip install -r requirements.txt
```
```bash
pip install -e .
```

### 4. Set up HuggingFace token

Create a `.env` file in the project root:

```bash
echo "HF_TOKEN=hf_your_token_here" > .env
```

Replace `hf_your_token_here` with your actual token.

### 5. Run the pipeline

```bash
# Full benchmark (accuracy, latency, memory) + robustness evaluation
python scripts/run_benchmark.py

# Layer sensitivity analysis
python scripts/run_sensitivity.py

# Run unit tests
pytest tests/ -v

# Launch Streamlit dashboard (opens at http://localhost:8501)
streamlit run app_streamlit.py
```
### 6. Output files

After running the scripts, results are saved in the `outputs/` folder:
- `results.csv` — Benchmark results (accuracy, latency, memory for all 4 modes)
- `robustness.csv` — Robustness comparison (FP32 vs INT8 under perturbations)
- `robustness.png` — Robustness comparison chart
- `sensitivity.png` — Layer sensitivity chart

## Setup & Run — Google Colab

### Cell 1 — Clone and install

```python
!git clone https://github.com/saketh-kuppili/Layer-Aware-Quantization-Pipeline-for-Efficient-DistilBERT-Deployment.git
```
```python
%cd Layer-Aware-Quantization-Pipeline-for-Efficient-DistilBERT-Deployment
```
```python
!pip install -r requirements.txt -q
```
```python
!pip install -e . -q
```

### Cell 2 — Set up HuggingFace token

```python
import os
os.environ["HF_TOKEN"] = "hf_your_token_here"
```
### Cell 3 — Run benchmark

```python
!python scripts/run_benchmark.py
```

### Cell 4 — Run sensitivity

```python
!python scripts/run_sensitivity.py
```

### Cell 5 — Run tests

```python
!pytest tests/ -v
```

### Cell 6 — Launch Streamlit

Cell 6a — Install tunnel:
```python
!pip install streamlit -q
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb
```

Cell 6b — Start Streamlit:
```python
!cd /content/Layer-Aware-Quantization-Pipeline-for-Efficient-DistilBERT-Deployment && streamlit run app_streamlit.py \
  --server.port 8501 \
  --server.headless true \
  &>/content/logs.txt &
```

Cell 6c — Get public URL:
```python
import subprocess, re
process = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://localhost:8501"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
)
for line in process.stdout:
    if "trycloudflare.com" in line:
        url = re.search(r"https://[-a-zA-Z0-9\.]+trycloudflare.com", line)
        if url:
            print("Open this link:", url.group(0))
            break
```
### 7. Output files

After running the scripts, results are saved in the `outputs/` folder:
- `results.csv` — Benchmark results (accuracy, latency, memory for all 4 modes)
- `robustness.csv` — Robustness comparison (FP32 vs INT8 under perturbations)
- `robustness.png` — Robustness comparison chart
- `sensitivity.png` — Layer sensitivity chart

## Pipeline Architecture

```
Raw Text (SST-2)
      │
      ▼
┌─────────────┐
│  Tokenizer   │  WordPiece vocabulary
└──────┬──────┘
       │
       ├──────────────┬──────────────┬──────────────┐
       ▼              ▼              ▼              ▼
   ┌───────┐    ┌───────┐    ┌─────────┐    ┌─────────┐
   │ FP32  │    │ FP16  │    │INT8 PTQ │    │INT8 QAT │
   └───┬───┘    └───┬───┘    └────┬────┘    └────┬────┘
       │            │             │              │
       └────────────┴─────────────┴──────────────┘
                          │
                          ▼
              ┌──────────────────────┐
              │  Evaluation Engine    │
              │  Accuracy + Latency  │
              │  + Memory Profiling  │
              └──────────┬───────────┘
                         │
              ┌──────────┴───────────┐
              ▼                      ▼
   ┌──────────────────┐  ┌──────────────────┐
   │  Sensitivity      │  │  Robustness      │
   │  Per-layer INT8   │  │  Typos, drops,   │
   │  quantization     │  │  noise injection │
   └──────────────────┘  └──────────────────┘
              │                      │
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │  Dashboard + CSV     │
              │  Streamlit           │
              └──────────────────────┘
```

## Sample Sizes

| Purpose | Split | Samples |
|---------|-------|---------|
| Benchmark evaluation | validation | 200 |
| QAT fine-tuning | train | 300 |
| QAT fine-tuning (Streamlit demo) | train | 300 |
| Robustness evaluation | validation | 100 |
| Sensitivity analysis | validation | 200 |

## Key Design Decisions

- **Dynamic quantization** for PTQ — targets only `nn.Linear` layers, avoids Embedding/LayerNorm crashes
- **Two-step QAT** — fine-tune FP32 model, then apply dynamic INT8 quantization (avoids fragile `prepare_qat` path)
- **Cross-platform fallback** — auto-detects backend (`fbgemm` for x86, `qnnpack` for ARM), validates with test forward pass, falls back to FP16 if INT8 fails
- **Block-level sensitivity** — quantizes entire transformer blocks (not individual Linear layers) for meaningful accuracy deltas
- **Single forward pass** in benchmark — no double inference
- **Memory includes buffers** — quantized models store scale/zero-point as buffers

## Troubleshooting

**"Dataset not found" error:** Your HuggingFace token may not be set. Re-run the token setup step.

**"NoQEngine" error on Mac:** Expected — Mac's `qnnpack` backend has limited INT8 support. The pipeline automatically falls back to FP16.

**Streamlit QAT takes long time:** First-time QAT inference trains the model on 300 samples. Subsequent predictions are instant (cached).

**Tests fail:** Make sure you installed the package with `pip install -e .` before running `pytest tests/ -v`.

## Authors

Saketh Kuppili (002014840) | Ashvath Cheppalli (002500670)

CS 5130 — Applied Programming and Data Processing for AI

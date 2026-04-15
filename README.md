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
├── app_gradio.py                    # Gradio demo interface
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
cd Layer-Aware-Quantization-Pipeline-for-Efficient-DistilBERT-Deployment
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
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

# Launch Gradio demo (opens at http://127.0.0.1:7860)
python app_gradio.py

# Launch Streamlit dashboard (opens at http://localhost:8501)
streamlit run app_streamlit.py
```

## Setup & Run — Google Colab

### 1. Clone the repository

```python
!git clone https://github.com/saketh-kuppili/Layer-Aware-Quantization-Pipeline-for-Efficient-DistilBERT-Deployment.git
%cd Layer-Aware-Quantization-Pipeline-for-Efficient-DistilBERT-Deployment
```

### 2. Install dependencies

```python
!pip install -r requirements.txt -q
!pip install -e . -q
!pip install plotly -q
```

### 3. Set up HuggingFace token

```python
import os
os.environ["HF_TOKEN"] = "hf_your_token_here"
```

Or use Colab Secrets (recommended — keeps token hidden):
1. Click the **key icon** (🔑) on the left sidebar
2. Add a new secret named `HF_TOKEN` with your token
3. Toggle **Notebook access** on
4. Then use:

```python
from google.colab import userdata
import os

try:
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    print("Token loaded from Colab secrets")
except Exception:
    os.environ["HF_TOKEN"] = input("Enter HuggingFace token: ")
    print("Token set manually")
```

### 4. Run the pipeline

```python
# Full benchmark + robustness
!python scripts/run_benchmark.py

# Layer sensitivity analysis
!python scripts/run_sensitivity.py

# Run unit tests
!pytest tests/ -v
```

### 5. Launch Gradio (in Colab)

```python
!python app_gradio.py
```

Click the public Gradio link that appears.

### 6. Launch Streamlit (in Colab)

```python
!pip install streamlit -q
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb
```

```python
!cd /content/Layer-Aware-Quantization-Pipeline-for-Efficient-DistilBERT-Deployment && streamlit run app_streamlit.py \
  --server.port 8501 \
  --server.headless true \
  &>/content/logs.txt &
```

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
              │  Streamlit / Gradio  │
              └──────────────────────┘
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Classification accuracy on clean + perturbed data |
| Avg Latency | Mean inference time per sample (ms) |
| P99 Latency | 99th percentile inference time (worst-case) |
| Memory | Model size in MB (parameters + buffers) |
| Sensitivity | Per-layer accuracy delta when quantized individually |
| Robustness Gap | Accuracy difference between FP32 and INT8 under perturbation |

## Key Design Decisions

- **Dynamic quantization** for PTQ — targets only `nn.Linear` layers, avoids Embedding/LayerNorm crashes
- **Two-step QAT** — fine-tune FP32 model, then apply dynamic INT8 quantization (avoids fragile `prepare_qat` path)
- **Cross-platform fallback** — auto-detects backend (`fbgemm` for x86, `qnnpack` for ARM), validates with test forward pass, falls back to FP16 if INT8 fails
- **Block-level sensitivity** — quantizes entire transformer blocks (not individual Linear layers) for meaningful accuracy deltas
- **Single forward pass** in benchmark — no double inference
- **Memory includes buffers** — quantized models store scale/zero-point as buffers

## Sample Sizes

| Purpose | Split | Samples |
|---------|-------|---------|
| Benchmark evaluation | train | 1,000 |
| QAT fine-tuning (benchmark) | train | 1,000 |
| QAT fine-tuning (Streamlit demo) | train | 300 |
| Robustness evaluation | train | 1,000 |
| Sensitivity analysis | train | 1,000 |





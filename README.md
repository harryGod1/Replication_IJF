
## Reproducibility Package Information

- **Date assembled:** June 2026
- **Authors:** Hao Wang(scxhw1@nottingham.edu.cn), Anthony Bellotti(Anthony-Graham.Bellotti@nottingham.edu.cn)


---

## Repository Structure
# Long Short-Term Memory Network with Adapted Attention Mechanism for Credit Risk Modelling

[![Python 3.8.19](https://img.shields.io/badge/python-3.8.19-blue.svg)](https://www.python.org/downloads/release/python-3819/)


**Reproducibility package for:**
> Hao Wang, Anthony Bellotti. *Long Short-Term Memory Network with Adapted Attention Mechanism for Credit Risk Modelling*. International Journal of Forecasting, 2026.

---

```
.
├── code/                       # Core source code for the paper
│   ├── Figure6.py              # reproduce the results shown in Figure 6                     
│   ├── Figure7.py              # reproduce the results shown in Figure 7
│   ├── Figure8.py              # reproduce the results shown in Figure 8
...
│   ├── Figure20.py             # reproduce the results shown in Figure 20
│   ├── Table2.py               # reproduce the descriptive statistics of the dataset used in analysis (2004-2013)
│   ├── Table3.py               # reproduce the descriptive statistics of the dataset used in analysis (2016-2024)
│   ├── Table4.py               # reproduce the results shown in Table4
...
│   └── Table7.py               # reproduce the results shown in Table7         
│
├── data/
│   ├── 2259/                   # All datasets used in this paper(raw data and intermediary data)
│   ├── washout_phase/          # data for washout testing
│   ├── DTSM.sql                # script designed to process raw data into the intermediate datasets; requires MySQL to run.
│   ├── Out_file.sql            # script for exporting datasets from the database as text data suitable for model execution.requires MySQL to run.
│   └── raw data.txt            # Data Usage Guidelines: Readers are strongly advised to read this document first. It contains download links for the raw data, as 
│                               # well as step-by-step instructions on how to use sample scripts to process the raw data into an intermediate dataset.
├── env/                        # Environment and Dependencies
│   ├── requirements.txt        # First, run this file to install the required Python dependencies with their specified versions.
│   └── requirements2.txt       # Second, run this file to install the required Python dependencies with their specified versions.
│
├── example/                    # Demonstration of Code Execution
│   └── example_figure14.png    # An example screenshot of the execution of Figure14.py in the real environment.
│
├── saved_model/                # All models used in the experiments
│
├── train_model/                # Demonstration of Code Execution
│   ├── 3LSTM_train.py          # Training code for the proposed state-of-the-art model (Attention + 3-layer LSTM).
│   └── DeepHit_train.py        # Training code for the baseline model (DeepHit).

```

---

## Computing Environment

| Component | Full Experiments (cluster) | Light Reproduction (laptop) |
|---|---|---|
| **OS** | Linux | Any (tested on Windows 11) |
| **CPU** | 40-core cluster node | Intel Core i7 |
| **RAM** | 377 GB | 32 GB |
| **Python** | 3.10 | 3.10 |
| **`--n_jobs`** | 40 | 4 |

**Language:** Python 3.10
**Package manager:** conda (miniforge recommended)
**Key dependencies** (see `requirements.txt` for full list with pinned versions):

| Package | Version | Purpose |
|---|---|---|
| `numpy` | 1.23.5 | Numerical computation |
| `pandas` | 1.5.3 | Data manipulation |
| `scikit-learn` | 1.6.1 | Machine learning metrics and classifiers |
| `imbalanced-learn` | 0.12.4 | `BalancedRandomForestClassifier` |
| `tigramite` | 5.2.10.1 | PCMCI implementation |
| `lingam` | 1.12.2 | VARLiNGAM implementation |
| `dcor` | 0.6 | Distance correlation (used by tigramite/GPDC) |
| `causalnex` | 0.12.1 | DYNOTEARS implementation |
| `matplotlib` | 3.7.5 | Plotting |
| `seaborn` | 0.13.2 | Statistical visualisation |
| `scipy` | 1.10.1 | Statistical tests (Wilcoxon, Friedman) |
| `networkx` | 3.1 | Graph utilities |

---

## Installation

### 1. Create conda environment

```bash
conda create -n replication_ijf python=3.8.19 -y
conda activate replication_ijf
```

### 2. Clone this repository and install dependencies

```bash
git clone https://github.com/harryGod1/Replication_IJF.git
cd Replication_IJF
pip install -r requirements.txt
pip install lifelines==0.27.8
pip install pygam==0.8.1
pip install -r requirements2.txt
```
- Optional
#### All code can be executed even without the following two installation commands. However, if readers wish to utilize GPU acceleration, please run the following two lines of code.
```bash
conda install -c conda-forge cudatoolkit=10.1.243
conda install -c conda-forge cudnn=7.6.5

```

### 3. Download the data folder

The `data/` directory (~500 MB) contains all datasets, pre-computed descriptors, and cached results. It is stored separately from the repository due to its size.

**Download link:** [data.zip (Google Drive)](https://drive.google.com/file/d/1z8cHkUTe7TlvWwqBlpoEsukWaSsvFC26/view?usp=sharing)

Via shell:
```bash
pip install gdown
gdown 1z8cHkUTe7TlvWwqBlpoEsukWaSsvFC26
python3 -m zipfile -e data.zip .
```

Place the resulting `data/` folder in the root of the repository.

> **⚠️ Important — verify your pre-computed files before running `05.py`, `06.py`, or `07.py`.**
> Running `04.py` **without** the `--skip_benchmark` flag will overwrite `data/causal_dfs/*.pkl` with freshly generated results, which introduces non-determinism from `BalancedRandomForestClassifier`. To reproduce Tables 4, G.13, and H.14 and Figure I.6 accurately, always use the pre-computed files from `data.zip`.
>
> You can verify you have the correct file with:
> ```bash
> md5sum data/causal_dfs/causal_dfs_TEST.pkl
> # Expected: 4b49870ad8685e2cb3885d3495d1b9a6
> ```
> If the checksum does not match, re-download `data.zip` and extract again.

---

## Data Description

### Synthetic data
Generated programmatically by `01.py` using NAR (Nonlinear AutoRegressive) processes. Nine processes (IDs: 1, 3, 5, 7, 9, 11, 13, 15, 19) with three noise distributions (Gaussian, Uniform, Laplace). No external source required.

- **Training set:** `data/observations/training_data_{gaussian,uniform,laplace}.pkl` — 120 time series per process per noise type (3240 total)
- **Test set:** `data/observations/testing_data_{gaussian,uniform,laplace}.pkl` — 40 time series per process per noise type (1080 total)

### Realistic data (DREAM3 and NetSim)
Pre-processed versions are included directly in the repository under `data/realistic/`. No registration, cost, or restricted access. Both datasets are publicly available:

- **DREAM3:** In-silico gene regulatory networks. Original source: [DREAM3 challenge (Synapse)](https://www.synapse.org/#!Synapse:syn2853594). Pre-processing notebook: `data/realistic/dream3/dream3.ipynb`
- **NetSim:** Simulated fMRI time series with known connectivity. Original source: [Smith et al. (2011), NeuroImage](https://www.sciencedirect.com/science/article/pii/S1053811910011602). Pre-processing notebook: `data/realistic/netsym/netsym.ipynb`

### Intermediary data files
The following files are intermediary (expensive to recompute) and are included in `data.zip` to allow skipping heavy steps:

| File(s) | Generated by | Description |
|---|---|---|
| `data/observations/*.pkl` | `01.py` | Raw synthetic time series |
| `data/descriptors/*.pkl` | `02.py` | TD2C descriptor matrices (hours to compute) |
| `data/before_d2c/*.pkl` | `04.py` | Competitor method predictions (hours to compute) |
| `data/causal_dfs/*.pkl` | `04.py` | Final combined results for all methods |

---

## Reproducing the Results

All scripts must be run from the `reproduce/py_scripts/` directory:

```bash
cd reproduce/py_scripts
```

### Quick reproduction (~5 minutes, laptop)

Use pre-computed intermediary data to skip the two heavy steps and regenerate all tables and figures:

```bash
python pipeline.py --n_jobs 4 --skip_data --skip_descriptors --skip_benchmark
```

This runs steps 3, 5, 6, 7, and 9 — producing all paper tables and figures.

### Full reproduction from scratch (cluster, ~2–3 days)

```bash
python pipeline.py --n_jobs 40
```

### Skip flags explained

| Flag | Skips | Requires |
|---|---|---|
| `--skip_data` | Step 01 (data generation) | `data/observations/*.pkl` present |
| `--skip_descriptors` | Step 02 (descriptor computation) | `data/descriptors/*.pkl` present |
| `--skip_benchmark` | Step 04 (benchmark execution) | `data/before_d2c/*.pkl` and `data/causal_dfs/*.pkl` present |

All three are satisfied by the provided `data.zip`.

### Running individual steps

```bash
python 03.py --n_jobs 4   # threshold selection
python 05.py              # synthetic analysis
python 06.py              # realistic analysis
python 07.py              # CD diagrams
python 08.py --n_jobs 4   # scalability benchmark (standalone, not in pipeline)
python 09.py --n_jobs 4   # feature importance
```

---

## Paper Tables and Figures — Output Mapping

| Paper Item | Script | Output file |
|---|---|---|
| **Table 1** — Path counting | `00.py` | Console output (theoretical; not in main pipeline) |
| **Table 2** — Generation parameters | — | Descriptive table (not computationally generated) |
| **Table 3** — Threshold selection metrics | `03.py` | Console output |
| **Table 4** — Synthetic overall results | `05.py` | `TEST_analysis/tables/overall_macro_summary.csv` |
| **Table 5** — Runtime per method | `08.py` | `data/benchmark_times_by_nvars.csv` |
| **Table 6** — Feature importance | `09.py` | Console output |
| **Figure 1** — Markov Blanket DAG | — | Conceptual illustration (not computationally generated) |
| **Figure 2** — Causal scenarios | — | Conceptual illustration (not computationally generated) |
| **Figure 3** — F1-score boxplot (realistic) | `06.py` | `REAL_analysis/figures/summary_boxplot_all_datasets_F1-Score.png` |
| **Figure 4a** — CD diagram (Precision) | `07.py` | `CD_PLOTS/cd_TEST_precision.png` |
| **Figure 4b** — CD diagram (Recall) | `07.py` | `CD_PLOTS/cd_TEST_recall.png` |
| *Appendix* — Per-process synthetic results | `05.py` | `TEST_analysis/tables/macro_process_*.csv` |
| *Appendix* — Realistic benchmark details | `06.py` | `REAL_analysis/tables/summary_macro_{NETSIM_5,NETSIM_10,DREAM3_10,DREAM3_50}.csv` |
| *Appendix* — CD diagrams (F1, Accuracy, Bal. Accuracy) | `07.py` | `CD_PLOTS/cd_TEST_{f1,accuracy,balanced_accuracy}.png` |

---

## Expected Runtimes

| Step | Script | Laptop | 
|---|---|---|
| 01 — Data generation | `Figure6.py` | ~2 min | 
| 02 — Descriptor computation | `Figure7.py` | ~2 min | 
| 03 — Threshold selection | `Figure8.py` |  ~6 min | 
| 04 — Benchmark execution | `Figure9.py` |  ~6 min | 
| 05 — Synthetic analysis | `Figure10.py` |  ~6 min | 
| 06 — Realistic analysis | `Figure11.py` | ~9 min |
| 07 — CD diagrams | `Figure12.py` | ~9 min |
| 08 — Scalability benchmark | `Figure13.py` | ~9 min |
| 09 — Feature importance | `Figure14.py` | ~3 min | 
| 09 — Feature importance | `Figure15.py` |  ~3 min | 
| **Total (from scratch)** | | **days** | 



Steps 02 and 04 are the bottleneck. All intermediary outputs are provided in `data.zip` to bypass them.

---

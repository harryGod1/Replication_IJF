
## 📅 Assembly date and authorship

- **Date assembled:** June 2026
- **Authors:** Hao Wang(scxhw1@nottingham.edu.cn), Anthony Bellotti(Anthony-Graham.Bellotti@nottingham.edu.cn)


---

## 🗂️ Repository Structure
# Long Short-Term Memory Network with Adapted Attention Mechanism for Credit Risk Modelling

[![Python 3.8.19](https://img.shields.io/badge/python-3.8.19-blue.svg)](https://www.python.org/downloads/release/python-3819/)

**⚠️Important:** 
> We have updated the documentation and file structure to enhance usability for the review process. All core analysis code, data, and empirical results remain identical. For the best experience, we recommend using the latest commit.

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
│   └── raw data.txt            # Data Usage Guidelines: It contains download links for the raw data, as 
│                               # well as step-by-step instructions on how to use sample scripts to process the raw data into an intermediate dataset.
├── env/                        # Environment and Dependencies
│   ├── requirements.txt        # First, run this file to install the required Python dependencies with their specified versions.
│   └── requirements2.txt       # Second, run this file to install the required Python dependencies with their specified versions.
│
├── example/                    # Demonstration of Code Execution
│   └── example_figure14.png    # An example screenshot of the execution of Figure14.py in the real environment.
│
├── output/                     # The execution results of the code (corresponding to the figures in the paper) are saved in this folder.
│
└── saved_model/                # All models used in the experiments



```

---

## 💻 Computing Environment

| Component | Reproduction (laptop) | 
|---|---|
| **OS** | Any (tested on Centos and Windows10) |
| **CPU** | Intel Core i7 |
| **GPU** | NVIDIA GeForce GTX 1650 |
| **RAM** | 32 GB |
| **Python** | 3.8.19 |


**Language:** Python 3.8.19
**Package manager:** conda 
**Key dependencies** (see `requirements.txt & requirements2.txt` for full list with pinned versions):

| Package | Version | Purpose |
|---|---|---|
| `numpy` | 1.18.0 | Numerical computation |
| `pandas` | 1.0.5 | Data manipulation |
| `scikit-learn` | 1.1.2 | Machine learning metrics and classifiers |
| `keras-preprocessing` | 1.1.2 | data preprocessing library for Keras |
| `tensorflow-gpu` | 2.2.0 | GPU-enabled TensorFlow |
| `lifelines` | 0.27.8 | Survival Analysis Library |
| `pygam` | 0.8.1 | Generalized Additive Models (GAMs) library |
| `matplotlib` | 3.5.5 | Plotting |
| `scipy` | 1.4.1 | Statistical tests |

---

## Installation

### 1. Create conda environment

```bash
conda create -n replication_ijf python=3.8.19 -y
conda activate replication_ijf
```
### 2. Installing git lfs for handling big data(>100m)
> **⚠️ Important — Please download the appropriate Git LFS package for your operating system.**
> 
> **⚠️ Important — Due to the presence of data files exceeding 100 MB, Git Large File Storage (LFS) is required to handle them. Readers should first download the appropriate Git LFS installation package for their operating system.**

#### Example for installing git lfs in Centos

##### Run the following command to install the EPEL repository (skip this step if it is already installed):
```bash
sudo yum install epel-release
#(Note: For CentOS 8/9 or newer versions, replace yum with sudo dnf install epel-release.)
```
##### Download, install, and initialize Git LFS:
```bash
sudo yum install git-lfs
git lfs install
```
#### Example for installing git lfs in unbuntu
```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
apt-get install git-lfs
git lfs install
```


### 3. Clone this repository and install dependencies
> **⚠️ Important — Please be sure to adhere to the following specified installation instructions to avoid any compatibility conflicts.**


```bash
git lfs clone https://github.com/harryGod1/Replication_IJF.git
#reader can also use the traditional command: git clone https://github.com/harryGod1/Replication_IJF.git. Mannually download the big data from github and put them #to data/2259/ folder
cd Replication_IJF
pip install -r env/requirements.txt
pip install lifelines==0.27.8
pip install pygam==0.8.1
pip install -r env/requirements2.txt
conda install -c conda-forge cudatoolkit=10.1.243
conda install -c conda-forge cudnn=7.6.5
```

### 4. Data folder

The `data/` directory (~2 GB) contains all the datasets. For copyright reasons, the original data are not provided directly in this repository. Readers can download the data by clicking the link below.

**Download link:** [Freddie Mac](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset)

---

## Reproducing the Results
Each figure and table in the paper can be reproduced by executing the corresponding Python script located in the code/ directory.

The following shows a running example of the Figure14.py script.：

```bash
python code/Figure14.py
```

## 🗃️ Data — what it is and how to get it

### Freddie Mac Mortgage data(2004 to 2024)
The Freddie Mac single-family loan-level dataset is a publicly available collection of residential mortgage origination and monthly performance data provided by Freddie Mac(https://www.freddiemac.com/home). The detailed data dictionary and descriptive statistics is available from this website. It covers a substantial portion of the U.S. mortgage market over multiple decades, including loan characteristics (e.g., credit score, loan-to-value ratio, debt-to-income ratio), property details, and dynamic repayment behavior (e.g., prepayments and defaults). This dataset has become a benchmark for research on mortgage default prediction, prepayment modeling, and other related tasks in finance and econometrics. Due to its large scale, temporal depth, and real-world relevance, it is widely used to evaluate state-of-the-art forecasting models.

For copyright reasons, the original data are not provided directly in this repository. Readers can download the data by clicking the link below.
-The download link for the raw data is provided here:[Freddie Mac Download](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset)

### Intermediary Datasets
The data/ folder contains all data used in the experiments and testing. Readers do not need to generate all data files from scratch; instead, they can directly utilize the intermediate data located in the `data/` folder to execute all code and generate the corresponding outputs.

---


## 📊 Which code produces which outputs

| Paper Item | Script | Output File |
|---|---|---|
| **Table 1** | — | Literature review (not computationally generated) |
| **Table 2** | `Table2.py` | Console Output |
| **Table 3** | `Table3.py` | Console Output |
| **Table 4** | `Table4.py` | Console Output |
| **Table 5** | `Table5.py` | Console Output |
| **Table 6** | `Table6.py` | Console Output |
| **Table 7** | `Table7.py` | Console Output |
| **Figure 1-5** | — | Conceptual illustration (not computationally generated) |
| **Figure 6**  | `Figure6.py` | output/Figure6.png|
| **Figure 7**  | `Figure7.py` | output/Figure7.png |
| **Figure 8**  | `Figure8.py` | output/Figure8.png |
| **Figure 9**  | `Figure9.py` | output/Figure9.png |
| **Figure 10**  | `Figure10.py` | output/Figure10.png |
| **Figure 11**  | `Figure11.py` | output/Figure11.png |
| **Figure 12**  | `Figure12.py` | output/Figure12.png |
| **Figure 13**  | `Figure13.py` | output/Figure13.png |
| **Figure 14**  | `Figure14.py` | output/Figure14.png |
| **Figure 15**  | `Figure15.py` | output/Figure15.png |
| **Figure 16**  | `Figure16.py` | output/Figure16.png |
| **Figure 17**  | `Figure17.py` | output/Figure17.png |
| **Figure 18**  | `Figure18.py` | output/Figure18.png |
| **Figure 19**  | `Figure19.py` | output/Figure19.png |
| **Figure 20**  | `Figure20.py` | output/Figure20.png |

---

## ⏱️ Expected Runtimes

| Script | Laptop | 
|---|---|
| `Figure6.py` | ~5 min | 
| `Figure7.py` | ~15 min | 
| `Figure8.py` |  ~6 min | 
| `Figure9.py` |  ~6 min | 
| `Figure10.py` |  ~6 min | 
| `Figure11.py` | ~9 min |
| `Figure12.py` | ~9 min |
| `Figure13.py` | ~9 min |
| `Figure14.py` | ~3 min | 
| `Figure15.py` | ~11 min | 
| `Figure16.py` | ~11 min | 
| `Figure17.py` | ~40 min | 
| `Figure18.py` | ~40 min |
| `Figure19.py` | ~40 min |
| `Table2.py` | ~2 min |
| `Table3.py` | ~2 min |
| `Table4.py` | ~5 min |
| `Table5.py` | ~5 min | 
| `Table6.py` | ~5 min | 
| `Table7.py` | ~5 min | 
| **Total (from scratch)** | **4hours** | 

---

## ⚙️ Special setup requirements

None

## References

Gian Marco Paldino, Gianluca Bontempi. Causal Discovery in Multivariate Time Series through Mutual Information Featurization. International Journal of Forecasting, 2025. https://doi.org/10.48550/arXiv.2508.01848

Timo Dimitriadis and Alexander I. Jordan. Replication package for "Evaluating Probabilistic Classifiers: The Triptych". International Journal of Forecasting, 2024. https://doi.org/10.1016/j.ijforecast.2023.09.007

Dennis van der Meer, Pierre Pinson, Simon Camal and Georges Kariniotakis. CRPS-based online learning for nonlinear probabilistic forecast combination. International Journal of Forecasting, 2024. https://doi.org/10.1016/j.ijforecast.2023.12.005

---

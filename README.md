Long Short-Term Memory Network with Adapted Attention Mechanism for Credit Risk Modelling
========================================================================================
Hao Wang & Anthony Bellotti

Overview:

The code in this reproducibility package can generate the 15 figures and 6 tables presented in this paper, covering all experimental results and data. 
Each figure and table is produced by its corresponding .py file: Figurexx.py and Tablexx.py, respectively. Readers are free to explore the content and
run all the code. The example directory contains a screenshot demonstrating the execution status of Figure13.py.

----------------------------------------------------------------------------------------
The main directories are organized as follows:

env/: contains all environment specifications and the corresponding version numbers required to run the code in this paper.

data/: includes all data used for the experiments and models.

train_model/: provides example code for model training, facilitating further research by future readers.

Figurexx.py: generates the experimental results presented in the corresponding figure of the paper.

Tablexx.py: produces the results shown in the corresponding table of the paper.

========================================================================================
Instructions & computational requirements：

Before using this code, please carefully read the .txt file in the env directory, which provides the specific version numbers of all required software packages 
to avoid compatibility issues. It is strongly recommended to first install the compatible versions of Python, NumPy, SciPy, pandas, TensorFlow-GPU, Pillow, and 
other base packages. Subsequently, install toolkits such as lifelines and pygam in order, as they are necessary for the implementation and testing described in the paper.
Importantly, installing the designated versions of pygam and lifelines may overwrite previously installed versions of NumPy, Scipy, Scikit-learn and other similar packages. 
Therefore, after completing the installation of pygam and lifelines, manually reinstall NumPy and any other overwritten packages to the exact versions specified in the env directory's .txt file  is required; 
failure to do so will cause compatibility problems.

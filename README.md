## Overview
This repository provides scripts to:

- create and use a Python virtual environment (venv)
- install dependencies from `requirements.txt`
- generate data (simulated or prepared format)
- train the model
- estimate parameter variance (Hessian, sandwich, score/OPG)

---

## Setup: Python + virtual environment
### Prerequisites
- Python **3.10+** recommended
- `pip` available

### Create and activate a venv
```
python -m venv .venv
source .venv/bin/activate
```
### Install dependencies (requirements.txt)

Install:
```
pip install -r requirements.txt
```

## Generate data

### Simulated antibody kinetic dataset
**Dosing schedule (simulation):** all subjects receive **3 injections**, with injections at **Day 0**, **Day 30** (2nd dose), and **Day 250** (3rd dose).

```
python antibody_multidose_data_scipy.py \
--irregular True \
--out antibody_datasets/
--seed 42 \
--measurements 10 \
--duration 400
```

```md
- `--irregular True/False`: generate irregular/regular visit times per subject  
- `--out antibody_datasets/`: save dataset files to this folder  
- `--seed 42`: set RNG seed for reproducible simulation
- `--measurements`: the number of measurements per subject
- `--duration`: the total duration
```
The **S** and **Ab** trajectories are stored in the output file.

The datasets used in the paper are in the folder data/

## Train on regular data + variance estimation (simulation loop)

We run training and uncertainty quantification on multiple simulated datasets. For each dataset replicate:

1. **Train** the model on *regularly-sampled* data and save the fitted parameters.
2. **Estimate the variance–covariance matrix** of the fitted parameters using a Monte Carlo procedure with `b_sample_size` samples.
3. **Save** the variance matrix to disk.
4. After all replicates, compute **pointwise analysis** and **coverage** based on the estimated variances.

### Key settings

- `N_subjects = 50`: number of subjects in each simulated dataset  
- `b_sample_size = 8000`: Monte Carlo sample size used in the variance estimation  
- `data_folder`: folder containing simulated datasets (e.g., `dataset1.npy`, `dataset2.npy`, …)  
- `model_folder`: folder containing model configuration / experiment setup  
- `variance_folder`: output folder where variance matrices are saved  
- `estimated_params_file`: text file where fitted parameters are appended (one line per replicate)

### What the script does

For `i = 1, 2, ...` (dataset replicate index):

- **Training**  
  Calls:
  - `antibody_multidoses_train_regular.main(i, model_folder=..., data_folder=...)`  
  This returns a serialized string of estimated parameters (stored in `parameters`), which is appended to:
  - `results/antibody_poitwise_regular_results.txt`

- **Variance estimation**  
  Loads the corresponding dataset:
  - `np.load(data_folder + 'dataset' + str(i) + '.npy')`  
  Then computes the variance matrix:
  - `variance_estimation(datas, parameters, N_subjects, b_sample_size=..., ni=15, T=400)`  
  and saves it as:
  - `EXPs/antibody_regular_deltaAB_variance_400d_15p_50s_8000/variance_matrix_<i>.npy`

- **Analysis (after all replicates)**  
  - `analyse_poitwise_estimation(estimated_params_file)` produces a dataframe summarizing pointwise estimates.  
  - `coverage_from_estimated_variance(...)` computes coverage statistics using the saved variance matrices.

### Outputs

- **Estimated parameters (text)**
  - `results/antibody_poitwise_regular_results.txt`  
  One line per dataset replicate.

- **Variance matrices (NumPy)**
  - `EXPs/antibody_regular_deltaAB_variance_400d_15p_50s_8000/variance_matrix_1.npy`
  - `EXPs/antibody_regular_deltaAB_variance_400d_15p_50s_8000/variance_matrix_2.npy`
  - …

- **Coverage summary (printed)**
  - `summary_df` printed to stdout (can be redirected to a file if needed).

### Example run

```bash
python simu_antibody_regular_pipeline.py
```





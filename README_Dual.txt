# DualML – Dual Machine Learning Forecasting Framework

This repository implements a Dual Machine Learning framework for macroeconomic forecasting, combining econometric models (e.g., OLS, Ridge, Kernel Ridge) and machine learning models (Random Forest, LightGBM, Neural Networks) with dual interpretation of forecasts through observation-level weights and contributions.

---

1) Repository Structure

```
.
├── US_data.csv                   # Example macroeconomic dataset for the US
├── dataTrain__infl_h1.csv        # Example training dataset (inflation, horizon 1)
├── dataTest__infl_h1.csv         # Example testing dataset (inflation, horizon 1)
├── dual_ML.py                    # Main pipeline – runs all models and saves results
├── auxiliaries.py                # Helper functions for weights, contributions, kernels
├── NN.py                         # Neural network (MLP) training + dual interpretation
├── MLP.py                        # MLP architecture, training loop, bootstrap
├── README_Dual.txt               # This documentation
```

---

2) Requirements

- Python 3.9+
- Install required Python packages:

pip install numpy pandas matplotlib scikit-learn statsmodels lightgbm tqdm scipy torch


- [Optional] RPY2 + R,  for exact replication of R-based methods:
  1. Install R on your system.
  2. Install the following R packages: `ranger`, `glmnet`, `kernlab`.
  3. Install Python interface:

     pip install rpy2

  4. In `dual_ML.py`, uncomment the `RPY2 OPTION` code blocks.

---

3) Usage

-> 1. Data Preparation
Place the input dataset in the repository root:
- `US_data.csv` for macroeconomic variables (default in scripts).
- Update the path in `dual_ML.py` OR/AND `NN.py`:

dir = "/path/to/US_data.csv"


-> 2. Running the Main Pipeline
Run all models from the main script:

python dual_ML.py

This will:
- Load and standardize data
- Train selected models (OLS, Random Forest, LightGBM, Kernel Ridge, Ridge)
- Compute dual interpretation metrics (weights, contributions)
- Save results in `.csv` files
- Generate plots

-> 3. Running the Neural Network (standalone)
If you want to train and interpret only the NN model [Do not forget to place the input dataset in the repository root] :

python NN.py

This will:
- Train an MLP with bootstrap averaging
- Compute observation weights & contributions
- Save predictions and plots

---

4) Model Overview

| Model              | Script      | Function in `auxiliaries.py`  |
|--------------------|-------------|-------------------------------|
| OLS                | `dual_ML.py`| `dualml_ols`                  | 
| Random Forest      | `dual_ML.py`| `dualml_rf` / `get_weights_RF`| 
| LightGBM           | `dual_ML.py`| `GeertsemaLu2023`             | 
| Kernel Ridge (KRR) | `dual_ML.py`| `krr_weight`                  | 
| Ridge Regression   | `dual_ML.py`| `dualml_rr`                   | 
| Neural Net (MLP)   | `NN.py`     | `dualml_nn`                   | 

---

5) Output Files

- Weights & Contributions
  - `results_OLS.csv`
  - `results_RF_sklearn.csv`
  - `results_LGB.csv`
  - `results_KRR.csv`
  - `results_RR.csv`
  - `results_NN.csv`

- Predictions
  - `OLS_Prediction.csv`
  - `predictions_RF_all.csv`
  - `predictions_KRR.csv`
  - `predictions_RR.csv`
  - `nn_predictions_all.csv`

Each file contains:
- `Date`
- `Weight` (importance of each in-sample observation for a given OOS point)*
- `Contribution` (cumulative effect on the prediction)*

---

6) Notes

- All scripts assume wide format CSV: variables in columns, time in rows.
- In-sample (`train`) = dates ≤ 2019-12-31, Out-of-sample (`test`) = dates > 2019-12-31 (default split).
- `auxiliaries.py` contains:
  - Weight & contribution computation for all models
  - Kernel computation for KRR
  - Custom LightGBM and RF leaf-matching functions
- The framework is modular – you can add/remove models without affecting the others.
- For exact replication of the original R-based results, use the `RPY2 OPTION` blocks with matching R versions.

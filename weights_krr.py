"""
weights_krr.py — Kernel Ridge Regression (KRR) observation weights.

Source: dual_ML.py, section "3. Kernel Ridge-Regression"
Helpers: auxiliaries.compute_kernel_r_py, auxiliaries.krr_weight

Weight formula (KRR dual):
    W = (K_test @ (K_train + lambda*I)^{-1})'
    rows = training observations, columns = test observations.

Contributions (cumulative):
    C_i = cumsum_t( w_t * (y_t - ybar) ) + ybar * sum(w_t)

Default hyperparameters (from dual_ML.py):
    kernel_type = "laplace"   (L1 / cityblock distance)
    sigma       = 1e-4
    lambda      = 1e-5
"""

import os
import numpy as np
import pandas as pd
from auxiliaries import compute_kernel_r_py, krr_weight


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_data(csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "US_data.csv")
    data = pd.read_csv(csv_path)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.drop(columns=[c for c in data.columns
                               if any(k in c for k in ("outputGap", "chan_outputgap"))])
    return data


# ---------------------------------------------------------------------------
# Weight computation  (dual_ML.py lines 533-602)
# ---------------------------------------------------------------------------

def compute_krr_weights(data, cutoff="2019-12-31",
                         kernel_type="laplace", sigma=1e-4, lmbda=1e-5):
    """
    Compute KRR observation weights and contributions.

    Parameters
    ----------
    data        : DataFrame returned by load_data()
    cutoff      : last in-sample date (inclusive)
    kernel_type : 'laplace' or 'gaussian'  (default: 'laplace')
    sigma       : kernel bandwidth          (default: 1e-4)
    lmbda       : ridge regularisation      (default: 1e-5)

    Returns
    -------
    weights_df  : DataFrame (n_train × n_test+1), last col = 'Date'
    contrib_df  : DataFrame same shape
    pred_df     : DataFrame with train+test predictions
    """
    idx_ins = data["Date"] <= cutoff
    idx_oos = data["Date"] > cutoff

    # Global standardisation (same as dual_ML.py lines 57-63)
    scaler_mean = data.loc[idx_ins, data.columns[2:]].mean()
    scaler_std  = data.loc[idx_ins, data.columns[2:]].std(ddof=1)
    Ytrain = data.loc[idx_ins, "y"].values
    Ytest  = data.loc[idx_oos, "y"].values
    Xtrain = ((data.loc[idx_ins, data.columns[2:]] - scaler_mean) / scaler_std).values
    Xtest  = ((data.loc[idx_oos, data.columns[2:]] - scaler_mean) / scaler_std).values

    # --- Kernel matrices (compute_kernel_r_py uses Python, no R dependency)
    K_train, K_test = compute_kernel_r_py(
        Xtrain=Xtrain, Xtest=Xtest,
        kernel_type=kernel_type, sigma=sigma,
    )

    backpack = {
        "type": "KRR",
        "params": {
            "Xtrain":      K_train,
            "Xtest":       K_test,
            "Ytrain":      Ytrain,
            "dates_ins":   data.loc[idx_ins, "Date"],
            "lmbda":       lmbda,
            "kernel_type": kernel_type,
            "sigma":       sigma,
        },
    }

    weights_df, contrib_df = krr_weight(backpack)

    # --- Predictions via alpha = (K_train + lambda*I)^{-1} y  (dual_ML.py lines 573-579)
    alpha     = np.linalg.solve(K_train + lmbda * np.eye(K_train.shape[0]), Ytrain)
    Y_hat_ins = K_train @ alpha
    Y_hat_oos = K_test  @ alpha

    pred_df = pd.concat([
        pd.DataFrame({
            "Date":   data.loc[idx_ins, "Date"].values,
            "set":    "train",
            "Y_true": Ytrain,
            "Y_hat":  Y_hat_ins,
        }),
        pd.DataFrame({
            "Date":   data.loc[idx_oos, "Date"].values,
            "set":    "test",
            "Y_true": Ytest,
            "Y_hat":  Y_hat_oos,
        }),
    ], ignore_index=True)

    return weights_df, contrib_df, pred_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data = load_data()
    weights_df, contrib_df, pred_df = compute_krr_weights(data)

    pd.DataFrame({
        "Date":         weights_df["Date"],
        "Weight":       weights_df.iloc[:, 0],
        "Contribution": contrib_df.iloc[:, 0],
    }).to_csv("results_KRR.csv", index=False)
    pred_df.to_csv("predictions_KRR.csv", index=False)
    print("Saved: results_KRR.csv, predictions_KRR.csv")
    print(f"Weight matrix shape : {weights_df.shape[0]} train obs × {weights_df.shape[1] - 1} test obs")

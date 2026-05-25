"""
weights_rr.py — Ridge Regression (RR) observation weights.

Source: dual_ML.py, section "4bis. Ridge Regression (Sklearn)"
Helper: auxiliaries.dualml_rr

Weight formula (Ridge dual):
    W = X_test @ (X_train'X_train + n*lambda*I)^{-1} @ X_train'
    rows = training observations, columns = test observations.

Contributions (cumulative):
    C_i = cumsum_t( w_t * (y_t - ybar) ) + ybar

Notes
-----
* X is standardised column-wise using in-sample mean / std (zero-std cols set to 1).
* Y is standardised for CV and model fitting; original Y is passed to dualml_rr
  (matching dual_ML.py exactly).
* Lambda selected via 10-fold CV with the 1-SE rule on a log-space grid.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from auxiliaries import dualml_rr


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
# Weight computation  (dual_ML.py lines 791-892)
# ---------------------------------------------------------------------------

def compute_rr_weights(data, cutoff="2019-12-31"):
    """
    Compute Ridge Regression observation weights and contributions.

    Parameters
    ----------
    data    : DataFrame returned by load_data()
    cutoff  : last in-sample date (inclusive)

    Returns
    -------
    df_weights       : DataFrame (n_train × n_test+1), last col = 'date'
    df_contributions : DataFrame same shape
    pred_df          : DataFrame with train+test predictions (original Y scale)
    alpha_1se        : float, selected regularisation parameter
    """
    idx_ins = data["Date"] <= cutoff
    idx_oos = data["Date"] > cutoff

    # --- Standardise X (replace zero std with 1.0, as in dual_ML.py line 796)
    X_cols    = data.columns[2:]
    X_ins     = data.loc[idx_ins, X_cols].copy()
    X_oos     = data.loc[idx_oos, X_cols].copy()
    scaler_m  = X_ins.mean()
    scaler_s  = X_ins.std(ddof=1).replace(0, 1.0)
    Xtrain_df = (X_ins - scaler_m) / scaler_s
    Xtest_df  = (X_oos - scaler_m) / scaler_s

    # --- Standardise Y (for CV and model fit only)
    Ytrain    = data.loc[idx_ins, "y"].astype(float).copy()
    Ytest     = data.loc[idx_oos, "y"].astype(float).copy()
    y_mu      = Ytrain.mean()
    y_sd      = Ytrain.std(ddof=1) or 1.0
    Ytrain_rr = (Ytrain - y_mu) / y_sd

    Xtr = Xtrain_df.values
    Xte = Xtest_df.values
    ytr = Ytrain_rr.values

    # --- 10-fold CV layout (same as dual_ML.py lines 812-814)
    n        = len(ytr)
    foldid   = np.repeat(np.arange(10), n // 10)
    foldid   = np.concatenate([foldid, np.full(n - len(foldid), 9)])[:n]
    unique_folds = np.unique(foldid)

    # --- Alpha grid (log-space, same as dual_ML.py line 818)
    alphas = np.logspace(-4, 2, 200)

    # --- Cross-validation
    mse_mat = np.zeros((len(unique_folds), len(alphas)))
    for fi, f in enumerate(tqdm(unique_folds, desc="CV Folds", unit="fold")):
        val_idx = np.where(foldid == f)[0]
        trn_idx = np.where(foldid != f)[0]
        X_tr, y_tr = Xtr[trn_idx], ytr[trn_idx]
        X_va, y_va = Xtr[val_idx], ytr[val_idx]
        for li, lam in enumerate(alphas):
            m = Ridge(alpha=lam, fit_intercept=False)
            m.fit(X_tr, y_tr)
            mse_mat[fi, li] = mean_squared_error(y_va, m.predict(X_va))

    mean_mse  = mse_mat.mean(axis=0)
    std_mse   = mse_mat.std(axis=0, ddof=1)
    imin      = np.argmin(mean_mse)
    se_min    = std_mse[imin] / np.sqrt(len(unique_folds))
    threshold = mean_mse[imin] + se_min

    # 1-SE rule: largest alpha whose CV error is within one SE of the minimum
    candidates = np.where(mean_mse <= threshold)[0]
    idx_1se    = candidates[-1] if len(candidates) > 0 else imin
    alpha_1se  = float(alphas[idx_1se])

    # --- Fit final model on standardised Y
    final_model = Ridge(alpha=alpha_1se, fit_intercept=False)
    final_model.fit(Xtr, ytr)

    # --- Predictions back-transformed to original Y scale
    y_hat_train = final_model.predict(Xtr) * y_sd + y_mu
    y_hat_test  = final_model.predict(Xte) * y_sd + y_mu

    # --- DualML weights (original Ytrain passed, matching dual_ML.py lines 874-882)
    backpack = {
        "type": "RR",
        "params": {
            "Xtrain":       Xtrain_df.values,
            "Xtest":        Xtest_df.values,
            "Ytrain":       Ytrain.values,
            "dates_ins":    data.loc[idx_ins, "Date"].tolist(),
            "model_object": final_model,
            "lmbda":        alpha_1se,
            "intercept":    False,
        },
    }
    res              = dualml_rr(backpack)
    df_weights       = res["weights"]
    df_contributions = res["contributions"]

    pred_df = pd.DataFrame({
        "Date":   np.concatenate([data.loc[idx_ins, "Date"].values,
                                  data.loc[idx_oos, "Date"].values]),
        "set":    ["train"] * len(y_hat_train) + ["test"] * len(y_hat_test),
        "y_true": np.concatenate([Ytrain.values, Ytest.values]),
        "y_hat":  np.concatenate([y_hat_train, y_hat_test]),
    })

    return df_weights, df_contributions, pred_df, alpha_1se


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data = load_data()
    df_w, df_c, pred_df, alpha = compute_rr_weights(data)

    pd.DataFrame({
        "Date":         df_w["date"],
        "Weight":       df_w.iloc[:, 0],
        "Contribution": df_c.iloc[:, 0],
    }).to_csv("results_RR.csv", index=False)
    pred_df.to_csv("predictions_RR.csv", index=False)
    print(f"alpha_1se = {alpha:.6f}")
    print("Saved: results_RR.csv, predictions_RR.csv")
    print(f"Weight matrix shape : {df_w.shape[0]} train obs × {df_w.shape[1] - 1} test obs")

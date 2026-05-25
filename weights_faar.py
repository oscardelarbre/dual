"""
weights_faar.py — Factor-Augmented AR (FAAR/OLS) observation weights.

Source: dual_ML.py, section "00. OLS - e.g., FAAR4"
Helper: auxiliaries.dualml_ols

Weight formula (OLS dual):
    W = X_test @ (X_train' X_train)^{-1} @ X_train'
    rows = training observations, columns = test observations.

Contributions (cumulative):
    C_i = cumsum_t( w_t * (y_t - ybar) ) + ybar
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from auxiliaries import dualml_ols


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
# PCA factor construction  (dual_ML.py lines 78-105)
# ---------------------------------------------------------------------------

def build_pca_factors(data, cutoff="2019-12-31", n_components=4):
    """
    Build in-sample PCA factors (fixed window) and expanding-window OOS factors.

    In-sample  : PCA fitted once on the full in-sample window.
    Out-of-sample : expanding window — PCA re-fitted up to each test date.

    Sign conventions from dual_ML.py:
        pc_train[:, [0, 3]] *= -1
        pc_test[:,  :3]     *= -1
    """
    idx_ins = data["Date"] <= cutoff
    idx_oos = data["Date"] > cutoff

    # L0_ columns, skipping the first 4 (same slice as dual_ML.py line 78)
    pca_cols = [c for c in data.columns if c.startswith("L0_")][4:]

    # --- In-sample PCA (fitted once)
    X_raw   = data.loc[idx_ins, pca_cols]
    pca     = PCA(n_components=n_components, svd_solver='full')
    pc_train = pca.fit_transform((X_raw - X_raw.mean()) / X_raw.std(ddof=1))
    pc_train[:, [0, 3]] *= -1

    # --- OOS PCA (expanding window)
    pc_test = []
    for d in data.loc[idx_oos, "Date"]:
        X_c  = data.loc[data["Date"] <= d, pca_cols]
        pca_tmp = PCA(n_components=n_components, svd_solver='full')
        pcs  = pca_tmp.fit_transform((X_c - X_c.mean()) / X_c.std(ddof=1))
        pc_test.append(pcs[-1])
    pc_test = np.vstack(pc_test)
    pc_test[:, :3] *= -1

    pc_cols     = [f"L0_PC{i}" for i in range(1, n_components + 1)]
    pc_train_df = pd.DataFrame(pc_train, columns=pc_cols, index=data.index[idx_ins])
    pc_test_df  = pd.DataFrame(pc_test,  columns=pc_cols, index=data.index[idx_oos])
    pc = pd.concat([pc_train_df, pc_test_df]).sort_index()

    for i in range(1, n_components + 1):
        pc[f"L1_PC{i}"] = pc[f"L0_PC{i}"].shift(1)

    base_cols = ["Date", "y"] + [c for c in data.columns if c.startswith("L_")]
    data_pca  = (data[base_cols]
                 .merge(pc, left_index=True, right_index=True)
                 .dropna()
                 .round(9))

    return data_pca


# ---------------------------------------------------------------------------
# Weight computation  (dual_ML.py lines 108-163)
# ---------------------------------------------------------------------------

def compute_faar_weights(data, cutoff="2019-12-31"):
    """
    Compute FAAR observation weights and contributions.

    Parameters
    ----------
    data    : DataFrame returned by load_data()
    cutoff  : last in-sample date (inclusive)

    Returns
    -------
    df_weights      : DataFrame (n_train × n_test+1), last col = 'date'
    df_contributions: DataFrame same shape
    pred_df         : DataFrame with train+test OLS predictions
    """
    data_pca = build_pca_factors(data, cutoff=cutoff)

    idx_ins = data["Date"] <= cutoff
    idx_oos = data["Date"] > cutoff

    # --- Design matrices with intercept prepended
    X_ar_train = data_pca.loc[data_pca["Date"] <= cutoff].iloc[:, 2:].to_numpy()
    y_ar_train = data_pca.loc[data_pca["Date"] <= cutoff, "y"].to_numpy()
    X_ar_train = np.column_stack([np.ones(X_ar_train.shape[0]), X_ar_train])

    X_ar_test  = data_pca.loc[data_pca["Date"] > cutoff].iloc[:, 2:].to_numpy()
    X_ar_test  = np.column_stack([np.ones(X_ar_test.shape[0]), X_ar_test])

    # Skip first 6 rows (lag initialisation), same as dual_ML.py line 123
    backpack = {
        'type': 'OLS',
        'params': {
            'Xtrain':    X_ar_train[6:],
            'Xtest':     X_ar_test,
            'Ytrain':    y_ar_train[6:],
            'dates_ins': data.loc[idx_ins, "Date"].reset_index(drop=True)[6:],
            'intercept': True,
        },
    }

    dual_res         = dualml_ols(backpack)
    df_weights       = dual_res['weights']
    df_contributions = dual_res['contributions']

    # --- Final OLS fit for predictions (statsmodels, same as dual_ML.py lines 145-163)
    ols_model  = sm.OLS(backpack['params']['Ytrain'],
                        backpack['params']['Xtrain']).fit()
    yhat_train = ols_model.fittedvalues
    yhat_test  = ols_model.predict(backpack['params']['Xtest'])

    dates_train = data_pca.loc[data_pca["Date"] <= cutoff,
                               "Date"].iloc[6:].reset_index(drop=True)
    dates_test  = data_pca.loc[data_pca["Date"] > cutoff,
                               "Date"].reset_index(drop=True)

    pred_df = pd.DataFrame({
        "date":   pd.concat([dates_train, dates_test], ignore_index=True),
        "set":    ["train"] * len(yhat_train) + ["test"] * len(yhat_test),
        "y_true": np.concatenate([
            backpack['params']['Ytrain'],
            data_pca.loc[data_pca["Date"] > cutoff, "y"].values
        ]),
        "y_hat":  np.concatenate([yhat_train, yhat_test])
    })

    return df_weights, df_contributions, pred_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data = load_data()
    df_w, df_c, pred_df = compute_faar_weights(data)

    pd.DataFrame({
        'Date':         df_w['date'],
        'Weight':       df_w.iloc[:, 0],
        'Contribution': df_c.iloc[:, 0],
    }).to_csv("results_OLS.csv", index=False)
    pred_df.to_csv("OLS_Prediction.csv", index=False)
    print("Saved: results_OLS.csv, OLS_Prediction.csv")
    print(f"Weight matrix shape : {df_w.shape[0]} train obs × {df_w.shape[1] - 1} test obs")

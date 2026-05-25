"""
weights_rf.py — Random Forest (RF) observation weights.

Source: dual_ML.py, section "1.bis Random Forest (Sklearn)"
Helper: auxiliaries.get_weights_RF

Weight formula:
    K[i, j] = (1/n_trees) * sum_b  count_b(i) / total_b(j)
    where count_b(i) is the bootstrap count of training obs i in tree b,
    and total_b(j) is the total bootstrap mass in the same leaf as test obs j.
    Columns are then normalised to sum to 1.
    Shape: (n_train, n_test) — rows = training obs, columns = test obs.

Contributions (cumulative, for a given test observation j):
    C_i = cumsum_t( w_t * (y_t - ybar) ) + ybar

Hyperparameters (from dual_ML.py):
    n_estimators=500, max_features=1/3, min_samples_leaf=5,
    bootstrap=True, random_state=42
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from auxiliaries import get_weights_RF


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
# Weight computation  (dual_ML.py lines 352-410)
# ---------------------------------------------------------------------------

def compute_rf_weights(data, cutoff="2019-12-31"):
    """
    Fit a Random Forest and compute its observation weight matrix.

    Parameters
    ----------
    data   : DataFrame returned by load_data()
    cutoff : last in-sample date (inclusive)

    Returns
    -------
    K          : ndarray (n_train, n_test), raw weight matrix
    obs_imp_rf : DataFrame — K with a leading 'date' column and named row/col index
    pred_df    : DataFrame with train+test predictions
    mod_rf     : fitted RandomForestRegressor
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

    # --- Hyperparameters (same as dual_ML.py lines 352-358)
    params_rf = {
        'n_estimators':     500,
        'max_features':     1 / 3,
        'min_samples_leaf': 5,
        'bootstrap':        True,
        'random_state':     42,
    }

    mod_rf = RandomForestRegressor(**params_rf)
    mod_rf.fit(Xtrain, Ytrain)

    # get_weights_RF returns K of shape (n_train, n_test), columns sum to 1
    K = get_weights_RF(mod_rf, Ytrain, Xtrain, Xtest)

    dates_ins  = data.loc[idx_ins, "Date"].values
    obs_imp_rf = pd.DataFrame(
        K,
        index=[f'obs_ins_{i+1}' for i in range(K.shape[0])],
        columns=[f'obs_oos_{j+1}' for j in range(K.shape[1])]
    )
    obs_imp_rf.insert(0, "date", dates_ins)

    pred_df = pd.DataFrame({
        "date":  np.concatenate([data.loc[idx_ins, "Date"],
                                 data.loc[idx_oos, "Date"]]),
        "set":   ["train"] * len(Ytrain) + ["test"] * len(Ytest),
        "y_true": np.concatenate([Ytrain, Ytest]),
        "y_hat": np.concatenate([mod_rf.predict(Xtrain), mod_rf.predict(Xtest)]),
    })

    return K, obs_imp_rf, pred_df, mod_rf


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data   = load_data()
    K, obs_imp_rf, pred_df, mod_rf = compute_rf_weights(data)

    idx_ins    = data["Date"] <= "2019-12-31"
    Ytrain     = data.loc[idx_ins, "y"].values
    mean_Y     = Ytrain.mean()
    centered_Y = Ytrain - mean_Y

    # Weights and contributions for the first test observation (OOS = 1)
    weights_oos1  = obs_imp_rf.iloc[:, 1].values   # col 0 = 'date'
    contributions = np.cumsum(weights_oos1 * centered_Y) + mean_Y

    pd.DataFrame({
        "Date":         obs_imp_rf["date"].values,
        "Weight":       weights_oos1,
        "Contribution": contributions,
    }).to_csv("results_RF_sklearn.csv", index=False)
    pred_df.to_csv("predictions_RF_all.csv", index=False)
    print("Saved: results_RF_sklearn.csv, predictions_RF_all.csv")
    print(f"Weight matrix shape : {K.shape[0]} train obs × {K.shape[1]} test obs")

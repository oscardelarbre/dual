"""
weights_lgb.py — LightGBM (LGB) observation weights.

Source: dual_ML.py, section "2. Boosted Trees (LightGBM)"
Helper: auxiliaries.GeertsemaLu2023  (Algorithm I + II from Geertsema & Lu 2023)

Weight formula:
    K = Algo-II accumulation over boosting rounds of
        leaf-coincidence matrices scaled by tree prediction weights.
    Shape: (n_train, n_test) — rows = training obs, columns = test obs.

Contributions (cumulative, for a given test observation j):
    C_i = cumsum_t( w_t * (y_t - ybar) ) + ybar

Hyperparameters (from dual_ML.py):
    boosting_type='gbdt', objective='regression', metric='rmse',
    num_leaves=14, min_data=2, learning_rate=0.01, num_boost_round=100
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from auxiliaries import GeertsemaLu2023


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
# Weight computation  (dual_ML.py lines 468-522)
# ---------------------------------------------------------------------------

def compute_lgb_weights(data, cutoff="2019-12-31"):
    """
    Train LightGBM and compute Geertsema-Lu observation weights.

    Parameters
    ----------
    data   : DataFrame returned by load_data()
    cutoff : last in-sample date (inclusive)

    Returns
    -------
    obs_imp    : ndarray (n_train, n_test), raw weight matrix
    obs_imp_df : DataFrame with datetime index (training dates) and named columns
    pred_df    : DataFrame with train+test predictions
    model      : fitted lightgbm Booster
    """
    idx_ins = data["Date"] <= cutoff
    idx_oos = data["Date"] > cutoff

    # Global standardisation (same as dual_ML.py lines 57-63)
    scaler_mean = data.loc[idx_ins, data.columns[2:]].mean()
    scaler_std  = data.loc[idx_ins, data.columns[2:]].std(ddof=1)
    Ytrain  = data.loc[idx_ins, "y"].values
    Ytest   = data.loc[idx_oos, "y"].values
    Xtrain  = ((data.loc[idx_ins, data.columns[2:]] - scaler_mean) / scaler_std).values
    Xtest   = ((data.loc[idx_oos, data.columns[2:]] - scaler_mean) / scaler_std).values
    dates_ins = data.loc[idx_ins, "Date"].reset_index(drop=True)

    # --- Hyperparameters (same as dual_ML.py lines 468-476)
    params_lgb = {
        "boosting_type": "gbdt",
        "objective":     "regression",
        "metric":        "rmse",
        "num_leaves":    14,
        "verbose":       0,
        "min_data":      2,
        "learning_rate": 0.01,
    }

    print("Fitting LGB...")
    dtrain = lgb.Dataset(Xtrain, label=Ytrain)
    model  = lgb.train(params_lgb, train_set=dtrain, num_boost_round=100)
    print("Done.")

    # --- Geertsema & Lu (2023) observation weights
    obs_imp = GeertsemaLu2023(model, Ytrain, Xtrain, X_test=Xtest)
    obs_imp_df = pd.DataFrame(
        obs_imp,
        index=dates_ins,
        columns=[f"obs_oos_{t}" for t in range(1, obs_imp.shape[1] + 1)]
    )

    pred_df = pd.DataFrame({
        "Date":   np.concatenate([data.loc[idx_ins, "Date"].values,
                                  data.loc[idx_oos, "Date"].values]),
        "set":    ["train"] * len(Ytrain) + ["test"] * len(Ytest),
        "y_true": np.concatenate([Ytrain, Ytest]),
        "y_hat":  np.concatenate([model.predict(Xtrain), model.predict(Xtest)]),
    })

    return obs_imp, obs_imp_df, pred_df, model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data = load_data()
    obs_imp, obs_imp_df, pred_df, model = compute_lgb_weights(data)

    idx_ins  = data["Date"] <= "2019-12-31"
    Ytrain   = data.loc[idx_ins, "y"].values
    mean_Y   = Ytrain.mean()
    centered = Ytrain - mean_Y

    # Weights and contributions for the first test observation (OOS = 1)
    weights = obs_imp_df.iloc[:, 0].values
    contrib = np.cumsum(weights * centered) + mean_Y

    pd.DataFrame({
        "Date":         obs_imp_df.index,
        "Weight":       weights,
        "Contribution": contrib,
    }).to_csv("results_LGB.csv", index=False)
    pred_df.to_csv("predictions_LGB.csv", index=False)
    print("Saved: results_LGB.csv, predictions_LGB.csv")
    print(f"Weight matrix shape : {obs_imp.shape[0]} train obs × {obs_imp.shape[1]} test obs")

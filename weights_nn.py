"""
weights_nn.py — Neural Network (NN) observation weights.

Source: NN.py (pipeline) + MLP.py (architecture + training) + auxiliaries.dualml_nn
Helper: MLP.MLP for training, auxiliaries.dualml_nn for weight extraction

Weight formula (NTK-inspired kernel ridge on last hidden layer):
    portfolio_weights[j, i] = mean_b( (H_test[j] @ H_train[i].T) @ K_inv_b )
    where H is the last-hidden-layer embedding and K_inv_b is the pseudo-inverse
    of the empirical NTK + lambda*I for bootstrap replicate b.
    Shape: (n_test, n_train); dualml_nn transposes to (n_train, n_test).

Contributions (cumulative, for a given test observation j):
    C_i = cumsum_t( w_t * (y_t - ybar) / y_std ) * y_std + ybar

Hyperparameters (from NN.py):
    nodes=[400,400,400], dropout=0.2, lr=0.001, epochs=100,
    patience=10, num_average=30, sampling_rate=0.85, batch_size=32,
    lambda_grid=linspace(1e-3, 50, 200), seed=1234

Warning
-------
Training 30 bootstrap replicates of a 3-layer MLP (400 units each) is slow.
On a CPU this may take 30–60+ minutes.
"""

import os
import numpy as np
import pandas as pd
from MLP import MLP
from auxiliaries import dualml_nn


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
# Weight computation  (NN.py lines 34-98)
# ---------------------------------------------------------------------------

def compute_nn_weights(data, cutoff="2019-12-31", seed=1234):
    """
    Train the MLP ensemble and extract observation portfolio weights.

    Parameters
    ----------
    data   : DataFrame returned by load_data()
    cutoff : last in-sample date (inclusive)
    seed   : random seed (default 1234, same as NN.py)

    Returns
    -------
    df_weights       : DataFrame (n_train × n_test)
    df_contributions : DataFrame same shape
    pred_df          : DataFrame with train+test predictions
    mod_NN           : raw dict returned by MLP()
    """
    idx_ins = data["Date"] <= cutoff
    idx_oos = data["Date"] > cutoff

    # Global standardisation (same as dual_ML.py / NN.py)
    scaler_mean = data.loc[idx_ins, data.columns[2:]].mean()
    scaler_std  = data.loc[idx_ins, data.columns[2:]].std(ddof=1)
    Ytrain  = data.loc[idx_ins, "y"].values
    Ytest   = data.loc[idx_oos, "y"].values
    Xtrain  = ((data.loc[idx_ins, data.columns[2:]] - scaler_mean) / scaler_std).values
    Xtest   = ((data.loc[idx_oos, data.columns[2:]] - scaler_mean) / scaler_std).values

    dates_ins = data.loc[idx_ins, "Date"].reset_index(drop=True)
    dates_oos = data.loc[idx_oos, "Date"].reset_index(drop=True)

    # --- Hyperparameters (same as NN.py lines 34-48)
    nn_hyps = {
        'n_features':    Xtrain.shape[1],
        'nodes':         [400, 400, 400],
        'patience':      10,
        'epochs':        100,
        'lr':            0.001,
        'tol':           0.01,
        'show_train':    2,
        'num_average':   30,
        'dropout_rate':  0.2,
        'sampling_rate': 0.85,
        'batch_size':    32,
        'num_batches':   None,
        'lambda_grid':   np.linspace(1e-3, 50, 200),
    }

    # X_OOS=Xtest ensures portfolio_weights covers the test set (NN.py line 61)
    mod_NN = MLP(
        X           = Xtrain,
        Y           = Ytrain,
        Xtest       = Xtest,
        Ytest       = Ytest,
        X_OOS       = Xtest,
        nn_hyps     = nn_hyps,
        standardize = True,
        seed        = seed,
    )

    # Rename key to match dualml_nn expectation (NN.py lines 67-68)
    if 'portfolio_weights' in mod_NN:
        mod_NN['portfolio.weights'] = mod_NN.pop('portfolio_weights')

    # Average OOS predictions across bootstrap replicates (NN.py lines 71-78)
    if 'pred_OOS_B' in mod_NN and mod_NN['pred_OOS_B'].shape[0] == len(dates_oos):
        mod_NN['pred'] = np.mean(mod_NN['pred_OOS_B'], axis=1)
    else:
        mod_NN['pred'] = np.mean(mod_NN['pred_test_B'], axis=1)

    backpack = {
        'type': 'NN',
        'params': {
            'Xtrain':       None,
            'Xtest':        None,
            'Ytrain':       Ytrain,
            'dates_ins':    dates_ins.values,
            'model_object': mod_NN,
            'lmbda':        None,
            'intercept':    None,
        },
    }

    df_weights, df_contributions = dualml_nn(backpack)

    pred_train = mod_NN["pred_train_B"].mean(axis=1)
    if "pred_OOS_B" in mod_NN and mod_NN["pred_OOS_B"].shape[0] == len(dates_oos):
        pred_test = mod_NN["pred_OOS_B"].mean(axis=1)
    else:
        pred_test = mod_NN["pred_test_B"].mean(axis=1)

    pred_df = pd.concat([
        pd.DataFrame({"Date": dates_ins, "set": "train",
                      "y_true": Ytrain, "y_hat": pred_train}),
        pd.DataFrame({"Date": dates_oos, "set": "test",
                      "y_true": Ytest,  "y_hat": pred_test}),
    ], ignore_index=True)

    return df_weights, df_contributions, pred_df, mod_NN


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data = load_data()
    df_w, df_c, pred_df, mod_NN = compute_nn_weights(data)

    dates_ins = data.loc[data["Date"] <= "2019-12-31", "Date"].reset_index(drop=True)
    pd.DataFrame({
        'Date':         dates_ins,
        'Weight':       df_w.iloc[:, 0].values,
        'Contribution': df_c.iloc[:, 0].values,
    }).to_csv("results_NN.csv", index=False)
    pred_df.to_csv("nn_predictions_all.csv", index=False)
    print("Saved: results_NN.csv, nn_predictions_all.csv")
    print(f"Weight matrix shape : {df_w.shape[0]} train obs × {df_w.shape[1]} test obs")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from MLP import MLP, dualml_nn


# ------------------------------
# 1. Load data exactly as specified
# ------------------------------
dir = "US_data direction"  # set the path to the US_data folder here
data = pd.read_csv(dir)
data["Date"] = pd.to_datetime(data["Date"])
data = data.drop(columns=[c for c in data.columns
                          if any(k in c for k in ("outputGap", "chan_outputgap"))])

idx_ins = data["Date"] <= "2019-12-31"
idx_oos = data["Date"] >  "2019-12-31"

scaler_mean = data.loc[idx_ins, data.columns[2:]].mean()
scaler_std  = data.loc[idx_ins, data.columns[2:]].std(ddof=1)

Ytrain = data.loc[idx_ins, "y"].values
Ytest  = data.loc[idx_oos, "y"].values
Xtrain = ((data.loc[idx_ins, data.columns[2:]] - scaler_mean) / scaler_std).values
Xtest  = ((data.loc[idx_oos, data.columns[2:]] - scaler_mean) / scaler_std).values

dates_ins = data.loc[idx_ins, "Date"].reset_index(drop=True)
dates_oos = data.loc[idx_oos, "Date"].reset_index(drop=True)

# --------------------------------------------------
# 5.1 | Set the Hyperparameters
# --------------------------------------------------

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
    'lambda_grid':   np.linspace(1e-3, 50, 200)
}


# --------------------------------------------------
# 5.2 | Run the Neural Net
# --------------------------------------------------

mod_NN = MLP(
    X           = Xtrain,
    Y           = Ytrain,
    Xtest       = Xtest,
    Ytest       = Ytest,
    X_OOS       = Xtest,
    nn_hyps     = nn_hyps,
    standardize = True,
    seed        = 1234
)

# rename portfolio weights key if nécessaire
if 'portfolio_weights' in mod_NN:
    mod_NN['portfolio.weights'] = mod_NN.pop('portfolio_weights')

# average out-of-sample predictions
if 'pred_OOS_B' in mod_NN:
    mod_NN['pred'] = np.mean(mod_NN['pred_OOS_B'], axis=1)
elif 'pred_test_B' in mod_NN:
    mod_NN['pred'] = np.mean(mod_NN['pred_test_B'], axis=1)
else:
    raise KeyError(
        "Impossible de trouver 'pred_OOS_B' ni 'pred_test_B' dans la sortie de MLP."
    )


# --------------------------------------------------
# 5.3 | Get Dual Interpretation
# --------------------------------------------------

backpack = {
    'type': 'NN',
    'params': {
        'Xtrain':    None,
        'Xtest':     None,
        'Ytrain':    Ytrain,
        'dates_ins': dates_ins.values,
        'model_object': mod_NN,
        'lmbda':     None,
        'intercept': None
    }
}

df_weights, df_contributions = dualml_nn(backpack)


# --------------------------------------------------
# 5.4 | Save weights & contributions
# --------------------------------------------------

results_df = pd.DataFrame({
    'Date':         dates_ins,
    'Weight':       df_weights.iloc[:, 0].values,
    'Contribution': df_contributions.iloc[:, 0].values
})
results_df.to_csv('results_NN.csv', index=False)


# --------------------------------------------------
# 5.5 | Save raw predictions
# --------------------------------------------------

# moyenne des prédictions in-sample bootstrap
pred_train = mod_NN["pred_train_B"].mean(axis=1)

# moyenne des prédictions out-of-sample bootstrap
if "pred_OOS_B" in mod_NN and mod_NN["pred_OOS_B"].shape[0] == len(dates_oos):
    pred_test = mod_NN["pred_OOS_B"].mean(axis=1)
else:
    pred_test = mod_NN["pred_test_B"].mean(axis=1)

# DataFrame pour l'échantillon d'entraînement
df_train = pd.DataFrame({
    "Date":    dates_ins,
    "set":     "train",
    "y_true":  Ytrain,
    "y_hat":   pred_train
})

# DataFrame pour l'échantillon out-of-sample
df_test = pd.DataFrame({
    "Date":    dates_oos,
    "set":     "test",
    "y_true":  Ytest,
    "y_hat":   pred_test
})

# Fusion et export
df_all = pd.concat([df_train, df_test], ignore_index=True)
df_all.to_csv("nn_predictions_all.csv", index=False)


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# If not already in memory, load the saved results:
# results_df = pd.read_csv('results_NN.csv')

# Ensure the Date column is datetime
results_df['Date'] = pd.to_datetime(results_df['Date'])

dates = results_df['Date']
weights = results_df['Weight']
contrib = results_df['Contribution']

# Configure date locator/formatter
locator   = mdates.YearLocator()
formatter = mdates.DateFormatter("%Y")

# ---- Observation Weights ----
plt.figure(figsize=(25, 10))
plt.plot(dates, weights, color='red', label='Weight')
plt.title("NN — Observation Weights")
plt.xlabel("Date")
plt.ylabel("Weight")
ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.xticks(rotation=45, ha="right")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.show()

# ---- Observation Contributions ----
plt.figure(figsize=(25, 10))
plt.plot(dates, contrib, color='blue', label='Contribution')
plt.title("NN — Observation Contributions")
plt.xlabel("Date")
plt.ylabel("Contribution")
ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.xticks(rotation=45, ha="right")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.show()




"""
df_r_w  = pd.read_csv(
    '/Users/silyamoussous/Desktop/Stage_Missions/Mission2_Traduction_R_a_Python/code_R/results_NN.csv'
).rename(columns={'Weight': 'Weight_R', 'Contribution': 'Contribution_R'})
df_py_w = pd.read_csv('/Users/silyamoussous/Desktop/Mission_Stage/results_NN.csv').rename(
    columns={'Weight': 'Weight_Py', 'Contribution': 'Contribution_Py'}
)

n = min(len(df_r_w), len(df_py_w))
merge_w = pd.concat(
    [df_r_w.iloc[:n][["Weight_R", "Contribution_R"]],
     df_py_w.iloc[:n][["Weight_Py", "Contribution_Py"]]],
    axis=1
).round(10)

cutoff_idx = pd.to_datetime(df_r_w.iloc[:n]['Date']).gt("2019-12-31").idxmax()
corr_w  = merge_w[["Weight_R", "Weight_Py"]].corr().iloc[0, 1]
corr_c  = merge_w[["Contribution_R", "Contribution_Py"]].corr().iloc[0, 1]

plt.figure(figsize=(15, 5))
plt.plot(merge_w.index, merge_w["Weight_R"],  label="Weight R")
plt.plot(merge_w.index, merge_w["Weight_Py"], "--", label="Weight Python")
plt.axvline(cutoff_idx, ls=":", color="grey")
plt.title(f"Weights – corr = {corr_w:.2%}")
plt.grid(); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(15, 5))
plt.plot(merge_w.index, merge_w["Contribution_R"],  label="Contribution R")
plt.plot(merge_w.index, merge_w["Contribution_Py"], "--", label="Contribution Python")
plt.axvline(cutoff_idx, ls=":", color="grey")
plt.title(f"Contributions – corr = {corr_c:.2%}")
plt.grid(); plt.legend(); plt.tight_layout(); plt.show()


# ------------------------------
# 5.6 Extract for two target dates in OOS
# ------------------------------

# Liste des dates test
oos_dates = pd.to_datetime(data_test.index)
target_dates = pd.to_datetime(["1962-09-01", "1964-06-01"])  # ✅ à adapter si tu veux

for td in target_dates:
    if td not in oos_dates:
        print(f"❌ Date {td} not in OOS sample")
        continue

    j = np.where(oos_dates == td)[0][0]  # index colonne dans df_weights

    df_out = pd.DataFrame({
        "Date": pd.to_datetime(data_train.index),
        "Weight": df_weights.iloc[:, j].values,
        "Contribution": df_contributions.iloc[:, j].values
    })

    output_path = f"/Users/silyamoussous/Desktop/Mission_Stage/results_NN_{td.strftime('%Y%m%d')}.csv"
    df_out.to_csv(output_path, index=False)
    print(f"✅ Fichier créé pour {td.date()} → {output_path}")
"""
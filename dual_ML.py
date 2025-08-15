#%%
# --------------------------------------------------
# Import
# --------------------------------------------------

import os

"""  ====================== RPY2 OPTION ======================== """

"""
R_home = os.popen('R RHOME').read().splitlines()[-1]
os.environ['R_HOME'] = R_home
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from auxiliaries import GeertsemaLu2023, block_sampler, dualml_ols, krr_weight, compute_kernel_r,compute_kernel_r_py , dualml_rr, dualml_nn, get_weights_RF
from sklearn.decomposition import PCA
import statsmodels.api as sm

"""  ====================== RPY2 OPTION ======================== """

"""
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, default_converter
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.vectors import FloatVector, IntVector
"""

from auxiliaries import dualml_rf
from MLP import MLP
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import PredefinedSplit
from tqdm import tqdm


# --------------------------------------------------
#  Load data and basic cleaning
# --------------------------------------------------

dir = "/Users/silyamoussous/Desktop/Dual/US_data.csv"  # set the path to the US_data folder here

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



#%%

# --------------------------------------------------
# 00. OLS - e.g., FAAR4
# --------------------------------------------------

# Build PCA factors L0_/L1_

pca_cols = [c for c in data.columns if c.startswith("L0_")][4:]
X_raw = data.loc[idx_ins, pca_cols]
pca = PCA(n_components=4, svd_solver='full')
pc_train = pca.fit_transform((X_raw - X_raw.mean()) / X_raw.std(ddof=1))
pc_train[:, [0, 3]] *= -1

pc_test = []
for d in data.loc[idx_oos, "Date"]:
    X_c = data.loc[data["Date"] <= d, pca_cols]
    pca = PCA(n_components=4, svd_solver='full')
    pcs = pca.fit_transform((X_c - X_c.mean()) / X_c.std(ddof=1))
    pc_test.append(pcs[-1])
pc_test = np.vstack(pc_test)
pc_test[:, :3] *= -1

pc_cols = [f"L0_PC{i}" for i in range(1, 5)]
pc_train_df = pd.DataFrame(pc_train, columns=pc_cols, index=data.index[idx_ins])
pc_test_df  = pd.DataFrame(pc_test,  columns=pc_cols, index=data.index[idx_oos])
pc = pd.concat([pc_train_df, pc_test_df]).sort_index()

for i in range(1, 5):
    pc[f"L1_PC{i}"] = pc[f"L0_PC{i}"].shift(1)

base_cols = ["Date", "y"] + [c for c in data.columns if c.startswith("L_")]
data_pca = (data[base_cols]
            .merge(pc, left_index=True, right_index=True)
            .dropna()
            .round(9))


# Prepare matrices for dualml_ols

X_ar_train = data_pca.loc[data_pca["Date"] <= "2019-12-31"].iloc[:, 2:].to_numpy()
y_ar_train = data_pca.loc[data_pca["Date"] <= "2019-12-31", "y"].to_numpy()
X_ar_train = np.column_stack([np.ones(X_ar_train.shape[0]), X_ar_train])

X_ar_test = data_pca.loc[data_pca["Date"] > "2019-12-31"].iloc[:, 2:].to_numpy()
X_ar_test = np.column_stack([
    np.ones(X_ar_test.shape[0]),
    X_ar_test
])

backpack = {
    'type': 'OLS',
    'params': {
        'Xtrain': X_ar_train[6:],
        'Xtest':  X_ar_test,
        'Ytrain': y_ar_train[6:],
        'dates_ins': data.loc[data["Date"] <= "2019-12-31",
                              "Date"].reset_index(drop=True)[6:],
        'intercept': True,
    },
}

dual_res = dualml_ols(backpack)

pd.DataFrame(
    {
        'Date':  dual_res['weights']['date'],
        'Weight': dual_res['weights'].iloc[:, 0],
        'Contribution': dual_res['contributions'].iloc[:, 0],
    }
).to_csv("results_OLS.csv", index=False)
print("results_OLS.csv Created")

# Fit final OLS and save predictions

ols_model = sm.OLS(backpack['params']['Ytrain'],
                   backpack['params']['Xtrain']).fit()
yhat_train = ols_model.fittedvalues
yhat_test  = ols_model.predict(backpack['params']['Xtest'])

dates_train = data_pca.loc[data_pca["Date"] <= "2019-12-31",
                           "Date"].iloc[6:].reset_index(drop=True)
dates_test  = data_pca.loc[data_pca["Date"] > "2019-12-31", "Date"]

pred_df = pd.DataFrame({
    "date":   pd.concat([dates_train, dates_test], ignore_index=True),
    "set":    ["train"] * len(yhat_train) + ["test"] * len(yhat_test),
    "y_true": np.concatenate([
        backpack['params']['Ytrain'],
        data_pca.loc[data_pca["Date"] > "2019-12-31", "y"]
    ]),
    "y_hat_py": np.concatenate([yhat_train, yhat_test])
})
pred_df.to_csv("OLS_Prediction.csv", index=False)
print("OLS_Prediction.csv Created")

weights_df      = dual_res['weights']
contrib_df      = dual_res['contributions']

# Convert the 'date' column to datetime
dates = pd.to_datetime(weights_df['date'])

# Setup date formatter and locator
locator   = mdates.YearLocator()
formatter = mdates.DateFormatter("%Y")

# ---- Observation Weights ----
plt.figure(figsize=(25, 10))
plt.plot(dates, weights_df.iloc[:, 0], color='red')
plt.title("OLS — Observation Weights")
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
plt.plot(dates, contrib_df.iloc[:, 0], color='blue')
plt.title("OLS — Observation Contributions")
plt.xlabel("Date")
plt.ylabel("Contribution")
ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.xticks(rotation=45, ha="right")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.show()


"""  ====================== RPY2 OPTION ======================== """

"""
#%%
# --------------------------------------------------
# 1. Random Forest (Rpy2)
# --------------------------------------------------

# Hyperparameters
my_blocksize = 8
N_trees = 500
mtry_denom = 3
my_minnodesize = 5
my_samplefraction = 0.75

# Draw blocks of observations for Out-Of-Bag computations
bs = block_sampler(Xtrain,
                   sampling_rate=0.8,
                   block_size=my_blocksize,
                   num_tree=N_trees)


ro.globalenv["Y"] = ro.FloatVector(Ytrain.tolist())
with localconverter(default_converter + pandas2ri.converter):
    ro.globalenv["X"] = ro.conversion.py2rpy(pd.DataFrame(Xtrain))
ro.globalenv["inbag"] = ro.r.list(
    *[ro.IntVector(inbag_row.tolist()) for inbag_row in bs["inbag1"]]
)

#mod_rf = ro.r(f"""
#  ranger::ranger(
#    Y ~ .,
#    data = data.frame(Y = Y, X),
#    num.trees      = {N_trees},
#    inbag          = inbag,
#    mtry           = {int(Xtrain.shape[1]/mtry_denom)},
#    min.node.size  = {my_minnodesize},
#    sample.fraction= {my_samplefraction},
#    keep.inbag     = TRUE,
#    num.threads    = 1
#  )
""")

# --- 1.3 Get Dual Interpretation
backpack = {
    'type': 'RF',
    'params': {
        'Xtrain': Xtrain,
        'Xtest': Xtest,
        'Ytrain': Ytrain,
        'dates_oos': data.loc[idx_oos, 'Date'].values,
        'model_object': mod_rf
    }
}

with localconverter(default_converter + pandas2ri.converter):
    ro.globalenv["X"] = ro.conversion.py2rpy(pd.DataFrame(Xtrain))
    ro.globalenv["Xtest"] = ro.conversion.py2rpy(pd.DataFrame(Xtest))

ro.globalenv["mod_RF"] = mod_rf

test_RF = dualml_rf(backpack)

# --- 1.3.b save results
df_weights = pd.DataFrame({
    'Date': data.loc[idx_ins, 'Date'].values,
    'Weight': test_RF['weights'].iloc[:, 0].values,
    'Contribution': test_RF['contributions'].iloc[:, 0].values,
})
df_weights.to_csv('results_RF.csv', index=False)
print("results_RF.csv Created")



#------- predictions -------#

# --- 1.4 Prédictions out-of-sample
with localconverter(default_converter + pandas2ri.converter):
    ro.globalenv["Xtrain"] = ro.conversion.py2rpy(pd.DataFrame(Xtrain))
    ro.globalenv["Xtest"]  = ro.conversion.py2rpy(pd.DataFrame(Xtest))

Y_hat_train = np.array(ro.r("predict(mod_RF, data = data.frame(Xtrain))$predictions"))
Y_hat_test  = np.array(ro.r("predict(mod_RF, data = data.frame(Xtest ))$predictions"))

df_pred = pd.concat(
    [
        pd.DataFrame({
            "Date": data.loc[idx_ins, "Date"].values,
            "Set":  "Train",
            "y_true": Ytrain,
            "y_hat_py":  Y_hat_train,
        }),
        pd.DataFrame({
            "Date": data.loc[idx_oos, "Date"].values,
            "Set":  "Test",
            "y_true": Ytest,
            "y_hat_py":  Y_hat_test,
        }),
    ],
    ignore_index=True,
)

df_pred.to_csv("RF_predictions.csv", index=False)
print("RF_predictions.csv Created")


# Convert the Date column to datetime if not already
dates = pd.to_datetime(df_weights['Date'])

# Configure date ticks
locator   = mdates.YearLocator()
formatter = mdates.DateFormatter("%Y")

# ---- Observation Weights ----
plt.figure(figsize=(25, 10))
plt.plot(dates, df_weights['Weight'], color='red')
plt.title("Random Forest — Observation Weights")
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
plt.plot(dates, df_weights['Contribution'], color='blue')
plt.title("Random Forest — Observation Contributions")
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

#%%
# --------------------------------------------------
# 1.bis Random Forest (Sklearn)
# --------------------------------------------------

params_rf = {
    'n_estimators': 500,
    'max_features': 1/3,
    'min_samples_leaf': 5,
    'bootstrap': True,
    'random_state': 42
}

mod_rf = RandomForestRegressor(**params_rf)
mod_rf.fit(Xtrain, Ytrain)

# --- Calcul des poids d’observation (à adapter selon ta fonction)
K_rf = get_weights_RF(mod_rf, Ytrain, Xtrain, Xtest)

# Vérif prédictions
pred_skl = mod_rf.predict(Xtest)
pred_via = K_rf.T @ Ytrain     # vecteur de taille n_test

diff = np.abs(pred_skl - pred_via).max()


# --- Mettre en forme avec les bonnes dates
dates_ins = data.loc[idx_ins, "Date"].values

obs_imp_rf = pd.DataFrame(
    K_rf,
    index=[f'obs_ins_{i+1}' for i in range(K_rf.shape[0])],
    columns=[f'obs_oos_{j+1}' for j in range(K_rf.shape[1])]
)

obs_imp_rf.insert(0, "date", dates_ins)


df_preds_py = pd.DataFrame({
    "date": np.concatenate([data.loc[idx_ins, "Date"], data.loc[idx_oos, "Date"]]),
    "y_hat_py": np.concatenate([mod_rf.predict(Xtrain), mod_rf.predict(Xtest)])
})
df_preds_py.to_csv("predictions_RF_all.csv", index=False)


#--------- Enregistrer Weights et Contributions ---------

# Récupérer les poids pour OOS = 1
weights = obs_imp_rf.iloc[:, 1].values  # 1ère colonne après "date"

# Calculer les contributions
mean_Y = Ytrain.mean()
centered_Y = Ytrain - mean_Y
contributions = np.cumsum(weights * centered_Y) + mean_Y

# Concaténer avec les dates
results_rf = pd.DataFrame({
    "Date": obs_imp_rf["date"].values,
    "Weight": weights,
    "Contribution": contributions
})

# Sauvegarder
results_rf.to_csv("results_RF_sklearn.csv", index=False)
print("results_RF_sklearn Created")

# ----------- PLOTS ----------- #
# === FORMAT DATE SEMESTRE === #

def semester_formatter(x, pos=None):
    date = mdates.num2date(x)
    semester = 1 if date.month <= 6 else 2
    return f"{date.year}-S{semester}"

locator = mdates.MonthLocator(interval=6)
formatter = plt.FuncFormatter(semester_formatter)

# === WEIGHTS === #

plt.figure(figsize=(25, 10))
plt.plot(obs_imp_rf["date"], obs_imp_rf.iloc[:, 1], color='red')  # 1ère colonne après "date"
plt.title("RF SkLearn — Observation Weights (OOS = 1)")
plt.xlabel("Date")
plt.ylabel("Weight")
plt.gca().xaxis.set_major_locator(locator)
plt.gca().xaxis.set_major_formatter(formatter)
plt.xticks(rotation=45, ha="right")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.show()

# === CONTRIBUTIONS === #

weights_rf = obs_imp_rf.iloc[:, 1:].values  # Toutes les colonnes numériques (n_train, n_OOS)
mean_Y = Ytrain.mean()
centered_Y = Ytrain - mean_Y
contributions_matrix = np.cumsum(weights_rf * centered_Y[:, np.newaxis], axis=0) + mean_Y

contributions_rf = pd.DataFrame({
    "date": obs_imp_rf["date"],
    "contribution": contributions_matrix[:, 0]  # OOS = 1
})

plt.figure(figsize=(25, 10))
plt.plot(contributions_rf["date"], contributions_rf["contribution"], color='blue')
plt.title("RF SkLearn — Observation Contributions (OOS = 1)")
plt.xlabel("Date")
plt.ylabel("Contribution")
plt.gca().xaxis.set_major_locator(locator)
plt.gca().xaxis.set_major_formatter(formatter)
plt.xticks(rotation=45, ha="right")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.show()

#%%
# --------------------------------------------------
#   2. Boosted Trees (LightGBM)
# --------------------------------------------------

# LGB parameters
params_lgb = {
    "boosting_type": "gbdt",
    "objective": 'regression',
    "metric": 'rmse',
    "num_leaves": 14,
    "verbose": 0,
    "min_data": 2,
    "learning_rate": 0.01
}

# Train
print("Start fitting LGB…")
dtrain = lgb.Dataset(Xtrain, label=Ytrain)
model = lgb.train(params_lgb, train_set=dtrain, num_boost_round=100)
print("Fitting terminé.")

# Observation importance OOS = 1
obs_imp = GeertsemaLu2023(model, Ytrain, Xtrain, X_test=Xtest)
obs_imp = pd.DataFrame(
    obs_imp,
    index=dates_ins,
    columns=[f"obs_oos_{t}" for t in range(1, obs_imp.shape[1] + 1)]
)

# Plot settings
locator   = mdates.YearLocator()
formatter = mdates.DateFormatter("%Y")

# ---- Weights ----
plt.figure(figsize=(25, 10))
plt.plot(dates_ins, obs_imp.iloc[:, 0], color='red')
plt.title("LGB — Observation Weights (OOS = 1)")
plt.xlabel("Date"); plt.ylabel("Weight")
plt.gca().xaxis.set_major_locator(locator)
plt.gca().xaxis.set_major_formatter(formatter)
plt.xticks(rotation=45, ha="right")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.show()

# ---- Contributions ----
weights = obs_imp.values
mean_Y = Ytrain.mean()
centered = Ytrain - mean_Y
contrib = np.cumsum(weights * centered[:, None], axis=0) + mean_Y

plt.figure(figsize=(25, 10))
plt.plot(dates_ins, contrib[:, 0], color='blue')
plt.title("LGB — Observation Contributions (OOS = 1)")
plt.xlabel("Date"); plt.ylabel("Contribution")
plt.gca().xaxis.set_major_locator(locator)
plt.gca().xaxis.set_major_formatter(formatter)
plt.xticks(rotation=45, ha="right")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.show()

#%%
# --------------------------------------------------
#   3. Kernel Ridge-Regression
# --------------------------------------------------


#  Kernel Ridge Regression : hyper-params

krr_kernel        = "laplace"
krr_kernel_sigma  = 1e-4
krr_lmbda         = 1e-5

#  Kernels from kernlab + DualML weights/contributions

K_train, K_test = compute_kernel_r_py(
    Xtrain=Xtrain,
    Xtest=Xtest,
    kernel_type=krr_kernel,
    sigma=krr_kernel_sigma,
)

backpack = {
    "type": "KRR",
    "params": {
        "Xtrain": K_train,
        "Xtest":  K_test,
        "Ytrain": Ytrain,
        "dates_ins": data.loc[idx_ins, "Date"],
        "lmbda":  krr_lmbda,
        "kernel_type": krr_kernel,
        "sigma": krr_kernel_sigma,
    },
}

weights_df, contrib_df = krr_weight(backpack)

pd.DataFrame(
    {
        "Date":        weights_df["Date"],
        "Weight":      weights_df.iloc[:, 0],
        "Contribution": contrib_df.iloc[:, 0],
    }
).to_csv("results_KRR.csv", index=False)
print("results_KRR.csv Created")


#  Predictions

alpha = np.linalg.solve(
    K_train + krr_lmbda * np.eye(K_train.shape[0]),
    Ytrain,
)

Y_hat_ins = K_train @ alpha
Y_hat_oos = K_test  @ alpha

df_pred = pd.concat(
    [
        pd.DataFrame(
            {
                "Date":  data.loc[idx_ins, "Date"].values,
                "Set":   "train",
                "Y_true": Ytrain,
                "Y_hat_py": Y_hat_ins,
            }
        ),
        pd.DataFrame(
            {
                "Date":  data.loc[idx_oos, "Date"].values,
                "Set":   "test",
                "Y_true": Ytest,
                "Y_hat_py": Y_hat_oos,
            }
        ),
    ],
    ignore_index=True,
)
df_pred.to_csv("predictions_KRR.csv", index=False)
print("predictions_KRR.csv Created")
# Convert the Date column to datetime
dates = pd.to_datetime(weights_df["Date"])

# Setup date ticks
locator   = mdates.YearLocator()
formatter = mdates.DateFormatter("%Y")

# ---- Observation Weights ----
plt.figure(figsize=(25, 10))
plt.plot(dates, weights_df.iloc[:, 0], color='red')
plt.title("KRR — Observation Weights")
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
plt.plot(dates, contrib_df.iloc[:, 0], color='blue')
plt.title("KRR — Observation Contributions")
plt.xlabel("Date")
plt.ylabel("Contribution")
ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.xticks(rotation=45, ha="right")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.show()

#%%
"""  ====================== RPY2 OPTION ======================== """

""""
# --------------------------------------------------
#   4. Ridge Regression (Rpy2)
# --------------------------------------------------

ro.r('library(glmnet)')
r = ro.r

#  Standardize X
X_cols   = data.columns[2:]
X_ins    = data.loc[idx_ins, X_cols]
X_oos    = data.loc[idx_oos, X_cols]
scaler_m = X_ins.mean()
scaler_s = X_ins.std(ddof=1)
Xtrain_df = (X_ins - scaler_m) / scaler_s
Xtest_df  = (X_oos - scaler_m) / scaler_s

#  Standardize Y
Ytrain    = data.loc[idx_ins, "y"]
Ytest     = data.loc[idx_oos, "y"]
y_mu      = Ytrain.mean()
y_sd      = Ytrain.std(ddof=1)
Ytrain_rr = (Ytrain - y_mu) / y_sd

# Fold IDs
cv_folds = np.repeat(np.arange(1,10), len(Ytrain)//10)
cv_folds = np.concatenate([cv_folds,
                           np.full(len(Ytrain)-len(cv_folds), 10)])
cv_folds_R = IntVector(cv_folds.tolist())

#  Call glmnet::cv.glmnet()
lmbda_grid = np.linspace(1e-3, 100, 1000)
lmbda_R    = FloatVector(lmbda_grid.tolist())
as_matrix  = r['as.matrix']

with localconverter(ro.default_converter + pandas2ri.converter):
    X_R_df     = ro.conversion.py2rpy(Xtrain_df)
    Xtest_R_df = ro.conversion.py2rpy(Xtest_df)
    Y_R        = ro.conversion.py2rpy(Ytrain_rr)

X_R_mat     = as_matrix(X_R_df)
Xtest_R_mat = as_matrix(Xtest_R_df)

cv_glmnet = r['cv.glmnet']
fit = cv_glmnet(
    X_R_mat, Y_R,
    **{"lambda": lmbda_R},
    intercept=False,
    alpha=0,
    nfolds=10,
    foldid=cv_folds_R,
    standardize=False,
    standardize_response=False,
    type_measure="mse"
)

#  Extract λ₁ₛₑ and predict
lmbda_1se = float(fit.rx2('lambda.1se')[0])

predict = r['predict']
with localconverter(ro.default_converter + pandas2ri.converter):
    y_hat_train_rr = np.array(predict(fit, X_R_mat,    s=lmbda_1se)).ravel()
    y_hat_test_rr  = np.array(predict(fit, Xtest_R_mat, s=lmbda_1se)).ravel()

y_hat_train = y_hat_train_rr * y_sd + y_mu
y_hat_test  = y_hat_test_rr  * y_sd + y_mu

#  Export predictions
pred_df = pd.DataFrame({
    "Date":   np.concatenate([data.loc[idx_ins, "Date"].values,
                              data.loc[idx_oos, "Date"].values]),
    "set":   ["train"] * len(y_hat_train) + ["test"] * len(y_hat_test),
    "y_true": np.concatenate([Ytrain.values, Ytest.values]),
    "y_hat_py":  np.concatenate([y_hat_train, y_hat_test])
})
pred_df.to_csv("predictions_RR.csv", index=False)
print("predictions_RR.csv Created")


# Compute weights & contributions via DualML

backpack = {
    "type": "RR",
    "params": {
        "Xtrain": Xtrain_df.values,
        "Xtest":  Xtest_df.values,
        "Ytrain": Ytrain.values,
        "dates_ins": data.loc[idx_ins, "Date"].tolist(),
        "model_object": fit,
        "lmbda": lmbda_1se,
        "intercept": False
    }
}

test_RR = dualml_rr(backpack)

# Extract and save weights & contributions
df_rr = pd.DataFrame({
    "Date":         test_RR["weights"]["date"],
    "Weight":       test_RR["weights"].iloc[:, 0],
    "Contribution": test_RR["contributions"].iloc[:, 0]
})
df_rr.to_csv("results_RR.csv", index=False)
print("results_RR.csv Created")

# Ensure the Date column is datetime
df_rr["Date"] = pd.to_datetime(df_rr["Date"])
dates = df_rr["Date"]

# Configure tick locator/formatter
locator   = mdates.YearLocator()
formatter = mdates.DateFormatter("%Y")

# ---- Observation Weights ----
plt.figure(figsize=(25, 10))
plt.plot(dates, df_rr["Weight"], color='red')
plt.title("RR — Observation Weights")
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
plt.plot(dates, df_rr["Contribution"], color='blue')
plt.title("RR — Observation Contributions")
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

#%%
# --------------------------------------------------
#   4bis. Ridge Regression (Sklearn)
# --------------------------------------------------


# --- Standardize X
X_cols   = data.columns[2:]
X_ins    = data.loc[idx_ins, X_cols].copy()
X_oos    = data.loc[idx_oos, X_cols].copy()
scaler_m = X_ins.mean()
scaler_s = X_ins.std(ddof=1).replace(0, 1.0)
Xtrain_df = (X_ins - scaler_m) / scaler_s
Xtest_df  = (X_oos - scaler_m) / scaler_s

# --- Standardize Y
Ytrain = data.loc[idx_ins, "y"].astype(float).copy()
Ytest  = data.loc[idx_oos, "y"].astype(float).copy()
y_mu   = Ytrain.mean()
y_sd   = Ytrain.std(ddof=1) or 1.0
Ytrain_rr = (Ytrain - y_mu) / y_sd

Xtr = Xtrain_df.values
Xte = Xtest_df.values
ytr = Ytrain_rr.values

# --- Fold layout 0..9
n = len(ytr)
foldid = np.repeat(np.arange(10), n // 10)
foldid = np.concatenate([foldid, np.full(n - len(foldid), 9)])[:n]
unique_folds = np.unique(foldid)

# --- Lambda grid (logspace)
alphas = np.logspace(-4, 2, 200)

# --- Cross-validation avec tqdm
mse_mat = np.zeros((len(unique_folds), len(alphas)))
for fi, f in enumerate(tqdm(unique_folds, desc="CV Folds", unit="fold")):
    val_idx = np.where(foldid == f)[0]
    trn_idx = np.where(foldid != f)[0]
    X_tr, y_tr = Xtr[trn_idx], ytr[trn_idx]
    X_va, y_va = Xtr[val_idx], ytr[val_idx]

    for li, lam in enumerate(alphas):
        model = Ridge(alpha=lam, fit_intercept=False)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_va)
        mse_mat[fi, li] = mean_squared_error(y_va, y_pred)

mean_mse = mse_mat.mean(axis=0)
std_mse  = mse_mat.std(axis=0, ddof=1)
imin = np.argmin(mean_mse)
mse_min = mean_mse[imin]
se_min  = std_mse[imin] / np.sqrt(len(unique_folds))
threshold = mse_min + se_min

# --- λ₁ₛₑ
candidates = np.where(mean_mse <= threshold)[0]
if len(candidates) == 0:
    idx_1se = imin
else:
    idx_1se = candidates[-1]
alpha_1se = float(alphas[idx_1se])

# --- Fit final
final_model = Ridge(alpha=alpha_1se, fit_intercept=False)
final_model.fit(Xtr, ytr)

# --- Predictions
y_hat_train_rr = final_model.predict(Xtr)
y_hat_test_rr  = final_model.predict(Xte)
y_hat_train = y_hat_train_rr * y_sd + y_mu
y_hat_test  = y_hat_test_rr  * y_sd + y_mu

# --- Save predictions
pred_df = pd.DataFrame({
    "Date":   np.concatenate([data.loc[idx_ins, "Date"].values,
                              data.loc[idx_oos, "Date"].values]),
    "set":    ["train"] * len(y_hat_train) + ["test"] * len(y_hat_test),
    "y_true": np.concatenate([Ytrain.values, Ytest.values]),
    "y_hat_py":  np.concatenate([y_hat_train, y_hat_test])
})
pred_df.to_csv("predictions_RR.csv", index=False)
print("predictions_RR.csv Created")

# --- DualML
backpack = {
    "type": "RR",
    "params": {
        "Xtrain": Xtrain_df.values,
        "Xtest":  Xtest_df.values,
        "Ytrain": Ytrain.values,
        "dates_ins": data.loc[idx_ins, "Date"].tolist(),
        "model_object": final_model,
        "lmbda": alpha_1se,
        "intercept": False
    }
}
test_RR = dualml_rr(backpack)

# --- Save results
df_rr = pd.DataFrame({
    "Date":         test_RR["weights"]["date"],
    "Weight":       test_RR["weights"].iloc[:, 0],
    "Contribution": test_RR["contributions"].iloc[:, 0]
})
df_rr.to_csv("results_RR.csv", index=False)
print("results_RR.csv Created")

# --- Plots
df_rr["Date"] = pd.to_datetime(df_rr["Date"])
dates = df_rr["Date"]
locator = mdates.YearLocator()
formatter = mdates.DateFormatter("%Y")

plt.figure(figsize=(25, 10))
plt.plot(dates, df_rr["Weight"], color='red')
plt.title("RR — Observation Weights")
plt.xlabel("Date")
plt.ylabel("Weight")
ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.xticks(rotation=45, ha="right")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.show()

plt.figure(figsize=(25, 10))
plt.plot(dates, df_rr["Contribution"], color='blue')
plt.title("RR — Observation Contributions")
plt.xlabel("Date")
plt.ylabel("Contribution")
ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.xticks(rotation=45, ha="right")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.show()

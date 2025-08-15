import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.ensemble._forest import _generate_sample_indices


"""  ====================== RPY2 OPTION ======================== """

"""
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, default_converter
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import numpy2ri, default_converter
"""

#-------------- OLS -------------#

def dualml_ols(run_model, Q=10):
    Xtrain = run_model['params']['Xtrain']
    Xtest = run_model['params']['Xtest']
    Ytrain = run_model['params']['Ytrain']
    intercept = run_model['params'].get('intercept', True)

    if 'dates_ins' in run_model['params']:
        full_dates = run_model['params']['dates_ins']
        full_dates = pd.to_datetime(full_dates).reset_index(drop=True)
        if len(full_dates) != Xtrain.shape[0]:
            full_dates = full_dates[-Xtrain.shape[0]:].reset_index(drop=True)
    else:
        full_dates = pd.Series(np.arange(Xtrain.shape[0]))

    XtX_inv = np.linalg.inv(Xtrain.T @ Xtrain)
    weight_matrix = Xtest @ XtX_inv @ Xtrain.T
    df_weights = pd.DataFrame(weight_matrix.T, columns=[f'obs_oos_{i+1}' for i in range(Xtest.shape[0])])

    # --- Contributions
    contrib_matrix = []
    Ymean = np.mean(Ytrain)
    for i in range(df_weights.shape[1]):
        contrib = np.cumsum(df_weights.iloc[:, i] * (Ytrain - Ymean)) + Ymean
        contrib_matrix.append(contrib)
    df_contributions = pd.DataFrame(contrib_matrix).T
    df_contributions.columns = df_weights.columns

    df_weights['date'] = full_dates
    df_contributions['date'] = full_dates

    k = int(df_weights.shape[0] * Q / 100)
    metric_FC = df_weights.iloc[:, :-1].apply(lambda col: np.sum(np.sort(np.abs(col))[-k:]) / np.sum(np.abs(col)), axis=0).mean().round(4)
    metric_FSP = df_weights.iloc[:, :-1].apply(lambda col: np.sum(col[col < 0]), axis=0).mean().round(4)
    metric_FT = df_weights.iloc[:, :-1].diff().iloc[1:].abs().sum().mean().round(4)
    metric_FL = df_weights.iloc[:, :-1].sum().round(2).values
    if not intercept:
        metric_FL += 1

    return {
        'weights': df_weights,
        'contributions': df_contributions,
        'concentration': metric_FC,
        'short_position': metric_FSP,
        'turnover': metric_FT,
        'leverage': metric_FL
    }

#-------- RANDOM FOREST ---------#

def block_sampler(X, sampling_rate, block_size, num_tree, idx_pos=[]):
    inbag = []
    inbag2 = []

    n_obs = X.shape[0]

    for j in range(num_tree):
        sample_index = np.arange(n_obs)
        n_blocks = int(np.floor(len(sample_index) / block_size))
        groups = np.sort(np.random.choice(np.arange(1, n_blocks + 1), size=len(sample_index), replace=True))
        rando_vec = np.random.exponential(scale=1.0, size=n_blocks)[groups - 1] + 0.1

        if len(idx_pos) > 0:
            pos_sample_true = groups[idx_pos]

            pos_train = []
            pos_sample = np.unique(pos_sample_true)
            while len(pos_train) < (len(pos_sample_true) * sampling_rate - block_size / 2):
                take_pos = np.random.choice(pos_sample, size=1)[0]
                pos_train.extend([take_pos] * np.sum(pos_sample_true == take_pos))
                pos_sample = pos_sample[pos_sample != take_pos]

            pos_val = pos_sample_true[~np.isin(pos_sample_true, np.unique(pos_train))]

            rando_vec[np.isin(groups, np.unique(pos_train))] += np.abs(np.max(rando_vec))
            rando_vec[np.isin(groups, np.unique(pos_val))] -= np.abs(np.max(rando_vec))

        chosen_ones_plus = rando_vec
        threshold = np.quantile(chosen_ones_plus, 1 - sampling_rate)
        selected = sample_index[chosen_ones_plus > threshold]

        boot = np.sort(selected)
        inbag_j = np.isin(np.arange(n_obs), boot).astype(int)
        inbag.append(inbag_j)

    for j in range(num_tree):
        sample_index = np.arange(n_obs)
        n_blocks = int(np.floor(len(sample_index) / block_size))
        groups = np.sort(np.random.choice(np.arange(1, n_blocks + 1), size=len(sample_index), replace=True))
        rando_vec = np.random.exponential(scale=1.0, size=n_blocks)[groups - 1] + 0.1

        chosen_ones_plus = rando_vec
        threshold = np.quantile(chosen_ones_plus, 1 - sampling_rate)
        selected = sample_index[chosen_ones_plus > threshold]

        boot = np.sort(selected)
        inbag_j = np.isin(np.arange(n_obs), boot).astype(int)
        inbag2.append(inbag_j)

    return {"inbag1": inbag, "inbag2": inbag2}

def dualml_rf(run_model):

    if run_model['type'] != 'RF':
        raise ValueError("dualml_rf is only for type='RF'")

    df_weights = rf_weights(
        mod_RF=run_model['params']['model_object'],
        Ytrain=run_model['params']['Ytrain'],
        Xtrain=run_model['params']['Xtrain'],
        Xtest=run_model['params']['Xtest'],
        show_progress=False
    )

    df_weights.columns = [f"obs_oos_{i+1}" for i in range(df_weights.shape[1])]

    # ------------------------------------- Observation Contributions ------------------------- #
    Ytrain = run_model['params']['Ytrain']
    mean_y = np.mean(Ytrain)

    df_contributions = pd.DataFrame({
        col: np.cumsum(df_weights[col] * (Ytrain - mean_y)) + mean_y
        for col in df_weights.columns
    })

    # ------------------------------------- Attach the Observation Label ------------------------------------- #
    if run_model['params'].get('dates_ins') is None:
        dates = np.arange(1, run_model['params']['Xtrain'].shape[0] + 1)
    else:
        dates = run_model['params']['dates_ins']

    df_weights['date'] = dates
    df_contributions['date'] = dates

    return {
        'weights': df_weights,
        'contributions': df_contributions
    }






def _r_matrix_to_np(mat_r):
    arr = np.asarray(mat_r, dtype=int, order="F")
    n_rows, n_cols = mat_r.dim
    return arr.reshape((n_rows, n_cols), order="F")


def rf_weights(mod_RF, Ytrain, Xtrain, Xtest, show_progress=False):
    import numpy as np, pandas as pd
    from tqdm import tqdm
    from rpy2 import robjects as ro


    leaf_ins_r = ro.r('predict(mod_RF, data = data.frame(X), '
                      'type="terminalNodes", predict.all=TRUE)$predictions')
    leaf_oos_r = ro.r('predict(mod_RF, data = data.frame(Xtest), '
                      'type="terminalNodes", predict.all=TRUE)$predictions')

    rf_assigned_leaf_ins = np.asarray(leaf_ins_r, dtype=int).reshape(len(Ytrain), -1, order='F')
    rf_assigned_leaf_oos = np.asarray(leaf_oos_r, dtype=int).reshape(len(Xtest),  -1, order='F')

    n_train, n_trees = rf_assigned_leaf_ins.shape
    n_test           = rf_assigned_leaf_oos.shape[0]

    inbag_counts = [np.asarray(vec, dtype=int) for vec in mod_RF.rx2("inbag.counts")]

    obs_weights_by_oos = pd.DataFrame(
        0.0,
        index=np.arange(n_train),
        columns=[f"obs_oos_{i+1}" for i in range(n_test)]
    )

    iterator = tqdm(range(n_test)) if show_progress else range(n_test)

    for oo in iterator:
        obs_weights_by_tree = pd.DataFrame(
            0.0,
            index=np.arange(n_train),
            columns=[f"Tree_{i+1}" for i in range(n_trees)]
        )

        for ll, inbag_ll in enumerate(inbag_counts):
            leaf_ll = rf_assigned_leaf_oos[oo, ll]
            ins_idx = np.where((rf_assigned_leaf_ins[:, ll] == leaf_ll) &
                               (inbag_ll == 1))[0]
            if ins_idx.size:
                obs_weights_by_tree.iloc[ins_idx, ll] = 1.0 / ins_idx.size

        obs_weights_by_oos.iloc[:, oo] = obs_weights_by_tree.mean(axis=1)

    return obs_weights_by_oos

def get_weights_RF(model, Y_train, X_train, X_test):
    """
    Exact RF weights matching RandomForestRegressor.predict():
     - bootstrap with replacement
     - duplicate counts matter
     - leaf mean is weighted by sample counts
    """
    n_train = X_train.shape[0]
    n_test  = X_test.shape[0]
    n_trees = len(model.estimators_)

    K = np.zeros((n_train, n_test), dtype=float)

    for tree in model.estimators_:
        idx_boot = _generate_sample_indices(
            tree.random_state, n_train, n_train
        )
        sample_count = np.bincount(idx_boot, minlength=n_train)

        leaves_train = tree.apply(X_train)
        leaves_test  = tree.apply(X_test)

        for j in range(n_test):
            leaf_j = leaves_test[j]
            mask   = (leaves_train == leaf_j)
            counts = sample_count[mask]
            total  = counts.sum()
            if total > 0:
                K[mask, j] += counts / total

    K /= n_trees

    col_sums = K.sum(axis=0, keepdims=True)
    K = np.divide(K, col_sums, where=col_sums>0)


    assert np.allclose(K.sum(axis=0), 1.0, atol=1e-8), "Weights do not sum to 1!"

    return K


# -------- Kernel Ridge-Regression -------------------- #

def compute_kernel_r_py(Xtrain, Xtest=None, kernel_type='laplace', sigma=1e-4):
    Xtrain = np.asarray(Xtrain, dtype=float)
    if Xtest is not None:
        Xtest = np.asarray(Xtest, dtype=float)

    if kernel_type.lower() == 'gaussian':
        D_train = cdist(Xtrain, Xtrain, metric='sqeuclidean')
        K_train = np.exp(-sigma * D_train)
        if Xtest is not None:
            D_test = cdist(Xtest, Xtrain, metric='sqeuclidean')
            K_test = np.exp(-sigma * D_test)
            return K_train, K_test
        return K_train, None

    elif kernel_type.lower() == 'laplace':
        # Cohérent avec ta version Python (cityblock / L1)
        D_train = cdist(Xtrain, Xtrain, metric='cityblock')
        K_train = np.exp(-sigma * D_train)
        if Xtest is not None:
            D_test = cdist(Xtest, Xtrain, metric='cityblock')
            K_test = np.exp(-sigma * D_test)
            return K_train, K_test
        return K_train, None

    else:
        raise ValueError(f"Unsupported kernel: {kernel_type}")

def compute_kernel_r(Xtrain, Xtest=None, kernel_type='laplace', sigma=1e-4):
    import rpy2.robjects as ro
    ro.r("library(kernlab)")

    with localconverter(default_converter + numpy2ri.converter):
        ro.globalenv["Xtrain"] = Xtrain
        if Xtest is not None:
            ro.globalenv["Xtest"] = Xtest
        ro.globalenv["sigma"] = sigma

    if kernel_type == "gaussian":
        ro.r("kernel <- rbfdot(sigma=sigma)")
    elif kernel_type == "laplace":
        ro.r("kernel <- laplacedot(sigma=sigma)")
    else:
        raise ValueError(f"Unsupported kernel: {kernel_type}")

    ro.r("K_train <- kernelMatrix(kernel, as.matrix(Xtrain))")
    with localconverter(default_converter + numpy2ri.converter):
        K_train = np.array(ro.r("K_train"))

    if Xtest is not None:
        ro.r("K_test <- kernelMatrix(kernel, as.matrix(Xtest), as.matrix(Xtrain))")
        with localconverter(default_converter + numpy2ri.converter):
            K_test = np.array(ro.r("K_test"))
        return K_train, K_test

    return K_train, None


from scipy.spatial.distance import cdist


def krr_weight(backpack):

    if backpack["type"] != "KRR":
        raise ValueError("backpack['type'] doit être 'KRR'")

    p = backpack["params"]

    X_train = p["Xtrain"]
    X_test  = p["Xtest"]
    Y_train = p["Ytrain"]
    lmbda   = p["lmbda"]

    sigma        = p.get("sigma", 1e-4)
    kernel_type  = p.get("kernel_type", "laplace").lower()
    if p.get("recompute_kernel", False):          # flag optionnel
        if kernel_type == "gaussian":
            def kernel(A, B):
                return np.exp(-sigma * cdist(A, B, "sqeuclidean"))
        else:  # laplace
            def kernel(A, B):
                return np.exp(-sigma * cdist(A, B, "cityblock"))
        X_train = kernel(X_train, X_train)
        X_test  = kernel(X_test,  X_train)

    n_train = X_train.shape[0]
    inv_K   = np.linalg.solve(X_train + lmbda * np.eye(n_train),
                              np.eye(n_train))

    # ---------- WEIGHTS  ----------------------
    W = (X_test @ inv_K).T
    weights_df = pd.DataFrame(
        W,
        columns=[f"obs_oos_{j+1}" for j in range(X_test.shape[0])]
    )

    # ---------- CONTRIBUTIONS ------------------
    y_bar = Y_train.mean()
    y_c   = Y_train - y_bar
    sum_w = weights_df.sum(axis=0).values

    C = np.empty_like(W)
    for j in range(W.shape[1]):
        C[:, j] = np.cumsum(W[:, j] * y_c) + y_bar * sum_w[j]

    contrib_df = pd.DataFrame(C, columns=weights_df.columns)

    # ---------- DATE ---------------------------
    if p.get("dates_ins") is not None:
        dates = pd.Series(p["dates_ins"]).reset_index(drop=True)[:n_train]
    else:
        dates = np.arange(1, n_train + 1)

    weights_df["Date"]  = dates
    contrib_df["Date"]  = dates

    return weights_df, contrib_df

# --------------- Ridge Regression ---------------- #

def dualml_rr(backpack):
    if backpack["type"].upper() != "RR":
        raise ValueError("backpack['type'] must be 'RR'")

    X_train = backpack["params"]["Xtrain"]
    Y_train = backpack["params"]["Ytrain"]
    X_test  = backpack["params"]["Xtest"]
    lmbda   = backpack["params"]["lmbda"]
    dates   = backpack["params"].get("dates_ins", None)


    n_train, d = X_train.shape
    XtX         = X_train.T @ X_train
    ridge_part  = n_train * lmbda * np.eye(d)
    project_mat = np.linalg.solve(XtX + ridge_part, np.eye(d))

    W = X_test @ project_mat @ X_train.T
    dfW = pd.DataFrame(W.T, columns=[f"obs_oos_{i+1}" for i in range(X_test.shape[0])])

    ȳ   = Y_train.mean()
    y_c = Y_train - ȳ
    C   = np.cumsum(dfW.values * y_c[:, None], axis=0) + ȳ
    dfC = pd.DataFrame(C, columns=dfW.columns)

    index_vals = np.arange(1, n_train + 1) if dates is None else dates
    dfW["date"] = index_vals
    dfC["date"] = index_vals

    return {
        "weights": dfW,
        "contributions": dfC
    }

# ---------------- NN ------------------

import pandas as pd
import numpy as np


def dualml_nn(backpack):
    run__model = backpack


    if run__model['type'] != 'NN':
        raise ValueError(f"Model type {run__model['type']} not supported. Expected 'NN'.")

    params = run__model['params']
    model_obj = params['model_object']


    if 'portfolio.weights' not in model_obj:
        raise ValueError(
            "I cannot find the 'portfolio.weights' in your 'model_object'.\n"
            "Are you sure your NN was fitted using the provided MLP function?\n"
        )

    Ytrain = params['Ytrain']
    preds = model_obj['pred']
    params['Xtrain'] = pd.DataFrame(np.full((len(Ytrain), 1), np.nan))
    params['Xtest'] = pd.DataFrame(np.full((len(preds), 1), np.nan))

    weights = np.array(model_obj['portfolio.weights'])
    df_weights = pd.DataFrame(
        weights.T,
        columns=[f"obs_oos_{i + 1}" for i in range(weights.shape[0])]
    )

    y = np.array(Ytrain)
    y_mean = np.mean(y)
    y_std = np.std(y, ddof=1)  # ddof=1 pour sd() de R

    std_y = (y - y_mean) / y_std
    contrib_dict = {}
    for col in df_weights.columns:
        cum = np.cumsum(df_weights[col].values * std_y)
        contrib_dict[col] = cum * y_std + y_mean

    df_contributions = pd.DataFrame(contrib_dict)

    return df_weights, df_contributions



# -------------------- LGB -------------------
def LCM(leaf_vec_1,leaf_vec_2):

    mat = np.nan*np.ones((len(leaf_vec_1),len(leaf_vec_2)))

    for rr in range(len(leaf_vec_1)):
        for cc in range(len(leaf_vec_2)):
            if leaf_vec_1[rr] == leaf_vec_2[cc]:
                mat[rr,cc] = 1
            else:
                mat[rr,cc] = 0

    return mat

def GeertsemaLu2023(mod, Y_train, X_train, X_test):

    # ------------------------------------- Algo I ------------------------------------- #

    # --- Number of trees &  Learning Rate
    N_trees = mod.num_trees()
    learning_rate = mod.params['learning_rate']

    # --- Number of Training-Samples:
    N_train = Y_train.shape[0]

    # --- Storage for the "Tree-Prediction-Weight-Matrix"
    P_dict = dict()

    # --- Initialize the prediction matrix
    G = 1 / N_train * np.ones((N_train, N_train))
    P_dict[0] = G

    # --- Run through the trees!
    for tt in range(N_trees):
        # --- For tree 'tt', which observations have been assigned to which leaf in-sample?
        v_tt = mod.predict(X_train, pred_leaf=True)[:, tt]

        # --- Construct Leaf-Coincidence Matrix
        D = LCM(v_tt, v_tt)

        # --- Scale the Leave Coincidence Matrix
        W = D / (np.ones((N_train, N_train)) @ D)

        # --- Update the Tree-Prediction-Weight-Matrix
        P = learning_rate * W @ (np.identity(N_train) - G)

        # --- Update the Prediction-Matrix for next iteration
        G += P

        # --- Collect: Tree-Prediction-Weight-Matrix
        P_dict[tt + 1] = P

    # ------------------------------------- Algo II ------------------------------------- #

    # --- Number of Test observations
    N_oos = X_test.shape[0]

    # --- Initialize the "Tree-Prediction-Weight-Matrix"
    P = P_dict[0]

    # --- Initialize 'L'
    L = np.ones((N_train, N_oos))

    # --- First-iteration prediction weights
    K = P.T @ (L / (np.ones((N_train, N_train)) @ L))

    # --- Run through the trees!
    for tt in range(N_trees):
        # --- For tree 'tt', which observations have been assigned to which leaf in-sample?
        v_tt = mod.predict(X_train, pred_leaf=True)[:, tt]

        # --- For tree 'tt', which observations have been assigned to which leaf out-of-sample?
        w_tt = mod.predict(X_test, pred_leaf=True)[:, tt]

        # --- Get the "Tree-Prediction-Weight-Matrix"
        P = P_dict[tt + 1]

        # --- Construct Leave-Coincidence Matrix
        L = LCM(v_tt, w_tt)

        # --- Scale the Leaf-Coincidence-Matrix
        W = L / (np.ones((N_train, N_train)) @ L)

        # --- Update/Accumulate the Prediction-weights
        K += P.T @ W

    return K
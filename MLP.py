import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd


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


# ──────────────────────────────────────────────────────────────────────────────
# 1. STANDARDISATION
# ──────────────────────────────────────────────────────────────────────────────
def scale_data(Xtrain, Ytrain, Xtest, Ytest):
    """Centre et réduit (ddof=1 pour coller à sd() de R)."""
    mu_x    = Xtrain.mean(axis=0)
    sigma_x = Xtrain.std(axis=0, ddof=1)
    mu_y = np.mean(Ytrain)
    sigma_y = np.sqrt(((Ytrain - mu_y) ** 2).sum() / (len(Ytrain) - 1))

    Xtrain_scaled = (Xtrain - mu_x) / sigma_x
    Xtest_scaled  = (Xtest  - mu_x) / sigma_x
    Ytrain_scaled = (Ytrain - mu_y) / sigma_y
    Ytest_scaled  = (Ytest  - mu_y) / sigma_y

    scaler = {"mu_x": mu_x, "sigma_x": sigma_x, "mu_y": mu_y, "sigma_y": sigma_y}
    return Xtrain_scaled, Ytrain_scaled, Xtest_scaled, Ytest_scaled, scaler

def invert_scaling(scaled, scaler):
    return scaled * scaler["sigma_y"] + scaler["mu_y"]

def predict_scale_data(X, scaler):
    return (X - scaler["mu_x"]) / scaler["sigma_x"]


# ──────────────────────────────────────────────────────────────────────────────
# 2. ARCHITECTURE MLP
# ──────────────────────────────────────────────────────────────────────────────
class MLPModel(nn.Module):
    def __init__(self, input_dim, nodes, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, nodes[0])])
        for i in range(len(nodes)-1):
            self.layers.append(nn.Linear(nodes[i], nodes[i+1]))
        self.output  = nn.Linear(nodes[-1], 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
            x = self.dropout(x)
        embeddings = x
        yhat = self.output(x).squeeze()
        return yhat, embeddings


# ──────────────────────────────────────────────────────────────────────────────
# 3.BOOTSTRAP
# ──────────────────────────────────────────────────────────────────────────────
def train_model(X_tensor, Y_tensor, train_idx, nn_hyps):
    model = MLPModel(
        input_dim   = X_tensor.shape[1],
        nodes       = nn_hyps["nodes"],
        dropout_rate= nn_hyps["dropout_rate"]
    ).double()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=nn_hyps["lr"])
    patience, tol = nn_hyps["patience"], nn_hyps["tol"]
    batch_size, epochs = nn_hyps["batch_size"], nn_hyps["epochs"]

    oob_idx = np.setdiff1d(np.arange(X_tensor.shape[0]), train_idx)
    best_loss = None
    wait = 0
    best_state = None

    for epoch in range(1, epochs+1):

        model.train()
        np.random.shuffle(train_idx)
        for i in range(0, len(train_idx), batch_size):
            batch = train_idx[i:i+batch_size]
            optimizer.zero_grad()
            yhat, _ = model(X_tensor[batch])
            loss = criterion(yhat, Y_tensor[batch])
            loss.backward()
            optimizer.step()


        model.eval()
        with torch.no_grad():
            y_oob, _ = model(X_tensor[oob_idx])
            val_loss = criterion(y_oob, Y_tensor[oob_idx]).item()


        if best_loss is None or val_loss < best_loss or epoch == 1:
            percent_change = float("inf") if epoch == 1 else (best_loss - val_loss) / val_loss
            best_loss = val_loss

            best_state = {k: v.clone() for k, v in model.state_dict().items()}

            wait = 0 if (percent_change > tol or epoch == 1) else (wait + 1)
        else:
            wait += 1

        if wait > patience:
            break

    model.load_state_dict(best_state)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 4. MLP
# ──────────────────────────────────────────────────────────────────────────────
def MLP(
    X, Y, Xtest, Ytest,
    nn_hyps,
    standardize=True,
    seed=42,
    X_OOS=None
):

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)

    if X_OOS is None:
        X_OOS = np.empty((0, X.shape[1]))

    if standardize:
        X, Y, Xtest, Ytest, scaler = scale_data(X, Y, Xtest, Ytest)
        if X_OOS.shape[0] > 0:
            X_OOS = predict_scale_data(X_OOS, scaler)
    else:
        scaler = None

    X_t     = torch.tensor(X,     dtype=torch.float64)
    Y_t     = torch.tensor(Y,     dtype=torch.float64)
    Xtest_t = torch.tensor(Xtest, dtype=torch.float64)
    Xoos_t  = torch.tensor(X_OOS, dtype=torch.float64)

    B         = nn_hyps["num_average"]
    samp_rate = nn_hyps["sampling_rate"]
    n_ins     = X.shape[0]
    n_test    = Xtest.shape[0]
    n_oos     = X_OOS.shape[0]

    pred_train_B = np.zeros((n_ins,     B))
    pred_test_B  = np.zeros((n_test,    B))
    pred_OOS_B   = np.zeros((n_oos,     B))
    inner_prod_B = np.zeros((n_test, n_ins, B))
    lambda_opt_B = np.zeros(B)
    mse_opt_B    = np.zeros(B)

    for b in range(B):
        # bootstrap
        np.random.seed(seed + b)
        torch.manual_seed(seed + b)
        train_idx = np.random.choice(n_ins, int(samp_rate * n_ins), replace=False)

        # train
        model = train_model(X_t, Y_t, train_idx, nn_hyps)
        model.eval()

        with torch.no_grad():
            y_train, H_train = model(X_t)
            y_test,  H_test  = model(Xtest_t)
            if n_oos > 0:
                y_oos, H_oos = model(Xoos_t)
            else:
                y_oos = torch.empty((0,))
                H_oos = torch.empty((0, n_ins))

        y_train = y_train.numpy()
        y_test  = y_test.numpy()
        y_oos   = y_oos.numpy() if n_oos>0 else np.empty((0,))
        H_train = H_train.numpy()
        H_test  = H_test.numpy()

        if standardize:
            y_train = invert_scaling(y_train, scaler)
            y_test  = invert_scaling(y_test,  scaler)
            if n_oos>0:
                y_oos = invert_scaling(y_oos, scaler)

        pred_train_B[:, b] = y_train
        pred_test_B[:,  b] = y_test
        if n_oos>0:
            pred_OOS_B[:, b] = y_oos

        K_train = H_train @ H_train.T
        mses = []

        for lam in nn_hyps["lambda_grid"]:
            try:
                alpha = np.linalg.solve(K_train + lam * np.eye(n_ins), Y_t.numpy())
            except np.linalg.LinAlgError:
                alpha = np.linalg.pinv(K_train + lam * np.eye(n_ins)) @ Y
            yk = K_train @ alpha
            if standardize:
                yk = invert_scaling(yk, scaler)
            mses.append(np.mean((y_train - yk) ** 2))

        best_i   = np.argmin(mses)
        lam_star = nn_hyps["lambda_grid"][best_i]
        lambda_opt_B[b] = lam_star
        mse_opt_B[b]    = mses[best_i]

        K_inv = np.linalg.pinv(K_train + lam_star * np.eye(n_ins))
        inner_prod_B[:, :, b] = (H_test @ H_train.T) @ K_inv

    portfolio_weights = inner_prod_B.mean(axis=2)

    return {
        "pred_train_B":     pred_train_B,
        "pred_test_B":      pred_test_B,
        "pred_OOS_B":       pred_OOS_B,
        "portfolio_weights":portfolio_weights,
        "lambda_opt":       lambda_opt_B,
        "mse_opt":          mse_opt_B,
        "trained_models":   None,
        "scaler":           scaler,
        "standardize":      standardize
    }

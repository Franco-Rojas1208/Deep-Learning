# -*- coding: utf-8 -*-

import os
import time

import numpy as np
import matplotlib
#matplotlib.use("Agg")          # sacar esta linea para ver las figuras en pantalla
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURACION
# =============================================================================

DATA_DIR = "./data"            # donde se cachea cifar10.npz
OUT_DIR = "./figuras"          # donde se guardan los graficos
SEED = 0

# Hiperparametros de la red
N_HIDDEN = 100                 # neuronas de la capa oculta (lo fija el enunciado)
LEARNING_RATE = 0.02           # rango util con esta arquitectura: ~0.005 a 0.05.
                               # Si el costo explota o se mueren casi todas las
                               # unidades en las primeras epocas, bajarlo.
                               # Con salida SIGMOIDE el gradiente que llega a Z2
                               # viene multiplicado por sigma' <= 0.25, asi que
                               # esa variante tolera (y necesita) un lr mayor que
                               # la de salida lineal.
MOMENTUM = 0 #0.9
LAMBDA_L2 = 1e-4               # peso del termino de regularizacion L2
BATCH_SIZE = 128
EPOCHS = 80

# Activacion de la capa de salida:
#   "sigmoid" -> lo que pide el ejercicio 5
#   "linear"  -> ReLU + salida lineal (la variante a analizar)
OUT_ACTIVATION = "sigmoid"

# Costo con cuyo gradiente se entrena: "mse" o "cce".
# (Las DOS metricas se registran siempre para graficarlas, entrene con la que
#  entrene; esto es lo que pide el enunciado.)
GRAD_METRIC = "cce"

# Si es True, ademas del entrenamiento principal corre las 4 combinaciones
# {sigmoid, linear} x {mse, cce} y arma los graficos comparativos.
RUN_COMPARACION = True


CLASSES = ["avion", "auto", "pajaro", "gato", "ciervo", "perro", "rana", "caballo", "barco", "camion"]


# =============================================================================
# 1. DATOS
# =============================================================================

def load_dataset_cached(name, loader_fn, data_dir=DATA_DIR):
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, f"{name}.npz")

    if os.path.exists(cache_path):
        with np.load(cache_path) as f:
            x_train, y_train = f["x_train"], f["y_train"]
            x_test, y_test = f["x_test"], f["y_test"]
        return (x_train, y_train), (x_test, y_test)

    (x_train, y_train), (x_test, y_test) = loader_fn()
    np.savez(cache_path,
             x_train=x_train, y_train=y_train,
             x_test=x_test, y_test=y_test)
    return (x_train, y_train), (x_test, y_test)


def load_cifar10():
    def _keras_loader():
        from tensorflow.keras.datasets import cifar10
        return cifar10.load_data()

    (x_tr, y_tr), (x_te, y_te) = load_dataset_cached("cifar10", _keras_loader)
    return x_tr, y_tr.reshape(-1), x_te, y_te.reshape(-1)


def preprocess(x_train, x_test):
    X_tr = x_train.reshape(x_train.shape[0], -1).astype(np.float32) / np.float32(255.0)
    X_te = x_test.reshape(x_test.shape[0], -1).astype(np.float32) / np.float32(255.0)

    mean_img = X_tr.mean(axis=0, keepdims=True)   # se calcula SOLO con train
    X_tr -= mean_img
    X_te -= mean_img
    return X_tr, X_te


def one_hot(y, n_classes=10, dtype=np.float64):
    Y = np.zeros((y.shape[0], n_classes), dtype=dtype)
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


# =============================================================================
# 2. GRAFO COMPUTACIONAL: FORWARD
# =============================================================================

def relu(z):
    return np.maximum(z, 0.0)


def relu_grad(z):
    return (z > 0.0).astype(z.dtype)


def sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def softmax(Z):
    Z_shift = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z_shift)
    return expZ / np.sum(expZ, axis=1, keepdims=True)


def out_activation(Z2, kind=OUT_ACTIVATION):
    if kind == "sigmoid":
        return sigmoid(Z2)
    elif kind == "linear":
        return Z2
    raise ValueError(f"OUT_ACTIVATION debe ser 'sigmoid' o 'linear', recibido: {kind!r}")


def out_activation_grad(Yhat, kind=OUT_ACTIVATION):
    if kind == "sigmoid":
        return Yhat * (1.0 - Yhat)
    elif kind == "linear":
        return np.ones_like(Yhat)
    raise ValueError(f"OUT_ACTIVATION debe ser 'sigmoid' o 'linear', recibido: {kind!r}")


def init_params(n_in=3072, n_hidden=N_HIDDEN, n_out=10, seed=SEED):
    rng = np.random.default_rng(seed)
    params = {
        "W1": rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_hidden)),
        "b1": np.full(n_hidden, 0.01),
        "W2": rng.normal(0.0, np.sqrt(2.0 / (n_hidden + n_out)), size=(n_hidden, n_out)),
        "b2": np.zeros(n_out),
    }
    return params


def forward(X, params, out_kind=OUT_ACTIVATION):
    Z1 = X @ params["W1"] + params["b1"]      # (N,100)
    H = relu(Z1)                              # (N,100)
    Z2 = H @ params["W2"] + params["b2"]      # (N,10)
    Yhat = out_activation(Z2, out_kind)       # (N,10)
    cache = {"X": X, "Z1": Z1, "H": H, "Z2": Z2, "Yhat": Yhat}
    return Yhat, cache


# --- funciones de costo ------------------------------------------------------

def reg_loss(params, lam):
    return lam * (np.sum(params["W1"] ** 2) + np.sum(params["W2"] ** 2))


def loss_mse(Yhat, Y, params, lam):
    N = Yhat.shape[0]
    return np.sum((Yhat - Y) ** 2) / N + reg_loss(params, lam)


def loss_cce(Yhat, Y, params, lam, eps=1e-12):
    N = Yhat.shape[0]
    P = softmax(Yhat)
    return -np.sum(Y * np.log(P + eps)) / N + reg_loss(params, lam)


def compute_loss(Yhat, Y, params, lam, metric="mse"):
    if metric == "mse":
        return loss_mse(Yhat, Y, params, lam)
    elif metric == "cce":
        return loss_cce(Yhat, Y, params, lam)
    raise ValueError(f"metric debe ser 'mse' o 'cce', recibido: {metric!r}")


# =============================================================================
# 3. GRAFO COMPUTACIONAL: BACKWARD
# =============================================================================

def grad_mse(Yhat, Y):
    N = Yhat.shape[0]
    return (2.0 / N) * (Yhat - Y)                      # (N,10)


def grad_cce(Yhat, Y):
    N = Yhat.shape[0]
    P = softmax(Yhat)
    return (1.0 / N) * (P - Y)                         # (N,10)


def compute_grad_output(Yhat, Y, metric="mse"):
    if metric == "mse":
        return grad_mse(Yhat, Y)
    elif metric == "cce":
        return grad_cce(Yhat, Y)
    raise ValueError(f"metric debe ser 'mse' o 'cce', recibido: {metric!r}")


def backward(cache, Y, params, lam, metric="mse", out_kind=OUT_ACTIVATION):
    X, Z1, H, Yhat = cache["X"], cache["Z1"], cache["H"], cache["Yhat"]

    # --- nodo costo: dL/dYhat ----------------------------------------------
    dYhat = compute_grad_output(Yhat, Y, metric=metric)       # (N,10)

    # --- activacion de salida: dL/dZ2 = dL/dYhat * g'(Z2) -------------------
    dZ2 = dYhat * out_activation_grad(Yhat, out_kind)         # (N,10)

    # --- capa 2 (afin): Z2 = H@W2 + b2 --------------------------------------
    dW2 = H.T @ dZ2 + 2.0 * lam * params["W2"]                # (100,10)
    db2 = dZ2.sum(axis=0)                                     # (10,)
    dH = dZ2 @ params["W2"].T                                 # (N,100)

    # --- ReLU: dH/dZ1 = 1[Z1>0] --------------------------------------------
    dZ1 = dH * relu_grad(Z1)                                  # (N,100)

    # --- capa 1 (afin): Z1 = X@W1 + b1 --------------------------------------
    dW1 = X.T @ dZ1 + 2.0 * lam * params["W1"]                # (3072,100)
    db1 = dZ1.sum(axis=0)                                     # (100,)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


# =============================================================================
# 4. METRICAS
# =============================================================================

def accuracy(Yhat, y):
    return float(np.mean(np.argmax(Yhat, axis=1) == y))


def evaluate(X, Y, y, params, lam, out_kind=OUT_ACTIVATION, batch=2000):
    N = X.shape[0]
    sum_mse = 0.0
    sum_cce = 0.0
    correct = 0
    off = 0.0
    activa = np.zeros(params["W1"].shape[1], dtype=bool)   # unidad que se prendio alguna vez
    for i in range(0, N, batch):
        Yb = Y[i:i + batch]
        Yhat, cache = forward(X[i:i + batch], params, out_kind)
        sum_mse += np.sum((Yhat - Yb) ** 2)
        P = softmax(Yhat)
        sum_cce += -np.sum(Yb * np.log(P + 1e-12))
        correct += int(np.sum(np.argmax(Yhat, axis=1) == y[i:i + batch]))
        off += float(np.sum(cache["H"] == 0.0))
        activa |= np.any(cache["H"] > 0.0, axis=0)
    reg = reg_loss(params, lam)
    # frac_off  = fraccion de activaciones en cero (lo normal es ~0.5)
    # frac_dead = fraccion de unidades MUERTAS: apagadas para todos los ejemplos,
    #             o sea con gradiente identicamente nulo, no se recuperan nunca
    frac_off = off / (N * params["W1"].shape[1])
    frac_dead = float(np.mean(~activa))
    return sum_mse / N + reg, sum_cce / N + reg, correct / N, frac_off, frac_dead


# =============================================================================
# 5. ENTRENAMIENTO (SGD por mini-lotes con momento)
# =============================================================================

def train(X_tr, Y_tr, y_tr, X_te, Y_te, y_te,
          n_hidden=N_HIDDEN, lr=LEARNING_RATE, momentum=MOMENTUM,
          lam=LAMBDA_L2, batch_size=BATCH_SIZE, epochs=EPOCHS, seed=SEED,
          grad_metric=GRAD_METRIC, out_kind=OUT_ACTIVATION, verbose=True):

    rng = np.random.default_rng(seed)
    params = init_params(X_tr.shape[1], n_hidden, Y_tr.shape[1], seed=seed)
    # trabajar en la misma precision que los datos (float32 ahorra ~600 MB con CIFAR)
    params = {k: v.astype(X_tr.dtype) for k, v in params.items()}
    velocity = {k: np.zeros_like(v) for k, v in params.items()}

    # para las curvas de train usamos una submuestra fija (evaluar 50k por epoca es caro)
    idx_eval = rng.choice(X_tr.shape[0], size=min(10000, X_tr.shape[0]), replace=False)

    hist = {"mse_train": [], "mse_test": [], "cce_train": [], "cce_test": [],
            "acc_train": [], "acc_test": [], "frac_off": [], "frac_dead": []}
    N = X_tr.shape[0]
    t0 = time.time()

    for ep in range(epochs):
        perm = rng.permutation(N)
        for i in range(0, N, batch_size):
            b = perm[i:i + batch_size]
            Xb, Yb = X_tr[b], Y_tr[b]

            _, cache = forward(Xb, params, out_kind)
            grads = backward(cache, Yb, params, lam,
                             metric=grad_metric, out_kind=out_kind)

            for k in params:                       # SGD con momento
                velocity[k] = momentum * velocity[k] - lr * grads[k]
                params[k] += velocity[k]

        mtr, ctr, atr, otr, dtr = evaluate(X_tr[idx_eval], Y_tr[idx_eval], y_tr[idx_eval],
                                           params, lam, out_kind)
        mte, cte, ate, _, _ = evaluate(X_te, Y_te, y_te, params, lam, out_kind)
        hist["mse_train"].append(mtr); hist["mse_test"].append(mte)
        hist["cce_train"].append(ctr); hist["cce_test"].append(cte)
        hist["acc_train"].append(atr); hist["acc_test"].append(ate)
        hist["frac_off"].append(otr); hist["frac_dead"].append(dtr)

        if verbose:
            print(f"  epoca {ep + 1:3d}/{epochs}  "
                  f"mse={mtr:.4f}/{mte:.4f}  cce={ctr:.4f}/{cte:.4f}  "
                  f"acc={atr:.4f}/{ate:.4f}  off={otr:.2f} muertas={dtr:.2f}  "
                  f"[{time.time() - t0:.0f}s]")

    return params, hist


# =============================================================================
# 6. GRAFICOS
# =============================================================================

def plot_curves(hist, path, out_kind=OUT_ACTIVATION, grad_metric=GRAD_METRIC):
    ep = np.arange(1, len(hist["acc_train"]) + 1)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    ax[0].plot(ep, hist["mse_train"], label="entrenamiento")
    ax[0].plot(ep, hist["mse_test"], label="validacion")
    ax[0].set_xlabel("epoca"); ax[0].set_ylabel("costo MSE + L2")
    ax[0].set_title("Funcion de costo MSE")

    ax[1].plot(ep, hist["cce_train"], label="entrenamiento")
    ax[1].plot(ep, hist["cce_test"], label="validacion")
    ax[1].set_xlabel("epoca"); ax[1].set_ylabel("costo CCE + L2")
    ax[1].set_title("Funcion de costo Categorical Cross Entropy")

    ax[2].plot(ep, 100 * np.array(hist["acc_train"]), label="entrenamiento")
    ax[2].plot(ep, 100 * np.array(hist["acc_test"]), label="validacion")
    ax[2].set_xlabel("epoca"); ax[2].set_ylabel("accuracy [%]")
    ax[2].set_title("Precision")

    for a in ax:
        a.legend(); a.grid(alpha=0.3)

    fig.suptitle(f"Ejercicio 5 - red 3072-100(ReLU)-10({out_kind}), "
                 f"entrenada con {grad_metric.upper()} + L2")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  figura guardada en {path}")


def plot_comparacion(hists, path):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for nombre, h in hists.items():
        ep = np.arange(1, len(h["acc_test"]) + 1)
        ax[0].plot(ep, 100 * np.array(h["acc_test"]), label=nombre)
        ax[1].plot(ep, h["mse_test"], label=nombre)
        ax[2].plot(ep, h["cce_test"], label=nombre)

    ax[0].set_ylabel("accuracy en validacion [%]"); ax[0].set_title("Precision")
    ax[0].axhline(10, ls="--", lw=1, color="gray")
    ax[1].set_ylabel("MSE + L2 (validacion)"); ax[1].set_title("Costo MSE")
    ax[2].set_ylabel("CCE + L2 (validacion)"); ax[2].set_title("Costo CCE")
    for a in ax:
        a.set_xlabel("epoca"); a.legend(fontsize=8); a.grid(alpha=0.3)

    fig.suptitle("Ejercicio 5 - activacion de salida (sigmoide vs lineal) y costo (MSE vs CCE)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  figura guardada en {path}")


def plot_barras_finales(resultados, path):
    names = list(resultados.keys())
    vals = [100 * resultados[k] for k in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, vals)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}%", ha="center", fontsize=10)
    ax.axhline(10, ls="--", lw=1, color="gray")
    ax.set_ylabel("accuracy en validacion [%]")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title("Ejercicio 5 - comparacion de configuraciones")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  figura guardada en {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 72)
    print("Practica 2 - Ejercicio 5: ReLU en la capa oculta, sigmoide en la salida")
    print("=" * 72)

    # --- 1) datos -----------------------------------------------------------
    print("Cargando CIFAR-10...")
    x_tr, y_tr, x_te, y_te = load_cifar10()
    X_tr, X_te = preprocess(x_tr, x_te)
    Y_tr, Y_te = one_hot(y_tr, dtype=X_tr.dtype), one_hot(y_te, dtype=X_te.dtype)
    print(f"  train: {X_tr.shape}   test: {X_te.shape}   clases: {len(CLASSES)}\n")

    # --- 2) entrenamiento principal ----------------------------------------
    print(f"Entrenando (lr={LEARNING_RATE}, momento={MOMENTUM}, lambda={LAMBDA_L2}, "
          f"batch={BATCH_SIZE}, epocas={EPOCHS}, salida={OUT_ACTIVATION}, "
          f"grad={GRAD_METRIC})")
    params, hist = train(X_tr, Y_tr, y_tr, X_te, Y_te, y_te,
                         grad_metric=GRAD_METRIC, out_kind=OUT_ACTIVATION)
    print(f"\nAccuracy final: train={hist['acc_train'][-1]:.4f}  "
          f"test={hist['acc_test'][-1]:.4f}\n")

    plot_curves(hist, os.path.join(OUT_DIR, "ej5_curvas.png"),
                out_kind=OUT_ACTIVATION, grad_metric=GRAD_METRIC)

    # --- 3) comparacion sigmoide vs lineal, MSE vs CCE ----------------------
    if RUN_COMPARACION:
        print("\nComparacion de configuraciones:")
        hists, finales = {}, {}
        for ok in ("sigmoid", "linear"):
            for m in ("mse", "cce"):
                nombre = f"ReLU + {ok} / {m.upper()}"
                print(f"\n  --- {nombre} ---")
                _, h = train(X_tr, Y_tr, y_tr, X_te, Y_te, y_te,
                             grad_metric=m, out_kind=ok, verbose=False)
                hists[nombre] = h
                finales[nombre] = h["acc_test"][-1]
                print(f"  acc test final = {h['acc_test'][-1]:.4f}   "
                      f"(mse={h['mse_test'][-1]:.4f}, cce={h['cce_test'][-1]:.4f}, "
                      f"activaciones en cero={h['frac_off'][-1]:.2f}, "
                      f"unidades muertas={h['frac_dead'][-1]:.2f})")

        plot_comparacion(hists, os.path.join(OUT_DIR, "ej5_comparacion.png"))
        plot_barras_finales(finales, os.path.join(OUT_DIR, "ej5_barras.png"))

    print("\nListo.")


if __name__ == "__main__":
    main()
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
LEARNING_RATE = 0.03           # rango util con este costo: ~0.01 a 0.05.
                               # Si el costo explota en las primeras epocas, bajarlo.
MOMENTUM = 0 #0.9
LAMBDA_L2 = 1e-3               # peso del termino de regularizacion L2
BATCH_SIZE = 128
EPOCHS = 80                    # ~1-3 s por epoca en CPU

# Metrica de costo a reportar/graficar y formula de gradiente a usar en el
# entrenamiento. Cada una puede ser "mse" o "softmax" (softmax = CCE),
# e independientemente una de la otra (para poder, por ej., entrenar con el
# gradiente de un costo pero reportar/graficar el otro).
LOSS_METRIC = "softmax"        # "mse" o "softmax"
GRAD_METRIC = "softmax"        # "mse" o "softmax"


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


def init_params(n_in=3072, n_hidden=N_HIDDEN, n_out=10, seed=SEED):
    rng = np.random.default_rng(seed)
    params = {
        "W1": rng.normal(0.0, np.sqrt(2.0 / (n_in + n_hidden)), size=(n_in, n_hidden)),
        "b1": np.zeros(n_hidden),
        "W2": rng.normal(0.0, np.sqrt(2.0 / (n_hidden + n_out)), size=(n_hidden, n_out)),
        "b2": np.zeros(n_out),
    }
    return params


def forward(X, params):
    Z1 = X @ params["W1"] + params["b1"]      # (N,100)
    H = sigmoid(Z1)                           # (N,100)
    Z2 = H @ params["W2"] + params["b2"]      # (N,10)
    Yhat = Z2                                 # activacion lineal
    cache = {"X": X, "Z1": Z1, "H": H, "Z2": Z2, "Yhat": Yhat}
    return Yhat, cache


def loss_mse(Yhat, Y, params, lam):
    N = Yhat.shape[0]
    data_loss = np.sum((Yhat - Y) ** 2) / N
    reg_loss = lam * (np.sum(params["W1"] ** 2) + np.sum(params["W2"] ** 2))
    return data_loss + reg_loss


def loss_softmax(Yhat, Y, params, lam, eps=1e-12):
    N = Yhat.shape[0]
    P = softmax(Yhat)
    data_loss = -np.sum(Y * np.log(P + eps)) / N
    reg_loss = lam * (np.sum(params["W1"] ** 2) + np.sum(params["W2"] ** 2))
    return data_loss + reg_loss


def compute_loss(Yhat, Y, params, lam, metric="mse"):
    if metric == "mse":
        return loss_mse(Yhat, Y, params, lam)
    elif metric == "softmax":
        return loss_softmax(Yhat, Y, params, lam)
    raise ValueError(f"metric debe ser 'mse' o 'softmax', recibido: {metric!r}")


# =============================================================================
# 3. GRAFO COMPUTACIONAL: BACKWARD
# =============================================================================

def grad_mse(Yhat, Y):
    N = Yhat.shape[0]
    return (2.0 / N) * (Yhat - Y)                      # (N,10)


def grad_softmax(Yhat, Y):
    N = Yhat.shape[0]
    P = softmax(Yhat)
    return (1.0 / N) * (P - Y)                         # (N,10)


def compute_grad_output(Yhat, Y, metric="mse"):
    if metric == "mse":
        return grad_mse(Yhat, Y)
    elif metric == "softmax":
        return grad_softmax(Yhat, Y)
    raise ValueError(f"metric debe ser 'mse' o 'softmax', recibido: {metric!r}")


def backward(cache, Y, params, lam, metric="mse"):
    X, H, Yhat = cache["X"], cache["H"], cache["Yhat"]

    # --- nodo costo + activacion lineal de la salida: dL/dZ2 ---------------
    dZ2 = compute_grad_output(Yhat, Y, metric=metric)  # (N,10)

    # --- capa 2 (afin): Z2 = H@W2 + b2 --------------------------------------
    dW2 = H.T @ dZ2 + 2.0 * lam * params["W2"]         # (100,10)
    db2 = dZ2.sum(axis=0)                              # (10,)
    dH = dZ2 @ params["W2"].T                          # (N,100)

    # --- sigmoide: dH/dZ1 = H*(1-H) -----------------------------------------
    dZ1 = dH * H * (1.0 - H)                           # (N,100)

    # --- capa 1 (afin): Z1 = X@W1 + b1 --------------------------------------
    dW1 = X.T @ dZ1 + 2.0 * lam * params["W1"]         # (3072,100)
    db1 = dZ1.sum(axis=0)                              # (100,)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


# =============================================================================
# 4. METRICA
# =============================================================================

def accuracy(Yhat, y):
    return float(np.mean(np.argmax(Yhat, axis=1) == y))


def evaluate(X, Y, y, params, lam, metric="mse", batch=2000):
    N = X.shape[0]
    total_data_loss = 0.0
    correct = 0
    for i in range(0, N, batch):
        Yb = Y[i:i + batch]
        Yhat, _ = forward(X[i:i + batch], params)
        if metric == "mse":
            total_data_loss += np.sum((Yhat - Yb) ** 2)
        elif metric == "softmax":
            P = softmax(Yhat)
            total_data_loss += -np.sum(Yb * np.log(P + 1e-12))
        else:
            raise ValueError(f"metric debe ser 'mse' o 'softmax', recibido: {metric!r}")
        correct += int(np.sum(np.argmax(Yhat, axis=1) == y[i:i + batch]))
    reg = lam * (np.sum(params["W1"] ** 2) + np.sum(params["W2"] ** 2))
    return total_data_loss / N + reg, correct / N


# =============================================================================
# 5. ENTRENAMIENTO (SGD por mini-lotes con momento)
# =============================================================================

def train(X_tr, Y_tr, y_tr, X_te, Y_te, y_te,
          n_hidden=N_HIDDEN, lr=LEARNING_RATE, momentum=MOMENTUM,
          lam=LAMBDA_L2, batch_size=BATCH_SIZE, epochs=EPOCHS, seed=SEED,
          loss_metric=LOSS_METRIC, grad_metric=GRAD_METRIC, verbose=True):

    rng = np.random.default_rng(seed)
    params = init_params(X_tr.shape[1], n_hidden, Y_tr.shape[1], seed=seed)
    # trabajar en la misma precision que los datos (float32 ahorra ~600 MB con CIFAR)
    params = {k: v.astype(X_tr.dtype) for k, v in params.items()}
    velocity = {k: np.zeros_like(v) for k, v in params.items()}

    # para las curvas de train usamos una submuestra fija (evaluar 50k por epoca es caro)
    idx_eval = rng.choice(X_tr.shape[0], size=min(10000, X_tr.shape[0]), replace=False)

    hist = {"loss_train": [], "loss_test": [], "acc_train": [], "acc_test": []}
    N = X_tr.shape[0]
    t0 = time.time()

    for ep in range(epochs):
        perm = rng.permutation(N)
        for i in range(0, N, batch_size):
            b = perm[i:i + batch_size]
            Xb, Yb = X_tr[b], Y_tr[b]

            _, cache = forward(Xb, params)
            grads = backward(cache, Yb, params, lam, metric=grad_metric)

            for k in params:                       # SGD con momento
                velocity[k] = momentum * velocity[k] - lr * grads[k]
                params[k] += velocity[k]

        ltr, atr = evaluate(X_tr[idx_eval], Y_tr[idx_eval], y_tr[idx_eval], params, lam, metric=loss_metric)
        lte, ate = evaluate(X_te, Y_te, y_te, params, lam, metric=loss_metric)
        hist["loss_train"].append(ltr)
        hist["loss_test"].append(lte)
        hist["acc_train"].append(atr)
        hist["acc_test"].append(ate)

        if verbose:
            print(f"  epoca {ep + 1:3d}/{epochs}  "
                  f"loss_train={ltr:.4f}  loss_test={lte:.4f}  "
                  f"acc_train={atr:.4f}  acc_test={ate:.4f}  "
                  f"[{time.time() - t0:.0f}s]")

    return params, hist


# =============================================================================
# 6. GRAFICOS
# =============================================================================

def plot_sample_images(x, y, n=3, path=None):
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2))
    if n == 1:
        axes = [axes]
    for i in range(n):
        axes[i].imshow(x[i])
        axes[i].set_title(CLASSES[y[i]])
        axes[i].axis("off")
    fig.suptitle("Primeras imagenes del set de entrenamiento")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=140)
        print(f"  figura guardada en {path}")
    plt.show()


def plot_curves(hist, path, loss_metric=LOSS_METRIC, grad_metric=GRAD_METRIC):
    ep = np.arange(1, len(hist["loss_train"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    nombre_costo = "CCE (softmax)" if loss_metric == "softmax" else "MSE"

    ax[0].plot(ep, hist["loss_train"], label="entrenamiento")
    ax[0].plot(ep, hist["loss_test"], label="test")
    ax[0].set_xlabel("epoca"); ax[0].set_ylabel(f"costo ({nombre_costo} + L2)")
    ax[0].set_title("Funcion de costo")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].plot(ep, 100 * np.array(hist["acc_train"]), label="entrenamiento")
    ax[1].plot(ep, 100 * np.array(hist["acc_test"]), label="test")
    ax[1].set_xlabel("epoca"); ax[1].set_ylabel("accuracy [%]")
    ax[1].set_title("Precision")
    ax[1].legend(); ax[1].grid(alpha=0.3)

    fig.suptitle(f"Ejercicio 4 - red 3072-100(sigmoide)-10(lineal), "
                 f"loss={loss_metric}, grad={grad_metric}, L2")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  figura guardada en {path}")


def plot_comparison(results, path):
    names = list(results.keys())
    vals = [100 * results[k] for k in names]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(names, vals)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}%",
                ha="center", fontsize=10)
    ax.axhline(10, ls="--", lw=1, color="gray")
    ax.text(len(names) - 0.5, 10.8, "azar (10%)", ha="right", fontsize=9, color="gray")
    ax.set_ylabel("accuracy en test [%]")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title("Comparacion de metodos (datos de test)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  figura guardada en {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 68)
    print("Practica 2 - Ejercicio 4: red de dos capas para CIFAR-10 (MSE/CCE)")
    print("=" * 68)

    # --- 1) datos -----------------------------------------------------------
    print("Cargando CIFAR-10...")
    x_tr, y_tr, x_te, y_te = load_cifar10()
    plot_sample_images(x_tr, y_tr, n=3, path=os.path.join(OUT_DIR, "ej4_muestras.png"))
    X_tr, X_te = preprocess(x_tr, x_te)
    Y_tr, Y_te = one_hot(y_tr, dtype=X_tr.dtype), one_hot(y_te, dtype=X_te.dtype)
    print(f"  train: {X_tr.shape}   test: {X_te.shape}   clases: {len(CLASSES)}\n")

    # --- 2) entrenamiento ---------------------------------------------------
    print(f"Entrenando (lr={LEARNING_RATE}, momento={MOMENTUM}, "
          f"lambda={LAMBDA_L2}, batch={BATCH_SIZE}, epocas={EPOCHS}, "
          f"loss_metric={LOSS_METRIC}, grad_metric={GRAD_METRIC})")
    params, hist = train(X_tr, Y_tr, y_tr, X_te, Y_te, y_te,
                          loss_metric=LOSS_METRIC, grad_metric=GRAD_METRIC)

    acc_red = hist["acc_test"][-1]
    print(f"\nAccuracy final de la red: train={hist['acc_train'][-1]:.4f}  "
          f"test={acc_red:.4f}\n")

    # --- 3) graficos --------------------------------------------------------
    plot_curves(hist, os.path.join(OUT_DIR, "ej4_curvas.png"),
                loss_metric=LOSS_METRIC, grad_metric=GRAD_METRIC)

    print("\nListo.")
    


if __name__ == "__main__":
    main()
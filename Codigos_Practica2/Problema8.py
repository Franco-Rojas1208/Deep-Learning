# -*- coding: utf-8 -*-

import os
import time

import numpy as np
import matplotlib.pyplot as plt

from activations import Sigmoid, Linear
from losses import MSE
from layers import Input, Dense
from models import Network
from optimizers import SGD
import metric


# =============================================================================
# CONFIGURACION (mismos hiperparametros que Problema3.py, para comparar)
# =============================================================================

DATA_DIR = "./data"
OUT_DIR = "./figuras"
SEED = 0

N_HIDDEN = 100
LEARNING_RATE = 0.03
MOMENTUM = 0.0
LAMBDA_L2 = 1e-5
BATCH_SIZE = 128
EPOCHS = 80

CLASSES = ["avion", "auto", "pajaro", "gato", "ciervo", "perro", "rana", "caballo", "barco", "camion"]


# =============================================================================
# 1. DATOS (identico a Problema3.py/Problema4.py)
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
# 2. ARQUITECTURAS (armadas apilando Dense del ejercicio 6)
# =============================================================================

def build_network_1_capa(n_in=3072, n_hidden=N_HIDDEN, n_out=10, lam=LAMBDA_L2, seed=SEED):
    net = Network(loss=MSE(), optimizer=SGD(lr=LEARNING_RATE, momentum=MOMENTUM))
    net.add(Input())
    net.add(Dense(n_in, n_hidden, Sigmoid(), seed=seed, l2=lam))
    net.add(Dense(n_hidden, n_out, Linear(), seed=seed + 1, l2=lam))
    return net


def build_network_2_capas(n_in=3072, n_hidden=N_HIDDEN, n_out=10, lam=LAMBDA_L2, seed=SEED):
    net = Network(loss=MSE(), optimizer=SGD(lr=LEARNING_RATE, momentum=MOMENTUM))
    net.add(Input())
    net.add(Dense(n_in, n_hidden, Sigmoid(), seed=seed, l2=lam))
    net.add(Dense(n_hidden, n_hidden, Sigmoid(), seed=seed + 1, l2=lam))
    net.add(Dense(n_hidden, n_out, Linear(), seed=seed + 2, l2=lam))
    return net


# =============================================================================
# 3. EVALUACION Y ENTRENAMIENTO
# =============================================================================

def compute_reg_loss(net):
    return sum(layer.reg_loss() for layer in net.layers if hasattr(layer, "reg_loss"))


def evaluate(net, X, Y, y, batch=2000):
    N = X.shape[0]
    total_sq = 0.0
    correct = 0
    for i in range(0, N, batch):
        y_pred = net.forward(X[i:i + batch])
        total_sq += np.sum((y_pred - Y[i:i + batch]) ** 2)
        correct += int(np.sum(np.argmax(y_pred, axis=1) == y[i:i + batch]))
    reg = compute_reg_loss(net)
    return total_sq / N + reg, correct / N


def train(net, X_tr, Y_tr, y_tr, X_te, Y_te, y_te,
          batch_size=BATCH_SIZE, epochs=EPOCHS, seed=SEED, verbose=True, nombre=""):
    rng = np.random.default_rng(seed)
    N = X_tr.shape[0]
    # para las curvas de train usamos una submuestra fija (evaluar 50k por epoca es caro)
    idx_eval = rng.choice(N, size=min(10000, N), replace=False)

    hist = {"loss_train": [], "loss_test": [], "acc_train": [], "acc_test": []}
    t0 = time.time()

    for ep in range(epochs):
        perm = rng.permutation(N)
        for i in range(0, N, batch_size):
            b = perm[i:i + batch_size]
            Xb, Yb = X_tr[b], Y_tr[b]

            y_pred = net.forward(Xb)
            net.backward(y_pred, Yb)
            net.optimizer.step(net.layers)

        ltr, atr = evaluate(net, X_tr[idx_eval], Y_tr[idx_eval], y_tr[idx_eval])
        lte, ate = evaluate(net, X_te, Y_te, y_te)
        hist["loss_train"].append(ltr)
        hist["loss_test"].append(lte)
        hist["acc_train"].append(atr)
        hist["acc_test"].append(ate)

        if verbose and ((ep + 1) % 10 == 0 or ep == 0 or ep == epochs - 1):
            print(f"  [{nombre}] epoca {ep + 1:3d}/{epochs}  "
                  f"loss_train={ltr:.4f}  loss_test={lte:.4f}  "
                  f"acc_train={atr:.4f}  acc_test={ate:.4f}  "
                  f"[{time.time() - t0:.0f}s]")

    return hist


# =============================================================================
# 4. GRAFICOS
# =============================================================================

def plot_curvas(hist, titulo, path):
    ep = np.arange(1, len(hist["loss_train"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    ax[0].plot(ep, hist["loss_train"], label="entrenamiento")
    ax[0].plot(ep, hist["loss_test"], label="test")
    ax[0].set_xlabel("epoca"); ax[0].set_ylabel("costo (MSE + L2)")
    ax[0].set_title("Funcion de costo")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].plot(ep, 100 * np.array(hist["acc_train"]), label="entrenamiento")
    ax[1].plot(ep, 100 * np.array(hist["acc_test"]), label="test")
    ax[1].set_xlabel("epoca"); ax[1].set_ylabel("accuracy [%]")
    ax[1].set_title("Precision")
    ax[1].legend(); ax[1].grid(alpha=0.3)

    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  figura guardada en {path}")


def plot_comparacion(hist_a, hist_b, path):
    ep_a = np.arange(1, len(hist_a["loss_test"]) + 1)
    ep_b = np.arange(1, len(hist_b["loss_test"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    ax[0].plot(ep_a, hist_a["loss_test"], label="A: 3072-100-10 (ej. 3)")
    ax[0].plot(ep_b, hist_b["loss_test"], label="B: 3072-100-100-10 (ej. 8)")
    ax[0].set_xlabel("epoca"); ax[0].set_ylabel("costo test (MSE + L2)")
    ax[0].set_title("Funcion de costo (test)")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].plot(ep_a, 100 * np.array(hist_a["acc_test"]), label="A: 3072-100-10 (ej. 3)")
    ax[1].plot(ep_b, 100 * np.array(hist_b["acc_test"]), label="B: 3072-100-100-10 (ej. 8)")
    ax[1].set_xlabel("epoca"); ax[1].set_ylabel("accuracy test [%]")
    ax[1].set_title("Precision (test)")
    ax[1].legend(); ax[1].grid(alpha=0.3)

    fig.suptitle("Ejercicio 8 - Efecto de agregar una segunda capa oculta (CIFAR-10)")
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
    print("Practica 2 - Ejercicio 8: CIFAR-10 con una capa oculta extra (POO)")
    print("=" * 68)

    # --- 1) datos -------------------------------------------------------------
    print("Cargando CIFAR-10...")
    x_tr, y_tr, x_te, y_te = load_cifar10()
    X_tr, X_te = preprocess(x_tr, x_te)
    Y_tr, Y_te = one_hot(y_tr, dtype=X_tr.dtype), one_hot(y_te, dtype=X_te.dtype)
    print(f"  train: {X_tr.shape}   test: {X_te.shape}   clases: {len(CLASSES)}\n")

    print(f"Hiperparametros: lr={LEARNING_RATE}, momento={MOMENTUM}, "
          f"lambda={LAMBDA_L2}, batch={BATCH_SIZE}, epocas={EPOCHS}\n")

    # --- 2) red A: la del ejercicio 3, reproducida con el motor del 6 --------
    print("Entrenando red A (3072-100-10, igual al ejercicio 3):")
    net_a = build_network_1_capa(seed=SEED)
    hist_a = train(net_a, X_tr, Y_tr, y_tr, X_te, Y_te, y_te, nombre="A")

    # --- 3) red B: la de este ejercicio, con la capa oculta extra -----------
    print("\nEntrenando red B (3072-100-100-10, capa oculta extra):")
    net_b = build_network_2_capas(seed=SEED)
    hist_b = train(net_b, X_tr, Y_tr, y_tr, X_te, Y_te, y_te, nombre="B")

    print(f"\nAccuracy final en test:")
    print(f"  A (3072-100-10)      : {hist_a['acc_test'][-1]:.4f}")
    print(f"  B (3072-100-100-10)  : {hist_b['acc_test'][-1]:.4f}")

    # --- 4) graficos -----------------------------------------------------------
    plot_curvas(hist_a, "Ejercicio 8 - Red A: 3072-100-10 (= ejercicio 3, motor del ej. 6)",
                os.path.join(OUT_DIR, "ej8_curvas_red_A.png"))
    plot_curvas(hist_b, "Ejercicio 8 - Red B: 3072-100-100-10 (capa oculta extra)",
                os.path.join(OUT_DIR, "ej8_curvas_red_B.png"))
    plot_comparacion(hist_a, hist_b, os.path.join(OUT_DIR, "ej8_comparacion.png"))

    print("\nListo.")


if __name__ == "__main__":
    main()

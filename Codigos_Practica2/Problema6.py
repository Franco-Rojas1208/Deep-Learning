# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt

from activations import Tanh
from losses import MSE
from layers import Input, Dense
from models import Network
from optimizers import SGD
import metric


# =============================================================================
# CONFIGURACION
# =============================================================================

OUT_DIR = "./figuras"
SEED = 0
LEARNING_RATE = 0.5
EPOCHS = 1000


# =============================================================================
# 1. DATOS
# =============================================================================

def make_xor_data():
    X = np.array([[-1., -1.],
                  [-1.,  1.],
                  [ 1., -1.],
                  [ 1.,  1.]])
    Y = np.array([[ 1.],
                  [-1.],
                  [-1.],
                  [ 1.]])
    return X, Y


# =============================================================================
# 2. ARQUITECTURA
# =============================================================================

def build_network(n_hidden, lr=LEARNING_RATE, seed=SEED):
    net = Network(loss=MSE(), optimizer=SGD(lr=lr))
    net.add(Input())
    net.add(Dense(2, n_hidden, Tanh(), seed=seed))
    net.add(Dense(n_hidden, 1, Tanh(), seed=seed + 1))
    return net


# =============================================================================
# 3. GRAFICOS
# =============================================================================

def plot_comparison(hist_a, hist_b, path):
    ep_a = np.arange(1, len(hist_a["loss"]) + 1)
    ep_b = np.arange(1, len(hist_b["loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

    ax[0].plot(ep_a, hist_a["loss"], label="A: 2-2-1")
    ax[0].plot(ep_b, hist_b["loss"], label="B: 2-1-1")
    ax[0].set_xlabel("epoca"); ax[0].set_ylabel("costo (MSE)")
    ax[0].set_title("Funcion de costo")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].plot(ep_a, 100 * np.array(hist_a["accuracy"]), label="A: 2-2-1")
    ax[1].plot(ep_b, 100 * np.array(hist_b["accuracy"]), label="B: 2-1-1")
    ax[1].set_xlabel("epoca"); ax[1].set_ylabel("accuracy [%]")
    ax[1].set_ylim(-5, 105)
    ax[1].set_title("Precision")
    ax[1].legend(); ax[1].grid(alpha=0.3)

    fig.suptitle("Ejercicio 6 - XOR con tanh + MSE: arquitectura 2-2-1 vs 2-1-1")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  figura guardada en {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X, Y = make_xor_data()

    print("=" * 68)
    print("Practica 2 - Ejercicio 6: XOR con backpropagation (POO)")
    print("=" * 68)
    print(f"\nDatos (batch completo, {X.shape[0]} ejemplos):")
    for xi, yi in zip(X, Y):
        print(f"  x={xi}  ->  y={yi}")

    print("\nEntrenando arquitectura A (2-2-1):")
    net_a = build_network(n_hidden=2)
    hist_a = net_a.fit(X, Y, epochs=EPOCHS, batch_size=None, seed=SEED, verbose=True)

    print("\nEntrenando arquitectura B (2-1-1):")
    net_b = build_network(n_hidden=1)
    hist_b = net_b.fit(X, Y, epochs=EPOCHS, batch_size=None, seed=SEED, verbose=True)

    print("\nPredicciones finales:")
    for nombre, net in [("A (2-2-1)", net_a), ("B (2-1-1)", net_b)]:
        pred = net.predict(X)
        acc = metric.accuracy(pred, Y)
        print(f"  {nombre}: pred={np.round(pred.ravel(), 3).tolist()}  accuracy={acc:.2f}")

    plot_comparison(hist_a, hist_b, os.path.join(OUT_DIR, "ej6_curvas.png"))

    print("\nListo.")


if __name__ == "__main__":
    main()

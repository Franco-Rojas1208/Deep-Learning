# -*- coding: utf-8 -*-

import os
import itertools

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
EPOCHS = 4000
N_RESTARTS = 6          # reintentos con distinta semilla por cada configuracion (N, N')


# =============================================================================
# 1. DATOS
# =============================================================================

def make_parity_data(N):
    X = np.array(list(itertools.product([-1., 1.], repeat=N)))
    Y = np.prod(X, axis=1, keepdims=True)
    return X, Y


# =============================================================================
# 2. ARQUITECTURA (igual a la de Problema6.build_network, generalizada a N entradas)
# =============================================================================

def build_network(n_inputs, n_hidden, lr=LEARNING_RATE, seed=SEED):
    net = Network(loss=MSE(), optimizer=SGD(lr=lr))
    net.add(Input())
    net.add(Dense(n_inputs, n_hidden, Tanh(), seed=seed))
    net.add(Dense(n_hidden, 1, Tanh(), seed=seed + 1))
    return net


def train_parity_restarts(N, n_hidden, epochs=EPOCHS, lr=LEARNING_RATE,
                           n_restarts=N_RESTARTS, base_seed=SEED):
    X, Y = make_parity_data(N)
    mejor_hist, mejor_loss = None, np.inf
    exitos = 0
    for r in range(n_restarts):
        seed = base_seed + r
        net = build_network(N, n_hidden, lr=lr, seed=seed)
        hist = net.fit(X, Y, epochs=epochs, batch_size=None, seed=seed)
        if hist["accuracy"][-1] >= 1.0:
            exitos += 1
        if hist["loss"][-1] < mejor_loss:
            mejor_loss, mejor_hist = hist["loss"][-1], hist
    return mejor_hist, exitos / n_restarts


# =============================================================================
# 3. EXPERIMENTOS: efecto de N y de N'
# =============================================================================

def experimento_variar_n_hidden(N, hidden_sizes, **kwargs):
    print(f"\n--- Efecto de N' (N={N} fijo, {2 ** N} ejemplos, "
          f"{N_RESTARTS} semillas por punto) ---")
    resultados = {}
    for n_hidden in hidden_sizes:
        hist, tasa_exito = train_parity_restarts(N, n_hidden, **kwargs)
        resultados[n_hidden] = (hist, tasa_exito)
        print(f"  N'={n_hidden:2d}   tasa_exito={tasa_exito:4.0%}   "
              f"mejor accuracy_final={hist['accuracy'][-1]:.2f}   "
              f"mejor loss_final={hist['loss'][-1]:.4f}")
    return resultados


def experimento_variar_n(n_values, n_hidden_fn, **kwargs):
    print("\n--- Efecto de N ---")
    resultados = {}
    for N in n_values:
        n_hidden = n_hidden_fn(N)
        hist, tasa_exito = train_parity_restarts(N, n_hidden, **kwargs)
        resultados[N] = (hist, tasa_exito)
        print(f"  N={N:2d}  N'={n_hidden:2d} ({2 ** N:3d} ejemplos)   "
              f"tasa_exito={tasa_exito:4.0%}   "
              f"mejor accuracy_final={hist['accuracy'][-1]:.2f}   "
              f"mejor loss_final={hist['loss'][-1]:.4f}")
    return resultados


# =============================================================================
# 4. GRAFICOS
# =============================================================================

def plot_curvas_y_exito(resultados, titulo_sufijo, xlabel, path, etiqueta_fn):
    claves = list(resultados.keys())
    ep_max = max(len(hist["loss"]) for hist, _ in resultados.values())
    ep = np.arange(1, ep_max + 1)

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.5))

    for clave in claves:
        hist, _ = resultados[clave]
        ax[0].plot(ep, hist["loss"], label=etiqueta_fn(clave))
        ax[1].plot(ep, 100 * np.array(hist["accuracy"]), label=etiqueta_fn(clave))

    ax[0].set_xlabel("epoca"); ax[0].set_ylabel("costo (MSE)")
    ax[0].set_title("Funcion de costo (mejor corrida)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    ax[1].set_xlabel("epoca"); ax[1].set_ylabel("accuracy [%]"); ax[1].set_ylim(-5, 105)
    ax[1].set_title("Precision (mejor corrida)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    tasas = [100 * resultados[c][1] for c in claves]
    bars = ax[2].bar([str(etiqueta_fn(c)) for c in claves], tasas)
    for b, v in zip(bars, tasas):
        ax[2].text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%", ha="center", fontsize=9)
    ax[2].set_ylim(0, 110)
    ax[2].set_xlabel(xlabel); ax[2].set_ylabel("tasa de exito [%]")
    ax[2].set_title(f"Exito sobre {N_RESTARTS} semillas")
    ax[2].grid(axis="y", alpha=0.3)

    fig.suptitle(f"Ejercicio 7 - Problema de paridad: {titulo_sufijo}")
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
    print("Practica 2 - Ejercicio 7: problema de paridad (generalizacion del XOR)")
    print("=" * 68)

    X4, Y4 = make_parity_data(4)
    print(f"\nEjemplo con N=4 ({X4.shape[0]} combinaciones):")
    for xi, yi in zip(X4, Y4):
        print(f"  x={xi}  ->  y={yi}")

    # --- experimento 1: N fijo, variar N' -----------------------------------
    res_hidden = experimento_variar_n_hidden(N=4, hidden_sizes=(1, 2, 3, 4, 6, 8))
    plot_curvas_y_exito(res_hidden, titulo_sufijo="efecto de N' (N=4 fijo)",
                         xlabel="N'", path=os.path.join(OUT_DIR, "ej7_variar_n_hidden.png"),
                         etiqueta_fn=lambda n_hidden: f"N'={n_hidden}")

    # --- experimento 2: N' = N, variar N -------------------------------------
    res_n = experimento_variar_n(n_values=(2, 3, 4, 5, 6), n_hidden_fn=lambda N: N)
    plot_curvas_y_exito(res_n, titulo_sufijo="efecto de N (con N'=N)",
                         xlabel="N", path=os.path.join(OUT_DIR, "ej7_variar_n.png"),
                         etiqueta_fn=lambda N: f"N={N}")

    # --- experimento 3: N' fijo y chico, variar N (deberia degradar) --------
    res_n_hidden_fijo = experimento_variar_n(n_values=(2, 3, 4, 5, 6), n_hidden_fn=lambda N: 2)
    plot_curvas_y_exito(res_n_hidden_fijo, titulo_sufijo="efecto de N (con N'=2 fijo)",
                         xlabel="N", path=os.path.join(OUT_DIR, "ej7_variar_n_nhidden_fijo.png"),
                         etiqueta_fn=lambda N: f"N={N}")

    print("\nListo.")


if __name__ == "__main__":
    main()

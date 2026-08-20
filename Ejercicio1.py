"""
Práctica 1 - Ejercicio 1: k-Nearest Neighbors sobre datos 2D
------------------------------------------------------------
1) Generamos un conjunto de datos bidimensional con 5 clases.
2) Clasificamos con k-NN tomando k = 1, 3 y 7 vecinos.
3) Graficamos las fronteras de decisión.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ----------------------------------------------------------------------
# 1) GENERACIÓN DEL DATASET
# ----------------------------------------------------------------------
def generar_datos(n_por_clase=80, dispersion=0.9, semilla=0):
    """
    Genera 5 clases gaussianas 2D alrededor de 5 centros distintos.
    Devuelve X (N, 2) con las coordenadas e y (N,) con las etiquetas 0..4.
    """
    rng = np.random.default_rng(semilla)

    # Centros de cada clase en el plano
    centros = np.array([
        [0.0,  0.0],
        [3.0,  3.0],
        [-3.0, 3.0],
        [3.0, -3.0],
        [-3.0, -3.0],
    ])

    X, y = [], []
    for etiqueta, centro in enumerate(centros):
        # Nube de puntos gaussiana centrada en 'centro'
        puntos = centro + dispersion * rng.standard_normal((n_por_clase, 2))
        X.append(puntos)
        y.append(np.full(n_por_clase, etiqueta))

    X = np.vstack(X)
    y = np.concatenate(y)
    return X, y


# ----------------------------------------------------------------------
# 2) CLASIFICADOR k-NN (implementado a mano)
# ----------------------------------------------------------------------
class KNN:
    """k-Nearest Neighbors con distancia euclídea y voto por mayoría."""

    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        # k-NN es "perezoso": solo memoriza los datos de entrenamiento.
        self.X_train = X
        self.y_train = y
        self.n_clases = len(np.unique(y))
        return self

    def predict(self, X):
        # Distancia euclídea de cada punto de X a cada punto de entrenamiento.
        # dists[i, j] = ||X[i] - X_train[j]||
        dists = np.sqrt(
            ((X[:, None, :] - self.X_train[None, :, :]) ** 2).sum(axis=2)
        )

        # Índices de los k vecinos más cercanos para cada punto.
        vecinos = np.argsort(dists, axis=1)[:, : self.k]

        # Etiquetas de esos vecinos.
        etiquetas_vecinos = self.y_train[vecinos]

        # Voto por mayoría: la clase más frecuente entre los k vecinos.
        pred = np.array([np.bincount(fila, minlength=self.n_clases).argmax() for fila in etiquetas_vecinos])
        return pred


# ----------------------------------------------------------------------
# 3) GRAFICADO DE FRONTERAS DE DECISIÓN
# ----------------------------------------------------------------------
def graficar_fronteras(X, y, valores_k=(1, 3, 7), paso=0.05):
    # Paleta de colores para 5 clases
    cmap_fondo = ListedColormap(
        ["#ffd6d6", "#d6e5ff", "#d6ffd9", "#fff2cc", "#eadcff"]
    )
    cmap_puntos = ListedColormap(
        ["#e63946", "#1d6fe0", "#2a9d3f", "#e0a400", "#8a4fdb"]
    )

    # Malla que cubre todo el plano de los datos
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, paso),
        np.arange(y_min, y_max, paso),
    )
    puntos_malla = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, len(valores_k), figsize=(15, 5))

    n_clases = len(np.unique(y))
    niveles = np.arange(n_clases + 1) - 0.5  # bandas centradas en cada entero de clase

    for ax, k in zip(axes, valores_k):
        modelo = KNN(k=k).fit(X, y)
        Z = modelo.predict(puntos_malla).reshape(xx.shape)

        # Colorea cada región del plano según la clase predicha
        ax.contourf(xx, yy, Z, levels=niveles, cmap=cmap_fondo, alpha=0.9)
        # Dibuja los puntos de entrenamiento
        ax.scatter(
            X[:, 0], X[:, 1], c=y, cmap=cmap_puntos,
            edgecolors="k", s=25, linewidths=0.4,
        )
        ax.set_title(f"k-NN con k = {k}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    fig.suptitle("Fronteras de decisión de k-NN (5 clases)", fontsize=14)
    fig.tight_layout()
    fig.savefig("fronteras_knn.pdf", dpi=130, bbox_inches="tight")
    print("Gráfico guardado en 'fronteras_knn.png'")
    plt.show()


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    X, y = generar_datos()

    # Chequeo rápido de exactitud sobre los propios datos (informativo)
    for k in (1, 3, 7):
        modelo = KNN(k=k).fit(X, y)
        acc = (modelo.predict(X) == y).mean()
        print(f"Exactitud (sobre entrenamiento) con k={k}: {acc:.3f}")

    graficar_fronteras(X, y)

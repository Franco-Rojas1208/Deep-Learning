

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def generar_datos(n_por_clase=80, dispersion=0.9, semilla=0):
    rng = np.random.default_rng(semilla)

    centros = np.array([
        [0.0,  0.0],
        [3.0,  3.0],
        [-3.0, 3.0],
        [3.0, -3.0],
        [-3.0, -3.0],
    ])

    X, y = [], []
    for etiqueta, centro in enumerate(centros):
        puntos = centro + dispersion * rng.standard_normal((n_por_clase, 2))
        X.append(puntos)
        y.append(np.full(n_por_clase, etiqueta))

    X = np.vstack(X)
    y = np.concatenate(y)
    return X, y


class KNN:

    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.n_clases = len(np.unique(y))
        return self

    def predict(self, X):
        dists = np.sqrt(
            ((X[:, None, :] - self.X_train[None, :, :]) ** 2).sum(axis=2)
        )

        vecinos = np.argsort(dists, axis=1)[:, : self.k]

        etiquetas_vecinos = self.y_train[vecinos]

        pred = np.array([np.bincount(fila, minlength=self.n_clases).argmax() for fila in etiquetas_vecinos])
        return pred


def graficar_fronteras(X, y, valores_k=(1, 3, 7), paso=0.05):
    cmap_fondo = ListedColormap(
        ["#ffd6d6", "#d6e5ff", "#d6ffd9", "#fff2cc", "#eadcff"]
    )
    cmap_puntos = ListedColormap(
        ["#e63946", "#1d6fe0", "#2a9d3f", "#e0a400", "#8a4fdb"]
    )

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, paso),
        np.arange(y_min, y_max, paso),
    )
    puntos_malla = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, len(valores_k), figsize=(15, 5))

    n_clases = len(np.unique(y))
    niveles = np.arange(n_clases + 1) - 0.5

    for ax, k in zip(axes, valores_k):
        modelo = KNN(k=k).fit(X, y)
        Z = modelo.predict(puntos_malla).reshape(xx.shape)

        ax.contourf(xx, yy, Z, levels=niveles, cmap=cmap_fondo, alpha=0.9)
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



X, y = generar_datos()

for k in (1, 3, 7):
    modelo = KNN(k=k).fit(X, y)
    acc = (modelo.predict(X) == y).mean()
    print(f"Exactitud (sobre entrenamiento) con k={k}: {acc:.3f}")

graficar_fronteras(X, y)

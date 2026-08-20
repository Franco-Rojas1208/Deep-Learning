"""
=============================================================================
 Clasificacion k-Nearest Neighbors (k-NN) para MNIST y CIFAR-10
=============================================================================

Consigna:
  Implementar el metodo de clasificacion k-nearest neighbors para clasificar
  las imagenes de las bases de datos MNIST y CIFAR-10. Evaluar la exactitud
  para el caso de MNIST con los primeros 20 ejemplos de test.
  (Usar keras para cargar los datos.)

Idea del metodo:
  k-NN no "entrena" en el sentido usual: simplemente memoriza el conjunto de
  entrenamiento. Para clasificar una imagen nueva, calcula su distancia a
  TODAS las imagenes de entrenamiento, toma las k mas cercanas y vota por
  mayoria la etiqueta. Aca usamos distancia euclidea (L2) sobre los pixeles.

Requisitos:
  pip install tensorflow numpy
=============================================================================
"""

import os
import numpy as np
from tensorflow.keras.datasets import mnist, cifar10

# Carpeta local donde se cachean los datasets ya descargados, para no
# tener que volver a bajarlos en cada ejecucion.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_dataset_cached(name, loader_fn):
    """
    Carga un dataset (mnist/cifar10) desde un .npz local si ya existe;
    si no, lo descarga con `loader_fn` (keras) y lo guarda en DATA_DIR
    para las proximas ejecuciones.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = os.path.join(DATA_DIR, f"{name}.npz")

    if os.path.exists(cache_path):
        with np.load(cache_path) as f:
            x_train, y_train = f["x_train"], f["y_train"]
            x_test, y_test = f["x_test"], f["y_test"]
        return (x_train, y_train), (x_test, y_test)

    (x_train, y_train), (x_test, y_test) = loader_fn()
    np.savez(
        cache_path,
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
    )
    return (x_train, y_train), (x_test, y_test)


# -----------------------------------------------------------------------------
# 1) Clasificador k-NN (implementado desde cero con numpy)
# -----------------------------------------------------------------------------
class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        """'Entrenar' = memorizar los datos (aplanados a vectores)."""
        self.X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
        self.y_train = np.asarray(y_train).reshape(-1)

    def _distances(self, X):
        """
        Distancia euclidea entre cada muestra de X y cada muestra de train,
        de forma vectorizada usando  ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        Devuelve una matriz (n_test, n_train).
        """
        X = X.reshape(X.shape[0], -1).astype(np.float32)
        cross = X @ self.X_train.T                        # (n_test, n_train)
        sq_test = np.sum(X ** 2, axis=1, keepdims=True)   # (n_test, 1)
        sq_train = np.sum(self.X_train ** 2, axis=1)      # (n_train,)
        d2 = sq_test + sq_train - 2.0 * cross
        np.maximum(d2, 0, out=d2)     # corregir negativos por error numerico
        return np.sqrt(d2)

    def predict(self, X):
        dists = self._distances(X)
        # indices de los k vecinos mas cercanos por cada fila (imagen de test)
        knn_idx = np.argpartition(dists, self.k, axis=1)[:, : self.k]
        knn_labels = self.y_train[knn_idx]                # (n_test, k)

        # voto por mayoria entre los k vecinos
        preds = np.array([np.bincount(row).argmax() for row in knn_labels])
        return preds


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return np.mean(y_true == y_pred)


# -----------------------------------------------------------------------------
# 2) MNIST  ->  exactitud sobre los primeros 20 ejemplos de test (consigna)
# -----------------------------------------------------------------------------
def run_mnist(k=3, n_test=20):
    print("=" * 62)
    print(f"MNIST   (k = {k})")
    print("=" * 62)

    (x_train, y_train), (x_test, y_test) = load_dataset_cached("mnist", mnist.load_data)
    print(f"Train: {x_train.shape}   Test: {x_test.shape}")

    # Normalizacion de pixeles a [0, 1] (castear a float32 antes de dividir
    # evita el paso intermedio en float64, que duplica la memoria usada)
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    clf = KNNClassifier(k=k)
    clf.fit(x_train, y_train)

    # Solo los primeros n_test ejemplos de test
    y_true = y_test[:n_test]
    y_pred = clf.predict(x_test[:n_test])

    acc = accuracy(y_true, y_pred)
    print(f"\nPrimeros {n_test} ejemplos de test:")
    print(f"  Reales      : {y_true.tolist()}")
    print(f"  Predicciones: {y_pred.tolist()}")
    print(f"  EXACTITUD   : {acc:.4f}  ({int(round(acc*n_test))}/{n_test})")
    return acc


# -----------------------------------------------------------------------------
# 3) CIFAR-10
# -----------------------------------------------------------------------------
def run_cifar10(k=3, n_test=20):
    print("\n" + "=" * 62)
    print(f"CIFAR-10   (k = {k})")
    print("=" * 62)

    (x_train, y_train), (x_test, y_test) = load_dataset_cached("cifar10", cifar10.load_data)
    print(f"Train: {x_train.shape}   Test: {x_test.shape}")

    # Castear a float32 antes de dividir evita el paso intermedio en
    # float64, que duplica la memoria usada (relevante en CIFAR-10)
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    clf = KNNClassifier(k=k)
    clf.fit(x_train, y_train)

    y_true = y_test[:n_test].reshape(-1)
    y_pred = clf.predict(x_test[:n_test])

    acc = accuracy(y_true, y_pred)
    print(f"\nPrimeros {n_test} ejemplos de test:")
    print(f"  Reales      : {y_true.tolist()}")
    print(f"  Predicciones: {y_pred.tolist()}")
    print(f"  EXACTITUD   : {acc:.4f}  ({int(round(acc*n_test))}/{n_test})")
    return acc


if __name__ == "__main__":
    # Consigna: exactitud de MNIST sobre los primeros 20 ejemplos de test
    run_mnist(k=3, n_test=20)

    # El mismo metodo aplicado a CIFAR-10
    run_cifar10(k=3, n_test=20)
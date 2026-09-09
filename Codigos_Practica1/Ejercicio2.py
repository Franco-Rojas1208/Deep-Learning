import os
import numpy as np
from tensorflow.keras.datasets import mnist, cifar10

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_dataset_cached(name, loader_fn):
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


class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
        self.y_train = np.asarray(y_train).reshape(-1)

    def _distances(self, X):
        X = X.reshape(X.shape[0], -1).astype(np.float32)
        cross = X @ self.X_train.T
        sq_test = np.sum(X ** 2, axis=1, keepdims=True)
        sq_train = np.sum(self.X_train ** 2, axis=1)
        d2 = sq_test + sq_train - 2.0 * cross
        np.maximum(d2, 0, out=d2)
        return np.sqrt(d2)

    def predict(self, X):
        dists = self._distances(X)
        knn_idx = np.argpartition(dists, self.k, axis=1)[:, : self.k]
        knn_labels = self.y_train[knn_idx]

        preds = np.array([np.bincount(row).argmax() for row in knn_labels])
        return preds


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return np.mean(y_true == y_pred)


def run_mnist(k=3, n_test=20):
    print("=" * 62)
    print(f"MNIST   (k = {k})")
    print("=" * 62)

    (x_train, y_train), (x_test, y_test) = load_dataset_cached("mnist", mnist.load_data)
    print(f"Train: {x_train.shape}   Test: {x_test.shape}")

    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    clf = KNNClassifier(k=k)
    clf.fit(x_train, y_train)

    y_true = y_test[:n_test]
    y_pred = clf.predict(x_test[:n_test])

    acc = accuracy(y_true, y_pred)
    print(f"\nPrimeros {n_test} ejemplos de test:")
    print(f"  Reales      : {y_true.tolist()}")
    print(f"  Predicciones: {y_pred.tolist()}")
    print(f"  EXACTITUD   : {acc:.4f}  ({int(round(acc*n_test))}/{n_test})")
    return acc


def run_cifar10(k=3, n_test=20):
    print("\n" + "=" * 62)
    print(f"CIFAR-10   (k = {k})")
    print("=" * 62)

    (x_train, y_train), (x_test, y_test) = load_dataset_cached("cifar10", cifar10.load_data)
    print(f"Train: {x_train.shape}   Test: {x_test.shape}")

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



run_mnist(k=3, n_test=20)

run_cifar10(k=3, n_test=20)

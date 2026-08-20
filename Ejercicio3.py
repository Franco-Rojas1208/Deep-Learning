import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def accuracy(y_pred, y_true):
    return float(np.mean(y_pred == y_true))


class LinearClassifier:

    def __init__(self, reg=1e-3, learning_rate=1e-2, weight_scale=1e-3,
                 momentum=0.0, seed=0):
        self.reg = reg
        self.learning_rate = learning_rate
        self.weight_scale = weight_scale
        self.momentum = momentum
        self.seed = seed
        self.W = None
        self.history = None

    @staticmethod
    def _add_bias(X):
        return np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])

    def _init_weights(self, D, K):
        rng = np.random.default_rng(self.seed)
        self.W = (self.weight_scale *
                  rng.standard_normal((D + 1, K))).astype(np.float32)

    def _l2_penalty(self):
        W_reg = self.W.copy()
        W_reg[-1, :] = 0.0
        loss = self.reg * np.sum(W_reg ** 2)
        grad = 2.0 * self.reg * W_reg
        return loss, grad

    def loss_gradient(self, X_bias, y):
        raise NotImplementedError("Implementar en la subclase")

    def scores(self, X):
        return self._add_bias(X.astype(np.float32)) @ self.W

    def predict(self, X):
        return np.argmax(self.scores(X), axis=1)

    def accuracy(self, X, y):
        return accuracy(self.predict(X), y)

    def evaluate(self, X, y):
        X_bias = self._add_bias(X.astype(np.float32))
        loss, _ = self.loss_gradient(X_bias, y)
        acc = accuracy(np.argmax(X_bias @ self.W, axis=1), y)
        return loss, acc

    def fit(self, X, y, X_val=None, y_val=None, epochs=20, batch_size=128,
            lr_decay=1.0, n_classes=None, eval_subset=5000, verbose=True):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.int64).ravel()
        N, D = X.shape
        K = n_classes if n_classes is not None else int(y.max()) + 1

        self._init_weights(D, K)
        velocity = np.zeros_like(self.W)

        X_bias = self._add_bias(X)
        rng = np.random.default_rng(self.seed)
        lr = self.learning_rate

        self.history = {"train_loss": [], "train_acc": [],
                        "val_loss": [], "val_acc": [], "epoch": []}

        idx_eval = rng.choice(N, size=min(eval_subset, N), replace=False)

        for epoch in range(1, epochs + 1):
            perm = rng.permutation(N)

            for start in range(0, N, batch_size):
                batch = perm[start:start + batch_size]
                _, dW = self.loss_gradient(X_bias[batch], y[batch])

                if self.momentum:
                    velocity = self.momentum * velocity - lr * dW
                    self.W += velocity
                else:
                    self.W -= lr * dW

            tr_loss, tr_acc = self._eval_bias(X_bias[idx_eval], y[idx_eval])
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)

            if X_val is not None:
                va_loss, va_acc = self.evaluate(X_val, y_val)
                self.history["val_loss"].append(va_loss)
                self.history["val_acc"].append(va_acc)
            else:
                va_loss = va_acc = np.nan

            if verbose:
                print(f"  epoca {epoch:3d}/{epochs} | "
                      f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                      f"test loss {va_loss:.4f} acc {va_acc:.4f}")

            lr *= lr_decay

        return self.history

    def _eval_bias(self, X_bias, y):
        loss, _ = self.loss_gradient(X_bias, y)
        return loss, accuracy(np.argmax(X_bias @ self.W, axis=1), y)


class SVMClassifier(LinearClassifier):

    def __init__(self, delta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta

    def loss_gradient(self, X_bias, y):
        N = X_bias.shape[0]
        scores = X_bias @ self.W

        correct = scores[np.arange(N), y][:, None]
        margins = np.maximum(0.0, scores - correct + self.delta)
        margins[np.arange(N), y] = 0.0

        loss = np.sum(margins) / N

        mask = (margins > 0).astype(X_bias.dtype)
        mask[np.arange(N), y] = -mask.sum(axis=1)
        dW = (X_bias.T @ mask) / N

        reg_loss, reg_grad = self._l2_penalty()
        return loss + reg_loss, dW + reg_grad


class SoftmaxClassifier(LinearClassifier):

    def loss_gradient(self, X_bias, y):
        N = X_bias.shape[0]
        scores = X_bias @ self.W

        scores = scores - scores.max(axis=1, keepdims=True)

        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        loss = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))

        dscores = probs.copy()
        dscores[np.arange(N), y] -= 1.0
        dscores /= N
        dW = X_bias.T @ dscores

        reg_loss, reg_grad = self._l2_penalty()
        return loss + reg_loss, dW + reg_grad


def load_data(dataset="mnist", n_train=None, n_test=None, seed=0):
    dataset = dataset.lower()
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    cache_path = os.path.join(data_dir, f"{dataset}.npz")

    if os.path.exists(cache_path):
        with np.load(cache_path) as f:
            X_train, y_train = f["x_train"], f["y_train"]
            X_test, y_test = f["x_test"], f["y_test"]
    else:
        from tensorflow.keras.datasets import mnist, cifar10

        loader = {"mnist": mnist, "cifar10": cifar10}[dataset]
        (X_train, y_train), (X_test, y_test) = loader.load_data()

        os.makedirs(data_dir, exist_ok=True)
        np.savez(cache_path, x_train=X_train, y_train=y_train,
                  x_test=X_test, y_test=y_test)

    y_train = y_train.ravel().astype(np.int64)
    y_test = y_test.ravel().astype(np.int64)

    rng = np.random.default_rng(seed)
    if n_train is not None and n_train < len(X_train):
        idx = rng.choice(len(X_train), n_train, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
    if n_test is not None and n_test < len(X_test):
        idx = rng.choice(len(X_test), n_test, replace=False)
        X_test, y_test = X_test[idx], y_test[idx]

    X_train = X_train.reshape(len(X_train), -1).astype(np.float32) / 255.0
    X_test = X_test.reshape(len(X_test), -1).astype(np.float32) / 255.0

    mean_image = X_train.mean(axis=0)
    X_train -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_test, y_test


COLORS = {"SVM": "#1f77b4", "SoftMax": "#d62728"}


def plot_histories(histories, dataset_name, fname=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for name, h in histories.items():
        c = COLORS.get(name, None)
        axes[0].plot(h["epoch"], h["train_loss"], "-", color=c,
                     label=f"{name} train")
        axes[0].plot(h["epoch"], h["val_loss"], "--", color=c,
                     label=f"{name} test")
        axes[1].plot(h["epoch"], h["train_acc"], "-", color=c,
                     label=f"{name} train")
        axes[1].plot(h["epoch"], h["val_acc"], "--", color=c,
                     label=f"{name} test")

    axes[0].set_xlabel("epoca"); axes[0].set_ylabel("loss")
    axes[0].set_title(f"Loss - {dataset_name}")
    axes[1].set_xlabel("epoca"); axes[1].set_ylabel("accuracy")
    axes[1].set_title(f"Accuracy - {dataset_name}")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150)
    return fig


def plot_templates(clf, dataset, class_names=None, fname=None):
    shape = (28, 28) if dataset == "mnist" else (32, 32, 3)
    W = clf.W[:-1]
    K = W.shape[1]

    fig, axes = plt.subplots(2, (K + 1) // 2, figsize=(1.6 * K / 2 + 2, 4))
    for k, ax in enumerate(axes.ravel()[:K]):
        w = W[:, k].reshape(shape)
        w = 255.0 * (w - w.min()) / (w.max() - w.min() + 1e-12)
        ax.imshow(w.astype(np.uint8), cmap="gray" if dataset == "mnist" else None)
        ax.set_title(class_names[k] if class_names else str(k), fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[K:]:
        ax.axis("off")

    fig.suptitle("Plantillas aprendidas (filas de W)")
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150)
    return fig


CIFAR_NAMES = ["avion", "auto", "pajaro", "gato", "ciervo",
               "perro", "rana", "caballo", "barco", "camion"]


def run_experiment(dataset="mnist", n_train=None, n_test=None,
                   epochs=25, batch_size=256, lr=0.1, reg=1e-4,
                   momentum=0.9, show_templates=True):
    print(f"\n{'=' * 70}\n  {dataset.upper()}\n{'=' * 70}")

    X_train, y_train, X_test, y_test = load_data(dataset, n_train, n_test)
    print(f"train: {X_train.shape}   test: {X_test.shape}   "
          f"clases: {len(np.unique(y_train))}\n")

    models = {
        "SVM":     SVMClassifier(delta=1.0, reg=reg, learning_rate=lr,
                                 momentum=momentum, seed=0),
        "SoftMax": SoftmaxClassifier(reg=reg, learning_rate=lr,
                                     momentum=momentum, seed=0),
    }

    histories = {}
    for name, clf in models.items():
        print(f"[{name}]")
        histories[name] = clf.fit(X_train, y_train, X_test, y_test,
                                  epochs=epochs, batch_size=batch_size)
        print()

    print(f"--- Resultados finales ({dataset}) ---")
    for name, clf in models.items():
        print(f"  {name:8s}  train acc = {clf.accuracy(X_train, y_train):.4f}"
              f"   test acc = {clf.accuracy(X_test, y_test):.4f}")

    plot_histories(histories, dataset.upper(), fname=f"curvas_{dataset}.png")
    if show_templates:
        names = CIFAR_NAMES if dataset == "cifar10" else None
        plot_templates(models["SoftMax"], dataset, names, fname=f"plantillas_{dataset}.png")

    return models, histories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="both",
                    choices=["mnist", "cifar10", "both"])
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()

    if args.dataset in ("mnist", "both"):
        run_experiment("mnist", epochs=args.epochs,
                       lr=1e-4, reg=1e-4, batch_size=256)

    if args.dataset in ("cifar10", "both"):
        run_experiment("cifar10", n_train=20000, epochs=args.epochs,
                       lr=1e-4, reg=5e-4, batch_size=256)

    plt.show()



main()

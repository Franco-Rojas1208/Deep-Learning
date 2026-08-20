"""
=============================================================================
Deep Learning - Practica 1 - Ejercicio 3
Clasificadores lineales SVM (hinge loss) y SoftMax (cross-entropy)
con regularizacion L2, implementados con el paradigma de objetos.
=============================================================================

Estructura:

    LinearClassifier            (clase base, implementa toda la maquinaria comun)
        |-- SVMClassifier       (solo redefine loss_gradient -> hinge loss)
        |-- SoftmaxClassifier   (solo redefine loss_gradient -> cross-entropy)

La idea central del ejercicio: *el modelo es el mismo* (scores = X @ W).
Lo unico que distingue a SVM de SoftMax es COMO se mide el error.
Por eso todo lo demas (fit, predict, accuracy, mini-batch, regularizacion)
vive en la clase base y se hereda.

Uso:
    python clasificadores_lineales.py            # corre MNIST y CIFAR-10

Requiere: numpy, matplotlib, tensorflow/keras (solo para bajar los datasets).
=============================================================================
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 1) METRICA
# =============================================================================

def accuracy(y_pred, y_true):
    """Fraccion de aciertos. y_pred e y_true son vectores de enteros (N,)."""
    return float(np.mean(y_pred == y_true))


# =============================================================================
# 2) CLASE BASE
# =============================================================================

class LinearClassifier:
    """
    Clasificador lineal generico:  scores = X_bias @ W

    Convenciones
    ------------
    X        : (N, D)    N ejemplos, D features (imagenes ya aplanadas)
    X_bias   : (N, D+1)  igual que X pero con una columna de unos al final
                         ("bias trick": permite escribir W@x + b como W'@x')
    W        : (D+1, K)  K = numero de clases. La ultima FILA es el bias.
    y        : (N,)      etiquetas enteras en [0, K-1]

    Las subclases SOLO deben implementar loss_gradient().
    """

    def __init__(self, reg=1e-3, learning_rate=1e-2, weight_scale=1e-3,
                 momentum=0.0, seed=0):
        """
        reg          : coeficiente lambda de la regularizacion L2
        learning_rate: paso del descenso por gradiente
        weight_scale : desvio de la inicializacion aleatoria de W.
                       Se elige CHICO a proposito: mantiene los scores
                       cerca de 0, que es lo que pide el enunciado para
                       que exp() no desborde en SoftMax.
        momentum     : 0.0 -> SGD puro (lo que pide el enunciado).
                       0.9 -> SGD con momento, converge bastante mas rapido.
        """
        self.reg = reg
        self.learning_rate = learning_rate
        self.weight_scale = weight_scale
        self.momentum = momentum
        self.seed = seed
        self.W = None
        self.history = None

    # ---------------------------------------------------------------- utils

    @staticmethod
    def _add_bias(X):
        """Agrega la columna de unos (bias trick)."""
        return np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])

    def _init_weights(self, D, K):
        rng = np.random.default_rng(self.seed)
        self.W = (self.weight_scale *
                  rng.standard_normal((D + 1, K))).astype(np.float32)

    def _l2_penalty(self):
        """
        Regularizacion L2:  L_R = lambda * sum(W_ij^2),  dL_R/dW = 2*lambda*W

        Importante: NO se regulariza la fila del bias. El bias solo corre el
        hiperplano, no controla su complejidad; penalizarlo sesga el modelo
        sin aportar nada.
        """
        W_reg = self.W.copy()
        W_reg[-1, :] = 0.0                      # excluye el bias
        loss = self.reg * np.sum(W_reg ** 2)
        grad = 2.0 * self.reg * W_reg
        return loss, grad

    # ------------------------------------------------------------ interfaz

    def loss_gradient(self, X_bias, y):
        """
        Devuelve (loss, dW) para el batch dado, INCLUYENDO regularizacion.
        X_bias ya trae la columna de bias (fit la agrega una sola vez).
        Metodo abstracto: lo define cada subclase.
        """
        raise NotImplementedError("Implementar en la subclase")

    def scores(self, X):
        """Scores sin normalizar, (N, K). Recibe X SIN columna de bias."""
        return self._add_bias(X.astype(np.float32)) @ self.W

    def predict(self, X):
        """Clase predicha = argmax de los scores."""
        return np.argmax(self.scores(X), axis=1)

    def accuracy(self, X, y):
        return accuracy(self.predict(X), y)

    def evaluate(self, X, y):
        """Devuelve (loss, accuracy) sobre un conjunto completo."""
        X_bias = self._add_bias(X.astype(np.float32))
        loss, _ = self.loss_gradient(X_bias, y)
        acc = accuracy(np.argmax(X_bias @ self.W, axis=1), y)
        return loss, acc

    # ---------------------------------------------------------- entrenamiento

    def fit(self, X, y, X_val=None, y_val=None, epochs=20, batch_size=128,
            lr_decay=1.0, n_classes=None, eval_subset=5000, verbose=True):
        """
        Entrenamiento por mini-batch gradient descent.

        Una EPOCA = una pasada completa por el set de entrenamiento,
        recorrido en batches de tamano batch_size en orden aleatorio.

        batch_size = N  -> Batch Gradient Descent (BGD) clasico
        batch_size = 1  -> SGD puro
        1 < bs < N      -> mini-batch (el que se usa en la practica)

        Devuelve un diccionario `history` con las curvas por epoca.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.int64).ravel()
        N, D = X.shape
        K = n_classes if n_classes is not None else int(y.max()) + 1

        self._init_weights(D, K)
        velocity = np.zeros_like(self.W)

        X_bias = self._add_bias(X)              # bias trick UNA sola vez
        rng = np.random.default_rng(self.seed)
        lr = self.learning_rate

        self.history = {"train_loss": [], "train_acc": [],
                        "val_loss": [], "val_acc": [], "epoch": []}

        # subconjunto fijo para medir el train loss/acc sin pagar el costo
        # de evaluar sobre todo el set en cada epoca
        idx_eval = rng.choice(N, size=min(eval_subset, N), replace=False)

        for epoch in range(1, epochs + 1):
            perm = rng.permutation(N)           # barajar en cada epoca

            for start in range(0, N, batch_size):
                batch = perm[start:start + batch_size]
                _, dW = self.loss_gradient(X_bias[batch], y[batch])

                # --- regla de actualizacion ---
                if self.momentum:
                    velocity = self.momentum * velocity - lr * dW
                    self.W += velocity
                else:
                    self.W -= lr * dW

            # ------------------ metricas de fin de epoca ------------------
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

            lr *= lr_decay                      # decaimiento opcional del lr

        return self.history

    def _eval_bias(self, X_bias, y):
        loss, _ = self.loss_gradient(X_bias, y)
        return loss, accuracy(np.argmax(X_bias @ self.W, axis=1), y)


# =============================================================================
# 3) SUBCLASE: SVM MULTICLASE (HINGE LOSS)
# =============================================================================

class SVMClassifier(LinearClassifier):
    r"""
    SVM multiclase (formulacion Weston-Watkins).

        L_i = sum_{j != y_i} max(0, s_j - s_{y_i} + delta)

    Interpretacion: se exige que el score de la clase correcta le gane a
    cada score incorrecto por al menos un margen delta. Si eso ya se cumple,
    la perdida de ese termino es EXACTAMENTE 0 y no aporta gradiente.
    """

    def __init__(self, delta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta

    def loss_gradient(self, X_bias, y):
        N = X_bias.shape[0]
        scores = X_bias @ self.W                       # (N, K)

        correct = scores[np.arange(N), y][:, None]     # (N, 1)
        margins = np.maximum(0.0, scores - correct + self.delta)
        margins[np.arange(N), y] = 0.0                 # j = y_i no cuenta

        loss = np.sum(margins) / N

        # --- gradiente ---
        # Para j != y_i:  dL/ds_j    = 1 si el margen esta violado, 0 si no
        # Para j == y_i:  dL/ds_{y_i} = -(cantidad de margenes violados)
        mask = (margins > 0).astype(X_bias.dtype)      # (N, K)
        mask[np.arange(N), y] = -mask.sum(axis=1)
        dW = (X_bias.T @ mask) / N                     # regla de la cadena

        reg_loss, reg_grad = self._l2_penalty()
        return loss + reg_loss, dW + reg_grad


# =============================================================================
# 4) SUBCLASE: SOFTMAX (CROSS-ENTROPY)
# =============================================================================

class SoftmaxClassifier(LinearClassifier):
    r"""
    Clasificador SoftMax (regresion logistica multinomial).

        p_i = exp(z_i) / sum_j exp(z_j)
        L_i = -log(p_{y_i}) = -z_{y_i} + log(sum_j exp(z_j))

    Minimizar esto equivale a minimizar la entropia cruzada H(p, q) entre la
    distribucion verdadera (one-hot) y la estimada, o a maximizar la
    verosimilitud de los datos.
    """

    def loss_gradient(self, X_bias, y):
        N = X_bias.shape[0]
        scores = X_bias @ self.W                       # (N, K)

        # --- ESTABILIDAD NUMERICA ---
        # exp(z) desborda para z grande. Restar el maximo por fila NO cambia
        # las probabilidades (numerador y denominador se dividen por el mismo
        # factor) pero garantiza que el mayor exponente sea exp(0) = 1.
        scores = scores - scores.max(axis=1, keepdims=True)

        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)   # (N, K)

        loss = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))

        # --- gradiente ---
        # dL/dz = p - y_onehot   (probabilidad predicha menos la verdadera)
        dscores = probs.copy()
        dscores[np.arange(N), y] -= 1.0
        dscores /= N
        dW = X_bias.T @ dscores

        reg_loss, reg_grad = self._l2_penalty()
        return loss + reg_loss, dW + reg_grad


# =============================================================================
# 5) CARGA Y PREPROCESADO DE DATOS
# =============================================================================

def load_data(dataset="mnist", n_train=None, n_test=None, seed=0):
    """
    Carga MNIST o CIFAR-10 con keras y los deja listos para un modelo lineal:
      1) aplana cada imagen a un vector
      2) escala a [0, 1]
      3) resta la imagen media del set de entrenamiento (centrado)

    El centrado importa: sin el, todos los pixeles son positivos, todas las
    componentes del gradiente de una fila de W tienen el mismo signo y el
    descenso zigzaguea.

    Los datos crudos se cachean en ./data/<dataset>.npz: si ya estan
    descargados se leen de ahi directamente, sin volver a bajarlos.
    """
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

    # submuestreo opcional (util para iterar rapido con CIFAR-10)
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


# =============================================================================
# 6) GRAFICOS
# =============================================================================

COLORS = {"SVM": "#1f77b4", "SoftMax": "#d62728"}


def plot_histories(histories, dataset_name, fname=None):
    """Curvas de loss y accuracy por epoca (train solido, test punteado)."""
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
    """
    Visualiza cada fila de W como una imagen: son las 'plantillas' que el
    clasificador lineal aprendio para cada clase. En CIFAR-10 se reconoce
    la silueta del avion sobre fondo azul, el caballo con dos cabezas, etc.
    """
    shape = (28, 28) if dataset == "mnist" else (32, 32, 3)
    W = clf.W[:-1]                                    # sin la fila del bias
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


# =============================================================================
# 7) EXPERIMENTO PRINCIPAL
# =============================================================================

CIFAR_NAMES = ["avion", "auto", "pajaro", "gato", "ciervo",
               "perro", "rana", "caballo", "barco", "camion"]

#0.1 en lr

def run_experiment(dataset="mnist", n_train=None, n_test=None,
                   epochs=25, batch_size=256, lr=0.1, reg=1e-4,
                   momentum=0.9, show_templates=True):
    """Entrena SVM y SoftMax sobre el mismo dataset y los compara."""
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
        # CIFAR-10 es mucho mas dificil para un modelo lineal:
        # se usa lr mas chico y mas regularizacion.
        run_experiment("cifar10", n_train=20000, epochs=args.epochs,
                       lr=1e-4, reg=5e-4, batch_size=256) #0.02

    plt.show()


if __name__ == "__main__":
    main()
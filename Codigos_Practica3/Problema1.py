# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import layers


SEED = 0

EPOCHS = 15
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1                           # 10% del train para monitorear
                                                 # el ajuste; el test queda intacto

keras.utils.set_random_seed(SEED)                # semillas de python, numpy y backend

print("backend:", keras.backend.backend(),
      "| GPUs:", keras.distribution.list_devices("gpu"))


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

input_shape = x_train.shape[1:]                  # (28, 28, 1)
n_clases = int(y_train.max()) + 1                # 10 digitos

print(f"train: {x_train.shape}   test: {x_test.shape}")
print(f"rango de x: [{x_train.min():.1f}, {x_train.max():.1f}]   "
      f"input_shape={input_shape}   n_clases={n_clases}")


n_filas, n_cols = 3, 6
fig, axes = plt.subplots(n_filas, n_cols, figsize=(1.5 * n_cols, 1.7 * n_filas))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_train[i].squeeze(), cmap="gray")  # squeeze() saca el eje de canal
    ax.set_title(f"y = {int(y_train[i])}", fontsize=10)
    ax.axis("off")
fig.suptitle("MNIST - imagenes y sus etiquetas")
fig.tight_layout()
plt.show()


Y_train = keras.utils.to_categorical(y_train, num_classes=n_clases)
Y_test = keras.utils.to_categorical(y_test, num_classes=n_clases)


model = keras.Sequential(
    [
        keras.Input(shape=input_shape),

        # bloque 1: 28x28 -> 14x14
        layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
        layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=2),

        # bloque 2: 14x14 -> 7x7
        layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
        layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=2),

        # bloque 3: sin pooling, solo mas canales
        layers.Conv2D(128, kernel_size=3, padding="same", activation="relu"),

        # cabeza: (7,7,128) -> (128,) -> (10,)
        layers.GlobalAveragePooling2D(),
        layers.Dense(n_clases, activation="softmax"),
    ],
    name="cnn_mnist",
)

model.summary()

model.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
)

hist = model.fit(
    x_train, Y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    verbose=2,
)


loss_test, acc_test = model.evaluate(x_test, Y_test, verbose=0)
print(f"\nperdida test  = {loss_test:.4f}")
print(f"accuracy test = {acc_test:.4f}  ({100 * acc_test:.2f} %)")


epocas = np.arange(1, len(hist.history["loss"]) + 1)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].plot(epocas, hist.history["loss"], "o-", label="train")
ax[0].plot(epocas, hist.history["val_loss"], "o-", label="validacion")
ax[0].set_xlabel("epoca"); ax[0].set_ylabel("perdida (cross-entropy)")
ax[0].set_title("Perdida"); ax[0].legend(); ax[0].grid(alpha=0.3)

ax[1].plot(epocas, 100 * np.array(hist.history["acc"]), "o-", label="train")
ax[1].plot(epocas, 100 * np.array(hist.history["val_acc"]), "o-", label="validacion")
ax[1].set_xlabel("epoca"); ax[1].set_ylabel("accuracy [%]")
ax[1].set_title("Precision"); ax[1].legend(); ax[1].grid(alpha=0.3)

fig.suptitle("Ejercicio 1 - CNN sobre MNIST (Keras)")
fig.tight_layout()
plt.show()


y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
errores = np.flatnonzero(y_pred != y_test)
print(f"errores: {len(errores)} sobre {len(y_test)}")

n_filas, n_cols = 2, 6
fig, axes = plt.subplots(n_filas, n_cols, figsize=(1.6 * n_cols, 1.9 * n_filas))
for ax in axes.flat:
    ax.axis("off")                               # por si hubiera menos de 12 errores
for ax, i in zip(axes.flat, errores):
    ax.imshow(x_test[i].squeeze(), cmap="gray")
    ax.set_title(f"pred {y_pred[i]} / real {y_test[i]}", fontsize=9)
fig.suptitle(f"Ejercicio 1 - Errores en test ({len(errores)} de {len(y_test)})")
fig.tight_layout()
plt.show()

# -*- coding: utf-8 -*-

import os
import glob
import tarfile
import urllib.request

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import layers


DATA_DIR = "/content/oxford-iiit-pet"
IMG_DIR = os.path.join(DATA_DIR, "images")
MASK_DIR = os.path.join(DATA_DIR, "annotations", "trimaps")

SEED = 0

IMG_SIZE = 128
N_CLASES = 3
BATCH_SIZE = 32
EPOCHS = 20
VAL_FRAC = 0.2

keras.utils.set_random_seed(SEED)

print("GPU:", tf.config.list_physical_devices("GPU"))


if not (os.path.isdir(IMG_DIR) and os.path.isdir(MASK_DIR)):
    os.makedirs(DATA_DIR, exist_ok=True)
    for url in ["https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz",
                "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz"]:
        tgz = os.path.join(DATA_DIR, url.split("/")[-1])
        if not os.path.exists(tgz):
            print(f"descargando {url} ...")
            urllib.request.urlretrieve(url, tgz)
        print(f"descomprimiendo {tgz} ...")
        with tarfile.open(tgz) as tar:
            tar.extractall(DATA_DIR)


pares = []
for f_img in sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg"))):
    base = os.path.basename(f_img)
    if base.startswith("._"):                    # basura del tar de macOS
        continue
    f_mask = os.path.join(MASK_DIR, os.path.splitext(base)[0] + ".png")
    if os.path.exists(f_mask):
        pares.append((f_img, f_mask))

rng = np.random.default_rng(SEED)
rng.shuffle(pares)                               # la lista viene ordenada por raza

n_val = int(VAL_FRAC * len(pares))
pares_val, pares_train = pares[:n_val], pares[n_val:]

print(f"pares (imagen, trimap): {len(pares)}")
print(f"train = {len(pares_train)}   validacion = {len(pares_val)}")


def cargar_par(ruta_img, ruta_mask):
    img = tf.io.decode_image(tf.io.read_file(ruta_img), channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))

    mask = tf.io.decode_png(tf.io.read_file(ruta_mask), channels=1)
    mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE), method="nearest")
    mask = tf.cast(mask, tf.int32) - 1

    return img, mask


def crear_dataset(pares, entrenamiento):
    rutas_img, rutas_mask = zip(*pares)
    ds = tf.data.Dataset.from_tensor_slices((list(rutas_img), list(rutas_mask)))
    if entrenamiento:
        ds = ds.shuffle(len(pares), seed=SEED)
    ds = ds.map(cargar_par, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


ds_train = crear_dataset(pares_train, entrenamiento=True)
ds_val = crear_dataset(pares_val, entrenamiento=False)


img, mask = cargar_par(*pares_train[0])
print(f"\nimagen {img.shape} en [{np.min(img):.2f}, {np.max(img):.2f}]"
      f"   trimap {mask.shape} con valores {np.unique(mask)}")

fig, ax = plt.subplots(1, 2, figsize=(7, 3.5))
ax[0].imshow(img); ax[0].set_title("imagen")
ax[1].imshow(mask[..., 0], cmap="viridis")
ax[1].set_title("trimap  (0=mascota, 1=fondo, 2=borde)")
for a in ax:
    a.axis("off")
fig.tight_layout()
plt.show()


def bloque_conv(x, filtros):
 
    for _ in range(2):
        x = layers.Conv2D(filtros, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    return x


entrada = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

c1 = bloque_conv(entrada, 32)                    # 128x128x 32
p1 = layers.MaxPooling2D(2)(c1)                  #  64x 64x 32
c2 = bloque_conv(p1, 64)                         #  64x 64x 64
p2 = layers.MaxPooling2D(2)(c2)                  #  32x 32x 64
c3 = bloque_conv(p2, 128)                        #  32x 32x128
p3 = layers.MaxPooling2D(2)(c3)                  #  16x 16x128
c4 = bloque_conv(p3, 256)                        #  16x 16x256
p4 = layers.MaxPooling2D(2)(c4)                  #   8x  8x256

cuello = bloque_conv(p4, 512)                    #   8x  8x512

u4 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(cuello)
u4 = layers.Concatenate()([u4, c4])              #  16x 16x512
u4 = bloque_conv(u4, 256)                        #  16x 16x256

u3 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(u4)
u3 = layers.Concatenate()([u3, c3])              #  32x 32x256
u3 = bloque_conv(u3, 128)                        #  32x 32x128

u2 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(u3)
u2 = layers.Concatenate()([u2, c2])              #  64x 64x128
u2 = bloque_conv(u2, 64)                         #  64x 64x 64

u1 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(u2)
u1 = layers.Concatenate()([u1, c1])              # 128x128x 64
u1 = bloque_conv(u1, 32)                         # 128x128x 32

# convolucion 1x1: un clasificador de 3 clases aplicado a cada pixel
salida = layers.Conv2D(N_CLASES, 1, activation="softmax")(u1)

model = keras.Model(entrada, salida, name="unet")
model.summary()

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

hist = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=EPOCHS,
    shuffle=False,                               # ya baraja el tf.data
    verbose=2,
)


loss_val, acc_val = model.evaluate(ds_val, verbose=0)
print(f"\nperdida validacion  = {loss_val:.4f}")
print(f"accuracy validacion = {acc_val:.4f}  ({100 * acc_val:.2f} % de pixeles bien)")


epocas = np.arange(1, len(hist.history["loss"]) + 1)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].plot(epocas, hist.history["loss"], "o-", label="train")
ax[0].plot(epocas, hist.history["val_loss"], "o-", label="validacion")
ax[0].set_xlabel("epoca"); ax[0].set_ylabel("perdida")
ax[0].set_title("Perdida"); ax[0].legend(); ax[0].grid(alpha=0.3)

ax[1].plot(epocas, 100 * np.array(hist.history["acc"]), "o-", label="train")
ax[1].plot(epocas, 100 * np.array(hist.history["val_acc"]), "o-", label="validacion")
ax[1].set_xlabel("epoca"); ax[1].set_ylabel("accuracy [%]")
ax[1].set_title("Precision por pixel"); ax[1].legend(); ax[1].grid(alpha=0.3)

fig.suptitle("Ejercicio 3 - U-net sobre oxford-iiit-pet")
fig.tight_layout()
plt.show()


imgs, masks = next(iter(ds_val))
n = 4
pred = np.argmax(model.predict(imgs[:n], verbose=0), axis=-1)

fig, ax = plt.subplots(n, 3, figsize=(8, 2.6 * n))
for i in range(n):
    ax[i, 0].imshow(imgs[i])
    ax[i, 1].imshow(masks[i][..., 0], cmap="viridis", vmin=0, vmax=2)
    ax[i, 2].imshow(pred[i], cmap="viridis", vmin=0, vmax=2)
    for a in ax[i]:
        a.axis("off")
ax[0, 0].set_title("imagen")
ax[0, 1].set_title("trimap verdadero")
ax[0, 2].set_title("prediccion")
fig.suptitle("Ejercicio 3 - segmentacion sobre datos de validacion")
fig.tight_layout()
plt.show()

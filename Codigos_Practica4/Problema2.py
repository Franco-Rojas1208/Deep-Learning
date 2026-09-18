

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(0)
np.random.seed(0)


LATENT_DIM = 10        
BATCH_SIZE = 128
EPOCHS     = 15
LR         = 1e-3
LOSS_TYPE  = 'mse'    # 'bce' (cross-entropy) o 'mse'


(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32')  / 255.0
x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)

print('train:', x_train.shape, ' validacion:', x_test.shape)

train_ds = (tf.data.Dataset.from_tensor_slices(x_train)
            .shuffle(60000).batch(BATCH_SIZE))
test_ds  = tf.data.Dataset.from_tensor_slices(x_test).batch(BATCH_SIZE)

encoder = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, strides=2, padding='same', activation='relu'),  # 14x14
    layers.Conv2D(64, 3, strides=2, padding='same', activation='relu'),  # 7x7
    layers.Flatten(),
    layers.Dense(2 * LATENT_DIM),  
], name='encoder')
encoder.summary()


decoder = keras.Sequential([
    keras.Input(shape=(LATENT_DIM,)),
    layers.Dense(7 * 7 * 64, activation='relu'),
    layers.Reshape((7, 7, 64)),
    layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu'),  # 14x14
    layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu'),  # 28x28
    layers.Conv2DTranspose(1, 3, padding='same'), 
], name='decoder')
decoder.summary()


def reparametrizar(mu, logvar):
    eps = tf.random.normal(shape=tf.shape(mu))
    return mu + tf.exp(0.5 * logvar) * eps

def calcular_loss(x):
    salida = encoder(x)
    mu, logvar = tf.split(salida, num_or_size_splits=2, axis=1)

    z = reparametrizar(mu, logvar)
    x_logits = decoder(z)

    if LOSS_TYPE == 'bce':
       
        ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_logits)
        recon = tf.reduce_sum(ce, axis=[1, 2, 3])
    else:
        
        x_rec = tf.sigmoid(x_logits)
        recon = tf.reduce_sum(tf.square(x - x_rec), axis=[1, 2, 3])

   
    kl = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)

    return tf.reduce_mean(recon + kl), tf.reduce_mean(recon), tf.reduce_mean(kl)

optimizador = keras.optimizers.Adam(LR)

@tf.function
def paso_entrenamiento(x):
    with tf.GradientTape() as tape:
        loss, recon, kl = calcular_loss(x)
    variables = encoder.trainable_variables + decoder.trainable_variables
    grads = tape.gradient(loss, variables)
    optimizador.apply_gradients(zip(grads, variables))
    return loss, recon, kl

hist_train, hist_val = [], []

for epoca in range(1, EPOCHS + 1):
    # entrenamiento
    acum = []
    for x in train_ds:
        loss, recon, kl = paso_entrenamiento(x)
        acum.append([loss.numpy(), recon.numpy(), kl.numpy()])
    l_tr, r_tr, k_tr = np.mean(acum, axis=0)

    # validacion
    acum = []
    for x in test_ds:
        loss, recon, kl = calcular_loss(x)
        acum.append(loss.numpy())
    l_va = np.mean(acum)

    hist_train.append(l_tr)
    hist_val.append(l_va)
    print(f'epoca {epoca:2d}  loss_train {l_tr:8.2f}  (recon {r_tr:7.2f}, KL {k_tr:6.2f})'
          f'  loss_val {l_va:8.2f}')


plt.figure(figsize=(5, 4))
plt.plot(hist_train, label='entrenamiento')
plt.plot(hist_val, label='validacion')
plt.xlabel('epoca'); plt.ylabel('-ELBO'); plt.legend(); plt.title('Funcion de costo')
plt.show()


x = x_test[:10]
mu, logvar = tf.split(encoder(x), 2, axis=1)
x_rec = tf.sigmoid(decoder(reparametrizar(mu, logvar))).numpy()

plt.figure(figsize=(10, 2.5))
for i in range(10):
    plt.subplot(2, 10, i + 1);      plt.imshow(x[i, :, :, 0], cmap='gray');     plt.axis('off')
    plt.subplot(2, 10, i + 11);     plt.imshow(x_rec[i, :, :, 0], cmap='gray'); plt.axis('off')
plt.suptitle('Arriba: originales   Abajo: reconstrucciones')
plt.show()


z = tf.random.normal(shape=(16, LATENT_DIM))
imgs = tf.sigmoid(decoder(z)).numpy()
plt.figure(figsize=(6, 6))
for i in range(16):
    plt.subplot(4, 4, i + 1); plt.imshow(imgs[i, :, :, 0], cmap='gray'); plt.axis('off')
plt.suptitle('Digitos generados desde z ~ N(0,I)')
plt.show()

if LATENT_DIM == 2:
    n = 15
    grilla = np.linspace(-2.5, 2.5, n)
    lienzo = np.zeros((28 * n, 28 * n))
    for i, zy in enumerate(grilla[::-1]):
        for j, zx in enumerate(grilla):
            z = np.array([[zx, zy]], dtype='float32')
            img = tf.sigmoid(decoder(z)).numpy()[0, :, :, 0]
            lienzo[i*28:(i+1)*28, j*28:(j+1)*28] = img
    plt.figure(figsize=(8, 8))
    plt.imshow(lienzo, cmap='gray'); plt.axis('off')
    plt.title('Espacio latente recorrido en grilla')
    plt.show()


    (_, y_test) = keras.datasets.mnist.load_data()[1]
    mu_test, _ = tf.split(encoder(x_test[:5000]), 2, axis=1)
    plt.figure(figsize=(6, 5))
    plt.scatter(mu_test[:, 0], mu_test[:, 1], c=y_test[:5000], cmap='tab10', s=4)
    plt.colorbar(); plt.xlabel('z1'); plt.ylabel('z2'); plt.title('mu(x) coloreado por digito')
    plt.show()
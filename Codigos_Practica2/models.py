# -*- coding: utf-8 -*-

import numpy as np
import metric


class Network:

    def __init__(self, loss, optimizer):
        self.layers = []
        self.loss = loss
        self.optimizer = optimizer

    def add(self, layer):
        self.layers.append(layer)
        return self

    def forward(self, x):
        a = x
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def backward(self, y_pred, y_true):
        grad = self.loss.gradient(y_pred, y_true)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def fit(self, X, Y, epochs, batch_size=None, seed=0, verbose=False):
        rng = np.random.default_rng(seed)
        N = X.shape[0]
        batch_size = N if batch_size is None else batch_size #Esto en realidad es para los próximos ejercicios

        hist = {"loss": [], "accuracy": []}
        for ep in range(epochs):
            perm = rng.permutation(N)
            for i in range(0, N, batch_size):
                b = perm[i:i + batch_size]
                y_pred = self.forward(X[b])
                self.backward(y_pred, Y[b])
                self.optimizer.step(self.layers)

            y_pred_full = self.forward(X)
            hist["loss"].append(self.loss(y_pred_full, Y))
            hist["accuracy"].append(metric.accuracy(y_pred_full, Y))

            if verbose and (ep == 0 or (ep + 1) % max(1, epochs // 10) == 0 or ep == epochs - 1):
                print(f"  epoca {ep + 1:4d}/{epochs}  "
                      f"loss={hist['loss'][-1]:.4f}  acc={hist['accuracy'][-1]:.3f}")

        return hist

    def predict(self, X):
        return self.forward(X)

# -*- coding: utf-8 -*-

import numpy as np


class Optimizador:

    def step(self, layers):
        raise NotImplementedError


class SGD(Optimizador):

    def __init__(self, lr=0.1, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self._velocity = {}

    def step(self, layers):
        for layer in layers:
            grads = getattr(layer, "grads", None)
            if not grads:
                continue                          # capas sin parametros (Input)

            params = layer.get_params()
            if id(layer) not in self._velocity:
                self._velocity[id(layer)] = {k: np.zeros_like(p) for k, p in params.items()}
            v = self._velocity[id(layer)]

            for k in params:
                v[k] = self.momentum * v[k] - self.lr * grads[k]
                params[k] = params[k] + v[k]
            layer.set_params(params)

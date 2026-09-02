# -*- coding: utf-8 -*-

import numpy as np


class BaseLayer:

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError


class Input(BaseLayer):

    def forward(self, x):
        self.output = x
        return x

    def backward(self, grad_output):
        return grad_output


class Layer(BaseLayer):

    def __init__(self, n_inputs, n_units, activation):
        self.n_inputs = n_inputs
        self.n_units = n_units
        self.activation = activation
        self.grads = {}

    def get_params(self):
        raise NotImplementedError

    def set_params(self, params):
        raise NotImplementedError


class Dense(Layer):

    def __init__(self, n_inputs, n_units, activation, seed=None, l2=0.0):
        super().__init__(n_inputs, n_units, activation)
        rng = np.random.default_rng(seed)
        limit = np.sqrt(1.0 / n_inputs)
        self.W = rng.uniform(-limit, limit, size=(n_inputs, n_units))
        self.b = np.zeros(n_units)
        self.l2 = l2

    def forward(self, x):
        self.x = x                                # (N, n_inputs), cache para el backward
        self.z = x @ self.W + self.b               # (N, n_units)
        self.a = self.activation(self.z)           # (N, n_units)
        return self.a

    def backward(self, grad_a):
        grad_z = grad_a * self.activation.gradient(self.z)   # (N, n_units)
        self.grads["W"] = self.x.T @ grad_z + 2.0 * self.l2 * self.W   # (n_inputs, n_units)
        self.grads["b"] = grad_z.sum(axis=0)                 # (n_units,)
        grad_x = grad_z @ self.W.T                           # (N, n_inputs)
        return grad_x

    def reg_loss(self):
        return self.l2 * np.sum(self.W ** 2)

    def get_params(self):
        return {"W": self.W, "b": self.b}

    def set_params(self, params):
        self.W = params["W"]
        self.b = params["b"]

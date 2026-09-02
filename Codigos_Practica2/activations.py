# -*- coding: utf-8 -*-

import numpy as np


class Activation:

    def __call__(self, z):
        raise NotImplementedError

    def gradient(self, z):
        raise NotImplementedError


class Sigmoid(Activation):

    def __call__(self, z):
        out = np.empty_like(z, dtype=float)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    def gradient(self, z):
        s = self(z)
        return s * (1.0 - s)


class Tanh(Activation):

    def __call__(self, z):
        return np.tanh(z)

    def gradient(self, z):
        t = np.tanh(z)
        return 1.0 - t ** 2


class ReLU(Activation):

    def __call__(self, z):
        return np.maximum(0.0, z)

    def gradient(self, z):
        return (z > 0).astype(z.dtype)


class Linear(Activation):

    def __call__(self, z):
        return z

    def gradient(self, z):
        return np.ones_like(z)

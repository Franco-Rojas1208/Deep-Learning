# -*- coding: utf-8 -*-

import numpy as np


class Loss:

    def __call__(self, y_pred, y_true):
        raise NotImplementedError

    def gradient(self, y_pred, y_true):
        raise NotImplementedError


class MSE(Loss):

    def __call__(self, y_pred, y_true):
        return float(np.mean(np.sum((y_pred - y_true) ** 2, axis=1)))

    def gradient(self, y_pred, y_true):
        N = y_pred.shape[0]
        return (2.0 / N) * (y_pred - y_true)

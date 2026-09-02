# -*- coding: utf-8 -*-

import numpy as np


def MSE(y_pred, y_true):
    return float(np.mean(np.sum((y_pred - y_true) ** 2, axis=1)))


def accuracy(y_pred, y_true, threshold=0.0):
    if y_pred.shape[1] == 1:
        pred = np.where(y_pred >= threshold, 1.0, -1.0)
        return float(np.mean(pred == y_true))
    pred = np.argmax(y_pred, axis=1)
    true = np.argmax(y_true, axis=1)
    return float(np.mean(pred == true))

    #Esta condición es para poder resolver el ejercicio 3 además del 6.

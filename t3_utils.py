import numpy as np


def target_transform(y, mu=1):
    return np.log(y + mu)


def target_inverse_transform(y_tr, mu=1):
    return np.exp(y_tr) - mu


def rmsle(y_true, y_pred):
    return np.sqrt(
        np.sum(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)) /
        y_true.size)


def xgb_eval(y_pred_tr, dmat):
    y_pred = target_inverse_transform(y_pred_tr)
    y_true = dmat.get_label()
    return 'rmsle', rmsle(y_true, y_pred)

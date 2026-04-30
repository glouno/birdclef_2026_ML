import numpy as np


def compute_pos_neg_weights(y, beta=0.5):
    classes = np.unique(y)
    n = len(y)

    def calc_weights_class(class_id):
        n_pos = (y == class_id).sum()
        n_neg = max(1, n - n_pos)

        w_pos = (n / n_pos) ** beta
        w_neg = (n / n_neg) ** beta

        mean = (w_pos + w_neg) / 2  # normalize for SGD stability
        return w_pos / mean, w_neg / mean

    weights = {c: calc_weights_class(c) for c in classes}
    return weights


def compute_pos_neg_weights_scopes(y, y_scope):
    scope_values = np.unique(y_scope)
    return {scope_value: compute_pos_neg_weights(y[y_scope == scope_value])
            for scope_value in scope_values}


def compute_pos_neg_sample_weights(y_bin, pos_neg_weights):
    w_pos, w_neg = pos_neg_weights
    return np.where(
        y_bin == 1,
        w_pos,
        w_neg,
    )


def compute_soft_oversampling_weights(y, beta=0.5):
    classes, counts = np.unique(y, return_counts=True)
    n = np.sum(counts)

    return dict(zip(classes, np.power(n / counts, 1-beta)))

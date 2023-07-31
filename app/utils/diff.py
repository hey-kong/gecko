import numpy as np


def hellinger_distance_multi(X, Y):
    if len(X) != len(Y):
        raise ValueError("X and Y must be the same-dimension vectors")

    distances = []
    for f in range(len(X)):
        X_feature = X[f]
        Y_feature = Y[f]

        min_val = min(min(X_feature), min(Y_feature))
        X_shifted = X_feature - min_val + 1e-9
        Y_shifted = Y_feature - min_val + 1e-9

        X_normalized = X_shifted / np.sum(X_shifted)
        Y_normalized = Y_shifted / np.sum(Y_shifted)

        distance = np.sqrt(np.sum((np.sqrt(X_normalized) - np.sqrt(Y_normalized)) ** 2))
        distances.append(distance)

    d_H = sum(distances) / len(X)
    return d_H

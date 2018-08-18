import os
import pandas as pd
import pickle as pk
import numpy as np

from sklearn.preprocessing import normalize

targets = ['In-hospital_death', 'Out-hospital_death_30', 'Out-hospital_death', 'Long_stay']


def load_icus(data_path, data_id, icu_list, should_normalize=True):
    icus = []

    if should_normalize:
        X, y = unpickle_and_normalize(data_path, data_id, icu_list[0])
    else:
        X, y = pk.load(open(data_path + "v2_%s/icu_%d.pk" % (data_id, icu_list[0]), "rb"))

    icus += ([icu_list[0]] * X.shape[0])

    for icu in icu_list[1:]:
        if should_normalize:
            X_icu, y_icu = unpickle_and_normalize(data_path, data_id, icu)
        else:
            X_icu, y_icu = pk.load(open(data_path + "v2_%s/icu_%d.pk" % (data_id, icu), "rb"))
        
        X = np.vstack((X, X_icu))
        y = np.concatenate((y, y_icu))

        icus += ([icu] * X_icu.shape[0])

    return X, y, icus


def unpickle_and_normalize(data_path, data_id, target_icu):
    X, y = pk.load(open(data_path + "%s/icu_%d.pk" % (data_id, target_icu), "rb"))
    # X, y = pk.load(open(data_path + "v2_%s/icu_%d_itp.pk" % (data_id, target_icu), "rb"))

    x_norm = np.zeros(X.shape)
    for ts in range(X.shape[1]):
        x_norm[:, ts, :] = normalize(X[:, ts, :], axis=0)
    X = x_norm

    return X, y


def subsampling(x, y, n_samples):
    one_ratio = len(y[y == 1]) / len(y)

    one_samples = int(n_samples * one_ratio)
    zero_samples = n_samples - one_samples

    random_idx_ones = list(range(y[y == 1].shape[0]))
    np.random.shuffle(random_idx_ones)
    x_ones = x[y == 1][random_idx_ones][:one_samples]
    y_ones = y[y == 1][random_idx_ones][:one_samples]

    random_idx_ones = list(range(y[y == 0].shape[0]))
    np.random.shuffle(random_idx_ones)
    x_zeroes = x[y == 0][random_idx_ones][:zero_samples]
    y_zeroes = y[y == 0][random_idx_ones][:zero_samples]

    xs = np.concatenate((x_ones, x_zeroes))
    ys = np.concatenate((y_ones, y_zeroes))

    return xs, ys


def oversample(x, y, ids):
    unq, unq_idx = np.unique(y, return_inverse=True)
    unq_cnt = np.bincount(unq_idx)
    cnt = np.max(unq_cnt)
    out = np.empty((cnt * len(unq),) + x.shape[1:], x.dtype)
    out_y = np.empty((cnt * len(unq), 1), y.dtype)
    out_ids = np.empty((cnt * len(unq), 1), ids.dtype)
    for j in range(len(unq)):
        if np.count_nonzero(unq_idx == j) == cnt:
            indices = unq_idx == j
        else:
            indices = np.random.choice(np.where(unq_idx == j)[0], cnt)
        out[j * cnt:(j + 1) * cnt] = x[indices]
        out_y[j * cnt:(j + 1) * cnt] = y[indices].reshape(y[indices].shape[0], 1)
        out_ids[j * cnt:(j + 1) * cnt] = ids[indices].reshape(ids[indices].shape[0], 1)
    return out, out_y, out_ids

def main():
    data_path = "../data/"
    load_icus(data_path, "60", [1])


if __name__ == '__main__':
    main()

import prepare_data as data
import models
import numpy as np
from datetime import datetime

# from keras import losses
from sklearn.model_selection import StratifiedKFold, train_test_split
from keras.preprocessing import sequence
from sklearn.preprocessing import normalize

n_epochs = 100
k_fold = 5

def today():
    return datetime.now().strftime('%Y-%m-%d')

def main():
    n_features = 46
    model_id = "m_conv_lstm_switch"

    with open("logs/log-permutation-" + today() + ".log", "w") as output:
        for n_perm in range(35):
            feature_sequence = np.random.permutation(list(range(n_features)))
            output.write("Permutation: %s" % feature_sequence + "\n")
            output.write("Fold,All,Coronary,Cardiac,Medical,Surgical\n")
            mixed(model_id, "60", output, 1500, feature_sequence)


def mixed(model_id, data_id, output, n_patients, feature_sequence):
    folds = load_data(data_id, [1, 2, 3, 4], n_patients, feature_sequence)
    for fold, (x_train, y_train, x_val, y_val, x_test, y_test, icu_ids_train, icu_ids_val, icu_ids_test) in enumerate(folds):
        layers, freezable = models.create_freezable_layers(model_id, x_train.shape[1], x_train.shape[2])
        checkpoint, early_stopping, model = models.create_model("mt-" + model_id + "-" + data_id, layers)

        model.fit(x_train, y_train,
                  epochs=n_epochs,
                  validation_data=(x_val, y_val),
                  callbacks=[early_stopping, checkpoint])

        score, auc_score = models.evaluate_model("mt-" + model_id + "-" + data_id, x_test, y_test)
        print("ALL " + "," + str(auc_score))
        output.write(str(fold) + "," + str(auc_score))

        for test_icu in np.unique(icu_ids_test):
            score, auc_score = models.evaluate_model("mt-" + model_id + "-" + data_id,
                                                     x_test[icu_ids_test == test_icu],
                                                     y_test[icu_ids_test == test_icu])
            print("ICU " + str(test_icu) + "," + str(auc_score))
            output.write("," + str(auc_score))
        output.write("\n")


def load_data(data_id, icus, n_patients, feature_sequence):
    x, y, icu_ids = data.load_icus(data_id, icus, data.targets[0])

    if data_id == 'X':
        x = sequence.pad_sequences(x, padding='pre', dtype='float64', maxlen=200)
    else:
        x = sequence.pad_sequences(x, padding='pre', dtype='float64')

    print(x.shape)
    x = x[:, :, np.array(feature_sequence)]

    x_norm = np.zeros(x.shape)
    for ts in range(x.shape[1]):
        x_norm[:, ts, :] = normalize(x[:, ts, :])
    x = x_norm

    y = np.asarray(y)
    icu_ids = np.asarray(icu_ids)

    folds = []
    x_train = [[] for i in range(k_fold)]
    y_train = [[] for i in range(k_fold)]
    x_val = [[] for i in range(k_fold)]
    y_val = [[] for i in range(k_fold)]
    x_test = [[] for i in range(k_fold)]
    y_test = [[] for i in range(k_fold)]
    icu_ids_train = [[] for i in range(k_fold)]
    icu_ids_val = [[] for i in range(k_fold)]
    icu_ids_test = [[] for i in range(k_fold)]

    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=33)
    for icu in np.unique(icu_ids):
        x_icu = x[icu_ids == icu]
        y_icu = y[icu_ids == icu]
        icu_ids_icu = icu_ids[icu_ids == icu]

        if n_patients < x_icu.shape[0]:
            x_icu, y_icu = subsampling(x_icu, y_icu, n_patients)

        for i, (train_idx, test_idx) in enumerate(skf.split(x_icu, y_icu)):
            # O erro ta aqui, eh por causa do conjunto de validacao
            x_icu_train, x_icu_val, y_icu_train, y_icu_val, icu_train, icu_val = train_test_split(x_icu[train_idx], y_icu[train_idx],
                                                                                                  icu_ids_icu[train_idx],
                                                                                                  test_size=0.2, random_state=42)
            x_train[i] += x_icu_train.tolist()
            y_train[i] += y_icu_train.tolist()
            x_val[i] += x_icu_val.tolist()
            y_val[i] += y_icu_val.tolist()
            x_test[i] += x_icu[test_idx].tolist()
            y_test[i] += y_icu[test_idx].tolist()
            icu_ids_train[i] += icu_train.tolist()
            icu_ids_val[i] += icu_val.tolist()
            icu_ids_test[i] += icu_ids_icu[test_idx].tolist()

    for i in range(k_fold):
        x_train[i], y_train[i], icu_ids_train[i] = oversample(np.asarray(x_train[i]), np.asarray(y_train[i]), np.asarray(icu_ids_train[i]))
        x_val[i] = np.asarray(x_val[i])
        y_val[i] = np.asarray(y_val[i])
        x_test[i] = np.asarray(x_test[i])
        y_test[i] = np.asarray(y_test[i])
        icu_ids_val[i] = np.asarray(icu_ids_val[i])
        icu_ids_test[i] = np.asarray(icu_ids_test[i])

        print("%d - Train - y_true: %d, y_false: %d" % (i, np.count_nonzero(y_train[i] == 0), np.count_nonzero(y_train[i] == 1)))
        print("%d - Test - y_true: %d, y_false: %d" % (i, np.count_nonzero(y_test[i] == 0), np.count_nonzero(y_test[i] == 1)))

        folds.append((x_train[i], y_train[i], x_val[i], y_val[i], x_test[i], y_test[i], icu_ids_train[i], icu_ids_val[i], icu_ids_test[i]))

    return folds


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


if __name__ == '__main__':
    main()

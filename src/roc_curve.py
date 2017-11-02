import prepare_data as data
import numpy as np

from keras.utils.generic_utils import get_custom_objects
from keras.models import load_model
from keras import backend as K
from keras.preprocessing import sequence
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=1)


def load_local_model(id):
    get_custom_objects().update({"precision": precision})
    get_custom_objects().update({"recall": recall})
    get_custom_objects().update({"fmeasure": fmeasure})
    model = load_model("models/" + id + ".h5")
    return model


def load_data(data_id, icus):
    x, y = data.load_icus(data_id, icus, data.targets[0])
    x = sequence.pad_sequences(x)
    y = np.asarray(y)

    skf = StratifiedKFold(n_splits=5, random_state=True)
    for i, (train_idx, test_idx) in enumerate(skf.split(x, y)):
        x_test = x[test_idx]
        y_test = y[test_idx]

    return x_test, y_test


def main():
    colors = ['r', 'g', 'b', 'y']

    x, y = load_data("60", [1, 2, 3, 4])

    model_mt = load_local_model("mt-m3r")
    y_pred = model_mt.predict_proba(x)
    score_mt = roc_auc_score(y, y_pred)
    fpr, tpr, _ = roc_curve(y, y_pred)
    color = 'c'
    plt.plot(fpr, tpr, color, label="Mixed Training")
    print(score_mt)

    for icu in [1, 2, 3, 4]:
        x, y = load_data("60", [1])

        model_ft = load_local_model("ft-m3r-" + str(icu))
        y_pred = model_ft.predict_proba(x)
        score_ft = roc_auc_score(y, y_pred)
        fpr, tpr, _ = roc_curve(y, y_pred)
        color = colors[icu-1]
        plt.plot(fpr, tpr, color, label="Fine Tuning - ICU " + str(icu))

        model_fc = load_local_model("fc-m3r-" + str(icu))
        y_pred = model_fc.predict_proba(x)
        fpr, tpr, _ = roc_curve(y, y_pred)
        score_fc = roc_auc_score(y, y_pred)
        color = colors[icu - 1]
        plt.plot(fpr, tpr, color + "--", label="Focused - ICU " + str(icu))
        print("\nICU %d - Focused: %f, Fine Tuning: %f" % (icu, score_fc, score_ft))

    plt.legend(loc='upper right')
    plt.title("ROC Curve")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.show()


if __name__ == '__main__':
    main()
import prepare_data as data
import visualize_data as visual
import conv_lstm
import models as mds

import numpy as np
from keras import backend as K
from keras import models
from keras.models import Model
from keras.preprocessing import sequence


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


def main():
    model_names = 'ft-m3r', 'fc-m3r'
    for model_name in model_names:
        for icu in [1, 2, 3, 4]:
            m = model_name + "-" + str(icu)
            model = models.load_model("models/m1-" + m + ".h5",
                                      custom_objects={"precision": precision, "recall": recall, "fmeasure": fmeasure})
            gen_model = Model(input=model.input, output=model.layers[2].output)
            # print(gen_model.layers)

            x, y = data.load_icus("60", [icu], data.targets[0])
            x = sequence.pad_sequences(x)
            y = np.asarray(y)

            neural_x = gen_model.predict(x)
            if 'ft' in model_name:
                title = "Fine Tuning - "
            else:
                title = "Focused - "
            plot_heatmap(title + " UTI - " + str(icu), neural_x, y)


def plot_heatmap_and_patient():
    icu_names = ["", "Coronary", "Cardiac", "Medical", "Surgical"]
    for icu in [1, 2, 3, 4]:
    # for icu in [1]:
        # for view_all in [True, False]:
        for view_all in [True]:
            if view_all:
                x_orig, y_orig, _ = data.load_icus("60", [icu], data.targets[0])
            else:
                folds = conv_lstm.load_data("60", [icu], data.targets[0])
                x_icu_train, y_icu_train, x_orig, y_orig = folds[4]

            max_1 = np.count_nonzero(np.asarray(y_orig) == 1)
            max_0 = np.count_nonzero(np.asarray(y_orig) == 0)
            for di in [0]:
                # dead_patient_x = np.asarray(x_orig)[np.asarray(y_orig) == 1][np.random.randint(max_1)]
                dead_patient_x = np.asarray(x_orig)[np.asarray(y_orig) == 1][di]
                # surv_patient_x = np.asarray(x_orig)[np.asarray(y_orig) == 0][np.random.randint(max_0)]
                surv_patient_x = np.asarray(x_orig)[np.asarray(y_orig) == 0][di]

                x_pad = sequence.pad_sequences(x_orig)
                y_pad = np.asarray(y_orig)

                dead_patient_x = np.floor(np.asarray(dead_patient_x))
                surv_patient_x = np.floor(np.asarray(surv_patient_x))

                model = models.load_model("models/ft-mvis-60-" + str(icu) + ".h5",
                                          custom_objects={"precision": precision, "recall": recall, "fmeasure": fmeasure})
                # print(mds.evaluate_model("ft-mvis-60-" + str(icu), x_pad, y_pad))

                gen_model = Model(input=model.input, output=model.layers[3].output)
                neural_x = gen_model.predict(x_pad)

                dead_patient_input = np.zeros((dead_patient_x.shape[0], x_pad.shape[1], x_pad.shape[2]))
                for i in range(dead_patient_x.shape[0]):
                    dead_patient_input[i, :i+1, :] = dead_patient_x[:i+1, :]
                dead_patient_x_neural = gen_model.predict(dead_patient_input)

                surv_patient_input = np.zeros((surv_patient_x.shape[0], x_pad.shape[1], x_pad.shape[2]))
                for i in range(surv_patient_x.shape[0]):
                    surv_patient_input[i, :i+1, :] = surv_patient_x[:i + 1, :]
                surv_patient_x_neural = gen_model.predict(surv_patient_input)

                y_pad[y_pad == 0] = -1
                y_orig[y_orig == 0] = -1
                name = "_all" if view_all else "_test_only"
                visual.heatmap_withpatient(neural_x, y_pad,
                                           icu_names[icu] + " ICU - Neural Death Heatmap", "in-hospital-death")
                # visual.heatmap_withpatient_gif(di, neural_x, y_pad, (dead_patient_x_neural, surv_patient_x_neural))
                # visual.heatmap(neural_x, y_orig, "neural_trajectory" + str(icu) + name, "in-hospital-death")


def plot_heatmap(filename, x, y):
    vectors = []
    indexes = []
    labels = []
    random_idx = list(range(x.shape[0]))
    np.random.shuffle(random_idx)
    n = x.shape[0]
    for i, p in enumerate(zip(x[random_idx][:n], y[random_idx][:n])):
        vectors.append(p[0])
        indexes.append(i)
        labels.append(p[1])
    visual.heatmap(vectors, np.asarray(labels), filename, data.targets)


if __name__ == '__main__':
    # main()
    plot_heatmap_and_patient()

import conv_lstm
import models
import numpy as np

import keras.backend as K
from keras.models import Model

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


folds = conv_lstm.load_data("60", [1, 2, 3, 4], 4000, do_oversample=False)
x_train, y_train, x_val, y_val, x_test, y_test, icu_ids_train, icu_ids_test = folds[4]

model_id = "mt-m_conv_lstm_switch_regl1-60"
model = models.load_model("models" + model_id + ".h5")
model_lstm = Model(input=model.input, output=model.layers[2].output)

sw = K.get_value(model.layers[3].switch_weights)
sb = K.get_value(model.layers[3].sb)

predictions_train = model_lstm.predict(x_train)
activations_train = np.dot(predictions_train, sw.T) + sb
sig_switch_train = sigmoid(activations_train)
tanh_switch_train = np.tanh(activations_train)


fig, ax = plt.subplots()
colors = ['', 'r', 'b', 'g', 'y']
for icu in np.unique(icu_ids_train):
    points = activations_train[icu_ids_train == icu]
    ax.scatter(points, np.zeros_like(points), c=colors[icu])
fig.savefig("switch_activation.png")

fig, ax = plt.subplots()
colors = ['', 'r', 'b', 'g', 'y']
for icu in np.unique(icu_ids_train):
    points = sig_switch_train[icu_ids_train == icu]
    ax.scatter(points, np.zeros_like(points), c=colors[icu])
fig.savefig("switch_sig_activation.png")

fig, ax = plt.subplots()
colors = ['', 'r', 'b', 'g', 'y']
for icu in np.unique(icu_ids_train):
    points = tanh_switch_train[icu_ids_train == icu]
    ax.scatter(points, np.zeros_like(points), c=colors[icu])
fig.savefig("switch_tanh_activation.png")

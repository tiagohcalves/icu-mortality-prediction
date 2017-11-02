import numpy as np
from keras import backend as K
from keras import callbacks
from keras import optimizers
from keras import regularizers
from keras.engine.topology import Layer
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Reshape, Flatten
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Activation
from keras.models import load_model, Sequential
from keras.utils.generic_utils import get_custom_objects
from sklearn.metrics import roc_auc_score


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


class Switch(Layer):
    def __init__(self, inner_dim, use_bias=True, kernel_regularizer=None, **kwargs):
        self.inner_dim = inner_dim
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer
        super(Switch, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.input_weights = self.add_weight(name='input_weights',
                                             shape=(input_shape[1], self.inner_dim),
                                             initializer='glorot_uniform',
                                             regularizer=self.kernel_regularizer,
                                             trainable=True)

        self.switch_weights = self.add_weight(name='switch_weights',
                                              shape=(self.inner_dim, input_shape[1]),
                                              initializer='glorot_uniform',
                                              regularizer=self.kernel_regularizer,
                                              trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.switch_weights,),
                                        initializer='zeros',
                                        name='bias')
        super(Switch, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        S = K.dot(inputs, self.input_weights)
        if self.use_bias:
            S = K.bias_add(S, self.bias)
        Z = K.sigmoid(K.dot(S, self.switch_weights)) * inputs
        return Z


archs = [
    # Conv1 Filters, Conv1 kernel, Conv1 act,
    # Conv2 Filters, Conv2 Kernels, Conv2 Act,
    # Max pool size,
    # Reshape,
    # Lstm Size,
    # Dense Size, Dense activation
    [32, 5, 64, 5, 3, 9 * 64, 120, 200],
    [64, 10, 128, 10, 3, 6 * 128, 300, 200],
    [128, 3, 256, 5, 5, 6 * 256, 500, 300],
]

acts = ['linear', 'elu', 'relu']


def create_freezable_layers(model_id, seq_size, n_features):
    return globals()[model_id](seq_size, n_features)


def m1(seq_size, n_features):
    layers = list()
    layers.append(Conv2D(archs[0][0], kernel_size=(1, archs[0][1]), activation=acts[0], input_shape=(seq_size, n_features, 1)))
    layers.append(Conv2D(archs[0][2], kernel_size=(1, archs[0][3]), activation=acts[0]))
    layers.append(MaxPooling2D(pool_size=(1, archs[0][4])))
    # layers.append(Dropout(0.3))
    layers.append(Flatten())
    layers.append(Reshape((55, archs[0][5])))

    layers.append(LSTM(archs[0][6]))
    layers.append(Dropout(0.3))
    layers.append(Dense(archs[0][7], activation=acts[0]))
    layers.append(Dropout(0.3))
    layers.append(Dense(1, activation='sigmoid'))

    freezable = [0,1,2]
    return layers, freezable


def m2(seq_size, n_features):
    layers = list()

    layers.append(LSTM(128, input_shape=(seq_size, n_features)))
    layers.append(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))

    freezable = [0]
    return layers, freezable


def m2_5(seq_size, n_features):
    layers = list()
    layers.append(Conv1D(64, kernel_size=5, padding='valid', strides=1, input_shape=(seq_size, n_features)))
    layers.append(MaxPooling1D(pool_size=4))
    layers.append(Flatten())
    layers.append(Dense(1))
    layers.append(Activation('sigmoid'))

    freezable = [0, 1]
    return layers, freezable


def m7(seq_size, n_features):
    layers = list()
    layers.append(Conv2D(64, kernel_size=(5, 5), padding='valid', input_shape=(seq_size, n_features, 1)))
    layers.append(MaxPooling2D(pool_size=(2,2)))
    layers.append(Flatten())
    layers.append(Dense(1))
    layers.append(Activation('sigmoid'))

    freezable = [0, 1]
    return layers, freezable


def m8(seq_size, n_features):
    layers = list()
    layers.append(Dense(30, input_shape=(seq_size, n_features)))
    layers.append(LSTM(70))
    layers.append(Dense(1))
    layers.append(Activation('sigmoid'))

    freezable = [0]
    return layers, freezable


def mswitch(seq_size, n_features):
    layers = list()
    layers.append(Conv1D(64, kernel_size=5, padding='valid', strides=1, input_shape=(seq_size, n_features)))
    layers.append(MaxPooling1D(pool_size=4))
    layers.append(LSTM(70))
    layers.append(Switch(1))
    layers.append(Dense(1))
    layers.append(Activation('sigmoid'))

    freezable = [0, 1]
    return layers, freezable


def m3(seq_size, n_features):
    layers = list()
    layers.append(Conv1D(64, kernel_size=5, padding='valid', strides=1, input_shape=(seq_size, n_features)))
    layers.append(MaxPooling1D(pool_size=4))
    layers.append(LSTM(70))
    layers.append(Dense(1))
    layers.append(Activation('sigmoid'))

    freezable = [0, 1]
    return layers, freezable


def m3rsl(seq_size, n_features):
    layers = list()
    layers.append(Conv1D(64, kernel_size=5,
                         activation='selu', padding='valid', strides=1,
                         input_shape=(seq_size, n_features)))
    layers.append(MaxPooling1D(pool_size=4))
    layers.append(LSTM(70, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Activation('selu'))
    layers.append(Dense(1, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Activation('sigmoid'))

    freezable = [0, 1]
    return layers, freezable


def m3r(seq_size, n_features):
    layers = list()
    layers.append(Conv1D(64, kernel_size=5,
                         activation='relu', padding='valid', strides=1,
                         input_shape=(seq_size, n_features)))
    layers.append(MaxPooling1D(pool_size=4))
    layers.append(LSTM(70, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Dense(1, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Activation('sigmoid'))

    freezable = [0, 1]
    return layers, freezable


def m6(seq_size, n_features):
    layers = list()
    layers.append(Dropout(0.2, input_shape=(seq_size, n_features)))
    layers.append(Conv1D(64, kernel_size=5))
    # layers.append(MaxPooling1D(pool_size=4))
    layers.append(Dropout(0.2))
    layers.append(LSTM(70, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Dropout(0.2))
    layers.append(Dense(1, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Activation('sigmoid'))

    freezable = [1]
    return layers, freezable


def m3r2(seq_size, n_features):
    layers = list()
    layers.append(Conv1D(64, kernel_size=5,
                         activation='relu', padding='valid', strides=1,
                         input_shape=(seq_size, n_features)))
    layers.append(MaxPooling1D(pool_size=4))
    layers.append(LSTM(70, kernel_regularizer=regularizers.l2(0.1)))
    layers.append(Dense(1, kernel_regularizer=regularizers.l2(0.1)))
    layers.append(Activation('sigmoid'))

    freezable = [0, 1]
    return layers, freezable


def m3rd(seq_size, n_features):
    layers = list()
    layers.append(Conv1D(64, kernel_size=5,
                         activation='relu', padding='valid', strides=1,
                         input_shape=(seq_size, n_features)))
    layers.append(MaxPooling1D(pool_size=4))
    layers.append(LSTM(70, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Dropout(0.3))
    layers.append(Dense(1, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Activation('sigmoid'))

    freezable = [0, 1]
    return layers, freezable


def m3r_freezeall(seq_size, n_features):
    layers = list()
    layers.append(Conv1D(64, kernel_size=5,
                         activation='relu', padding='valid', strides=1,
                         input_shape=(seq_size, n_features)))
    layers.append(MaxPooling1D(pool_size=4))
    layers.append(LSTM(70, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Dense(1, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Activation('sigmoid'))

    freezable = [0, 1, 2]
    return layers, freezable


def m3r_nonfreeze(seq_size, n_features):
    layers = list()
    layers.append(Conv1D(64, kernel_size=5,
                         activation='relu', padding='valid', strides=1,
                         input_shape=(seq_size, n_features)))
    layers.append(MaxPooling1D(pool_size=4))
    layers.append(LSTM(70, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Dense(1, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Activation('sigmoid'))

    freezable = []
    return layers, freezable


def m3bn(seq_size, n_features):
    layers = list()
    layers.append(Conv1D(64, kernel_size=5,
                         activation='relu', padding='valid', strides=1,
                         input_shape=(seq_size, n_features)))
    layers.append(MaxPooling1D(pool_size=4))
    layers.append(LSTM(70))
    layers.append(BatchNormalization())
    layers.append(Activation('relu'))
    layers.append(Dense(1, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Activation('sigmoid'))

    freezable = [0, 1]
    return layers, freezable


def mvis(seq_size, n_features):
    layers = list()
    layers.append(Conv1D(64, kernel_size=5,
                         activation='relu', padding='valid', strides=1,
                         input_shape=(seq_size, n_features)))
    layers.append(MaxPooling1D(pool_size=4))
    layers.append(LSTM(70, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Dense(2, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Dense(1, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Activation('sigmoid'))

    freezable = [0, 1]
    return layers, freezable


def m3s(seq_size, n_features):
    layers = list()
    layers.append(Conv1D(64, kernel_size=10,
                         activation='relu', padding='valid', strides=1,
                         input_shape=(seq_size, n_features)))
    layers.append(MaxPooling1D(pool_size=5))
    layers.append(LSTM(64, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Dense(1, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Activation('sigmoid'))

    freezable = [0, 1]
    return layers, freezable


def m32c(seq_size, n_features):
    layers = list()
    layers.append(Conv1D(64, kernel_size=10,
                         activation='relu', padding='valid', strides=1,
                         input_shape=(seq_size, n_features)))
    layers.append(Conv1D(128, kernel_size=5,
                         activation='relu', padding='valid', strides=1,
                         input_shape=(seq_size, n_features)))
    layers.append(MaxPooling1D(pool_size=5))
    layers.append(LSTM(64, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Dense(1, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Activation('sigmoid'))

    freezable = [0, 1, 2]
    return layers, freezable


def m3br(seq_size, n_features):
    layers = list()
    layers.append(Conv1D(64, kernel_size=3,
                         activation='relu', padding='valid', strides=1,
                         input_shape=(seq_size, n_features)))
    layers.append(MaxPooling1D(pool_size=2))
    layers.append(LSTM(300, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Activation('relu'))
    layers.append(Dense(1, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Activation('sigmoid'))

    freezable = [0, 1]
    return layers, freezable


def m3brl(seq_size, n_features):
    layers = list()
    layers.append(Conv1D(64, kernel_size=3, padding='valid', strides=1,
                         input_shape=(seq_size, n_features)))
    layers.append(MaxPooling1D(pool_size=2))
    layers.append(LSTM(300, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Dense(1, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Activation('sigmoid'))

    freezable = [0, 1]
    return layers, freezable


def m3brl2(seq_size, n_features):
    layers = list()
    layers.append(Dropout(0.2, input_shape=(seq_size, n_features)))
    layers.append(Conv1D(64, kernel_size=3, padding='valid', strides=1))
    layers.append(MaxPooling1D(pool_size=2))
    layers.append(Dropout(0.3))
    layers.append(LSTM(300, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Dropout(0.3))
    layers.append(Dense(1, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Activation('sigmoid'))

    freezable = [1, 2]
    return layers, freezable


def m3brl4(seq_size, n_features):
    layers = list()
    layers.append(Dropout(0.2, input_shape=(seq_size, n_features)))
    layers.append(Conv1D(64, kernel_size=3, padding='valid', strides=1))
    layers.append(Activation('selu'))
    layers.append(MaxPooling1D(pool_size=2))
    layers.append(Dropout(0.3))
    layers.append(LSTM(300, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Activation('selu'))
    layers.append(Dropout(0.3))
    layers.append(Dense(1, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Activation('sigmoid'))

    freezable = [1, 2]
    return layers, freezable


def m3br2(seq_size, n_features):
    layers = list()
    layers.append(Conv1D(128, kernel_size=3, input_shape=(seq_size, n_features)))
    layers.append(MaxPooling1D(pool_size=2))
    layers.append(LSTM(300, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Activation('relu'))
    layers.append(Dropout(0.3))
    layers.append(Dense(1, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Activation('sigmoid'))

    freezable = [0, 1]
    return layers, freezable


def m4(seq_size, n_features):
    layers = list()
    layers.append(Conv1D(64, kernel_size=5,
                         activation='relu', padding='valid', strides=1,
                         input_shape=(seq_size, n_features)))
    layers.append(Conv1D(128, kernel_size=5,
                         activation='relu', padding='valid', strides=1,
                         input_shape=(seq_size, n_features)))
    layers.append(MaxPooling1D(pool_size=4))
    layers.append(LSTM(200))
    layers.append(Dense(100))
    layers.append(Activation('relu'))
    layers.append(Dense(1))
    layers.append(Activation('sigmoid'))

    freezable = [0, 1, 2]
    return layers, freezable


def m4r(seq_size, n_features):
    layers = list()
    layers.append(Conv1D(64, kernel_size=5,
                         activation='relu', padding='valid', strides=1,
                         input_shape=(seq_size, n_features)))
    layers.append(Conv1D(128, kernel_size=5,
                         activation='relu', padding='valid', strides=1,
                         input_shape=(seq_size, n_features)))
    layers.append(MaxPooling1D(pool_size=4))
    layers.append(LSTM(200, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Dense(100, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Activation('relu'))
    layers.append(Dense(1, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Activation('sigmoid'))

    freezable = [0, 1, 2]
    return layers, freezable


def m3b(seq_size, n_features):
    layers = list()
    layers.append(Conv1D(128, kernel_size=3,
                         activation='relu', padding='valid', strides=1,
                         input_shape=(seq_size, n_features)))
    layers.append(MaxPooling1D(pool_size=2))
    layers.append(LSTM(128))
    layers.append(Dense(1))
    layers.append(Activation('sigmoid'))

    freezable = [0, 1]
    return layers, freezable


def m4d(seq_size, n_features):
    layers = list()
    layers.append(Conv1D(64, kernel_size=3,
                         activation='relu', padding='valid', strides=1,
                         input_shape=(seq_size, n_features)))
    layers.append(Conv1D(128, kernel_size=5,
                         activation='relu', padding='valid', strides=1,
                         input_shape=(seq_size, n_features)))
    layers.append(MaxPooling1D(pool_size=2))
    layers.append(LSTM(256))
    layers.append(Dropout(0.5))
    layers.append(Dense(100))
    layers.append(Activation('relu'))
    layers.append(Dropout(0.5))
    layers.append(Dense(1))
    layers.append(Activation('sigmoid'))

    freezable = [0, 1, 2]
    return layers, freezable


    layers.append(Conv1D(64, kernel_size=3,
                         activation='relu', padding='valid', strides=1,
                         input_shape=(seq_size, n_features)))
    layers.append(Conv1D(128, kernel_size=2,
                         activation='relu', padding='valid', strides=1,
                         input_shape=(seq_size, n_features)))
    layers.append(MaxPooling1D(pool_size=4))
    layers.append(Flatten())
    layers.append(Dense(64, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Activation('relu'))
    layers.append(Dense(1, kernel_regularizer=regularizers.l2(0.01)))
    layers.append(Activation('sigmoid'))

    freezable = [0, 1, 2]
    return layers, freezable


def build_model(layers):
    model = Sequential()

    for layer in layers:
        model.add(layer)
    return model


def create_model(id, layers):
    model = build_model(layers)

    optimizer = optimizers.Adam()
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', precision, recall, fmeasure])
    # model.summary()

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=15)
    checkpoint = callbacks.ModelCheckpoint("models/" + id + ".h5", monitor='val_loss', save_best_only=True)
    return checkpoint, early_stopping, model


def evaluate_model(id, x, y):
    model = get_model(id)
    y_pred = model.predict_proba(x)
    auc_score = roc_auc_score(y, y_pred)
    return model.evaluate(x, y, verbose=0), auc_score


def get_model(id):
    get_custom_objects().update({"precision": precision})
    get_custom_objects().update({"recall": recall})
    get_custom_objects().update({"fmeasure": fmeasure})
    model = load_model("models/" + id + ".h5")
    return model


def shuffle_weights(layer, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.
    This is a fast approximation of re-initializing the weights of a model.
    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).
    :param Model layer: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = layer.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    layer.set_weights(weights)

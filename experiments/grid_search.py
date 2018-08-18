import pickle as pk
from keras import backend as K
from keras import callbacks
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Activation
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from models import Switch
from conv_lstm import load_data


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


def create_model(seq_size, n_features,
                 switch_size, switch_reg, switch_activation,
                 use_batch_norm,
                 dropout_rate,
                 conv_size,
                 lstm_size,
                 optimizer):
    # create model
    model = Sequential()

    model.add(Switch(switch_size, kernel_regularizer=switch_reg, activation=switch_activation, input_shape=(seq_size, n_features)))
    if use_batch_norm:
        model.add(BatchNormalization())
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    model.add(Conv1D(conv_size, kernel_size=5, padding='valid', strides=1))
    model.add(MaxPooling1D(pool_size=4))
    if use_batch_norm:
        model.add(BatchNormalization())
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    model.add(LSTM(lstm_size))
    if use_batch_norm:
        model.add(BatchNormalization())
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    model.add(Switch(switch_size, kernel_regularizer=switch_reg, activation=switch_activation))
    if use_batch_norm:
        model.add(BatchNormalization())
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))
    if use_batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', precision, recall, fmeasure])
    return model


def main():
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=15)

    folds = load_data("60", [1, 2, 3, 4], 1500)
    x_train, y_train, x_val, y_val, x_test, y_test, icu_ids_test = folds[2]
    y_train = y_train.reshape(y_train.shape[0],)

    model = KerasClassifier(build_fn=create_model, epochs=25)

    param_grid = {
        "seq_size": [x_train.shape[1]],
        "n_features": [x_train.shape[2]],
        "switch_size": [1, 10],
        "switch_reg": ['l1', 'l2'],
        "switch_activation": ['selu', 'linear'],
        "use_batch_norm": [True, False],
        "dropout_rate": [0, 0.2, 0.5],
        "conv_size": [64, 128],
        "lstm_size": [64, 128],
        "optimizer": ['Adam', 'Nadam']
    }
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
    grid_result = grid.fit(x_train, y_train)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    pk.dump(open("grid_result", "wb"), grid_result)


if __name__ == '__main__':
    main()

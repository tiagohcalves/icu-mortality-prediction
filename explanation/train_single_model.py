# Imports

import sys
import pickle as pk
import numpy as np

import keras
from keras import callbacks
from keras import optimizers

from keras.utils.generic_utils import get_custom_objects
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Local imports

sys.path.append("/home/tiago/Documentos/research/mestrado/Medical/medical/src/")
import models
import prepare_data as data
from models import precision, recall, fmeasure, Switch


# Load and split data
X, y, icu = data.load_icus("60", [1,2,3,4], "", should_normalize=True)

x_train, x_test, y_train, y_test, icu_train, icu_test = train_test_split(X, y, icu, train_size=0.7, random_state=7, stratify=y)
x_train, x_val, y_train, y_val, icu_train, icu_val = train_test_split(x_train, y_train, icu_train, train_size=0.7, random_state=7, stratify=y_train)

# x_train, y_train, icu_train = data.oversample(x_train, y_train, np.array(icu_train))

# Define model
model_id = "m_conv_lstm_switch"

layers, freezable = models.create_freezable_layers(model_id, x_train.shape[1], x_train.shape[2])
model = models.build_model(layers)
optimizer = optimizers.Adam()

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', precision, recall, fmeasure])

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=15)
checkpoint = callbacks.ModelCheckpoint(model_id + ".h5", monitor='val_loss', save_best_only=True)

# Fit 
hist = model.fit(
    x_train, y_train,
    epochs=100,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, checkpoint]
)

bn_epochs = np.argmin(hist.history['val_loss'])

get_custom_objects().update({
    "precision": precision,
    "recall": recall,
    "fmeasure": fmeasure,
    "Switch": Switch
})

model = keras.models.load_model(model_id + ".h5")

y_pred_val = model.predict_proba(x_val)
val_auc_score = roc_auc_score(y_val, y_pred_val)

y_pred_test = model.predict_proba(x_test)
test_auc_score = roc_auc_score(y_test, y_pred_test)

print("Base model score for \nVal: %s\nTest: %s" % (val_auc_score, test_auc_score))

# # Redefine model
# layers, freezable = models.create_freezable_layers(model_id, x_train.shape[1], x_train.shape[2])
# model_val = models.build_model(layers)
# optimizer = optimizers.Adam()

# model_val.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', precision, recall, fmeasure])

# # Refit with more data
# hist_val = model_val.fit(
#     np.vstack([x_train, x_val]), np.concatenate([y_train, y_val]),
#     epochs=bn_epochs
# )

# print("Final model score for test: %s" % (test_auc_score))
# model_val.save(model_id + ".h5")


model.save(model_id + ".h5")


pk.dump((
    x_train, x_val, x_test, 
    y_train, y_val, y_test, 
    icu_train, icu_val, icu_test
), open("data.pk", "wb"))
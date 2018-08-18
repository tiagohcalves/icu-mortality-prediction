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


def slice_patients(patient_features, patient_outcomes, patient_icus):
    """Create slices with the first N selected time steps"""
    new_patient_features = []
    new_patient_outcomes = []
    new_patient_icus = []
    for features, outcome, icu in zip(patient_features, patient_outcomes, patient_icus):
        for i in range(0, features.shape[0] - 2, 2): # sample only half the timesteps to avoid creating too much inputs
            new_input = np.zeros(features.shape)
            new_input[:i+1, :] = features[:i+1, :]
            
            new_patient_features.append(new_input)
            new_patient_outcomes.append(outcome)
            new_patient_icus.append(icu)

    new_patient_features = np.asarray(new_patient_features)
    new_patient_outcomes = np.asarray(new_patient_outcomes)
    new_patient_icus = np.asarray(new_patient_icus)
    
    patient_features = np.vstack([patient_features, new_patient_features])
    patient_outcomes = np.concatenate([patient_outcomes, new_patient_outcomes])
    patient_icus = np.concatenate([patient_icus, new_patient_icus])
    
    return patient_features, patient_outcomes, patient_icus


# Load and split data
X, y, icu = data.load_icus("60", [1,2,3,4], "", should_normalize=True)

x_train, x_test, y_train, y_test, icu_train, icu_test = train_test_split(X, y, icu, train_size=0.7, random_state=7, stratify=y)
x_train, x_val, y_train, y_val, icu_train, icu_val = train_test_split(x_train, y_train, icu_train, train_size=0.7, random_state=7, stratify=y_train)

# Create slices
print("Creating slices...")
x_train, y_train, icu_train = slice_patients(x_train, y_train, icu_train)
x_val, y_val, icu_val = slice_patients(x_val, y_val, icu_val)
x_test, y_test, icu_test = slice_patients(x_test, y_test, icu_test)
print("Done")

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
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

###########
random_state = 13

# Load and split data
X, y, icu = data.load_icus("60", [1,2,3,4], "", should_normalize=True)

x_train, x_test, y_train, y_test, icu_train, icu_test = train_test_split(
    X, y, icu, train_size=0.7, random_state=random_state, stratify=y
)

x_train, x_val, y_train, y_val, icu_train, icu_val = train_test_split(
    x_train, y_train, icu_train, train_size=0.7, random_state=random_state, stratify=y_train
)

icu_train = np.array(icu_train)
icu_val = np.array(icu_val)
icu_test = np.array(icu_test)

x_train, y_train, icu_train = data.oversample(x_train, y_train, icu_train)
icu_train = icu_train.flatten()

############

# Define model
model_id = "m_conv_lstm_switch"

layers, freezable = models.create_freezable_layers(model_id, x_train.shape[1], x_train.shape[2])
model = models.build_model(layers)
optimizer = optimizers.Adam()

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', precision, recall, fmeasure])

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=15)
checkpoint = callbacks.ModelCheckpoint(model_id + ".h5", monitor='val_loss', save_best_only=True)

# Fit 
model.fit(
    x_train, y_train,
    epochs=100,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, checkpoint]
)

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

###########

auc_scores = { 
   "all": [0]*4, "ft": [0]*4
}

for icu_id in range(1, 5):
    x_icu_test = x_test[icu_test == icu_id]
    y_icu_test = y_test[icu_test == icu_id]
    
    y_pred_val = model.predict_proba(x_icu_test)
    test_auc_score = roc_auc_score(y_icu_test, y_pred_val)
    
    auc_scores["all"][icu_id-1] = test_auc_score
    print("Base model score for ICU %d: \t %.4f" % (icu_id, test_auc_score))


####
# Fine tuning
####

for icu_id in range(1, 5):
    print("ICU %d" % icu_id)
    x_icu_train = x_train[icu_train == icu_id]
    y_icu_train = y_train[icu_train == icu_id]
    
    x_icu_val = x_val[icu_val == icu_id]
    y_icu_val = y_val[icu_val == icu_id]
    
    x_icu_test = x_test[icu_test == icu_id]
    y_icu_test = y_test[icu_test == icu_id]
    
    model = keras.models.load_model(model_id + ".h5")
    
    for l in freezable:
        model.layers[l].trainable = False
    
    optimizer = optimizers.Adam()
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', precision, recall, fmeasure])
    
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=15)
    checkpoint = callbacks.ModelCheckpoint("ft-%s-%d.h5" % (model_id, icu_id), monitor='val_loss', save_best_only=True)
    
    model.fit(
        x_icu_train, y_icu_train,
        epochs=100,
        validation_data=(x_icu_val, y_icu_val),
        callbacks=[early_stopping, checkpoint]
    )
    
    y_pred_val = model.predict_proba(x_icu_test)
    test_auc_score = roc_auc_score(y_icu_test, y_pred_val)
    auc_scores["ft"][icu_id-1] = test_auc_score
    
    print("Fine tuned model score for ICU %d: \t %.4f" % (icu_id, test_auc_score))
    model.save("ft-%s-%d.h5" % (model_id, icu_id))

    
print("\n\n")
for mode, aucs in auc_scores.items():
    print("Scores for mode %s" % mode)
    for icu_id, auc_value in enumerate(aucs):
        print("%s \t %.4f" % (icu_id + 1, auc_value))

    
pk.dump((
    x_train, x_val, x_test, 
    y_train, y_val, y_test, 
    icu_train, icu_val, icu_test
), open("data_ft.pk", "wb"))
import sys
import keras
import shap
import tqdm

import pickle as pk
import numpy as np

from keras.utils.generic_utils import get_custom_objects
from multiprocessing import Pool

sys.path.append("/home/tiago/Documentos/research/mestrado/Medical/medical/src/")
from models import precision, recall, fmeasure, Switch

MODEL_PATH = "/home/tiago/Documentos/research/mestrado/Medical/medical/src/explanation"
output_dir = "/home/tiago/Documentos/research/mestrado/Medical/medical/explain/"


model_id = "m_conv_lstm_switch"
# mode = "ft"
mode = "all"

icu_id = 2

for mode in ["all", "ft"]:
    for icu_id in range(1, 5):
        print("Mode: %s - ICU: %d" % (mode, icu_id))
        get_custom_objects().update({
            "precision": precision,
            "recall": recall,
            "fmeasure": fmeasure,
            "Switch": Switch
        })

        if mode == "all":
            trained_model = keras.models.load_model("%s/%s.h5" % (MODEL_PATH, model_id))
        else:
            trained_model = keras.models.load_model("%s/ft-%s-%d.h5" % (MODEL_PATH, model_id, icu_id))

        def predict_proba_flatten(patient_input):
            return trained_model.predict_proba(
                patient_input.reshape(patient_input.shape[0], 48, 47)
            )

        print("Loading data...")
        x_train, x_val, x_test, y_train, y_val, y_test, icu_train, icu_val, icu_test = pk.load(
            open("data_ft.pk", "rb")
        )

        x_icu_train = x_train[icu_train == icu_id]
        x_icu_val = x_val[icu_val == icu_id]
        x_icu_test = x_test[icu_test == icu_id]
        y_icu_test = y_test[icu_test == icu_id]

        flattened_x_train = []
        flattened_x_test = []

        for x in np.vstack([x_icu_train, x_icu_val]):
            flattened_x_train.append(x.flatten())

        for x in x_icu_test:
            flattened_x_test.append(x.flatten())

        flattened_x_train = np.asarray(flattened_x_train)
        flattened_x_test = np.asarray(flattened_x_test)

        print("Creating explainer...")
        explainer = shap.KernelExplainer(predict_proba_flatten, shap.kmeans(flattened_x_train, 10), link="logit")

        input_batch = list(zip(
            list(range(len(flattened_x_test))), 
            flattened_x_test, y_icu_test, 
            icu_test[icu_test == icu_id])
        )

        progress_bar = tqdm.tqdm(total=len(input_batch))
        
        # Define function to enable parallelism
        def get_shap(x_input):
            i, x, y, icu = x_input
            # print(i)
            progress_bar.update(1)
            return x, explainer.shap_values(x, nsamples=2000), y, icu

        print("Calculating shaps")
        pool = Pool(processes=8)
        icu_shap_values = pool.map(get_shap, input_batch)

        pool.close()
        pool.join()

        print("Dumping results...")
        if mode == "all":
            pk.dump(icu_shap_values, open("all-%d-shap_values.pk" % icu_id, "wb"))
        else:
            pk.dump(icu_shap_values, open("%d-shap_values.pk" % icu_id, "wb"))

        print("Done")
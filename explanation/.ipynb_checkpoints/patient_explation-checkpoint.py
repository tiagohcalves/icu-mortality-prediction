import sys
import keras
import shap

import pickle as pk
import numpy as np

from keras.utils.generic_utils import get_custom_objects
from multiprocessing import Pool

sys.path.append("/home/tiago/Documentos/research/mestrado/Medical/medical/src/")
from models import precision, recall, fmeasure, Switch

MODEL_PATH = "/home/tiago/Documentos/research/mestrado/Medical/medical/src/explanation/model.h5"
output_dir = "/home/tiago/Documentos/research/mestrado/Medical/medical/explain/"


get_custom_objects().update({
    "precision": precision,
    "recall": recall,
    "fmeasure": fmeasure,
    "Switch": Switch
})

trained_model = keras.models.load_model(MODEL_PATH)
def predict_proba_flatten(patient_input):
    return trained_model.predict_proba(
        patient_input.reshape(patient_input.shape[0], 48, 47)
    )


# def main():
print("Loading data...")
x_train, x_val, x_test, y_train, y_val, y_test, icu_train, icu_val, icu_test = pk.load(
    open("data.pk", "rb")
)

flattened_x_train = []
flattened_x_test = []

for x in np.vstack([x_train, x_val]):
    flattened_x_train.append(x.flatten())

for x in x_test:
    flattened_x_test.append(x.flatten())

flattened_x_train = np.asarray(flattened_x_train)
flattened_x_test = np.asarray(flattened_x_test)

print("Creating explainer...")
explainer = shap.KernelExplainer(predict_proba_flatten, shap.kmeans(flattened_x_train, 10), link="logit")

input_batch = list(zip(list(range(len(flattened_x_test))), flattened_x_test, y_test, icu_test))

# Define function to enable parallelism
def get_shap(x_input):
    i, x, y, icu = x_input
    print(i)
    return x, explainer.shap_values(x, nsamples=2000), y, icu

print("Calculating shaps")
pool = Pool(processes=8)
shap_values = pool.map(get_shap, input_batch)

pool.close()
pool.join()

print("Dumping results...")
pk.dump(shap_values, open("shap_values.pk", "wb"))
print("Done")


# if __name__ == '__main__':
#     main()

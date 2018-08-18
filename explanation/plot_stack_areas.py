# Imports

import os
import sys
import shap
import matplotlib

import pickle as pk
import pandas as pd
import numpy as np
import seaborn as sns

import keras
from keras import callbacks
from keras import optimizers

from keras.utils.generic_utils import get_custom_objects
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt

# Local imports

sys.path.append("/home/tiago/Documentos/research/mestrado/Medical/medical/src/")
import models
import prepare_data as data
from explanation.time_visualization import visualize as time_vis
from models import precision, recall, fmeasure, Switch

# CONSTANTS

BASE_DIR = "/home/tiago/Documentos/research/mestrado/Medical/medical/"
RAW_FILE_PATH = BASE_DIR + "data/v2_60/"
SHAP_FILE_PATH = BASE_DIR + "explain/"
EXPLAIN_PATH = BASE_DIR + "src/explanation/"
MODEL_PATH = EXPLAIN_PATH + "model.h5"

HEADER_FILE = BASE_DIR + "data/header.csv"
PATIENT_FILE = "icu_1/1_227.csv"

with open(HEADER_FILE) as h_file:
    header = h_file.read().splitlines()

print("Header: ")
print(header)

treatable_list = ["ALP", "ALT", "AST", "Albumin", "Cholesterol", "Creatinine", "DiasABP", "Glucose", "HCO3", "HCT", "HR", "K", "Lactate", "MAP", "Mg", "NIDiasABP", "NIMAP", "NISysABP", "Na", "PaCO2", "PaO2", "Platelets", "RespRate", "SaO2", "SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC"]
print("Variables that can be intervened:")
print(treatable_list)

treatable_header_index = [i for i, v in enumerate(header) if v in treatable_list]

x_train, x_val, x_test, y_train, y_val, y_test, icu_train, icu_val, icu_test = pk.load(
    open(EXPLAIN_PATH + "data_ft.pk", "rb")
)
print("Loaded %d patients" % len(y_test))

_, _, non_norm_x_test, _, _, non_norm_y_test, _, _, non_norm_icu_test = pk.load(
    open(EXPLAIN_PATH + "data_non_norm.pk", "rb")
)

model_id = "m_conv_lstm_switch"

def plot_stacked_area(icu_shap, icu_id, ax, n_features):
    fontsize = 30
    tmp_df = pd.DataFrame(icu_shap[20:], index=range(20, 48), columns=treatable_list)
    
    tmp_df.index = tmp_df.index * 10
    tmp_df = tmp_df.reindex(index=range(200, 480)).interpolate(method='cubic')
    tmp_df.index = tmp_df.index / 10
    
    top_features = tmp_df.clip(lower=0).sum().sort_values(ascending=False).index.values[:n_features]
    tail_features = tmp_df.clip(lower=0).sum().sort_values(ascending=False).index.values[n_features:]
    tmp_df["Others"] = tmp_df[tail_features].sum(axis=1)
    
    ax = tmp_df[list(top_features) + ["Others"]].clip(lower=0).plot(
        kind="area", 
        stacked=True, 
        figsize=(8, 5),
        cmap='viridis',
#         title="Feature Impact on Time for Mortality in ICU " + icu_names[icu_id]
    )
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    
    tmp_df = tmp_df.drop("Others", axis=1)
#     ax.set_xticklabels([0, 20, 30, 40, 50])
    ax.set_xlabel("Hours since admission", fontsize=fontsize)
    
    leg = ax.get_legend()
    for l, text in enumerate(leg.get_texts()):
        text.set_backgroundcolor((1, 1, 1, 0.5))
    for hand in leg.legendHandles: 
        hand.set_edgecolor('w')
    
    plt.tight_layout()
    plt.savefig("figs/04-%d-area-icu-mortality.eps" % icu_id)
    plt.show()
    
    ############
    
    top_features = tmp_df.clip(upper=0).sum().sort_values(ascending=True).index.values[:n_features]
    tail_features = tmp_df.clip(upper=0).sum().sort_values(ascending=True).index.values[n_features:]
    tmp_df["Others"] = tmp_df[tail_features].sum(axis=1)
    
    ax = tmp_df[list(top_features) + ["Others"]].clip(upper=0).plot(
        kind="area", 
        stacked=True, 
#         ax=ax[icu_id, 1],
        figsize=(8, 5),
        cmap='viridis',
#         title="Feature Impact on Time for Survival in ICU " + icu_names[icu_id]
    )
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    tmp_df = tmp_df.drop("Others", axis=1)
    ax.set_xlabel("Hours since admission", fontsize=fontsize)
    ax.legend(loc="lower left")
#     ax.set_xticklabels([0, 20, 30, 40, 50])

    leg = ax.get_legend()
    for l, text in enumerate(leg.get_texts()):
        text.set_backgroundcolor((1, 1, 1, 0.5))
    for hand in leg.legendHandles: 
        hand.set_edgecolor('w')

    plt.tight_layout()
    plt.savefig("figs/04-%d-area-icu-survival.eps" % icu_id)
    # plt.show()

    
for icu_id in range(1, 5):
    print("ICU - %d" % icu_id)
    get_custom_objects().update({
        "precision": precision,
        "recall": recall,
        "fmeasure": fmeasure,
        "Switch": Switch
    })

    ft_model = keras.models.load_model(EXPLAIN_PATH + "ft-%s-%d.h5" % (model_id, icu_id))
    all_model = keras.models.load_model(EXPLAIN_PATH + "%s.h5" % (model_id))
    print("Loaded Convolutional-Recurrent model")
    
    ft_shap_list = pk.load(open(EXPLAIN_PATH + "%d-shap_values.pk" % icu_id, "rb"))
    all_shap_list = pk.load(open(EXPLAIN_PATH + "all-%d-shap_values.pk" % icu_id, "rb"))
    
    def get_shap_from_list(pid):
        return shap_list[pid][1][0][:-1].reshape(48, 47)
    
    ft_shap_x, ft_shap_values, ft_shap_y, ft_shap_icu = zip(*ft_shap_list)
    all_shap_x, all_shap_values, all_shap_y, all_shap_icu = zip(*all_shap_list)
    
    ft_shap_y = np.array(ft_shap_y)
    ft_shap_icu = np.array(ft_shap_icu)

    all_shap_y = np.array(all_shap_y)
    all_shap_icu = np.array(all_shap_icu)
    
    n_patients = len(ft_shap_values)
    ft_shap_values = np.array(ft_shap_values).reshape((n_patients, 2257))[:, :-1].reshape((n_patients, 48, 47))
    all_shap_values = np.array(all_shap_values).reshape((n_patients, 2257))[:, :-1].reshape((n_patients, 48, 47))
    
    ft_icu_shap_sum = ft_shap_values.mean(axis=0)
    ft_treatable_icu_shap_sum = ft_icu_shap_sum[:, treatable_header_index]

    all_icu_shap_sum = all_shap_values.mean(axis=0)
    all_treatable_icu_shap_sum = all_icu_shap_sum[:, treatable_header_index]
    
    plot_stacked_area(ft_treatable_icu_shap_sum, 1, None, 7)
    plot_stacked_area(all_treatable_icu_shap_sum, 1, None, 7)
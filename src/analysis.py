# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import numpy as np
import prepare_data as data
import matplotlib.pyplot as plt

from keras.preprocessing import sequence

mean_pat = []
max_x = []
min_x = []
mean_x = []
for icu in [1,2,3,4]:
    x, y, ids = data.load_icus("60", [icu], data.targets[0])
    x = sequence.pad_sequences(x)
    x = x[:, 1:, :]
    y = np.asarray(y)
    
    mean_pat.append(np.mean(x, axis=0))
    max_x.append(np.max(np.max(x, axis=0), axis=0))
    min_x.append(np.min(np.min(x, axis=0), axis=0))
    mean_x.append(np.mean(np.mean(x, axis=0), axis=0))
    
mean_pat = np.asarray(mean_pat)
max_x = np.asarray(max_x)
min_x = np.asarray(min_x)
mean_x = np.asarray(mean_x)

max_x = np.max(max_x, axis=0)
min_x = np.min(min_x, axis=0)
mean_x = np.mean(mean_x, axis=0)

import pandas as pd
for icu in range(4):
    df = pd.DataFrame(mean_pat[icu])
    df.to_csv("heatmap_data_icu_" + str(icu) + ".csv")
    # np.savetxt("heatmap_data_icu_" + str(icu) + ".csv", mean_pat, delimiter=",")

# norm_x = np.zeros((4, 48, 40))
# for i in range(mean_pat.shape[0]):
#     norm_x[i] = ((mean_pat[i] - mean_x) / (max_x - min_x))


# x_axis = np.array(list(range(40)))
# y_axis = np.array(list(range(49)))

# icu_name_list = ['Coronary', 'Cardiac', 'Medical', 'Surgical']
# for icu in range(4):
#     hmap = norm_x[icu]
#     np.savetxt("heatmap_data_icu" + str(icu) + ".csv", hmap, delimiter=",")
    
#     plt.pcolor(x_axis, y_axis, hmap, cmap='seismic', vmin=np.min(hmap), vmax=np.max(hmap))
#     plt.axis([x_axis.min(), x_axis.max(), y_axis.min(), y_axis.max()])
#     plt.colorbar()
    
#     plt.title("Feature heatmap: " + icu_name_list[icu] + " ICU")
#     plt.savefig("Feature heatmap: " + icu_name_list[icu] + " ICU.png")
#     plt.close()
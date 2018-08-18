# -*- coding: utf-8 -*-
import numpy as np
import prepare_data as data
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import normalize
from keras.preprocessing import sequence

#%%

x, y, ids = data.load_icus("60", [1, 2, 3, 4], data.targets[0])

#%%
x = sequence.pad_sequences(x, padding='post')
y = np.asarray(y)
ids = np.asarray(ids)

#%%

x = x[:, :47, :42]

#%%
x_norm = np.zeros(x.shape)
for ts in range(x.shape[2]):
    x_norm[:, :, ts] = normalize(x[:, :, ts])
x = x_norm

#%%

#fig = plt.figure()
#ax1 = plt.subplot2grid((10, 10), (0, 0), colspan=10)
#ax2 = plt.subplot2grid((10, 10), (1, 0), rowspan=9)
#for j in range(2):
#    for i in range(1, 5):
#        plt.subplot2grid((10, 10), (j+1, (i-1)*2+1), colspan=2, rowspan=2)
##ax3 = plt.subplot2grid((10, 10), (1, 0), colspan=2, rowspan=2)
##ax4 = plt.subplot2grid((10, 10), (1, 2), rowspan=2)
#
#plt.show()

#%%

from matplotlib.transforms import Bbox
def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles.
    https://stackoverflow.com/questions/4325733/save-a-subplot-in-matplotlib
    """
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels() 
#    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)

#%%

feature_names = np.array(["Age","Gender","Height","SAPS-I","SOFA","Albumin","ALP","ALT","AST","Bilirubin","BUN","Cholesterol","Creatinine","DiasABP","FiO2","GCS","Glucose","HCO3","HCT","HR","K","Lactate","MAP","MechVent","Mg","Na","NIDiasABP","NIMAP","NISysABP","PaCO2","PaO2","pH","Platelets","RespRate","SaO2","SysABP","Temp","TroponinI","TroponinT","Urine","WBC","Weight"])

feature_indexes = [7, 8, 16, 30, 39]

icu_names = ["Coronary", "Cardiac", "Medical", "Surgical"]
fig, axs = plt.subplots(2, 4)

for i in range(1, 5):
    x_icu = x[ids == i]
    x_icu = x_icu[y[ids ==i] == 1]
    x_mean = np.mean(x_icu, axis=0)
    x_mean = normalize(x_mean, axis=1)[:, feature_indexes]
    
    axs[0, i-1] = sns.heatmap(x_mean, vmin=0, vmax=1, cmap="Reds", ax=axs[0, i-1], xticklabels=feature_names[feature_indexes], cbar=False)
    axs[0, i-1].set_title(icu_names[i-1] + " ICU - Non-surviving patients", fontdict={"fontsize": 17})
       
    x_icu = x[ids == i]
    x_icu = x_icu[y[ids ==i] == 0]
    x_mean = np.mean(x_icu, axis=0)
    x_mean = normalize(x_mean, axis=1)[:, feature_indexes]

    axs[1, i-1] = sns.heatmap(x_mean, vmin=0, vmax=1, cmap="Reds", ax=axs[1, i-1], xticklabels=feature_names[feature_indexes], cbar=False)
    axs[1, i-1].set_title(icu_names[i-1] + " ICU - Surviving patients", fontdict={"fontsize": 17})
    
#    x_icu = x[ids == i]
#    x_mean = np.mean(x_icu, axis=0)
#    x_mean = normalize(x_mean, axis=1)
#
#    axs[2, i-1] = sns.heatmap(x_mean, vmin=0, vmax=1, cmap="Reds", ax=axs[2, i-1], xticklabels=feature_names)
#    axs[2, i-1].set_title(icu_names[i-1] + " ICU - All patients")


#plt.tight_layout()
fig.set_size_inches(20, 14)
fig.savefig("heatmap/feature_heatmap.eps", format="eps", orientation="portrait", bbox_inches="tight")

#for i in range(1, 5):
#    for j in range(2):
#        extent = full_extent(axs[j, i-1], pad=0.01).transformed(fig.dpi_scale_trans.inverted())
#        fig.savefig('heatmaps/hm_%s_%d_figure.eps' % (icu_names[i-1], j), bbox_inches=extent)


#%%

mean_pat = []
max_x = []
min_x = []
mean_x = []
for icu in [1,2,3,4]:
    x, y, ids = data.load_icus("60", [icu], data.targets[0])
    x = sequence.pad_sequences(x)
    x = x[:, :, :]
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

#%%
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
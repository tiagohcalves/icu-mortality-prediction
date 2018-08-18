#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 13:31:51 2018

@author: raul
"""

import shap
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from scipy.interpolate import spline
from sklearn import datasets
from sklearn.model_selection import train_test_split

'''
Plot
'''

def visualize(X, y, y_pred, shap_norm, x_offset=5, max_point_size=400, cmap='viridis'):
    #Color map 
    red = (255/255, 0, 82/255)
    blue = (0, 137/255, 231/255)
    violet = (145/255, 85/255, 158/255)
    
    cmap = matplotlib.cm.get_cmap(cmap)

    """
    Basic timeline
    """

    fig, ax = plt.subplots(1, 1)

    #Stile ticks
    fig.autofmt_xdate()
    fig.set_size_inches(18, 16)

    #Hide y axis 
    # ax.set_ylabel((
    #     'Features.\n Diameter proportional to Feature Value'
    #     '\n Color indicates Shap Value (scale z-score)'))

    #ax.yaxis.set_visible(False)
    ax.get_yaxis().set_ticklabels([])

    #Hide spines
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)


    #Leave left/right marging on x-axis
    min_x = X.index.min()
    max_x = X.index.max()
    ax.set_xlim(min_x - x_offset, max_x + 1)

    #Time line
    timeline_y = 0.1
    # ax.set_ylim(timeline_y, 4)

    y_step = .1
    y_true_line = timeline_y + y_step

    #Feature lines    
    for it, f in enumerate(X.columns):
        feature_line_y = y_true_line + ((1 + it) * y_step)
        ax.axhline(
            feature_line_y, 
            linewidth=1,
            c='#CCCCCC')

        ax.annotate(
            '%s' % f,
            (min_x - x_offset, feature_line_y),
            color='black'
        )

    cmap_norm = matplotlib.colors.Normalize(vmin=shap_norm.min().min(), vmax=shap_norm.max().max())
    
    #Plot feature/SHAP
    for it, feature in enumerate(X.columns):
        #y-height
        feature_line_y = y_true_line + ((1 + it) * y_step)
        y_plot = [feature_line_y] * X.shape[0]

        #Color 
        c = cmap(cmap_norm(shap_norm[feature]))

        #Size
        s = X[feature].copy()
        s -= s.min()
        s /= s.max() - s.min()
        # s += 0.01

        scatter = ax.scatter(
            X.index,
            y_plot,
            c=(c),
            s=(s * max_point_size).clip(100, max_point_size),
            alpha=.8)


    xnew = np.linspace(
        ax.get_xlim()[0], 
        ax.get_xlim()[1], 300)

    power_smooth = spline(
        list(X.index), 
        y_pred, 
        xnew)

    y_ = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
    y_ = pd.Series(power_smooth, index=xnew)
    y_.name = 'score'
    (y_ / 4).clip(0).plot(
        kind='area', 
        ax=ax, 
        color="orange"
        )
    
    ax.annotate(
        "Score", 
        (min_x - x_offset, 0.1)
    )
    
    ax.set_xlabel("Hours since admission")
    ax.get_xaxis().set_ticklabels([0, "", 0, 10, 20, 30, 40])
    
    """
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=cmap_norm)
    sm.set_array([])
    
    cbar = plt.colorbar(
        sm,
        ticks=[-0.8, 1],
        aspect=40,
        shrink=0.8
        # orientation="horizontal",
        # pad = 0.1
    )
    cbar.ax.set_yticklabels(['Survival', 'Mortality'], rotation=90)
    cbar.ax.set_xticklabels(['Survival', 'Mortality'])
    """
    
    return ax
    
    
def main():
    data = datasets.load_boston()
    #data = datasets.load_iris()

    X = pd.DataFrame(
        data['data'],
        columns=data.feature_names)

    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        train_size=X.shape[0] - 50)

    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    X = X_test[:47]
    X.index = [
        pd.to_datetime('2010/01/01') + pd.DateOffset(hours=x) 
        for x in range(X.shape[0])]

    y = y_test[:47]

    y_pred = model.predict(X_train)

    tree_expl = shap.TreeExplainer(model)
    shap_val = pd.DataFrame(
        tree_expl.shap_values(X)[:, :-1], 
        columns=X.columns)

    shap_norm = shap_val.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    
    visualize(X, y, y_pred, shap_norm)

    
if __name__ == '__main__':
    main()
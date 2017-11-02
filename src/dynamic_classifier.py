import os

import pandas as pd
import xgboost as xgb
import numpy as np

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score


base_path = os.path.realpath(__file__).split("src")[0]
data_path = base_path + 'data/'


features = ["SPEED", "ACCELERATION", "AVG_SPEED", "AVG_ACCELERATION", "DC0", "DC1"]
target = "DEATH"


def load_raw_data(icu, data_id):
    raw_df = pd.read_csv(data_path + "raw_dyn_features/icu_" + icu + "/features_" + data_id + ".csv", delimiter=",",
                         names=["TIME", "ID", "ICU", "DEATH", "SPEED", "ACCELERATION", "AVG_SPEED",
                                "AVG_ACCELERATION", "DC0", "DC1"])
    raw_x = raw_df[features].as_matrix()
    raw_y = raw_df[target].as_matrix()

    return raw_x, raw_y


def load_neural_data(icu, data_id):
    neural_df = pd.read_csv(data_path + "neural_dyn_features/icu_" + icu + "/features_" + data_id + ".csv", delimiter=";",
                            names=["DEATH", "SPEED", "ACCELERATION", "DC0", "DC1", "AVG_SPEED", "AVG_ACCELERATION"])
    neural_x = neural_df[features].as_matrix()
    neural_y = neural_df[target].as_matrix()

    return neural_x, neural_y


def train(x, y):
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_auc_list = np.zeros(n_folds)
    for fold, (train_idx, test_idx) in enumerate(skf.split(x, y)):
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_test = x[test_idx]
        y_test = y[test_idx]

        classifier = xgb.XGBClassifier()
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
        classifier.fit(x_train, y_train,
                       eval_set=[(x_train, y_train), (x_val, y_val)],
                       eval_metric="auc",
                       early_stopping_rounds=50)

        y_pred_proba = classifier.predict_proba(x_test, ntree_limit=classifier.best_ntree_limit)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        fold_auc_list[fold] = auc

        print("Fold %i - Test AUC: %f" % (fold, auc))

    final_auc = np.mean(fold_auc_list)
    final_auc_var = np.var(fold_auc_list)
    important_features = ",".join([features[ft] for ft in np.argsort(classifier.feature_importances_)])

    return final_auc, final_auc_var, important_features


def main():
    with open("logs/dynamic_classifier.log", "w") as log:
        log.write("Mode,ICU,Time,AUC,Var,Feature_1,Feature_2,Feature_3,Feature_4,Feature_5,Feature_6")
        for icu in range(1, 5):
            for t in range(1, 48):
                try:
                    raw_x, raw_y = load_raw_data(str(icu), str(t))
                    raw_auc, raw_var, important_features = train(raw_x, raw_y)
                    print("RAW - AUC: %f - VAR: %f - Features: %s" % (raw_auc, raw_var, important_features))
                    log.write("RAW,%i,%i,%f,%f,%s\n" % (icu, t, raw_auc, raw_var, important_features))
                except Exception as e:
                    print("ERROR - %s" % e)

        for icu in range(1, 5):
            for t in range(1, 48):
                try:
                    neural_x, neural_y = load_neural_data(str(icu), str(t))
                    neural_auc, neural_var, important_features = train(neural_x, neural_y)
                    print("NEURAL - AUC: %f - VAR: %f - Feature: %s" % (neural_auc, neural_var, important_features))
                    log.write("NEURAL,%i,%i,%f,%f,%s\n" % (icu, t, neural_auc, neural_var, important_features))
                except Exception as e:
                    print("ERROR - %s" % e)


if __name__ == '__main__':
    main()
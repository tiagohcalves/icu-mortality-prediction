import argparse
import json
import models
import os
import numpy as np
import pandas as pd
import prepare_data as data

from keras.preprocessing import sequence
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import normalize

n_epochs = 100
k_fold = 5


def get_args():
    parser = argparse.ArgumentParser(description='Run experiment')
    parser.add_argument('--data_path', '-d', help='Path to pre-processed dataset')
    parser.add_argument('--output_dir', '-o', help='Directory to save output files')
    parser.add_argument('--config_file', '-c', help='Json with experiments configuration')
    args = parser.parse_args()

    data_path = args.data_path
    output_dir = args.output_dir
    config_file = args.config_file

    return data_path, output_dir, config_file


def main():
    data_path, output_dir, config_file = get_args()
    with open(config_file, "r") as config_file_in:
        config_json = json.load(config_file_in)

    global n_epochs
    global k_fold

    all_results_map = {}
    for experiment_id, experiment_config in config_json.items():
        model_list = experiment_config["model_ids"]
        aggregation_list = experiment_config["aggregation_ids"]
        max_patient_per_icu_list = experiment_config["max_patient_per_icu"]
        mode_list = experiment_config["mode_ids"]
        shuffle_list = experiment_config["shuffle"]
        use_target_list = experiment_config["use_target"]
        n_epochs = experiment_config["n_epochs"]
        k_fold = experiment_config["k_fold"]

        os.makedirs(f"models", exist_ok=True)                    
        
        result_map = {}
        for model_id in model_list:
            print(f"=== MODEL: {model_id} ===")
            for data_id in aggregation_list:
                print(f"=== AGGREGATION: {data_id} ===")
                for n_patients in max_patient_per_icu_list:
                    print(f"=== MAX PATIENTS PER ICU: {n_patients} ===")
                    key = model_id + "-" + data_id + "-" + str(n_patients)
                    result_map[key] = {}


                    output_file_name = f"{output_dir}/log-{key}.log"                    
                    with open(output_file_name, "w") as output:
                        output.write("Mode,fold,y_true,y_false,loss,accuracy,precision,recall,fmeasure\n")
                        if "general_training" in mode_list:
                            result_map[key]["general_training"] = general_training(
                                data_path, model_id, data_id, output, n_patients
                            )
                        if "general_training_target_test" in mode_list:
                            result_map[key]["general_training_target_test"] = general_training_target_test(
                                data_path, model_id, data_id, output, n_patients
                            )
                        if "train_on_target" in mode_list:
                            result_map[key]["train_on_target"] = train_on_target(
                                data_path, model_id, data_id, output, n_patients
                            )
                        if "domain_adaptation" in mode_list:
                            for shuffle in shuffle_list:
                                shuffle = bool(shuffle)
                                for use_target in use_target_list:
                                    use_target = bool(use_target)
                                    shuffle_str = "s" if shuffle else "no-s"
                                    target_str = "ut" if shuffle else "no-ut"
                                    result_map[key]["domain_adaptation_%s_%s" % (shuffle_str, target_str)] = domain_adaptation(
                                        data_path, model_id, data_id, output, n_patients, shuffle, use_target, None
                                    )
        all_results_map[experiment_id] = result_map

        print("========= FINAL AUCS FOR EXPERIMENT %s: ==========" % experiment_id)
        for k, mode in result_map.items():
            print("Model: {0}".format(k))
            for t, value in mode.items():
                print("{0};{1}".format(t, value))
            print("\n")

    with open(f"{output_dir}/results.json", "w") as results_file:
        json.dump(all_results_map, results_file)


def general_training_target_test(data_path, model_id, data_id, output, n_patients, target_icu=None):
    """
    Train on all ICU and evaluate on test ICU
    """
    auc = []
    try:
        print("\n\n\n=== General training ===")
        folds = load_data(data_path, data_id, [1, 2, 3, 4], n_patients)
        for fold, (x_train, y_train, x_val, y_val, x_test, y_test, icu_ids_train, icu_ids_val, icu_ids_test) in enumerate(folds):
            layers, freezable = models.create_freezable_layers(model_id, x_train.shape[1], x_train.shape[2])
            checkpoint, early_stopping, model = models.create_model("mt-" + model_id + "-" + data_id, layers)

            model.fit(
                x_train, y_train,
                epochs=n_epochs,
                validation_data=(x_val, y_val),
                callbacks=[early_stopping, checkpoint]
            )

            icu_test_list = [target_icu] if target_icu is not None else np.unique(icu_ids_test)
            icu_auc = []

            score, general_score = models.evaluate_model(
                "mt-" + model_id + "-" + data_id,
                x_test, y_test
            )

            print("\n\n=== General Score - \tFold %d - \tAUC %f" % (fold, general_score))

            for test_icu in icu_test_list:
                score, auc_score = models.evaluate_model(
                    "mt-" + model_id + "-" + data_id,
                    x_test[icu_ids_test == test_icu],
                    y_test[icu_ids_test == test_icu]
                )
                icu_auc.append(auc_score)

                print("\n=== ICU Test %d - \tFold %d - \tAUC %f" % (test_icu, fold, auc_score))

                output.write("Mixed-%d,%d,%d,%d,%f,%f,%f,%f,%f,%f\n" % (
                    test_icu, fold, np.count_nonzero(y_test == 0), np.count_nonzero(y_test == 1),
                    score[0], score[1], score[2], score[3], score[4], auc_score
                ))
            auc.append(icu_auc)

    except Exception as e:
        print(e)

    avg_auc = np.array(auc).mean(axis=0).tolist()
    return avg_auc


def general_training(data_path, model_id, data_id, output, n_patients):
    """
    Train and evaluate on all ICU
    """
    auc = []
    try:
        print("=== General training and testing ===")
        folds = load_data(data_path, data_id, [1, 2, 3, 4], n_patients)
        for fold, (x_train, y_train, x_val, y_val, x_test, y_test, icu_ids_train, icu_ids_val, icu_ids_test) in enumerate(folds):
            layers, freezable = models.create_freezable_layers(model_id, x_train.shape[1], x_train.shape[2])
            checkpoint, early_stopping, model = models.create_model("mtt-" + model_id + "-" + data_id, layers)

            model.fit(x_train, y_train,
                      epochs=n_epochs,
                      validation_data=(x_val, y_val),
                      callbacks=[early_stopping, checkpoint])
            score, auc_score = models.evaluate_model("mtt-" + model_id + "-" + data_id, x_test, y_test)
            auc.append(auc_score)

            print("=== General Test, Fold: " + str(fold) + ",Loss: " + str(score[0]) +
                  ",Acc: " + str(score[1]) + ",Prec: " + str(score[2]) + ",Rec: " + str(score[3]) +
                  ",F1: " + str(score[4]) + ",AUC: " + str(auc_score) + " ====")
            
            output.write("General-Test-%d,%d,%d,%f,%f,%f,%f,%f,%f\n" % (
                    fold, np.count_nonzero(y_test == 0), np.count_nonzero(y_test == 1),
                    score[0], score[1], score[2], score[3], score[4], auc_score
            ))

    except Exception as e:
        print(e)

    avg_auc = np.array(auc).mean().tolist()
    return avg_auc


def train_on_target(data_path, model_id, data_id, output, n_patients):
    """
    Train and evaluate only on target ICU
    """
    auc = []
    try:
        icu_types = [1, 2, 3, 4]

        for held_out_icu in icu_types:
            print("=== Target: " + str(held_out_icu) + " ===")
            folds = load_data(data_path, data_id, [held_out_icu], n_patients)
            fold_auc = []
            for fold, (x_train, y_train, x_val, y_val, x_test, y_test, icu_ids_train, icu_ids_val, icu_ids_test) in enumerate(folds):
                layers, freezable = models.create_freezable_layers(model_id, x_train.shape[1], x_train.shape[2])

                print("=== Train on Target ===")
                checkpoint, early_stopping, model = models.create_model(
                    "fc-" + model_id + "-" + data_id + "-" + str(held_out_icu), layers
                )
                model.fit(x_train, y_train,
                          epochs=n_epochs,
                          validation_data=(x_val, y_val),
                          callbacks=[early_stopping, checkpoint])

                score, auc_score = models.evaluate_model(
                    "fc-" + model_id + "-" + data_id + "-" + str(held_out_icu),
                    x_test, y_test
                )
                fold_auc.append(auc_score)

                print("=== Fold: " + str(fold) + ",Loss: " + str(score[0]) + ",Acc: " + str(score[1]) +
                      ",Prec: " + str(score[2]) + ",Rec: " + str(score[3]) + ",F1: " + str(score[4]) + ",AUC: " + str(
                    auc_score) + " ====")
                
                output.write("Target-%d,%d,%d,%d,%f,%f,%f,%f,%f,%f\n" % (
                    held_out_icu, fold, np.count_nonzero(y_test == 0), np.count_nonzero(y_test == 1),
                    score[0], score[1], score[2], score[3], score[4], auc_score
                ))
            auc.append(fold_auc)
    except Exception as e:
        print(e)

    avg_auc = np.array(auc).mean(axis=1).tolist()
    return avg_auc


def domain_adaptation(data_path, model_id, data_id, output, n_patients, shuffle, use_target, target_icu=None):
    """
    Train on all ICU, fine tunning and evaluate on target ICU
    """
    auc = []
    try:
        icu_types = [1, 2, 3, 4]

        target_icu_list = [target_icu] if target_icu is not None else icu_types
        for held_out_icu in target_icu_list:
            icus = list(icu_types)
            if not use_target:
                icus.remove(held_out_icu)
            folds = load_data(data_path, data_id, icus, n_patients)
            icu_folds = load_data(data_path, data_id, [held_out_icu], n_patients)
            model_name = "ft-" + model_id + "-" + data_id + "-" + str(held_out_icu)

            fold_auc = []
            for fold in range(k_fold):
                x_train, y_train, x_val, y_val, x_test, y_test, icu_ids_train, icu_ids_val, icu_ids_test = folds[fold]

                print("=== Held out: " + str(held_out_icu) + " ===")
                layers, freezable = models.create_freezable_layers(model_id, x_train.shape[1], x_train.shape[2])

                print("=== General training ===")
                checkpoint, early_stopping, model = models.create_model(model_name, layers)
                model.fit(x_train, y_train,
                          epochs=n_epochs,
                          validation_data=(x_val, y_val),
                          callbacks=[early_stopping, checkpoint])
                if use_target:
                    score, auc_score_gt = models.evaluate_model(
                        model_name,
                        x_test[icu_ids_test == held_out_icu],
                        y_test[icu_ids_test == held_out_icu]
                    )
                else:
                    score, auc_score_gt = models.evaluate_model(model_name, x_test, y_test)

                print("=== Fine tuning ===")
                model = models.get_model(model_name)
                # layers = model.layers

                x_icu_train, y_icu_train, x_val, y_val, x_icu_test, y_icu_test, icu_ids_train, icu_ids_val, icu_ids_test = icu_folds[fold]
                checkpoint, early_stopping, new_model = models.create_model(
                    model_name, model.layers, freezable, shuffle
                )

                new_model.fit(
                    x_icu_train, y_icu_train,
                    epochs=n_epochs,
                    validation_data=(x_val, y_val),
                    callbacks=[early_stopping, checkpoint]
                )

                score, auc_score = models.evaluate_model(model_name, x_icu_test, y_icu_test)
                fold_auc.append([auc_score_gt, auc_score])

                print("\n=== ICU %d - \tFold %d - \tAUC GT %f - \tAUC FT %f" % (
                    held_out_icu, fold, auc_score_gt, auc_score)
                )

                y_zero_count = np.count_nonzero(y_icu_test == 0)
                y_one_count = np.count_nonzero(y_icu_test == 1)

                output.write("DA-%d-%s-%s,%d,%d,%d,%f,%f,%f,%f,%f,%f\n" % (
                    held_out_icu, shuffle, use_target, fold,
                    y_zero_count, y_one_count,
                    score[0], score[1], score[2], score[3], score[4], auc_score
                ))
            auc.append(fold_auc)
    except Exception as e:
        print(e)

    avg_auc = np.array(auc).mean(axis=1)

    auc_df = pd.DataFrame(
        avg_auc.mean(axis=1)
    )
    print(auc_df)

    return avg_auc.tolist()


def load_data(data_path, data_id, icus, n_patients):
    x, y, icu_ids = data.load_icus(data_path, data_id, icus, should_normalize=True)

    y = np.asarray(y)
    icu_ids = np.asarray(icu_ids)

    folds = []
    x_train = [[] for i in range(k_fold)]
    y_train = [[] for i in range(k_fold)]
    x_val = [[] for i in range(k_fold)]
    y_val = [[] for i in range(k_fold)]
    x_test = [[] for i in range(k_fold)]
    y_test = [[] for i in range(k_fold)]
    icu_ids_train = [[] for i in range(k_fold)]
    icu_ids_val = [[] for i in range(k_fold)]
    icu_ids_test = [[] for i in range(k_fold)]

    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=33)
    for icu in np.unique(icu_ids):
        x_icu = x[icu_ids == icu]
        y_icu = y[icu_ids == icu]
        icu_ids_icu = icu_ids[icu_ids == icu]

        if n_patients < x_icu.shape[0]:
            x_icu, y_icu = data.subsampling(x_icu, y_icu, n_patients)

        for i, (train_idx, test_idx) in enumerate(skf.split(x_icu, y_icu)):
            x_icu_train, x_icu_val, y_icu_train, y_icu_val, icu_train, icu_val = train_test_split(
                x_icu[train_idx], y_icu[train_idx],
                icu_ids_icu[train_idx],
                test_size=0.2, random_state=42
            )

            x_train[i] += x_icu_train.tolist()
            y_train[i] += y_icu_train.tolist()
            x_val[i] += x_icu_val.tolist()
            y_val[i] += y_icu_val.tolist()
            x_test[i] += x_icu[test_idx].tolist()
            y_test[i] += y_icu[test_idx].tolist()
            icu_ids_train[i] += icu_train.tolist()
            icu_ids_val[i] += icu_val.tolist()
            icu_ids_test[i] += icu_ids_icu[test_idx].tolist()

    for i in range(k_fold):
        x_train[i], y_train[i], icu_ids_train[i] = data.oversample(np.asarray(
            x_train[i]), np.asarray(y_train[i]), np.asarray(icu_ids_train[i]))
        x_val[i] = np.asarray(x_val[i])
        y_val[i] = np.asarray(y_val[i])
        x_test[i] = np.asarray(x_test[i])
        y_test[i] = np.asarray(y_test[i])
        icu_ids_val[i] = np.asarray(icu_ids_val[i])
        icu_ids_test[i] = np.asarray(icu_ids_test[i])

        folds.append((x_train[i], y_train[i], x_val[i], y_val[i], x_test[i],
                      y_test[i], icu_ids_train[i], icu_ids_val[i], icu_ids_test[i]))

    return folds


if __name__ == '__main__':
    main()

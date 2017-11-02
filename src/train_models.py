import prepare_data as data
import models
import numpy as np

# from keras import losses
from sklearn.model_selection import StratifiedKFold, train_test_split
from keras.preprocessing import sequence

n_epochs = 100
k_fold = 5


def main():
    models = ["m3brl2", "m2", "m6", "m3r_freezeall", "m3r_nonfreeze", "m4"]
    for model_id in models:
        print("=== MODEL: " + model_id + " ===")
        # for data_id in ['X', '5', '15', '30', '60']:
        for data_id in ['60']:
            print("=== AGGREGATION: " + data_id + " ===")
            # for n_patients in [288, 575, 874, 1068, 1500]:
            for n_patients in [1500]:
                print("=== PATIENTS PER ICU: " + str(n_patients) + " ===")
                # for i in range(3):
                i = 0
                with open("logs/log-" + model_id + "-" + data_id + "-" + str(n_patients) + "-" + str(i) + "-training.log", "w") as output:
                    output.write("Mode,fold,y_true,y_false,loss,accuracy,precision,recall,fmeasure\n")
                    # mixed(model_id, data_id, output, n_patients)
                    # focused(model_id, data_id, output, n_patients)
                    fine_tunning(model_id, data_id, output, n_patients)


def mixed(model_id, data_id, output, n_patients):
    try:
        print("=== Mixed training ===")
        folds = load_data(data_id, [1, 2, 3, 4], n_patients)
        model_name = "mt-" + model_id + "-" + data_id
        for fold, (x_train, y_train, x_val, y_val, x_test, y_test, icu_ids_test) in enumerate(folds):
            layers, freezable = models.create_freezable_layers(model_id, x_train.shape[1], x_train.shape[2])
            checkpoint, early_stopping, model = models.create_model(model_name, layers)

            model.fit(x_train, y_train,
                      epochs=n_epochs,
                      validation_data=(x_val, y_val),
                      callbacks=[early_stopping, checkpoint])
            for test_icu in np.unique(icu_ids_test):
                score, auc_score = models.evaluate_model(model_name,
                                                         x_test[icu_ids_test == test_icu],
                                                         y_test[icu_ids_test == test_icu])

                print("=== ICU Test: " + str(test_icu) + ",Fold: " + str(fold) + ",Loss: " + str(score[0]) +
                      ",Acc: " + str(score[1]) + ",Prec: " + str(score[2]) + ",Rec: " + str(score[3]) +
                      ",F1: " + str(score[4]) + ",AUC: " + str(auc_score) + " ====")
                output.write("Mixed-" + str(test_icu) + "," +
                             str(fold) + "," +
                             str(np.count_nonzero(y_test == 0))+","+str(np.count_nonzero(y_test == 1))+"," +
                             str(score[0])+","+str(score[1])+","+str(score[2])+","+str(score[3])+"," +
                             str(score[4])+","+str(auc_score)+"\n")

    except Exception as e:
        print("ERRO!!!")
        print(e)


def focused(model_id, data_id, output, n_patients):
    try:
        icu_types = [1, 2, 3, 4]

        for held_out_icu in icu_types:
            print ("=== Target: " + str(held_out_icu) + " ===")
            folds = load_data(data_id, [held_out_icu], n_patients)
            for fold, (x_train, y_train, x_val, y_val, x_test, y_test, icu_ids_test) in enumerate(folds):
                layers, freezable = models.create_freezable_layers(model_id, x_train.shape[1], x_train.shape[2])

                print ("=== Focused training ===")
                checkpoint, early_stopping, model = models.create_model("fc-" + model_id + "-" + data_id + "-" + str(held_out_icu), layers)
                model.fit(x_train, y_train,
                          epochs=n_epochs,
                          validation_data=(x_val, y_val),
                          callbacks=[early_stopping, checkpoint])

                score, auc_score = models.evaluate_model("fc-" + model_id + "-" + data_id + "-" + str(held_out_icu), x_test, y_test)
                print("=== Fold: " + str(fold) + ",Loss: " + str(score[0]) + ",Acc: " + str(score[1]) +
                      ",Prec: " + str(score[2]) + ",Rec: " + str(score[3]) + ",F1: " + str(score[4]) + ",AUC: " + str(auc_score) + " ====")
                output.write("Focused-"+str(held_out_icu)+"," +
                             str(fold) + "," +
                             str(np.count_nonzero(y_test == 0))+","+str(np.count_nonzero(y_test == 1))+"," +
                             str(score[0])+","+str(score[1])+","+str(score[2])+","+str(score[3])+","+str(score[4])+","+str(auc_score)+"\n")
    except Exception as e:
        print("ERRO!!!")
        print(e)


def fine_tunning(model_id, data_id, output, n_patients):
    # print(",".join([str(x) for x in arc]) + " --- " + act)
    try:
        icu_types = [1, 2, 3, 4]

        for held_out_icu in icu_types:
            icus = list(icu_types)
            # icus.remove(held_out_icu)
            folds = load_data(data_id, icus, n_patients)
            icu_folds = load_data(data_id, [held_out_icu], n_patients)
            for fold in range(k_fold):
                x_train, y_train, x_val, y_val, x_test, y_test, icu_ids_test = folds[fold]

                print ("=== Held out: " + str(held_out_icu) + " ===")
                layers, freezable = models.create_freezable_layers(model_id, x_train.shape[1], x_train.shape[2])

                print ("=== General training ===")
                checkpoint, early_stopping, model = models.create_model("ft-" + model_id + "-" + data_id + "-" + str(held_out_icu), layers)
                model.fit(x_train, y_train,
                          epochs=n_epochs,
                          validation_data=(x_val, y_val),
                          callbacks=[early_stopping, checkpoint])
                score, auc_score = models.evaluate_model("ft-" + model_id + "-" + data_id + "-" + str(held_out_icu), x_test, y_test)
                print("Loss: " + str(score[0]) + ",Acc: " + str(score[1]) +
                      ",Prec: " + str(score[2]) + ",Rec: " + str(score[3]) + ",F1: " + str(score[4]) + ",AUC: " + str(auc_score) + " ====")

                print("=== Fine tuning ===")
                x_icu_train, y_icu_train,  x_val, y_val, x_icu_test, y_icu_test, icu_ids_test = icu_folds[fold]
                for l in freezable:
                    layers[l].trainable = False
                # for l in range(len(layers)):
                #     if l not in freezable:
                #         models.shuffle_weights(layers[l])

                checkpoint, early_stopping, model = models.create_model("ft-" + model_id + "-" + data_id + "-" + str(held_out_icu), layers)
                model.fit(x_icu_train, y_icu_train,
                          epochs=n_epochs,
                          validation_data=(x_val, y_val),
                          callbacks=[early_stopping, checkpoint])

                score, auc_score = models.evaluate_model("ft-" + model_id + "-" + data_id + "-" + str(held_out_icu), x_icu_test, y_icu_test)
                print("=== Fold: " + str(fold) + ",Loss: " + str(score[0]) + ",Acc: " + str(score[1]) +
                      ",Prec: " + str(score[2]) + ",Rec: " + str(score[3]) + ",F1: " + str(score[4]) + ",AUC: " + str(auc_score) + " ====")

                output.write("FT-"+str(held_out_icu)+"," +
                             str(fold) + "," +
                             str(np.count_nonzero(y_icu_test == 0))+","+str(np.count_nonzero(y_icu_test == 1))+"," +
                             str(score[0])+","+str(score[1])+","+str(score[2])+","+str(score[3])+","+str(score[4])+","+str(auc_score)+"\n")
    except Exception as e:
        print("ERRO!!!")
        print(e)


def test():
    folds = load_data("60", [1])
    for fold, (x_train, y_train, x_test, y_test, icu_ids_test) in enumerate(folds):
        layers = models.create_freezable_layers("m3r", x_train.shape[1], x_train.shape[2])

        checkpoint, early_stopping, model = models.create_model("0", layers)
        model.fit(x_train, y_train,
                  epochs=10,
                  batch_size=128, validation_data=(x_test, y_test),
                  callbacks=[early_stopping, checkpoint])
        score, auc_score = models.evaluate_model("0", x_test, y_test)
        print("=== Fold: " + str(fold) + ",Loss: " + str(score[0]) + ",Acc: " + str(score[1]) +
              ",Prec: " + str(score[2]) + ",Rec: " + str(score[3]) + ",F1: " + str(score[4]) +
              ",AUC: " + str(auc_score) + " ====")


def load_data(data_id,  icus, n_patients):
    x, y, icu_ids = data.load_icus(data_id, icus, data.targets[0])
    if data_id == 'X':
        x = sequence.pad_sequences(x, padding='pre', maxlen=200)
    else:
        x = sequence.pad_sequences(x, padding='pre')
    # x = x.reshape((x.shape[0], x.shape[1], 36, 1))
    y = np.asarray(y)
    icu_ids = np.asarray(icu_ids)

    folds = []
    x_train = [[] for i in range(k_fold)]
    y_train = [[] for i in range(k_fold)]
    x_val = [[] for i in range(k_fold)]
    y_val = [[] for i in range(k_fold)]
    x_test = [[] for i in range(k_fold)]
    y_test = [[] for i in range(k_fold)]
    icu_ids_test = [[] for i in range(k_fold)]

    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
    for icu in np.unique(icu_ids):
        x_icu = x[icu_ids == icu]
        y_icu = y[icu_ids == icu]
        icu_ids_icu = icu_ids[icu_ids == icu]

        if n_patients < x_icu.shape[0]:
            x_icu, y_icu = subsampling(x_icu, y_icu, n_patients)

        for i, (train_idx, test_idx) in enumerate(skf.split(x_icu, y_icu)):
            x_icu_train, x_icu_val, y_icu_train, y_icu_val = train_test_split(x_icu[train_idx], y_icu[train_idx], test_size=0.2, random_state=42)
            x_train[i] += x_icu_train.tolist()
            y_train[i] += y_icu_train.tolist()
            x_val[i] += x_icu_val.tolist()
            y_val[i] += y_icu_val.tolist()
            x_test[i] += x_icu[test_idx].tolist()
            y_test[i] += y_icu[test_idx].tolist()
            # icu_ids_train = icu_ids[train_idx]
            icu_ids_test[i] += icu_ids_icu[test_idx].tolist()

    for i in range(k_fold):
        x_train[i], y_train[i] = oversample(np.asarray(x_train[i]), np.asarray(y_train[i]))
        x_val[i] = np.asarray(x_val[i])
        y_val[i] = np.asarray(y_val[i])
        x_test[i] = np.asarray(x_test[i])
        y_test[i] = np.asarray(y_test[i])
        icu_ids_test[i] = np.asarray(icu_ids_test[i])

        print("%d - Train - y_true: %d, y_false: %d" % (i, np.count_nonzero(y_train[i] == 0), np.count_nonzero(y_train[i] == 1)))
        print("%d - Test - y_true: %d, y_false: %d" % (i, np.count_nonzero(y_test[i] == 0), np.count_nonzero(y_test[i] == 1)))

        folds.append((x_train[i], y_train[i], x_val[i], y_val[i], x_test[i], y_test[i], icu_ids_test[i]))

    return folds


def subsampling(x, y, n_samples):
    one_ratio = len(y[y == 1])/len(y)

    one_samples = int(n_samples * one_ratio)
    zero_samples = n_samples - one_samples

    random_idx_ones = list(range(y[y == 1].shape[0]))
    np.random.shuffle(random_idx_ones)
    x_ones = x[y == 1][random_idx_ones][:one_samples]
    y_ones = y[y == 1][random_idx_ones][:one_samples]

    random_idx_ones = list(range(y[y == 0].shape[0]))
    np.random.shuffle(random_idx_ones)
    x_zeroes = x[y == 0][random_idx_ones][:zero_samples]
    y_zeroes = y[y == 0][random_idx_ones][:zero_samples]

    xs = np.concatenate((x_ones, x_zeroes))
    ys = np.concatenate((y_ones, y_zeroes))

    return xs, ys


def oversample(x, y):
    unq, unq_idx = np.unique(y, return_inverse=True)
    unq_cnt = np.bincount(unq_idx)
    cnt = np.max(unq_cnt)
    out = np.empty((cnt*len(unq),) + x.shape[1:], x.dtype)
    out_y = np.empty((cnt*len(unq), 1), y.dtype)
    for j in range(len(unq)):
        if np.count_nonzero(unq_idx == j) == cnt:
            indices = unq_idx == j
        else:
            indices = np.random.choice(np.where(unq_idx == j)[0], cnt)
        out[j*cnt:(j+1)*cnt] = x[indices]
        out_y[j*cnt:(j+1)*cnt] = y[indices].reshape(y[indices].shape[0], 1)
    return out, out_y


if __name__ == '__main__':
    main()
    # test()

import os
import numpy as np
import pandas as pd

base_path = os.path.realpath(__file__).split("src")[0]
data_path = base_path + "data/"


targets = ['In-hospital_death', 'Out-hospital_death_30', 'Out-hospital_death', 'Long_stay']


def load_patient_features(path):
    patient_features = pd.read_csv(path, sep=",")
    return patient_features


def load_patient_sequences(target, patients_features, path):
    sequences = []
    outcomes = []
    for i, f in enumerate(os.listdir(path)):
        patient_id = f.split(".")[0]
        try:
            patient_sequence, patient_outcome = get_patient_features(path, patient_id, patients_features, target)

            sequences.append(patient_sequence)
            outcomes.append(patient_outcome[0].tolist())
        except Exception as e:
            print("Falha para o id " + str(patient_id))
            print(e)
    return sequences, outcomes


def get_patient_features(path, patient_id, patients_features, target):
    n_cols = 40
    usefull_cols = list(range(3, n_cols))
    # usefull_cols = [11,12,17,20,21,27,31,32,33,34]  # medicao continua
    # usefull_cols = [17, 18, 20, 22, 23, 34]  # 6 mais frequentes
    # usefull_cols = [11, 12, 13, 16, 17, 18, 20, 21, 24, 25, 26, 27, 28, 29, 31, 33, 34, 37, 39]  # 19 mais frequentes

    patient_seq = np.loadtxt(path + "/" + patient_id + ".csv", delimiter=",", usecols=usefull_cols, skiprows=1)
    if patient_id == '132547':
        print(patient_seq)

    patient_features = patients_features[patients_features['ID'] == int(patient_id)]
    patient_outcome = patient_features[target].as_matrix()

    main_features = ['Age', 'Gender', 'Height', 'SAPS-I', 'SOFA']
    patient_main_features = patient_features[main_features].as_matrix()
    sequences_main_features = np.ones((patient_seq.shape[0], len(main_features))) * patient_main_features
    patient_sequence = np.hstack((sequences_main_features, patient_seq))

    return patient_sequence, patient_outcome


def load_patient(patient_id, data_id, icu, target):
    patient_features = load_patient_features(data_path + "patient_features.csv")
    x, y = get_patient_features(data_path + "data_" + data_id + "/icu_" + str(icu), patient_id, patient_features, target)

    return x, y


def load_icus(data_id, icu_list, target):
    patient_features = load_patient_features(data_path + "patient_features.csv")
    sequences = []
    outcomes = []
    icus = []
    for i in icu_list:
        s, o = load_patient_sequences(target, patient_features, data_path + "data_" + data_id + "/icu_" + str(i))
        sequences += s
        outcomes += o
        icus += ([i] * len(s))

    return sequences, outcomes, icus


def main():
    # pf = load_patient_features("/media/Dados/Documentos/Mestrado/Pesquisa/Medical/data/patient_features.csv")
    # seq, out = load_patient_sequences(pf, "/media/Dados/Documentos/Mestrado/Pesquisa/Medical/data/icu_1")

    load_icus("60", [1], targets[0])


if __name__ == '__main__':
    main()

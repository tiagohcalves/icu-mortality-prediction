import os
import argparse
import pickle as pk
from tqdm import tqdm

import pandas as pd
import numpy as np

def get_time_from_stamp(timestamp, agg_time):
    minutes = int(timestamp.split(":")[0]) * 60 + int(timestamp.split(":")[1])
    return int((minutes - (minutes % agg_time)) / agg_time)


def get_pat_matrix(path, outcomes_df, seq_len, agg_time):
    df = pd.read_csv(path)
    
    record_id = int(df.query("Parameter == 'RecordID'")["Value"].values[0])
    icu_id = int(df.query("Parameter == 'ICUType'")["Value"].values[0])
    weight = int(df.query("Parameter == 'Weight'")["Value"].values[0])
    height = int(df.query("Parameter == 'Height'")["Value"].values[0])
    gender = int(df.query("Parameter == 'Gender'")["Value"].values[0])
    
    df["Time"] = df.Time.apply(
        lambda t: get_time_from_stamp(t, agg_time)
    )
    
    pat_matrix = (
        df.groupby(["Time", "Parameter"])["Value"]
        .max()
        .unstack()
        .interpolate()
        # .fillna(method='ffill')
        .fillna(method='bfill')
        .reindex(range(seq_len), axis='index', method='ffill', tolerance=999)
    )
    pat_matrix = pat_matrix.drop("RecordID", axis=1)
    
    if weight == -1:
        pat_matrix = pat_matrix.drop("Weight", axis=1)
        
    if height == -1:
        pat_matrix = pat_matrix.drop("Height", axis=1)
        
    pat_matrix["Gender_is_0"] = 1 if gender == 0 else 0
    pat_matrix["Gender_is_1"] = 1 if gender == 1 else 0
    
    outcome = outcomes_df.query("RecordID == @record_id")
    pat_matrix["SAPS-I"] = outcome["SAPS-I"].values[0]
    pat_matrix["SOFA"] = outcome["SOFA"].values[0]
    
    for i in range(1, 5):
        pat_matrix[("ICUType_is_%d" % i)] = 1 if i == icu_id else 0
    
    pat_matrix = pat_matrix.drop("ICUType", axis=1)
    pat_matrix = pat_matrix.drop("Gender", axis=1)
    
    for col in pat_matrix:
        if pat_matrix[col].isnull().values.any():
            print(str(record_id) + " Dropping " + col)
            pat_matrix = pat_matrix.drop(col, axis=1)
    
    y = outcome["In-hospital_death"].values[0]
    return pat_matrix, record_id, icu_id, y


def main():
    parser = argparse.ArgumentParser(description='Create dataset with time aggregation')
    parser.add_argument('--mimic_dir', '-d', help='Path to Physionet\'s set-a directory')
    parser.add_argument('--outcome_file', '-c', help='Path to Physionet\'s outcome file')
    parser.add_argument('--output_dir', '-o', help='Directory to save output files')
    parser.add_argument('--agg_times', '-t', type=int, nargs='+', help='Minutes to aggregate the time series')
    args = parser.parse_args()
    
    mimic_dir = args.mimic_dir
    outcome_file = args.outcome_file
    output_dir = args.output_dir
    agg_times = args.agg_times

    outcomes_df = pd.read_csv(outcome_file)
    pat_paths = os.listdir(mimic_dir)

    print("Agg times: %s" % agg_times)
    for agg_time in agg_times:
        print("=== AGGREGATING IN %d MINUTES ===\n\n" % agg_time)
        
        max_seq = int(48 * 60/agg_time)
        min_seq = int(40 * 60/agg_time)
        
        print("Loading patients...")
        pat_map = {}
        for i, path in enumerate(tqdm(pat_paths)):
            pat_matrix, pid, icu_id, y = get_pat_matrix(mimic_dir + path, outcomes_df, seq_len=max_seq, agg_time=agg_time)
            pat_map[pid] = [icu_id, pat_matrix, y]

        all_columns = set()
        column_avg = {
            1: {}, 2: {}, 3: {}, 4: {}
        }

        print("Calculating feature average values...")
        for p_id, p_item in tqdm(pat_map.items()):
            icu_id, pat_matrix, y = p_item
            all_columns = all_columns.union(set(list(pat_matrix)))
            for col in pat_matrix:
                if col not in column_avg[icu_id]:
                    column_avg[icu_id][col] = {}
                for hour, value in pat_matrix[col].iteritems():
                    if hour not in column_avg[icu_id][col]:
                        column_avg[icu_id][col][hour] = {"total": 0, "count": 0}
                    column_avg[icu_id][col][hour]["total"] += value
                    column_avg[icu_id][col][hour]["count"] += 1

        print("Applying feature average values...")
        for icu, cols in tqdm(column_avg.items()):
            for col, hours in cols.items():
                series = np.zeros(max_seq)
                for hour in range(max_seq):
                    series[hour] = column_avg[icu][col][hour]["total"] / column_avg[icu][col][hour]["count"]

                column_avg[icu][col]["avg"] = series

        print("Filling missing columns...")
        for p_id, p_item in tqdm(pat_map.items()):
            icu_id, pat_matrix, y = p_item

            for col in all_columns:
                if col not in pat_matrix:
                    pat_map[p_id][1][col] = column_avg[icu_id][col]['avg']
        
        print("Initializing ICU tensors...")
        x_count = [0] * 4
        for p_id, p_item in tqdm(pat_map.items()):
            icu_id, pat_matrix, y = p_item
            if len(pat_matrix) != max_seq:
                continue

            x_count[icu_id-1] += 1
            n_seq = pat_matrix.shape[0]
            n_feat = pat_matrix.shape[1]

        X = [
            np.zeros((x_count[0], n_seq, n_feat)),
            np.zeros((x_count[1], n_seq, n_feat)),
            np.zeros((x_count[2], n_seq, n_feat)),
            np.zeros((x_count[3], n_seq, n_feat)),
        ]
        y = [
            np.zeros(x_count[0]), 
            np.zeros(x_count[1]), 
            np.zeros(x_count[2]), 
            np.zeros(x_count[3])
        ]

        icu_counter = [0] * 4

        print("Filling tensors...")
        columns_order = sorted(list(all_columns))
        for p_id, p_item in tqdm(pat_map.items()):
            icu_id, pat_matrix, pat_y = p_item
            if len(pat_matrix) != max_seq:
                continue

            i = icu_counter[icu_id-1]
            X[icu_id-1][i] = pat_matrix[columns_order].values
            y[icu_id-1][i] = pat_y

            icu_counter[icu_id-1] += 1

        print("Saving Data...")
        for i, (X_icu, y_icu) in enumerate(tqdm(zip(X, y))):
            os.makedirs(output_dir + "%d" % agg_time, exist_ok=True)
            pk.dump(
                (X_icu, y_icu), 
                open(output_dir + "%d/icu_%d.pk" % (agg_time, (i+1)), "wb")
            )

        pk.dump(
            column_avg, open(output_dir + "%d/column_avg.pk" % (agg_time), "wb")
        )

        print("Done.")
    
    
main()
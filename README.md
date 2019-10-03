# ICU Mortality Prediction

## Download the data:
The data for this project comes from the PhysioNet challenge, found on the link:
https://physionet.org/challenge/2012/

### [Patient time series](https://archive.physionet.org/challenge/2012/set-a/)

### [Patient outcomes](https://archive.physionet.org/challenge/2012/Outcomes-a.txt)

## Building the dataset:

```
cd data
python create_dataset.py --mimic_dir path/to/data/set-a/ --outcome_file path/to/data/Outcomes-a.txt --output_dir directory/to/save/output/ --agg_times 15 30 60
```

## Running experiments:

```
cd experiments
python conv_lstm.py --data_path path/to/data/ --output_dir path/to/save/output --config_file config/experiments_01.json
```

## Publications

Master Thesis: /publications/ufmg-master-thesis.pdf

Big Data 2018: /publications/bigdata2018.pdf

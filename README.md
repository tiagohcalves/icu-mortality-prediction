# ICU Mortality Prediction

1. Download the data:
The data for this project comes from the PhysioNet challenge, found on the link:
https://physionet.org/challenge/2012/

1.1 [Patient time series](https://physionet.org/challenge/2012/set-a/)

1.2 [Patient outcomes](https://physionet.org/challenge/2012/Outcomes-a.txt)

2. Building the dataset:

```
cd data
python create_dataset.py --mimic_dir path/to/data/set-a/ --outcome_file path/to/data/Outcomes-a.txt --output_dir directory/to/save/output/ --agg_times 15 30 60
```

3. Running experiments:

```
cd experiments
python conv_lstm.py --data_path path/to/data/ --output_dir path/to/save/output --config_file config/experiments_01.json
```

## Publications

Master Thesis: /publications/ufmg-master-thesis.pdf

Big Data 2018: /publications/bigdata2018.pdf

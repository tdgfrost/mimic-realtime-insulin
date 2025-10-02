# mimic-realtime-insulin
Open-source code for reproducing the MIMIC RealTime Insulin dataset for healthcare reinforcement learning.

## Pre-requisites
Package management is done using [Pixi](https://pixi.sh/latest/) and can be reinstalled using the provided pixi.toml file. Alternatively, the environment can be reproduced by manually (uv) pip installing the requirements found within the pixi.toml file. When generated, the entire set of data files takes up less than 2.5GB (excluding the original MIMIC-IV files, which take up an additional 4GB).

## Steps
1. Clone the repo
2. With the repo as your working directory, run `python convert_to_parquet.py --path PATH_TO_MIMIC_FOLDER`, which will convert the .csv.gz files to the .parquet format. The MIMIC folder is expected to be laid out exactly as found in PhysioNet. Depending on the speed of your computer, this may take a while to run (ESPECIALLY for icu/chartevents), but will be faster and more space-efficient afterwards. You should now have a `data` folder containing the .parquet files.
3. Run `python generate_dataset.py` to reproduce the dataset. See optional arguments to tune this. This generates the encoded input data and labels for inspection.
4. Run `python convert_to_training_data.py` to convert into (NaN right-padded) sequential training data/labels in .hdf5 format, which can be easily converted for use in RNN inference. The original .parquet version of the dataset is also made available. See optional arguments to tune this script.

## Optional arguments
`generate_dataset.py` accepts the following arguments:
- `--train_test_split`, the proportion of data allocated to training vs val/test (default is 0.8 i.e., 80% training, 10% validation, 10% testing)
- `--eligible_input_window`, the window of eligibility for input data in hours prior to the label (default is 168 hours i.e., 7 days)
- `--delay`, the delay (in minutes) between adjacent labelled states during training *in addition to* the mandatory offset of +5 minutes.

`convert_to_training_data.py` accepts the following arguments:
- `--context`, the maximum number of events included in the input, with NaN right-padding (default is 400)

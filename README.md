# mimic-realtime-insulin
Open-source code for reproducing the MIMIC RealTime Insulin dataset for healthcare reinforcement learning.

## Pre-requisites
Package management is done using [Pixi](https://pixi.sh/latest/) and can be reinstalled using the provided pixi.toml file. Alternatively, the environment can be reproduced by manually (uv) pip installing the requirements found within the pixi.toml file.

## Steps
1. Clone the repo
2. With the repo as your working directory, run `python convert_to_parquet.py --path PATH_TO_MIMIC_FOLDER`, which will convert the .csv.gz files to the .parquet format. The MIMIC folder is expected to be laid out exactly as found in PhysioNet. Depending on the speed of your computer, this may take a while to run (ESPECIALLY for icu/chartevents), but will be faster and more space-efficient afterwards. You should now have a `data` folder containing the .parquet files.
3. Run `python generate_dataset.py` to reproduce the dataset.

## Optional arguments
`generate_dataset.py` accepts the following arguments:
- `--input_window`, the eligible retrospective window in hours for input data to be included (default is 168 hours i.e., 7 days)
- `--context`, the maximum number of events included in the input, with NaN right-padding (default is 400)
- `--delay`, the delay (in minutes) between adjacent labelled states during training *in addition to* the mandatory offset of +5 minutes.

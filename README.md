# Insulin4RL
Open-source code for reproducing the Insulin4RL dataset for healthcare-based offline reinforcement learning.

## Pre-requisites
Package management is done using [Pixi](https://pixi.sh/latest/) and can be reinstalled using the provided pixi.toml file. Alternatively, the environment can be reproduced by manually (uv) pip installing the requirements found within the pixi.toml file. When generated, the entire set of insulin4rl files takes up 4.16GB (excluding the .parquet MIMIC-IV files, which take up an additional 4.2GB).

## Steps
1. Clone the repo
2. With the repo as your working directory, run `python convert_to_parquet.py --path PATH_TO_MIMIC_FOLDER`, which will convert the .csv.gz files to the .parquet format. The MIMIC folder is expected to be laid out exactly as found in PhysioNet. Depending on the speed of your computer, this may take a while to run (ESPECIALLY for icu/chartevents), but will be faster and more space-efficient afterwards. You should now have a `data` folder containing the .parquet files.
3. Run `pixi run python generate_data.py` to generate the all_data.parquet and SafeTensor binaries, as well as any metadata (e.g., feature mapping, outlier statistics, normalisation constants). See optional arguments to tune this.

## Optional arguments
`generate_data.py` accepts the following arguments:
- `--train_test_split`, the proportion of data allocated to training vs val/test (default is 0.8 i.e., 80% training, 10% validation, 10% testing)
- `--eligible_input_window`, the window of eligibility for input data in hours prior to the label (default is 168 hours i.e., 7 days)
- `--context`, the maximum number of events to include in the input, with NaN right-padding (default is 400)
- `--low_memory`, add this flag if you are getting out-of-memory issues with this script.

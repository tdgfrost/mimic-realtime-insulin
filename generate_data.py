import yaml
from utils.preprocessing.tools import *

DEFAULT_SPLIT = 0.8

# Define the argument parser and possible arguments globally
parser = argparse.ArgumentParser()
parser.add_argument('--train_test_split', type=float, default=DEFAULT_SPLIT)
parser.add_argument('--eligible_input_window', type=int, default=168,
                    help="Specify the window of eligibility for input data (in hours prior to the label) "
                         "- default is 168 hours i.e., 7 days")
parser.add_argument('--context', type=int, default=400,
                    help="Specify the context window size for the model - default is 400")
parser.add_argument('--low_memory', action='store_true')


def build_dataset(input_window_size=168, inclusion_hours=24, train_test_split=0.8, context_length=400, low_memory=False
                  ) -> None:
    """
    Generates the data.parquet and labels.parquet files.

    1. Loads MIMIC-IV with the relevant variables + patient train/val/test split
    2. Cleans the data, including removing outliers based on the train cohort percentiles
    3. Identifies decision labels
    4. Gets the scaling for the data (again, based on the train cohort)
    5. Saves final data
    """
    intermediate_path = './data/insulin4rl/_intermediate_data'
    metadata_path = './data/insulin4rl/metadata'
    os.makedirs(intermediate_path, exist_ok=True)
    os.makedirs(metadata_path, exist_ok=True)

    # Get our dict of target variables
    target_variables = get_variable_names_for_mimic()
    nutrition_variables = get_nutrition_variables_for_mimic()

    target_variables.update(nutrition_variables)

    # Get the required DataFrames
    announce_progress('Loading the data...')
    (admissions, combined_data, patients,
     (train_patient_ids, val_patient_ids, test_patient_ids)) = load_mimic(variable_names=target_variables,
                                                                          train_test_split=train_test_split)

    # Do the cleaning steps
    announce_progress('Cleaning the data...')
    combined_data = clean_combined_data(combined_data=combined_data,
                                        train_patient_ids=train_patient_ids)

    # Get our labels DataFrame
    announce_progress('Creating the labels...')
    (labels,
     train_patient_ids,
     val_patient_ids,
     test_patient_ids) = create_glucose_labels_for_mimic(combined_data=combined_data,
                                                         admissions=admissions,
                                                         patients=patients,
                                                         input_window_size=input_window_size,
                                                         inclusion_hours=inclusion_hours,
                                                         train_patient_ids=train_patient_ids,
                                                         val_patient_ids=val_patient_ids,
                                                         test_patient_ids=test_patient_ids)

    # Rebalance the strata
    train_patient_ids, val_patient_ids, test_patient_ids = rebalance_to_train(labels, train_patient_ids,
                                                                              val_patient_ids, test_patient_ids)

    # Save the patient IDs
    save_patient_ids(train_patient_ids=train_patient_ids,
                     val_patient_ids=val_patient_ids,
                     test_patient_ids=test_patient_ids)

    # Encode the combined_data DataFrame
    (encoded_input_data, features,
     feature_encoding, encodings) = encode_combined_data_for_mimic(combined_data=combined_data)


    # Save all our data thus far
    # - intermediate data (not intended for further use by user)
    feature_encoding_path = os.path.join(intermediate_path, 'feature_encoding.parquet')
    encoded_input_path = os.path.join(intermediate_path, 'encoded_input_data.parquet')
    label_path = os.path.join(intermediate_path, 'labels.parquet')
    encodings_path = os.path.join(intermediate_path, 'encodings.yaml')

    feature_encoding.write_parquet(feature_encoding_path)
    encoded_input_data.write_parquet(encoded_input_path)
    labels.write_parquet(label_path)
    with open(encodings_path, 'w') as f:
        yaml.safe_dump(encodings, f, default_flow_style=False)

    # - metadata data (intended for use by user)
    feature_mapping_path = os.path.join(metadata_path, 'feature_mapping.yaml')
    label_features_path = os.path.join(metadata_path, 'label_features.yaml')

    feature_mapping = dict(zip(feature_encoding["feature"], feature_encoding["str_feature"]))
    with open(feature_mapping_path, 'w') as f:
        yaml.safe_dump(feature_mapping, f, sort_keys=False)

    label_features = [
        # Unique identifiers
        'episode_num', 'step_num', 'label_id', 'label_id_next',

        # Temporal context
        'steps_per_episode', 'steps_remaining', 'minutes_remaining', 'is_done',

        # Physiological state
        'current_bm', 'prev_bm', 'time_since_prev_bm', 'next_bm', 'time_until_next_bm',

        # Intervention (insulin)
        'insulin_old_rate', 'insulin_new_rate', 'insulin_maintain', 'insulin_change', 'insulin_stop',
        'insulin_delta_change', 'insulin_maintain_prev', 'insulin_change_prev', 'insulin_stop_prev',
        'insulin_delta_change_prev', 'insulin_maintain_next', 'insulin_change_next', 'insulin_stop_next',
        'insulin_delta_change_next',

        # Mortality outcomes
        '1-day-alive', '1-day-alive-final', '3-day-alive', '3-day-alive-final', '7-day-alive', '7-day-alive-final',
        '14-day-alive', '14-day-alive-final', '28-day-alive', '28-day-alive-final'
    ]

    with open(label_features_path, 'w') as f:
        yaml.safe_dump(label_features, f, default_flow_style=False)

    # Load our input data and labels lazily
    encoded_input_data = pl.scan_parquet(encoded_input_path).drop('str_feature')
    labels = pl.scan_parquet(label_path)

    # Create the finalised input data and label data
    announce_progress('Constructing and saving final dataframe...')
    create_final_dataframe(encoded_input_data=encoded_input_data,
                           labels=labels,
                           encodings=encodings,
                           train_patient_ids=train_patient_ids,
                           val_patient_ids=val_patient_ids,
                           test_patient_ids=test_patient_ids,
                           sorting_columns=['subject_id', 'episode_num', 'step_num'],
                           grouping_columns=['subject_id', 'episode_num', 'step_num', 'targets'],
                           context_length=context_length,
                           low_memory=low_memory)

    # Save our feature normalisation stats as a YAML file
    announce_progress('Getting scaling data...')
    normalisation_stats_dict = create_scaling_dict()

    normalisation_dict_path = os.path.join(metadata_path, 'feature_stats.yaml')
    with open(normalisation_dict_path, 'w') as f:
        yaml.safe_dump(normalisation_stats_dict, f, sort_keys=False, default_flow_style=False)

    # We want to add another pytorch binary - this one normalises states/next_states
    announce_progress('Converting the data to SafeTensors...')
    convert_dataframe_to_mmap_safetensors(parquet_path='./data/insulin4rl/all_data.parquet',
                                          output_dir='./data/insulin4rl')

    # Create the demographics .parquet file
    announce_progress("Collecting demographics data...")
    build_demographics()

    # Clean up the intermediate folder
    shutil.rmtree('./data/insulin4rl/_intermediate_data')
    announce_progress('Done!')


if __name__ == "__main__":
    # Parse the command-line arguments inside the block
    args = parser.parse_args()

    if args.train_test_split != DEFAULT_SPLIT:
        raise UserWarning("Custom split entered - please be aware that this will be an approximate split due to"
                          "stratified sampling (to ensure balanced mortality/gender in each segment).")

    # Execute the conversion function
    build_dataset(input_window_size=args.eligible_input_window,
                  inclusion_hours=24,
                  train_test_split=args.train_test_split,
                  context_length=args.context,
                  low_memory=args.low_memory)

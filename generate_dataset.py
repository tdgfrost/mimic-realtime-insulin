from utils.preprocessing.tools import *
import json

# Define the argument parser and possible arguments globally
parser = argparse.ArgumentParser()
parser.add_argument('--train_test_split', type=float, default=0.8)
parser.add_argument('--eligible_input_window', type=int, default=168,
                    help="Specify the window of eligibility for input data (in hours prior to the label) "
                         "- default is 168 hours i.e., 7 days")
parser.add_argument('--delay', default=0,
                    help="Specify the delay (in minutes) between states during training - "
                         "default is 0 minutes.")


def build_mimic_data(input_window_size=168, next_state_delay=15, next_state_window=24*60, inclusion_hours=24,
                     train_test_split=0.8):
    # Get our dict of target variables
    target_variables = get_variable_names_for_mimic()  # <- returns a dict
    nutrition_variables = get_nutrition_variables_for_mimic()  # <- returns a list, needs to be mapped key:key

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
                                                         state_delay=next_state_delay,
                                                         next_state_window=next_state_window,
                                                         inclusion_hours=inclusion_hours,
                                                         train_patient_ids=train_patient_ids,
                                                         val_patient_ids=val_patient_ids,
                                                         test_patient_ids=test_patient_ids)

    # Save the patient IDs
    save_patient_ids(train_patient_ids=train_patient_ids,
                     val_patient_ids=val_patient_ids,
                     test_patient_ids=test_patient_ids)

    # Encode the combined_data DataFrame
    (encoded_input_data, features,
     feature_encoding, encodings) = encode_combined_data_for_mimic(combined_data=combined_data)

    # Get the delta value/time for the encoded input data
    encoded_input_data = get_delta_value_time(encoded_input_data=encoded_input_data,
                                              encodings=encodings)

    # Get the scaling data
    scaling_data = get_scaling_data_for_mimic(encoded_input_data=encoded_input_data,
                                              labels=labels,
                                              train_patient_ids=train_patient_ids,
                                              input_window_size=input_window_size,
                                              age_encoding=encodings['age'],
                                              gender_encoding=encodings['gender'],
                                              weight_encoding=encodings['weight'])

    # All labels should be given a unique index (different from id_num) specific to the train/val/test split
    train_labels = labels.filter(pl.col('subject_id').is_in(train_patient_ids))
    val_labels = labels.filter(pl.col('subject_id').is_in(val_patient_ids))
    test_labels = labels.filter(pl.col('subject_id').is_in(test_patient_ids))
    labels = labels.with_row_index().head(0)
    for current_label, current_idxs in [
        (train_labels, train_patient_ids),
        (val_labels, val_patient_ids),
        (test_labels, test_patient_ids)
    ]:
        labels = (
            pl.concat([labels,
                       (
                           current_label
                           .filter(pl.col('subject_id').is_in(current_idxs))
                           .sort(['subject_id', 'episode_num', 'step_num'])
                           .with_row_index()
                       )], how='diagonal')
        )

    # Save all our data thus far
    combined_data.write_parquet('./data/mimic/combined_data.parquet')
    encoded_input_data.write_parquet('./data/mimic/encoded_input_data.parquet')
    feature_encoding.write_parquet('./data/feature_encoding.parquet')
    labels.write_parquet('./data/mimic/labels.parquet')
    scaling_data.write_parquet('./data/scaling_data.parquet')
    with open('./data/encodings.json', 'w') as f:
        json.dump(encodings, f)
        f.close()

    with open('./data/features.txt', 'w') as f:
        for feature in features:
            if feature not in ['subject_id', 'charttime']:
                f.write(feature + '\n') if feature != features[-1] else f.write(feature)
        f.close()

    label_features = ['episode_num', 'step_num', 'label_id_num', 'label_id_num_next', 'steps_per_episode', 'steps_remaining',
                      'minutes_remaining', 'is_first_state', 'is_last_state', 'current_bm', 'prev_bm',
                      'time_since_prev_bm', 'bm_next', 'time_until_bm_next', 'n_future_hypers', 'insulin_default_rate',
                      'insulin_new_rate', 'insulin_maintain', 'insulin_change', 'insulin_stop', 'insulin_delta_change',
                      'insulin_maintain_prev', 'insulin_change_prev', 'insulin_stop_prev', 'insulin_delta_change_prev',
                      'insulin_maintain_next', 'insulin_change_next', 'insulin_stop_next', 'insulin_delta_change_next',
                      '1-day-alive', '1-day-alive-final', '3-day-alive', '3-day-alive-final', '7-day-alive',
                      '7-day-alive-final', '14-day-alive', '14-day-alive-final', '28-day-alive', '28-day-alive-final']

    with open('./data/label_features.txt', 'w') as f:
        for label_feature in label_features:
            f.write(label_feature + '\n') if label_feature != label_features[-1] else f.write(label_feature)
        f.close()


if __name__ == "__main__":
    # Parse the command-line arguments inside the block
    args = parser.parse_args()
    args.delay = parse_delay(args.delay)

    # Execute the conversion function
    build_mimic_data(input_window_size=args.eligible_input_window,
                     next_state_delay=args.delay,
                     next_state_window=24*60,
                     inclusion_hours=24,
                     train_test_split=args.train_test_split)

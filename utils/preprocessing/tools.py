import polars as pl
import numpy as np
import h5py

import os
import pickle
import sys
import time
import argparse
import yaml

pl.Config.set_tbl_cols(10000)
pl.Config.set_tbl_width_chars(10000)
pl.Config.set_tbl_rows(50)
pl.Config.set_fmt_str_lengths(10000)


# ========================================================================== #
# This contains all the processing functions for generating the datasets.
# The functions are sorted in alphabetical order.
# ========================================================================== #


def announce_progress(announcement):
    """
        Print-out of update statements

        :param announcement: String to be printed
        :return: None
    """

    print('\n', '=' * 50, '\n')
    print(' ' * 2, announcement)
    print('\n', '=' * 50, '\n')


def change_antibiotic_doses(df):
    """
    Remove the dose value for antibiotics - just treat as binary event (they are on 'x' antibiotic or not).
    """

    antibiotics = get_antibiotic_names()
    current_labels = df.select('label').unique().to_series().to_list()
    for antibiotic in antibiotics:
        if antibiotic in current_labels:
            criteria = pl.col('label') == antibiotic
            df = (
                df
                .with_columns([
                    # Set bolus to 1.0 for antibiotics
                    pl.when(criteria)
                    .then(pl.lit(1.0).alias('amount'))
                    .otherwise(pl.col('amount')),

                    pl.when(criteria)
                    .then(pl.lit('dose').alias('amountuom'))
                    .otherwise(pl.col('amountuom')),

                    # Set endtime to null
                    pl.when(criteria)
                    .then(pl.lit(None).alias('endtime'))
                    .otherwise(pl.col('endtime')),

                    # Set Bolus / IsSingleDose to 1
                    pl.when(criteria)
                    .then(pl.lit('Bolus').alias('ordercategorydescription'))
                    .otherwise(pl.col('ordercategorydescription')),
                ])
            )
    return df


def change_drug_units(df, variables, unit_str, old_albumin):
    for variable in variables.keys():
        default_unit = variables[variable]['label']
        is_default_unit = pl.col(unit_str) == default_unit
        is_variable = pl.col('label') == variable

        for unit in variables[variable]['convert'].keys():
            is_current_unit = pl.col(unit_str) == unit
            criteria = is_variable & is_current_unit

            df = (
                df.with_columns([
                    pl.when(criteria)
                    .then(pl.col('amount') * variables[variable]['convert'][unit])
                    .otherwise(pl.col('amount')),

                    pl.when(criteria)
                    .then(pl.lit(default_unit).alias(unit_str))
                    .otherwise(pl.col(unit_str))
                ])
            )

        df = df.filter((is_variable & is_default_unit) | is_variable.not_())

    df = (
        df.with_columns([pl.col('label').replace(old_albumin, 'Human Albumin Solution')])
    )

    return df


def change_drug_units_for_mimic(df):
    """
    Where there is inconsistency, group all drug units into a single unit.
    """

    variables = {
        'Adrenaline':
            {'convert': {'mg': 1000, 'mcg': 1}, 'label': 'mcg'},
        'Amino Acids':
            {'convert': {'grams': 1}, 'label': 'grams'},
        'Beneprotein':
            {'convert': {'grams': 1}, 'label': 'grams'},  # Filter out the rest of the units (small numbers)
        'Cisatracurium':
            {'convert': {'mg': 1000, 'mcg': 1}, 'label': 'mcg'},
        'Dexmedetomidine':
            {'convert': {'mg': 1000, 'mcg': 1}, 'label': 'mcg'},
        'Dextrose PN':
            {'convert': {'grams': 1}, 'label': 'grams'},
        'Fentanyl':
            {'convert': {'mg': 1000, 'mcg': 1}, 'label': 'mcg'},
        'Furosemide':
            {'convert': {'mg': 1}, 'label': 'mg'},
        'Human Albumin Solution 25%':
            {'convert': {'ml': 0.25}, 'label': 'grams'},  # this is a special case where we wish to convert to grams
        'Ketamine':
            {'convert': {'mcg': 1 / 1000, 'mg': 1}, 'label': 'mg'},
        'Labetalol':
            {'convert': {'mg': 1}, 'label': 'mg'},
        'Levetiracetam':
            {'convert': {'grams': 1000, 'mcg': 1 / 1000, 'mg': 1}, 'label': 'mg'},
        'Phenytoin':
            {'convert': {'grams': 1000, 'mg': 1}, 'label': 'mg'},
        'Promote with Fiber (Full)':
            {'convert': {'mL': 1, 'L': 1000}, 'label': 'mL'},
        'Propofol':
            {'convert': {'mg': 1, 'mcg': 1 / 1000}, 'label': 'mg'},
        'Rocuronium':
            {'convert': {'mcg': 1 / 1000, 'mg': 1}, 'label': 'mg'},
        'Sodium Bicarbonate 8.4%':
            {'convert': {'ml': 1, 'mEq': 1}, 'label': 'ml'},
    }

    df = change_drug_units(df, variables, 'amountuom', 'Human Albumin Solution 25%')

    return df


def change_lab_units(df, variables):
    for col in variables.keys():
        df = (
            df.with_columns([
                pl.when(pl.col('label') == col)
                .then(pl.col('valuenum') * variables[col]['valuenum'])
                .otherwise(pl.col('valuenum')),

                pl.when(pl.col('label') == col)
                .then(pl.lit(variables[col]['valueuom']).alias('valueuom'))
                .otherwise(pl.col('valueuom'))
            ])
        )

    return df


def change_lab_units_for_mimic(df):
    """
    Goal is to change units to standardised units.
    """
    variables = {'Albumin':  # from g/dL to g/L
                     {'valuenum': 10, 'valueuom': 'g/L'},

                 'Bedside Glucose':  # from mg/dL to mmol/L
                     {'valuenum': 1 / 18, 'valueuom': 'mmol/L'},

                 'Glucose':  # from mg/dL to mmol/L
                     {'valuenum': 1 / 18, 'valueuom': 'mmol/L'},

                 'Bilirubin':  # from mg/dL to µmol/L
                     {'valuenum': 17.1, 'valueuom': 'µmol/L'},

                 'Blood Gas pO2':  # from mmHg to kPa
                     {'valuenum': 0.133322, 'valueuom': 'kPa'},

                 'Blood Gas pCO2':  # from mmHg to kPa
                     {'valuenum': 0.133322, 'valueuom': 'kPa'},

                 'Calcium':
                     {'valuenum': 0.2495, 'valueuom': 'mmol/L'},

                 'Creatinine':  # from mg/dL to µmol/L
                     {'valuenum': 88.42, 'valueuom': 'µmol/L'},

                 'Dialysis Blood Flow Rate':  # from ml/min to ml/hr
                     {'valuenum': 60, 'valueuom': 'ml/hr'},

                 'Dialysis Fluid Removal Rate':  # change label from ml to ml/hr (which it effectively already is)
                     {'valuenum': 1, 'valueuom': 'ml/hr'},

                 'Haemoglobin':  # from g/dL to g/L
                     {'valuenum': 10, 'valueuom': 'g/L'},

                 'Urea':  # from mg/dL to mmol/L
                     {'valuenum': 0.357, 'valueuom': 'mmol/L'}}

    df = change_lab_units(df, variables)

    return df


def clean_combined_data(combined_data, train_patient_ids=None):
    cleaning_functions = get_cleaning_functions()

    combined_data = cleaning_functions['change_lab_units'](combined_data)
    combined_data = cleaning_functions['change_antibiotic_doses'](combined_data)
    combined_data = cleaning_functions['change_drug_units'](combined_data)
    combined_data = cleaning_functions['process_nutritional_info'](combined_data)
    combined_data = cleaning_functions['change_basal_bolus'](combined_data)
    combined_data = cleaning_functions['merge_overlapping_rates'](combined_data)
    combined_data = cleaning_functions['remove_outliers'](combined_data, train_patient_ids)

    select_cols = ['subject_id', 'label', 'starttime', 'valuenum', 'valueuom', 'bolus', 'bolusuom', 'rate',
                   'rateuom', 'patientweight']
    sort_cols = ['subject_id', 'starttime']

    combined_data = (
        combined_data
        .select(select_cols)
        .rename({'label': 'feature'})
        .unique()
        .sort(by=sort_cols)
        # Add unique ids for each discrete measurement
        .with_row_index()
        .rename({'index': 'input_id_num'})
    )

    return combined_data


def convert_dataframe_to_hdf5(context_length: int = 400):
    # Load the features.txt and label_features.txt files
    with open('./data/features.txt', 'r') as f:
        features = f.read().splitlines()
        f.close()

    with open('./data/label_features.txt', 'r') as f:
        label_features = f.read().splitlines()
        f.close()

    segments = ['train', 'val', 'test']

    for segment in segments:
        # Define our target paths
        dataframe_dir = f'./data/mimic/{segment}/dataframe_{segment}'
        array_dir = f'./data/mimic/{segment}'
        h5_array_path = os.path.join(array_dir, f'h5_array_{segment}.hdf5')

        # Create the directories if necessary
        os.makedirs(array_dir, exist_ok=True)

        # Create the hdf5 binary
        create_hdf5_array(h5_array_path, dataframe_dir, features, label_features, context_length)


def create_final_dataframe(encoded_input_data, labels, encodings, train_patient_ids, val_patient_ids, test_patient_ids,
                           sorting_columns, grouping_columns, context_length):
    """
    This is the final pipeline for taking the encoded data and labels, and creating dedicated input data
    for the model.

    N.B. We are creating a "sliding window" of data (with measurements being repeated across potentially
    many rows), which means the data is LARGE even with .parquet compression, and may be slow to complete.
    """
    # Iterate through each data segment
    for patient_ids, group in [[train_patient_ids, 'train'], [val_patient_ids, 'val'], [test_patient_ids, 'test']]:
        if patient_ids is None:
            continue
        print(f'Processing {group}...')
        path = f'./data/mimic/{group}'
        dataframe_path = os.path.join(path, f'dataframe_{group}')
        os.makedirs(dataframe_path, exist_ok=True)

        chunk_size = 100  # number of patient ids to process at once
        chunk_size = len(patient_ids) // chunk_size + 1

        join_cols = ['subject_id']

        # All labels should be given a unique index (different from id_num) specific to the train/val/test split.
        # We also want the order of patient_ids to match the index order
        patient_ids = (
            labels
            .filter(pl.col('subject_id').is_in(patient_ids))
            .select('subject_id', 'index')
            .sort('index')
            .select('subject_id')
            .unique(maintain_order=True)
            .collect()
            .to_series()
            .to_numpy()
        )

        # Iterate through each chunk of patient ids
        for idx, ids_chunk in progress_bar(np.array_split(patient_ids, chunk_size), with_time=True):

            # Start by creating our base "input data", i.e., for each label, we want to find all eligible measurements
            # within the inclusion window (e.g., the last 7 days of measurements)
            temp = (
                # Join the input data and the labels together
                encoded_input_data
                .filter(pl.col('subject_id').is_in(ids_chunk))
                .join(labels.drop('label', strict=False), on=join_cols, how='inner', suffix='_labels')
                # Filter out insulin labels from the input data
                .filter((pl.col('input_id_num') != pl.col('input_id_num_labels')) | (pl.col('input_id_num_labels').is_null()))
                .drop('input_id_num', 'input_id_num_labels')
                # Filter the measurements to the inclusion period for each label
                .filter(
                    (pl.col('featuretime') >= pl.col('start_inclusion'))
                    & (pl.col('featuretime') <= pl.col('end_inclusion')))
                .drop('start_inclusion')
                # Select the desired columns
                .select(sorting_columns + [
                    # Featuretime is now minutes UNTIL the end of the inclusion period
                    # (i.e., how many minutes ago, relative to right now)
                    (pl.col('end_inclusion') - pl.col('featuretime')).dt.total_minutes().cast(pl.Int16).alias(
                        'featuretime'),
                    'value', 'encoded_feature',
                    # For all our labels, just group these together into a Polars Struct for simplicity
                    pl.struct(pl.exclude(*sorting_columns, 'featuretime', 'value', 'encoded_feature')).alias('targets')
                ])
                .collect()
                .lazy()
            )

            # As we are limited to 400 measurements in a 7-day window, we want to make sure all current drug infusions
            # are included in the input data. The risk is that in edge cases where we have LOADS of recent measurements,
            # ongoing drug infusions might get missed out as a result.
            current_drug_infusions = (
                temp
                .filter(pl.col('encoded_feature').is_in(encodings['drug_names']))
                .rename({'encoded_feature': 'current_drug_feature', 'value': 'current_drug_value'})
                .sort(by=sorting_columns + ['featuretime'])
                # Following is equivalent to group_by, but is MUCH faster than using group_by directly
                .select(pl.all().first().over(grouping_columns + ['current_drug_feature']))
                .unique()
                # Filter out drugs that have been stopped (i.e., value = 0)
                .filter(pl.col('current_drug_value') > 0)
                # Fill in our delta_time/delta_value and time columns with 0
                .with_columns([pl.lit(0).cast(pl.Int16).alias('current_drug_time'),
                               pl.lit(0).cast(pl.Int16).alias('current_drug_time_delta'),
                               pl.lit(0.).alias('current_drug_value_delta')])
                .drop('featuretime')
                .group_by(grouping_columns)
                .agg(pl.all())
            )

            # Next, we will identify all historic drug rates and all other measurements.
            # To calculate delta_value/delta_time, we want to compare to rows >= 15 minutes ago.
            # To do this (as we did with scaling data), we will use join_asof to find the nearest approximate row.
            historic_events = (
                temp
                # Sorted from new to old (because increasing featuretime = older measurement)
                .sort(by=sorting_columns + ['featuretime', 'encoded_feature'])
            )

            historic_events = (
                historic_events
                .with_columns([
                    # Get the 'rank' of each measurement
                    # i.e., the number of times the feature has appeared
                    # (we want to prioritise "unseen" features before repeat features)
                    pl.col('encoded_feature').cum_count().over(sorting_columns + ['encoded_feature']).alias(
                        'feature_rank'),

                    # Calculate the delta_value and delta_time for each feature
                    # (excluding bolus/event values, hence the is_in check)
                    pl.when(pl.col('encoded_feature').is_in(encodings['lab_names'] + encodings['drug_names']))
                    .then(pl.col('value') - pl.col('value').shift(-1).over(sorting_columns + ['encoded_feature']))
                    .otherwise(pl.lit(None)).alias('delta_value'),

                    # Because larger featuretime = older row, we swap around the subtraction compared to delta_value
                    pl.when(pl.col('encoded_feature').is_in(encodings['lab_names'] + encodings['drug_names']))
                    .then(pl.col('featuretime').shift(-1).over(sorting_columns + ['encoded_feature']) - pl.col('featuretime'))
                    .otherwise(pl.lit(None)).alias('delta_time')
                ])
            )

            # Sort all historic measurements by 1) the feature rank, and then 2) the feature time
            historic_events = (
                historic_events
                .sort(sorting_columns + ['feature_rank', 'featuretime'])
                .group_by(grouping_columns)
                .agg(pl.all())
            )

            # Add back in our "current drug" measurements
            target_data = (
                historic_events
                .join(current_drug_infusions, on=grouping_columns, how='full', coalesce=True)
            )

            #
            target_data = (
                target_data
                .with_columns([
                    # Use if-then to avoid concatenating nulls unnecessarily
                    # - if one col is empty, use the other col on its own
                    pl.when(pl.col(first_col).is_null())
                    .then(pl.col(second_col))
                    # - otherwise, if the other col is empty, use the first col on its own
                    .otherwise(pl.when(pl.col(second_col).is_null())
                               .then(pl.col(first_col))
                               # - otherwise, concatenate the two
                               .otherwise(pl.concat_list(pl.col(first_col), pl.col(second_col)))
                               ).alias(second_col)

                    for first_col, second_col in [['current_drug_time', 'featuretime'],
                                                  ['current_drug_value', 'value'],
                                                  ['current_drug_feature', 'encoded_feature'],
                                                  ['current_drug_time_delta', 'delta_time'],
                                                  ['current_drug_value_delta', 'delta_value']]
                ])
                .drop('current_drug_feature', 'current_drug_value', 'current_drug_time', 'current_drug_time_delta',
                      'current_drug_value_delta', 'feature_rank')
            )

            input_feature_columns = ['featuretime', 'encoded_feature', 'value', 'delta_time', 'delta_value']

            # Now for each label, filter to just 397 measurements (i.e., 400 - 3 for age/gender/weight),
            # with nulls added if necessary
            target_data = (
                target_data
                .with_columns([
                    pl.col(col)
                    # We do (max_context_window - 3) because we will be adding age/gender/weight at the start
                    .list.concat([None for _ in range(context_length - 3)])
                    .list.slice(0, context_length - 3)
                    for col in input_feature_columns
                ])
            )

            # Merge gender/age/weight into the lab/event columns
            target_data = (
                target_data
                .unnest('targets')
                # - concat [age, gender, patientweight] to the start of value
                .with_columns([pl.concat_list(pl.concat_list(pl.col('age').cast(pl.Float64),
                                                             pl.col('gender').cast(pl.Float64),
                                                             pl.col('patientweight').cast(pl.Float64)
                                                             # pl.col('minutes_since_admission').cast(pl.Float64)
                                                             ),
                                              pl.col('value'))
                              .alias('value')])
                .drop('age', 'gender', 'patientweight')  # 'minutes_since_admission')
                # - concat [0, 0, 0] to the start of featuretime
                .with_columns([pl.concat_list(pl.lit([np.int16(0) for _ in range(3)]),
                                              pl.col('featuretime'))
                              .alias('featuretime')])
                # - concat encoded feature values for [age, gender, patientweight] to the start of feature
                .with_columns([pl.concat_list(pl.lit([encodings['age'], encodings['gender'], encodings['weight']
                                                      ]),
                                              pl.col('encoded_feature'))
                              .alias('encoded_feature')])
                # - concat [Null, Null, Null] to the start of delta_time and delta_value
                .with_columns([pl.concat_list(pl.lit([None for _ in range(3)]).cast(pl.List(pl.Int16)),
                                              pl.col('delta_time')).alias('delta_time'),

                               pl.concat_list(pl.lit([None for _ in range(3)]),
                                              pl.col('delta_value')).alias('delta_value')])
            )

            # For dtype efficiency, change some of the nulls to -1 (and NaN for the rest)
            target_data = (
                target_data
                .with_columns([
                    pl.col('featuretime').list.eval(pl.element().fill_null(-1)),
                    pl.col('value').list.eval(pl.element().fill_null(np.nan)),
                    pl.col('encoded_feature').list.eval(pl.element().fill_null(0)),
                    pl.col('delta_time').list.eval(pl.element().fill_null(-1)),
                    pl.col('delta_value').list.eval(pl.element().fill_null(np.nan))
                ])
                .collect()
                .lazy()
            )

            # We need to then get our "next state", defined as 24hrs from now (same as our viewing window)
            has_next_state = target_data.filter(pl.col('label_id_num_next').is_not_null())

            has_next_state = (
                has_next_state
                .join(target_data.select(['labeltime', 'label_id_num'] + input_feature_columns),
                      left_on=['labeltime_next', 'label_id_num_next'],
                      right_on=['labeltime', 'label_id_num'], how='inner', suffix='_next')
            )

            no_next_state = target_data.filter(pl.col('label_id_num_next').is_null())

            no_next_state = (
                no_next_state
                .with_columns([
                    pl.lit(None).alias(col + '_next') for col in input_feature_columns
                ])
            )

            final_target_data = pl.concat([has_next_state, no_next_state])

            final_target_data = (
                final_target_data
                # Again, sort out the "nulls" for each in way that is consistent with the dtypes
                .with_columns([
                    pl.when(pl.col(col + '_next').is_null())
                    .then(pl.lit([-1 for _ in range(context_length)]).cast(pl.List(pl.Int16)).alias(col + '_next'))
                    .otherwise(col + '_next') for col in input_feature_columns
                    if 'time' in col  # this covers both featuretime and delta_time
                ])

                .with_columns([
                    pl.when(pl.col(col + '_next').is_null())
                    .then(pl.lit([0 for _ in range(context_length)]).cast(pl.List(pl.Int16)).alias(col + '_next'))
                    .otherwise(col + '_next') for col in input_feature_columns
                    if 'encoded_feature' in col
                ])

                .with_columns([
                    pl.when(pl.col(col + '_next').is_null())
                    .then(pl.lit([np.nan for _ in range(context_length)]).alias(col + '_next'))
                    .otherwise(col + '_next') for col in input_feature_columns if 'value' in col
                ])
            )

            # Add a column that has the next context_length (400) indices of that episode. Right-pad using -1.
            future_index_df = final_target_data.sort('index').select('episode_num', 'step_num', 'index')
            if (
                    future_index_df
                            .select((pl.col('step_num') - pl.col('step_num').shift(1).over('episode_num') != 1).any())
                            .collect()
                            .item()
            ) or (
                    future_index_df
                            .select((pl.col('index') - pl.col('index').shift(1) != 1).any())
                            .collect()
                            .item()
            ):
                raise ValueError("Indices are not sequential within each episode_num. Please check the data.")

            future_index_df = (
                future_index_df
                .join(
                    future_index_df, on='episode_num', how='inner',
                    maintain_order='left_right'
                )
                .sort('index', 'index_right')
                .filter(pl.col('step_num_right') >= pl.col('step_num'))
                .drop('step_num_right')
                .group_by(pl.exclude('index_right'), maintain_order=True)
                .agg(pl.col('index_right').alias('future_indices'))
                .with_columns([
                    pl.concat_list(pl.col('future_indices'), pl.lit(-1).repeat_by(context_length)).list.slice(1, context_length)
                ])
            )

            # Add future_index_df back to target_data
            final_target_data = (
                final_target_data
                .join(future_index_df, on=['episode_num', 'step_num', 'index'], how='inner')
            )

            columns = final_target_data.collect_schema().names()
            columns.remove('future_indices')
            column_idx = columns.index('index')
            columns.insert(column_idx + 1, 'future_indices')

            (
                final_target_data
                # Sort, collect, and save the data
                .sort('index')
                .select(columns)
                .collect()
                .write_parquet(os.path.join(dataframe_path, f'dataframe{idx:03}.parquet'))
            )


def create_hdf5_array(target_path, batch_dir, features, label_features, context_window):
    # Create the .hdf5 file and some blank placeholders (for use later)
    is_train = 'train' in target_path
    h5_array = h5py.File(target_path, 'w')
    h5_array = create_hdf5_datasets(h5_array, features, label_features, context_window, is_train=is_train)
    next_size = next_real_size = 0

    batch_dir_files = sorted(os.listdir(batch_dir))
    scaling_path = './data/scaling_data.parquet'

    # Iterate through each batch of data (i.e., each .parquet file)
    for batch_file_idx, batch_file in progress_bar(batch_dir_files, with_time=True):
        # Load the .parquet dataframe
        dataframe_batch = pl.scan_parquet(os.path.join(batch_dir, batch_file))

        # Convert the dataframe to numpy arrays
        (future_indices_vals, feature_vals, timepoint_vals, value_vals, delta_time_vals, delta_value_vals,
         label_vals, next_feature_vals, next_timepoint_vals, next_value_vals,
         next_delta_time_vals, next_delta_value_vals) = create_numpy_arrays_from_dataframe(dataframe_batch,
                                                                                           context_window,
                                                                                           label_features)

        # Resize the h5_array datasets to accommodate the new data
        current_size = next_size
        next_size = current_size + value_vals.shape[0]

        h5_array = resize_hdf5_datasets(h5_array, next_size, label_features, context_window)

        # Compress and write data to the datasets
        for embedding, (data, next_data), dtype in [
            ['future_indices', (future_indices_vals, None), np.int64],
            ['features', (feature_vals, next_feature_vals), np.int16],
            ['timepoints', (timepoint_vals, next_timepoint_vals), np.int16],
            ['values', (value_vals, next_value_vals), np.float32],
            ['delta_time', (delta_time_vals, next_delta_time_vals), np.int16],
            ['delta_value', (delta_value_vals, next_delta_value_vals), np.float32]
        ]:
            h5_array[embedding][current_size:] = data.astype(dtype=dtype)[:, :context_window]
            if embedding != 'future_indices':
                h5_array[f'{embedding}_next'][current_size:] = next_data.astype(dtype=dtype)[:, :context_window]

        h5_array['labels'][current_size:] = label_vals

        if not is_train:
            continue

        # Update the scaling data

        # We want to find min, max, mean, and std values for our (relative) timepoints (using sse to update std)
        # For timepoints, we can use all the data at once (all features share a similar scale)

        # Flatten the data and remove all the nans
        real_mask = np.where(feature_vals.flatten() > -1)[0]
        timepoint_vals = timepoint_vals.flatten()[real_mask]

        # Update the scaling values for timepoints and delta_time, as the simpler step
        current_min = h5_array['min']['timepoints'][:]
        current_max = h5_array['max']['timepoints'][:]
        current_mean = h5_array['mean']['timepoints'][:]
        current_sse = h5_array['sse']['timepoints'][:]

        next_real_size += real_mask.size

        # Now we can calculate the min, max, mean, and std
        min_vals = np.nanmin(np.concatenate((timepoint_vals, current_min), 0), 0).reshape(1)
        max_vals = np.nanmax(np.concatenate((timepoint_vals, current_max), 0), 0).reshape(1)

        h5_array['min']['timepoints'][:] = min_vals
        h5_array['max']['timepoints'][:] = max_vals

        # Calculate the mean and std for each feature - we will use a special formula for continual updates
        # Calculate our update terms
        update_term = timepoint_vals - current_mean

        update_mean_term = np.nansum(update_term / np.maximum(next_real_size, 1), 0)

        update_mean = current_mean + update_mean_term

        update_sse_term = np.nansum(update_term * (timepoint_vals - update_mean), 0)

        update_sse = current_sse + update_sse_term

        update_std = np.sqrt(update_sse / np.maximum(next_real_size, 1))

        h5_array['mean']['timepoints'][:] = update_mean
        h5_array['std']['timepoints'][:] = update_std
        h5_array['sse']['timepoints'][:] = update_sse

    # Update the scaling for all other parameters (using our pre-prepared scaling.parquet file)
    if not is_train:
        h5_array.close()
        return

    # Delete the sse dataset, as this is no longer required
    del h5_array['sse']

    scaling_df = pl.read_parquet(scaling_path)
    for idx, feature in enumerate(features):
        current_feature_df = scaling_df.filter(pl.col('str_feature') == feature)
        if current_feature_df.is_empty():
            continue
        # Make sure our idx/feature mapping is intact
        assert current_feature_df.select('encoded_feature').item() == idx
        for group in ['mean', 'std', 'max', 'min']:
            for embedding in ['values', 'delta_value', 'delta_time']:
                h5_array[group][embedding][feature][:] = (
                    current_feature_df
                    .select(f'{embedding}_{group}')
                    .to_series().item()
                )

    # Close the newly created h5 array
    h5_array.close()
    return


def create_hdf5_datasets(h5_array: h5py.File, features: list = None, label_features: list = None,
                         context_window: int = 400, is_train: bool = True):
    empty_data_shape = (0, context_window, 1)
    max_data_shape = (10 ** 8, context_window, 1)

    empty_label_shape = (0, len(label_features))
    max_label_shape = (10 ** 8, len(label_features))

    # Create a dataset for encoded features for each token tuple
    # Use int16 when possible to save space (we will use -1 instead of nan)
    for embedding, dtype in [['features', np.int16], ['timepoints', np.int16], ['values', np.float32],
                             ['delta_time', np.int16], ['delta_value', np.float32], ['future_indices', np.int64]]:
        for i in ['', '_next']:
            if embedding == 'future_indices' and i == '_next':
                continue
            h5_array.create_dataset(name=embedding + i,
                                    shape=empty_data_shape,
                                    maxshape=max_data_shape,
                                    compression='gzip',
                                    compression_opts=9,
                                    dtype=dtype)

    # Create a dataset for the labels
    h5_array.create_dataset(name='labels',
                            shape=empty_label_shape,
                            maxshape=max_label_shape,
                            compression='gzip',
                            compression_opts=9,
                            dtype=np.float32)

    # Create datasets for standardisation/normalisation values
    if is_train:
        for scale in ['min', 'max', 'mean', 'std', 'sse']:
            h5_array.create_group(scale)

            if scale in ['mean', 'std', 'sse']:
                h5_array[scale].create_dataset(name='timepoints',
                                               data=np.zeros(1),
                                               dtype=np.float32)
            else:
                h5_array[scale].create_dataset(name='timepoints',
                                               # We want these values overwritten immediately,
                                               # so max is set to -inf and min is set +inf
                                               data=np.inf * np.ones(1) * -1 if scale == 'max' else np.inf * np.ones(1),
                                               dtype=np.float32)

            for embedding in ['values', 'delta_value', 'delta_time']:
                h5_array[scale].create_group(embedding)

                for feature in features:
                    if scale in ['mean', 'std', 'sse']:
                        h5_array[scale][embedding].create_dataset(name=feature,
                                                                  data=np.zeros(1),
                                                                  dtype=np.float32)
                    else:
                        h5_array[scale][embedding].create_dataset(name=feature,
                                                                  # We want these values overwritten immediately,
                                                                  # so max is set to -inf and min is set +inf
                                                                  data=np.inf * np.ones(1) *
                                                                       -1 if scale == 'max' else np.inf * np.ones(1),
                                                                  dtype=np.float32)

    # Lastly, store the actual feature names and label names
    h5_array.create_dataset(name='feature_names',
                            data=features)
    h5_array.create_dataset(name='label_names',
                            data=label_features)

    return h5_array


def create_numpy_arrays_from_dataframe(df: pl.LazyFrame, context_window: int = 2000, label_features: list = None):
    f = lambda x, column: x.select(column).explode(column).collect().to_numpy().reshape(-1, context_window, 1)
    future_indices_vals = f(df, 'future_indices')
    feature_vals = f(df, 'encoded_feature')
    timepoint_vals = f(df, 'featuretime')
    value_vals = f(df, 'value')
    delta_time_vals = f(df, 'delta_time')
    delta_value_vals = f(df, 'delta_value')
    next_feature_vals = f(df, 'encoded_feature_next')
    next_timepoint_vals = f(df, 'featuretime_next')
    next_value_vals = f(df, 'value_next')
    next_delta_time_vals = f(df, 'delta_time_next')
    next_delta_value_vals = f(df, 'delta_value_next')

    label_vals = df.select(label_features).collect().to_numpy()

    return (future_indices_vals, feature_vals, timepoint_vals, value_vals, delta_time_vals, delta_value_vals,
            label_vals, next_feature_vals, next_timepoint_vals, next_value_vals, next_delta_time_vals,
            next_delta_value_vals)


def create_glucose_labels_for_mimic(combined_data, admissions, patients, input_window_size, state_delay,
                                    next_state_window=24 * 60, inclusion_hours=24, train_patient_ids=None,
                                    val_patient_ids=None, test_patient_ids=None):
    # Create our labels DataFrame - label is every new measurement starttime (the "state marker")
    labels = (
        combined_data
        .filter((pl.col('feature') == "Bedside Glucose") | ((pl.col('feature') == "Regular Insulin")
                                                            & (pl.col('rate').is_not_null())))
        .select(['input_id_num', 'subject_id', 'starttime', 'feature', 'valuenum', 'rate', 'patientweight'])
        .unique()
    )

    # Get the insulin labels - we can experiment with broader inclusion hours if desired later
    (labels,
     train_patient_ids,
     val_patient_ids,
     test_patient_ids) = get_insulin_labels_for_mimic(labels, inclusion_hours=inclusion_hours,
                                                      next_state_start=state_delay, next_state_window=next_state_window,
                                                      train_ids=train_patient_ids, val_ids=val_patient_ids,
                                                      test_ids=test_patient_ids)

    # Get the death labels
    labels = get_death_labels_for_mimic(labels, admissions, patients)

    label_columns = ['subject_id', 'episode_num', 'step_num', 'label_id_num', 'label_id_num_next', 'is_first_state',
                     'is_last_state', 'starttime', 'starttime_next', 'patientweight', 'current_bm', 'prev_bm',
                     'time_since_prev_bm', 'bm_next', 'time_until_bm_next', 'insulin_changetime',
                     'insulin_default_rate', 'insulin_new_rate', 'insulin_maintain', 'insulin_change', 'insulin_stop',
                     'insulin_delta_change', 'insulin_maintain_next', 'insulin_change_next', 'insulin_stop_next',
                     'insulin_delta_change_next', '1-day-alive', '1-day-alive-final', '3-day-alive',
                     '3-day-alive-final', '7-day-alive', '7-day-alive-final', '14-day-alive', '14-day-alive-final',
                     '28-day-alive', '28-day-alive-final', 'input_id_num']

    # Create our start/end inclusion times for input data (and save the DataFrame for the next step)
    (
        labels
        .select(label_columns +
                [(pl.col('starttime') - pl.duration(hours=input_window_size)).alias('start_inclusion'),
                 pl.col('starttime').alias('end_inclusion')])
        .unique()
        .collect()
        .write_parquet('./data/mimic/parquet/labels.parquet')
    )

    labels = pl.scan_parquet('./data/mimic/parquet/labels.parquet')

    # Add in our previous actions (if they exist)
    labels = (
        labels
        .sort(by=['episode_num', 'step_num'])
        .with_columns([
            pl.col('insulin_maintain').shift(1).over(['episode_num']).alias('insulin_maintain_prev'),
            pl.col('insulin_change').shift(1).over(['episode_num']).alias('insulin_change_prev'),
            pl.col('insulin_stop').shift(1).over(['episode_num']).alias('insulin_stop_prev'),
            pl.col('insulin_delta_change').shift(1).over(['episode_num']).alias('insulin_delta_change_prev'),
        ])
    )

    # Add age and gender data
    # To get age, you must do the following:
    # - Get the anchor_age and anchor_year from `patients`
    # - Calculate 'starttime' - 'anchor_year'
    # - Add this result to anchor_age
    # To get gender, you must do the following:
    # - Get the 'gender' column from patients.csv, and convert 'F'/'M' to binary values
    labels = (
        labels
        .lazy()
        .join(patients.lazy()
              .select(['subject_id', 'gender', 'anchor_age', 'anchor_year']), on='subject_id', how='inner')
        .with_columns([
            # Turn gender into a binary value
            pl.when(pl.col('gender') == "F")
            .then(pl.lit(1).cast(pl.UInt8).alias('gender'))
            .otherwise(pl.lit(0)),

            # Calculate age
            (pl.col('starttime').dt.year() - pl.col('anchor_year') + pl.col('anchor_age')).cast(pl.UInt8).alias('age')
        ])
        .drop(['anchor_age', 'anchor_year'])
    )

    # Add time/steps remaining until the end of the episode + steps per episode
    labels = (
        labels
        .sort(by=['episode_num', 'step_num'])
        .with_columns([
            # Get steps remaining until the end of the episode
            (pl.col('step_num').max().over('episode_num') - pl.col('step_num')).alias('steps_remaining'),

            # Get minutes remaining until the end of the episode
            (pl.col('starttime').max().over('episode_num') - pl.col('starttime')
             ).dt.total_minutes().alias('minutes_remaining'),

            # Get the total steps per episode
            pl.col('step_num').max().over('episode_num').alias('steps_per_episode'),
        ])
    )

    # Add total number of future hyper episodes (where BM is >10)
    labels = (
        labels
        .sort('episode_num', 'step_num')
        .with_columns([
            pl.when(pl.col('bm_next') > 10)
            .then(pl.lit(1.).alias('n_future_hypers'))
            .otherwise(pl.lit(0.))
        ])
        .with_columns([
            pl.col('n_future_hypers').cum_sum(reverse=True).over('episode_num')
        ])
    )

    # Set the column order
    label_col_order = ['subject_id', 'label_id_num', 'labeltime', 'label_id_num_next', 'labeltime_next', 'start_inclusion',
                       'end_inclusion', 'episode_num', 'step_num', 'steps_per_episode', 'steps_remaining',
                       'minutes_remaining', 'is_first_state', 'is_last_state', 'current_bm', 'prev_bm',
                       'time_since_prev_bm', 'bm_next', 'time_until_bm_next', 'n_future_hypers', 'insulin_changetime',
                       'insulin_default_rate', 'insulin_new_rate', 'insulin_maintain', 'insulin_change', 'insulin_stop',
                       'insulin_delta_change', 'insulin_maintain_prev', 'insulin_change_prev', 'insulin_stop_prev',
                       'insulin_delta_change_prev', 'insulin_maintain_next', 'insulin_change_next', 'insulin_stop_next',
                       'insulin_delta_change_next', '1-day-alive', '1-day-alive-final', '3-day-alive',
                       '3-day-alive-final', '7-day-alive', '7-day-alive-final', '14-day-alive', '14-day-alive-final',
                       '28-day-alive', '28-day-alive-final', 'age', 'gender', 'patientweight', 'input_id_num']

    # Collect the labels
    labels = (
        labels
        .rename({'starttime': 'labeltime', 'starttime_next': 'labeltime_next'})
        .select(label_col_order).unique()
        # Fill the blank patient weights - start by filling with the previous value,
        # and insert a default based on gender for the rest.
        # Taken from train_ids: mean: F = 74, M = 86
        .with_columns([pl.col('patientweight').fill_null(strategy='forward').over('subject_id')])
        .with_columns([pl.when(pl.col('patientweight').is_null())
                      .then(pl.when(pl.col('gender') == 0)
                            .then(pl.lit(86.).alias('patientweight'))
                            .otherwise(pl.lit(74.).alias('patientweight')))
                      .otherwise(pl.col('patientweight'))])
        .sort('episode_num', 'step_num')
        .collect()
    )

    return labels, train_patient_ids, val_patient_ids, test_patient_ids


def encode_combined_data_for_mimic(combined_data):
    """
    Create the encodings for our features
    """
    # First, create separate drug variables for rate and bolus, and create a single 'value' column
    encoded_input_data = split_labels_to_rate_and_bolus(df=combined_data)
    encoded_input_data = encoded_input_data.rename({'starttime': 'featuretime'})

    # Convert our feature names to integer encodings
    features = encoded_input_data.select(pl.col('feature').unique()).to_series().sort().to_list()
    features.append('age')
    features.append('gender')
    features.append('patientweight')

    feature_encoding = (
        pl.DataFrame({'str_feature': features,
                      'encoded_feature': np.array([i for i in range(len(features))], dtype=np.int16)})
        .with_columns([pl.col('str_feature')])
    )

    # Encode the features in combined_data
    encoded_input_data = (
        encoded_input_data
        .rename({'feature': 'str_feature'})
        .join(feature_encoding, on='str_feature', how='inner')
    )

    # Keep track of the encodings for all our features (by category), for use later on
    age_encoding = np.int16(feature_encoding.filter(pl.col('str_feature') == 'age').select('encoded_feature').item())
    gender_encoding = np.int16(
        feature_encoding.filter(pl.col('str_feature') == 'gender').select('encoded_feature').item())
    weight_encoding = np.int16(
        feature_encoding.filter(pl.col('str_feature') == 'patientweight').select('encoded_feature').item())

    str_features, all_encoded_features = (encoded_input_data.select('str_feature', 'encoded_feature')
                                          .unique().sort('str_feature'))
    drug_names_encoded, lab_names_encoded = [], []
    for str_feature, encoded_feature in zip(str_features, all_encoded_features):
        if 'rate' in str_feature:
            drug_names_encoded.extend([encoded_feature])
        elif 'bolus' in str_feature:
            continue
        else:
            lab_names_encoded.extend([encoded_feature])

    encodings = {'age': age_encoding, 'gender': gender_encoding, 'weight': weight_encoding,
                 'drug_names': drug_names_encoded, 'lab_names': lab_names_encoded, 'all_features': all_encoded_features}

    return encoded_input_data, features, feature_encoding, encodings


def get_antibiotic_names():
    return ['Aciclovir', 'Ambisome', 'Amikacin', 'Ampicillin', 'Ampicillin-Sulbactam', 'Azithromycin', 'Aztreonam',
            'Co-trimoxazole', 'Caspofungin', 'Cefazolin', 'Cefepime', 'Ceftazidime', 'Ceftriaxone', 'Ciprofloxacin',
            'Chloramphenicol', 'Clindamycin', 'Colistin', 'Daptomycin', 'Doxycycline', 'Ertapenem', 'Erythromycin',
            'Gentamicin', 'Levofloxacin', 'Meropenem', 'Linezolid', 'Micafungin', 'Metronidazole', 'Nafcillin',
            'Oxacillin', 'Piperacillin', 'Piperacillin-Tazobactam', 'Rifampin', 'Tigecycline', 'Tobramycin',
            'Vancomycin', 'Voriconazole', 'IVIG', 'Mannitol']


def get_cleaning_functions():
    return {'change_lab_units': change_lab_units_for_mimic,
            'change_antibiotic_doses': change_antibiotic_doses,
            'change_drug_units': change_drug_units_for_mimic,
            'process_nutritional_info': process_nutritional_info,
            'change_basal_bolus': separate_basal_bolus_in_mimic,
            'merge_overlapping_rates': merge_overlapping_rates,
            'remove_outliers': lambda x, ids: remove_outliers_from_mimic(x, ids)}


def get_death_labels_for_mimic(df, admissions, patients):
    """
    Get the labels for death events.

    Admissions contains the deathtime for patients who died in hospital.
    Patients contains the date of death for patients who died after discharge.
    """
    df, admissions, patients = df.lazy(), admissions.lazy(), patients.lazy()
    # Refine admission DataFrame
    admissions = (admissions
                  .select(['subject_id', 'deathtime'])
                  .filter((pl.col('deathtime').is_null().all().over('subject_id'))
                          | (pl.col('deathtime').is_not_null()))
                  .unique())

    # Add in death times
    df = (
        df
        # Join with the `admissions` DataFrame, which contains admission deaths
        .join(admissions, on=['subject_id'], how='inner')
        # Join with the `patients` DataFrame (for deaths just after discharge)
        .join(patients.select(['subject_id', 'dod']).unique(), on='subject_id', how='inner')
        # When 'deathtime' is null but 'dod' occurs on the same day as some labels, use the final label as deathtime
        .with_columns([
            pl.when(pl.col('deathtime').is_null()
                    & pl.col('dod').is_not_null()
                    & (pl.col('starttime') >= pl.col('dod')).any().over('subject_id'))
            .then(pl.col('starttime').last().over('subject_id').alias('deathtime'))
            .otherwise(pl.col('deathtime'))
        ])
        # Now combine dod and deathtime into a single column
        .with_columns([pl.when(pl.col('deathtime').is_not_null())
                      .then(pl.col('deathtime'))
                      .otherwise(pl.col('dod'))])
        # If 'deathtime' appears 2+ times, choose the later time
        .sort(by=['subject_id', 'starttime', 'deathtime'])
        .filter((pl.col('deathtime').is_null())
                | (pl.col('deathtime') == pl.col('deathtime').last().over('subject_id')))
    )

    # Convert 1-day, 3-day, 7-day, 14-day, and 28-day mortality into reward labels
    df = (
        df
        .with_columns([
            pl.when(pl.col('deathtime').is_not_null()
                    & (pl.col('deathtime') <= pl.col('starttime') + pl.duration(days=day)))
            # Predict death as a value of 0
            .then(pl.lit(0).cast(pl.UInt8).alias(f'{day}-day-alive'))
            # Represent survival as a value of 1
            .otherwise(pl.lit(1).alias(f'{day}-day-alive')) for day in [1, 3, 7, 14, 28]
        ])
        .unique()
        .drop('deathtime', 'dod')
    )

    # Add x-day-alive-final value for the end of each self-contained episode
    df = (
        df
        .sort('subject_id', 'episode_num', 'step_num')
        .with_columns([
            pl.when((pl.col('is_first_state').shift(-1) == 1)
                    | (pl.col('is_first_state').shift(-1).is_null()))
            .then(pl.col(f'{day}-day-alive').alias(f'{day}-day-alive-final'))
            .otherwise(pl.lit(None)) for day in [1, 3, 7, 14, 28]
        ])
        .with_columns([
            pl.col(f'{day}-day-alive-final').fill_null(strategy='backward').over('subject_id')
            for day in [1, 3, 7, 14, 28]
        ])
    )

    return df


def get_insulin_labels_for_mimic(labels, inclusion_hours, next_state_start, next_state_window, train_ids, val_ids,
                                 test_ids):
    # We only want labels when patients are getting IV insulin infusions.
    # Start by identifying inclusion windows according to the following principles:
    # 1) Start_inclusion from when insulin is started for the first time (minus inclusion_hours),
    # or restarted after a gap of >2 * inclusion_hours (minus inclusion_hours)
    # 2) End_inclusion when insulin is stopped for 2 * inclusion_hours (i.e., at the point when rate is set to 0)

    labels = labels.lazy()

    # Get just the insulin rows, identified according to our start/end inclusion rules
    # Start inclusion rules:
    first_insulin_ever = pl.col('rate').shift(1).over('subject_id').is_null()
    prev_insulin_stopped = (pl.col('rate').shift(1).over('subject_id') == 0)
    large_delay_in_restart = (pl.col('insulin_changetime') - pl.col('insulin_changetime').shift(1).over('subject_id')
                              > pl.duration(hours=inclusion_hours * 2))
    insulin_restart_after_large_delay = prev_insulin_stopped & large_delay_in_restart

    # End inclusion rules:
    insulin_stopped = pl.col('rate') == 0
    stopped_for_long_period = (pl.col('insulin_changetime').shift(-1).over('subject_id') - pl.col('insulin_changetime')
                              > pl.duration(hours=inclusion_hours * 2))
    stopped_forever = pl.col('insulin_changetime').shift(-1).over('subject_id').is_null()

    insulin_rate_changes = (
        labels
        .filter(pl.col('feature') == "Regular Insulin")
        .drop('valuenum')
        .rename({'starttime': 'insulin_changetime'})
        .sort('subject_id', 'insulin_changetime')
        .with_columns([
            # Identify according to our start/end inclusion rules:

            # If first insulin ever, or first insulin after it has been 0.0 for >48hrs,
            # then set as start_inclusion (minus relevant inclusion_hours to include recent glucose checks)
            pl.when(first_insulin_ever | insulin_restart_after_large_delay)
            .then(pl.col('insulin_changetime').alias('start_inclusion') - pl.duration(hours=inclusion_hours))
            .otherwise(pl.lit(None).alias('start_inclusion')),

            # If insulin has been stopped for >48 hrs, set as end_inclusion
            pl.when(insulin_stopped & (stopped_for_long_period | stopped_forever))
            .then(pl.col('insulin_changetime').alias('end_inclusion') + pl.duration(hours=inclusion_hours))
            .otherwise(pl.lit(None).alias('end_inclusion'))
        ])
        .with_columns([
            pl.col('start_inclusion').fill_null(strategy='forward').over('subject_id'),
            pl.col('end_inclusion').fill_null(strategy='backward').over('subject_id')
        ])
    )

    # Start identifying our insulin labels
    # Create our filtering criteria - where closed="left", this is to avoid clashes with the next BM check
    valid_insulin_after_bm_check = pl.col('insulin_changetime').is_between('starttime', 'valid_insulin_post',
                                                                           closed="left")
    valid_insulin_before_bm_check = pl.col('insulin_changetime').is_between('valid_insulin_pre', 'starttime',
                                                                            closed="both")

    # Reduce labels to just the BM checks
    labels = (
        labels
        .filter(pl.col('feature') == "Bedside Glucose")
        .drop('patientweight', 'rate', 'input_id_num')
        .unique()
        .sort('subject_id', 'starttime')
        .rename({'valuenum': 'current_bm'})
        # We will apply group_by_dynamic to round glucose labels to the nearest 5 minutes
        .group_by_dynamic(
            index_column='starttime',
            every='5m',
            closed='right',
            by='subject_id',
            label='right'  # Aggregated data from the last 5 minutes relative to the label
        )
        .agg([
            # Take the latest BM/rate (and minimum weight) over the last 5 minutes (where multiple measurements exist)
            pl.col('current_bm').last()
        ])
    )

    # Join with the insulin rate changes
    labels = (
        labels
        .join_asof(
            # Find our next starttime label
            # - should be between next_state_start and (next_state_start + next_state_window)
            labels.select(
                ['subject_id',
                 pl.col('starttime') - pl.duration(minutes=next_state_start + 5)
                 ]).rename({'starttime': 'starttime_next'}),
            left_on='starttime',
            right_on='starttime_next',
            by='subject_id',
            check_sortedness=False,  # Already sorted
            strategy='forward',
            tolerance=f'{next_state_window + next_state_start + 5}m'
        )
        # Move 'next_starttime' back to its correct timestamp
        .with_columns([pl.col('starttime_next') + pl.duration(minutes=next_state_start + 5)])
        # Join with our insulin rate changes
        .join(insulin_rate_changes.lazy().drop('feature'), on='subject_id', how='inner')
        # Make sure we only have BM labels in our overall valid inclusion window (see previous section)
        .filter(pl.col('starttime').is_between('start_inclusion', 'end_inclusion', closed="both"))
        .unique()
    )

    # Identify eligible insulin labels
    labels = (
        labels
        # Identify the "valid insulin" range - you essentially have 15 minutes either side of the BM check
        # (remember that group_by_dynamic with 'right' label means that "-10 minutes" is inclusive of the last -15 mins
        .with_columns([
            (pl.col('starttime') - pl.duration(minutes=15)).alias('valid_insulin_pre'),
            (pl.col('starttime') + pl.duration(minutes=15)).alias('valid_insulin_post')
        ])
        .with_columns([
            # When the next BM check happens sooner than 15 minutes, ignore insulin changes after this time
            pl.when(pl.col('starttime_next') < pl.col('valid_insulin_post'))
            .then(pl.col('starttime_next').alias('valid_insulin_post'))
            .otherwise(pl.col('valid_insulin_post'))
        ])
        # Identify potential valid insulin changes
        .with_columns([
            pl.when(
                valid_insulin_after_bm_check | (valid_insulin_before_bm_check & valid_insulin_after_bm_check.not_()))
            .then(pl.lit(True).alias('valid_insulin_row'))
            .otherwise(pl.lit(False))
        ])
        .sort('subject_id', 'starttime', 'insulin_changetime')
        .with_columns([
            # The insulin label for this row should be the last valid insulin change
            # (i.e., closest to +15 mins from this BM)
            pl.when(pl.col('valid_insulin_row'))
            .then(pl.col('valid_insulin_row').is_last_distinct().over('subject_id', 'starttime'))
            .otherwise(pl.lit(False)).alias('this_row'),
        ])
        .with_columns([
            # The reference (for delta change) should be the last insulin change BEFORE this BM check...
            ((pl.col('insulin_changetime') <= pl.col('starttime'))
             # ... that is also BEFORE the target insulin change
             & (pl.col('this_row').not_().cum_prod().cast(pl.Boolean).over('subject_id', 'starttime'))
             ).alias('last_row'),
        ])
        .with_columns([
            # Restrict to the very last row that meets the above criteria
            (pl.col('last_row') & pl.col('last_row').is_last_distinct().over('subject_id', 'starttime')).alias(
                'last_row')
        ])
        .drop('valid_insulin_row')
        # Have our "previous rate" and our "new rate" for insulin changes
        .with_columns([
            pl.when(pl.col('last_row'))
            .then(pl.col('rate').alias('insulin_default_rate'))
            .otherwise(pl.lit(None)),

            pl.when(pl.col('this_row'))
            .then(pl.col('rate').alias('insulin_new_rate'))
            .otherwise(pl.lit(None))
        ])
        .drop('rate')
        # If there is no valid insulin change, set the new rate to the default rate
        .with_columns([
            pl.when(pl.col('insulin_new_rate').is_null().all().over('subject_id', 'starttime'))
            .then(pl.col('insulin_default_rate').alias('insulin_new_rate'))
            .otherwise('insulin_new_rate')
        ])
        # Forward/backward fill over the nulls
        .sort('subject_id', 'starttime', 'insulin_changetime')
        .with_columns([
            pl.col(col)
            .fill_null(strategy='forward')
            .fill_null(strategy='backward')
            .fill_null(value=0.0)  # If missing default rate info completely, rate will be zero.
            .over('subject_id', 'starttime')
            for col in ['insulin_default_rate', 'insulin_new_rate']
        ])
    )

    # Prepare our column selection expressions
    valid_insulin_doses = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    insulin_delta_rate = pl.col('insulin_new_rate') - pl.col('insulin_default_rate')
    rounded_insulin_delta = (
        pl.when(insulin_delta_rate.abs() < np.max(valid_insulin_doses) + 0.5)
        .then(
            # Find the closest value from the list
            pl.lit(valid_insulin_doses).list.get(
                (valid_insulin_doses - insulin_delta_rate.abs()).list.eval(pl.element().abs()).list.arg_min()
            ) * insulin_delta_rate.sign())
        .otherwise(
            # If delta change is outside the list, leave unchanged
            insulin_delta_rate
        )
    )

    valid_insulin_doses.remove(0.0)
    insulin_maintained = rounded_insulin_delta.abs() == 0
    insulin_stopped = (rounded_insulin_delta <= -0.5) & (pl.col('insulin_new_rate') < 0.25)
    in_range_insulin_delta_change = rounded_insulin_delta.abs().is_in(valid_insulin_doses)
    out_of_bounds_insulin_delta_change = (
            insulin_maintained.not_() & insulin_stopped.not_() & in_range_insulin_delta_change.not_()
    )

    # Select our valid insulin labels
    labels = (
        labels
        .filter(
            # Either filter to the only valid insulin change
            pl.col('this_row')
            |
            # Or, if there are no valid insulin changes, just choose the first row as a placeholder
            (pl.col('this_row').not_().all()
             & pl.col('this_row').is_first_distinct()).over('subject_id', 'starttime')
        )
        .with_columns([
            # For rows with no valid insulin changes, set the insulin changetime and input_id_num to null
            pl.when(pl.col('this_row').not_().all().over('subject_id', 'starttime'))
            .then(pl.lit(None))
            .otherwise('insulin_changetime').alias('insulin_changetime'),

            pl.when(pl.col('this_row').not_().all().over('subject_id', 'starttime'))
            .then(pl.lit(None))
            .otherwise('input_id_num').alias('input_id_num'),
        ])
        # And now we can create our insulin change/stop/delta_change columns
        .with_columns([
            # Add insulin maintain
            pl.when(insulin_maintained)
            .then(pl.lit(1))
            .otherwise(pl.lit(0)).alias('insulin_maintain'),

            # Add insulin change
            pl.when(in_range_insulin_delta_change & insulin_stopped.not_())
            .then(pl.lit(1))
            .otherwise(pl.lit(0)).alias('insulin_change'),

            # Add insulin stop
            pl.when(insulin_stopped)
            .then(pl.lit(1))
            .otherwise(pl.lit(0)).alias('insulin_stop'),

            # Add insulin delta change (rounded to nearest item in our valid_insulin_doses list)
            rounded_insulin_delta.alias('insulin_delta_change')
        ])
        # Remove completely the out-of-bounds labels
        .filter(out_of_bounds_insulin_delta_change.not_())
    )

    # Address multiple BMs that share the same insulin action label
    labels = (
        labels
        # 1a) If there is an insulin change after BMs e.g., BM -> BM -> insulin -> (BM) ..., select the last BM BEFORE the insulin
        # 1b) If there is an insulin change before BMs only e.g., insulin -> BM -> ..., select the first BM AFTER the insulin
        # 2) If there is no insulin change, select the last BM overall?
        .sort('subject_id', 'starttime', 'insulin_changetime')
        .with_columns([
            # Find all non-null duplicated insulin changetimes
            pl.when(pl.struct('subject_id', 'insulin_changetime').is_duplicated()
                    & pl.col('insulin_changetime').is_not_null())
            .then(
                # If there are BMs before the insulin change, prioritise the last one
                pl.when((pl.col('starttime') <= pl.col('insulin_changetime')).any().over('insulin_changetime'))
                .then((pl.col('starttime') <= pl.col('insulin_changetime')).cum_prod().is_last_distinct().over(
                    'insulin_changetime')
                      # (ignore BMs after the insulin change in this case)
                      & (pl.col('starttime') <= pl.col('insulin_changetime')).cum_prod().over('insulin_changetime'))
                .otherwise(
                    # If the only BMs are after the insulin change, prioritise the first one
                    pl.col('insulin_changetime').is_first_distinct().over('subject_id')
                )
            )
            # For all the null duplicated insulin changetimes, keep by default?
            .otherwise(pl.lit(True)).cast(pl.Boolean).alias('rows_to_keep')
        ])
        .filter(pl.col('rows_to_keep'))
        .drop('this_row', 'last_row', 'rows_to_keep')
        # A small number of labels now have the wrong starttime_next after filtering, so these should be updated
        .sort('subject_id', 'starttime')
        .with_columns([
            pl.when(
                # If the next starttime is not null
                pl.col('starttime').shift(-1).is_not_null()
                # and it doesn't match the actual next starttime...
                & (pl.col('starttime_next') != pl.col('starttime').shift(-1))
            )
            .then(pl.col('starttime').shift(-1))
            .otherwise('starttime_next')
            .over('subject_id', 'start_inclusion', 'end_inclusion').alias('starttime_next')
        ])
        # If the starttime_next is actually outside the end_inclusion range, change it to null
        # (because there won't be any starttime labels available to match with it!)
        .with_columns([
            pl.when(pl.col('starttime_next') > pl.col('end_inclusion'))
            .then(pl.lit(None))
            .otherwise('starttime_next').alias('starttime_next')
        ])
        .collect()
    )

    # Convert the start_/end_inclusions to a unique episode number
    episode_nums = (
        labels
        .select('subject_id', 'start_inclusion', 'end_inclusion')
        .unique()
        .sort('subject_id', 'start_inclusion', 'end_inclusion')
        .with_columns([
            pl.struct('subject_id', 'start_inclusion', 'end_inclusion')
            .rank(method='ordinal').alias('episode_num')
        ])
    )

    # Add label episode and id_nums
    labels = (
        labels
        # Add episode nums
        .join(episode_nums, on=['subject_id', 'start_inclusion', 'end_inclusion'], how='inner')
        .drop('start_inclusion', 'end_inclusion')
        # Get our id_num (unique ID for all rows)
        .sort('subject_id', 'starttime')
        .with_row_index().rename({'index': 'label_id_num'})
    )

    # Bring in the id_num and action labels for the next label
    labels = (
        labels
        .join(
            labels.select(['subject_id', 'starttime', 'label_id_num', 'insulin_maintain', 'insulin_change',
                           'insulin_stop', 'insulin_delta_change']),
            left_on=['subject_id', 'starttime_next'],
            right_on=['subject_id', 'starttime'],
            how='left',
            suffix='_next'
        )
    )

    # Add in the step nums and first/last state columns
    labels = (
        labels
        # Get our step_nums
        .sort('subject_id', 'episode_num', 'starttime')
        .with_columns([
            pl.col('starttime').rank(method='ordinal').over('episode_num').alias('step_num')
        ])

        # Finally, identify our "first state" and "last state"
        .with_columns([
            pl.when(pl.col('step_num') == 1)
            .then(pl.lit(1))
            .otherwise(pl.lit(0)).alias('is_first_state')
        ])
        .with_columns([
            pl.when(pl.col('step_num') == pl.col('step_num').max().over('episode_num'))
            .then(pl.lit(1))
            .otherwise(pl.lit(0)).alias('is_last_state')
        ])
    )

    # Add in our previous and next BM values
    labels = (
        labels
        .sort('episode_num', 'step_num')
        .with_columns([
            pl.col('current_bm').shift(1).over('episode_num').alias('prev_bm'),
            (pl.col('starttime') - pl.col('starttime').shift(1).over('episode_num')).alias('time_since_prev_bm'),

            pl.col('current_bm').shift(-1).over('episode_num').alias('bm_next'),
            (pl.col('starttime').shift(-1).over('episode_num') - pl.col('starttime')).alias('time_until_bm_next'),
        ])
        .with_columns([
            # Convert to total_minutes (leave null as they are)
            pl.when(pl.col('time_since_prev_bm').is_not_null())
            .then(pl.col('time_since_prev_bm').dt.total_minutes())
            .otherwise(pl.duration(minutes=0)),

            pl.when(pl.col('time_until_bm_next').is_not_null())
            .then(pl.col('time_until_bm_next').dt.total_minutes())
            .otherwise(pl.duration(minutes=0)),
        ])
    )

    # N.B. we need to update our train/val/test ids to reflect the smaller labels DF
    train_ids, val_ids, test_ids = train_ids.tolist(), val_ids.tolist(), test_ids.tolist()
    filtered_ids = set(labels.select('subject_id').unique().to_series())
    all_ids = set(train_ids + val_ids + test_ids)
    removed_ids = all_ids - filtered_ids

    new_train_ids = np.array(list(set(train_ids) - removed_ids))
    new_val_ids = np.array(list(set(val_ids) - removed_ids))
    new_test_ids = np.array(list(set(test_ids) - removed_ids))

    return labels, new_train_ids, new_val_ids, new_test_ids


def get_drug_names():
    return ['Amiodarone', 'Amiodarone', 'Amiodarone', 'Amiodarone', 'Aciclovir', 'Ambisome', 'Amikacin', 'Ampicillin',
            'Ampicillin-Sulbactam', 'Azithromycin', 'Aztreonam', 'Co-trimoxazole', 'Caspofungin', 'Cefazolin',
            'Cefepime', 'Ceftazidime', 'Ceftriaxone', 'Ciprofloxacin', 'Chloramphenicol', 'Clindamycin', 'Colistin',
            'Daptomycin', 'Dextrose 5%', 'Dextrose 10%', 'Dextrose 20%', 'Dextrose 50%',
            'Doxycycline', 'Ertapenem', 'Erythromycin', 'Gentamicin', 'Levofloxacin', 'Meropenem',
            'Linezolid', 'Micafungin', 'Metronidazole', 'Nafcillin', 'Oxacillin', 'Piperacillin',
            'Piperacillin-Tazobactam', 'Rifampin', 'Tigecycline', 'Tobramycin', 'Vancomycin', 'Voriconazole',
            'Unfractionated Heparin', 'Levetiracetam', 'Phenytoin', 'Nitroglycerin', 'Nitroprusside', 'Labetalol',
            'FFP', 'Human Albumin Solution 25%', 'IVIG', 'Platelet infusion', 'PRBC', 'Furosemide', 'Furosemide',
            'Bumetanide', 'Regular Insulin', 'Insulin (TPN)', 'Hypertonic Saline', 'Mannitol', 'Aminophylline',
            'Sodium Bicarbonate 8.4%', 'Sodium Bicarbonate 8.4%', 'Fentanyl', 'Fentanyl', 'Fentanyl', 'Morphine',
            'Cisatracurium', 'Rocuronium', 'Vecuronium', 'Dexmedetomidine', 'Ketamine', 'Ketamine', 'Lorazepam',
            'Midazolam', 'Propofol', 'Propofol', 'Alteplase', 'Adrenaline', 'Dobutamine', 'Dopamine', 'Milrinone',
            'Noradrenaline', 'Vasopressin',
            # And our steroids drugs
            'Methylprednisolone_IV', 'Methylprednisolone_PO_or_NG', 'Prednisolone_PO_or_NG', 'Dexamethasone_IV',
            'Dexamethasone_PO_or_NG']


def get_lab_names():
    return ['ALT', 'AST', 'Albumin', 'ALP', 'Amylase', 'Anion Gap', 'Base Excess', 'Blood Gas pCO2', 'Blood Gas SpO2',
            'Blood Gas pO2', 'Urea', 'Ionised Calcium', 'Calcium', 'CRP', 'Chloride', 'Chloride', 'Creatinine',
            'Glucose', 'Bedside Glucose', 'Bedside Glucose', 'HCO3', 'Haematocrit', 'Haemoglobin', 'Lactate', 'LDH',
            'Lipase', 'pH', 'pH', 'Platelets', 'Potassium', 'Potassium', 'Prothrombin Time', 'Sodium', 'Sodium',
            'Bilirubin', 'Troponin - T', 'Blood Gas pCO2', 'Blood Gas pO2', 'WBC', 'Dialysis Blood Flow Rate',
            'Dialysate Rate', 'Dialysis Fluid Removal Rate']


def get_nutrition_names():
    return ['carbs_enteral', 'carbs_parenteral', 'protein_enteral', 'protein_parenteral']


def get_patient_ids(combined_data):
    # Check if train/val/test ids already exist:
    if os.path.exists(f'./data/mimic/patient_ids/test_patient_ids.npy'):
        segments = ['train', 'val', 'test']
        return tuple([np.load(f'./data/mimic/patient_ids/{segment}_patient_ids.npy')
                      for segment in segments])

    # Define our patient_ids
    patient_ids = combined_data.select('subject_id').unique().to_series().to_list()

    np.random.seed(42)
    np.random.shuffle(patient_ids)

    train_idx = round(0.8 * len(patient_ids))  # Set train proportion to 80%
    val_idx = round(0.9 * len(patient_ids))  # Set val (and test) proportion to 10%

    train_patient_ids, val_patient_ids, test_patient_ids = np.split(
        patient_ids, [train_idx, val_idx]
    )

    return train_patient_ids, val_patient_ids, test_patient_ids


def get_scaling_data_for_mimic(encoded_input_data, labels, train_patient_ids, input_window_size,
                               age_encoding, gender_encoding, weight_encoding):
    def create_feature_stats(df, feature_name, encoded_feature_number):
        return (
            df
            .select(
                [pl.lit(feature_name).alias('str_feature'),
                 pl.lit(encoded_feature_number).cast(pl.Int16).alias('encoded_feature'),
                 pl.col(feature_name).max().alias('values_max'),
                 pl.col(feature_name).min().alias('values_min'),
                 pl.col(feature_name).mean().alias('values_mean'),
                 pl.col(feature_name).std().alias('values_std')] +
                [pl.lit(val).alias(col) for val, col in fill_null_scales]
            )
        )

    train_labels = labels.filter(pl.col('subject_id').is_in(train_patient_ids))
    age_stats = train_labels.select(pl.col('age').cast(pl.Float64))
    gender_stats = train_labels.select(pl.col('gender').cast(pl.Float64))
    weight_stats = train_labels.select(pl.col('patientweight').cast(pl.Float64))

    encoded_input_data = (
        encoded_input_data
        # Very important that we only calculate for the train patient ids to avoid data leakage
        .filter(pl.col('subject_id').is_in(train_patient_ids))
        .sort('subject_id', 'featuretime', 'str_feature')
    )

    scaling_data = (
        encoded_input_data
        # We want to calculate delta_value relative to any previous value >= 15 minutes ago for the same subject/feature
        # (we filter out based on maximum input_window_size later)
        # To achieve this, we can do "join_asof" to get the "nearest" approximate time.
        .join_asof(
            encoded_input_data
            .select('subject_id', 'str_feature', pl.col('featuretime') + pl.duration(minutes=15), 'value'),
            on='featuretime', by=['subject_id', 'str_feature'], strategy='backward',
            coalesce=False,  # <- preserves featuretime_right even though we are joining on it
            check_sortedness=False, # <- polars can't do this when by=[] is given, but we have already sorted above
        )
        .with_columns([
            (pl.col('value') - pl.col('value_right')).alias('delta_value'),
            # Remember, featuretime_right is currently artificially shifted 15 minutes forward.
            (pl.col('featuretime') - (pl.col('featuretime_right') - pl.duration(minutes=15))).alias('delta_time')
        ])
        .drop('^*_right$')
        .with_columns([
            pl.when(pl.col('delta_time') > pl.duration(hours=input_window_size))
            .then(pl.lit(None).alias('delta_value'))
            .otherwise(pl.col('delta_value'))
        ])
        # Delta time and delta value done separately to avoid any interactions
        .with_columns([
            pl.when(pl.col('delta_time') > pl.duration(hours=input_window_size))
            .then(pl.lit(None).alias('delta_time'))
            .otherwise(pl.col('delta_time').dt.total_minutes())
        ])
        .group_by('str_feature', 'encoded_feature')
    )
    # Assemble our required columns
    aggregate_columns = []
    for col in ['value', 'delta_value', 'delta_time']:
        aggregate_columns.extend([
            pl.col(col).max().alias(f'{col}_max'),
            pl.col(col).min().alias(f'{col}_min'),
            pl.col(col).mean().alias(f'{col}_mean'),
            pl.col(col).std().alias(f'{col}_std')
        ])

    scaling_data = (
        scaling_data
        .agg(aggregate_columns)
        .select('str_feature', 'encoded_feature', pl.exclude('str_feature', 'encoded_feature'))
        .rename({f'value_{scale}': f'values_{scale}' for scale in ['max', 'min', 'mean', 'std']})
    )

    # Add in age/gender/weight
    fill_null_scales = []
    for group in ['delta_value', 'delta_time']:
        for val, scale in [[1, 'max'], [0, 'min'], [0, 'mean'], [1, 'std']]:
            fill_null_scales.extend([
                [val, f'{group}_{scale}']
            ])

    age_stats = create_feature_stats(age_stats, 'age', age_encoding)
    gender_stats = create_feature_stats(gender_stats, 'gender', gender_encoding)
    weight_stats = create_feature_stats(weight_stats, 'patientweight', weight_encoding)
    scaling_data = pl.concat([scaling_data, pl.concat([age_stats, gender_stats, weight_stats])], how='vertical_relaxed')
    # Fill std nulls for binary/rare events etc
    scaling_data = scaling_data.with_columns([pl.col('values_std').fill_null(1.)]
                                             + [pl.col(col).fill_null(val) for val, col in fill_null_scales])

    return scaling_data


def get_nutrition_values_for_mimic():
    with open('./nutrition_conversion_values.yaml', 'r') as f:
        nutrition_conversion_values = yaml.safe_load(f)
    return nutrition_conversion_values


def get_nutrition_variables_for_mimic():
    with open('./nutrition_variables.yaml', 'r') as f:
        nutrition_variables = yaml.safe_load(f)
    nutrition_variables = {key: key for key in nutrition_variables}
    return nutrition_variables


def get_variable_names_for_mimic():
    with open('./drug_lab_variable_names.yaml', 'r') as f:
        variable_names = yaml.safe_load(f)
    return variable_names


def join_dfs(first_df, second_df):
    df = (
        first_df
        .join(second_df,
              on='itemid',
              how='inner')
        .with_columns([pl.col('label')])
        .drop('itemid')
    )

    return df


def load_files(directory, filename):
    filepath = os.path.join('./data', directory, filename + '.parquet')
    return pl.scan_parquet(filepath)


def load_mimic(variable_names: dict = None):
    admissions = load_files('mimic/parquet', 'admissions')
    chartevents = load_files('mimic/parquet', 'chartevents')
    d_items = load_files('mimic/parquet', 'd_items')
    emar = load_files('mimic/parquet', 'emar')
    emar_detail = load_files('mimic/parquet', 'emar_detail')
    prescriptions = load_files('mimic/parquet', 'prescriptions')
    inputevents = load_files('mimic/parquet', 'inputevents')
    patients = load_files('mimic/parquet', 'patients')

    # Get our emar and non-emar variable names
    emar_variable_names = variable_names.pop('emar')

    # Select only the necessary columns from the chartevents and inputevents DataFrames
    chartevents = chartevents.select(
        ['subject_id', 'itemid', 'charttime', 'valuenum', 'valueuom']
    ).rename({'charttime': 'starttime'})

    inputevents = inputevents.select(
        ['subject_id', 'itemid', 'starttime', 'endtime', 'amount',
         pl.col('amountuom'),
         'originalrate', 'rate', pl.col('rateuom'), 'patientweight',
         pl.col('ordercategoryname'),
         pl.col('ordercategorydescription'),
         pl.col('statusdescription'), 'orderid']
    )

    emar = emar.select([
        'pharmacy_id', 'emar_id', 'charttime', 'event_txt'
    ]).rename({'charttime': 'starttime'})

    emar_detail = emar_detail.select([
        'emar_id', 'dose_given', 'dose_given_unit'
    ]).rename({'dose_given': 'amount', 'dose_given_unit': 'amountuom'})

    prescriptions = prescriptions.select([
        'subject_id', 'pharmacy_id', 'route', 'drug'
    ]).rename({'drug': 'label'})

    # Filter and rename the d_items DataFrame according to the variable_names dictionary
    d_items = (
        d_items
        .filter(pl.col('label').is_in(variable_names.keys()))
        .select(['itemid', 'label'])
        .with_columns(
            pl.col('label').replace(variable_names)
        )
    )

    chartevents = join_dfs(chartevents, d_items)
    inputevents = join_dfs(inputevents, d_items)

    # We want a special insulin for TPN, for the input data only (not part of the labels)
    inputevents = (
        inputevents
        .with_columns([
            pl.when((pl.col('label') == "Regular Insulin") & (pl.col('ordercategoryname') == "12-Parenteral Nutrition"))
            .then(pl.col('label').replace({'Regular Insulin': 'Insulin (TPN)'}))
            .otherwise(pl.col('label'))
        ])
    )

    # Don't need to join emar/prescriptions because they already have the medication name
    # - however, we do still need to filter for our required variables
    prescriptions = (
        prescriptions
        # Filter for the required variables using our special 'emar' tag (and rename)
        .filter(pl.col('label').is_in(emar_variable_names.keys()))
        # Filter for the route
        .filter(pl.col('route').is_in(['PO/NG', 'IV', 'PO']))
        # Rename the variables
        .with_columns(
            pl.col('label').replace(emar_variable_names)
        )
        .with_columns([
            pl.col('route').replace({'PO/NG': 'PO_or_NG', 'PO': 'PO_or_NG'}),
        ])
        .with_columns([
            pl.col('label') + '_' + pl.col('route'),
            pl.lit('Bolus').alias('ordercategorydescription')
        ])
        .drop('route')
        # Join with emar for the charttime
        .join(emar.filter(pl.col('event_txt') == "Administered").drop('event_txt'),
              on=['pharmacy_id'], how='inner')
        # Join with emar_detail for the drug dose
        .join(emar_detail.filter(pl.col('amount') != "___"), on=['emar_id'], how='inner')
        .drop('emar_id')
        .with_columns([pl.col('amount').cast(pl.Float64)])
        # Some drug doses are recorded as duplicates (e.g., 40mg pred might be recorded as 8 x 5mg tablets),
        # so aggregate these into a single dose
        .group_by(['subject_id', 'pharmacy_id', 'label', 'starttime', 'ordercategorydescription'])
        .agg(pl.col('amount').sum(), pl.col('amountuom').first())
        .drop('pharmacy_id')
        .unique()
    )

    # Create our combined_data DataFrame
    combined_data = (
        chartevents
        .join(inputevents, on=['subject_id', 'label', 'starttime'], how='full', coalesce=True)
        .join(prescriptions, on=['subject_id', 'label', 'starttime', 'amount', 'amountuom', 'ordercategorydescription'],
              how='full', coalesce=True)
        .select(
            ['subject_id', 'label', 'starttime', 'endtime', 'valuenum', 'valueuom', 'amount', 'amountuom',
             'originalrate', 'rate', 'rateuom', 'ordercategoryname', 'ordercategorydescription',
             'statusdescription', 'orderid', 'patientweight']
        )
        .collect()
    )

    # Define our patient_ids
    train_patient_ids, val_patient_ids, test_patient_ids = get_patient_ids(combined_data)

    return admissions, combined_data, patients, (train_patient_ids, val_patient_ids, test_patient_ids)


def merge_overlapping_rates(df):
    """
    Optimizes the merging of overlapping infusion rates using an event-based sweep line algorithm.

    Args:
        df (pl.DataFrame): The input DataFrame containing infusion records.

    Returns:
        pl.DataFrame: A DataFrame with merged, non-overlapping infusion intervals and summed rates.
    """
    # Drop unnecessary rows - originalrate, ordercategoryname, ordercategorydescription, statusdescription, orderid
    df = df.drop('originalrate', 'ordercategoryname', 'ordercategorydescription', 'statusdescription',
                 'orderid', strict=False)
    # Filter rows with non-null rates
    rate_df = df.filter(pl.col("rate").is_not_null())
    non_rate_df = df.filter(pl.col("rate").is_null())

    # Create start and end events
    start_rates = rate_df.drop('endtime')
    end_rates = (rate_df.drop('starttime').rename({'endtime': 'starttime'})
                 .with_columns([(-pl.col('rate')).alias('rate')]))

    # Combine start and end events
    rate_df = pl.concat([start_rates, end_rates])

    # Sort events by subject_id, label, and timestamp
    rate_df = rate_df.sort(["subject_id", "label", "starttime", "rate"])

    # Group by subject_id and label, then compute cumulative sum of rate changes
    rate_df = rate_df.with_columns([
        pl.col("rate").cum_sum().over(["subject_id", "label"])
    ])

    # Select the 'last' (most up-to-date) rate for each starttime
    rate_df = (
        rate_df
        .filter(pl.col('rate') == pl.col('rate').last().over('subject_id', 'label', 'starttime'))
        .unique()
    )

    # For the very rare occasion where patient weight differs between two rates, just set the average
    rate_df = (
        rate_df
        .with_columns([
            pl.when(pl.col('starttime').is_duplicated().over('subject_id', 'label'))
            .then(pl.col('patientweight').mean().over('subject_id', 'label', 'starttime'))
            .otherwise(pl.col('patientweight'))
        ])
        .unique()
    )

    # Finally, set any negative rates (due to float errors e.g., -1e-17) to 0.
    rate_df = rate_df.with_columns([
        pl.when(pl.col('rate') < 0)
        .then(pl.lit(0.0).alias('rate'))
        .otherwise(pl.col('rate'))])

    # Bring back to main DataFrame
    df = pl.concat([rate_df, non_rate_df.drop('endtime')])

    return df


def process_nutritional_info(df):
    """
    Several steps involved.
    1) For enteral feeding, we need to separate out enteral feeds into separate rates of delivery
    of protein, fat, and carbohydrates.
    2) For parenteral feeding, we need to do the same, but this has already been partially done
    - the patient will get "TPN w/ Lipids" as a volume and rate, and then an amount of Amino Acids (for example)
    which has the same orderid, but no rate. We need to separate these out.

    N.B. Currently there is NO lipid information available for TPN, and this is excluded from mimic.
    Fortunately, AFAIK, between protein and lipids, protein has the bigger effect on endogenous insulin / glucose spikes.
    """
    announce_progress('Processing nutritional information...')
    nutritional_value_conversion = get_nutrition_values_for_mimic()

    # Iterate through each nutritional feed
    for mode in ['enteral', 'parenteral']:
        for feed_key, feed_dict in nutritional_value_conversion[mode].items():
            # - separate out the protein and carbohydrate components (ignore lipids)
            feed_df = df.filter(pl.col('label') == feed_key)
            df = df.filter(pl.col('label') != feed_key)
            new_feed_df = df.filter(pl.col('amount').is_infinite())  # <- empty DataFrame

            for nutritional_key, conversion_factor in feed_dict.items():
                if nutritional_key not in ['protein', 'carbs']:
                    continue

                # - create a new DataFrame with the protein/carbs separated out
                new_feed_df = pl.concat([
                    new_feed_df,
                    feed_df.with_columns([
                        pl.lit(nutritional_key + f'_{mode}').alias('label'),
                        pl.col('amount') * conversion_factor,
                        pl.lit('grams').alias('amountuom'),
                    ])
                ])
            # Rejoin the two dataframes
            df = pl.concat([df, new_feed_df])

    # Remove any drink supplements - we only care about enteral/parenteral feeds!
    df = df.filter((pl.col('ordercategoryname') == "15-Supplements").not_() | pl.col('ordercategoryname').is_null())
    return df


def parse_delay(value):
    if isinstance(value, str):
        if value.lower() == 'none':
            return None
        try:
            return int(float(value))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid value for --delay: '{value}'. Must be an integer/float or 'None'."
            )
    else:
        return int(value)


def parse_bool(value, name: str = ''):
    if isinstance(value, str) and value.lower() in ['true', 'false']:
        return value.lower() == "true"
    elif isinstance(value, (int, float)) and value in [0, 1]:
        return bool(value)
    elif isinstance(value, bool):
        return value
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid value for {name}: '{value}'. Must be a boolean e.g., True or true.")


def progress_bar(iterable, with_time: bool = False):
    total_length = len(iterable)
    bar_length = 20
    bar = ' ' * bar_length
    sys.stdout.write('\r[>' + bar + '] ' + '0%')
    sys.stdout.flush()
    start = time.time() if with_time else None
    remaining = None
    for step, item in enumerate(iterable):
        yield step, item
        if with_time:
            end = time.time()
            duration = end - start  # in seconds
            speed = (step + 1) / duration  # in steps per second
            remaining = round((total_length - step + 1) / speed / 60, 1)  # in minutes

        progress = step / total_length
        filled_length = int(bar_length * progress)
        bar = '=' * filled_length + '>' + ' ' * (bar_length - filled_length)
        if with_time:
            statement = '\r[' + bar + '] ' + str(int(progress * 100)) + '%, ETA: ' + str(remaining) + 'min'
        else:
            statement = '\r[' + bar + '] ' + str(int(progress * 100)) + '%'
        sys.stdout.write(statement.ljust(50))
        sys.stdout.flush()
    if total_length > 0:
        sys.stdout.write('\r[' + '=' * bar_length + '] 100%'.ljust(50) + '\n')
        sys.stdout.flush()


def remove_outliers_from_mimic(df, train_patient_ids):
    """
    Goal is to remove outliers in the data.

    To choose these, we have to balance having lots of data (avoid extreme quantiles) with having a good range
    of states (include extreme quantiles). We set drug quantiles to 0.005/0.995, and labs to 0.001/0.999.
    """

    def find_min_max(measurement, column, quantile):
        if quantile == 1:
            minimum, maximum = -np.inf, np.inf
        else:
            measurements = df.filter((pl.col('label') == column)
                                     & (pl.col('subject_id').is_in(train_patient_ids))).select(pl.col(measurement))
            minimum, maximum = measurements.quantile(1 - quantile).item(), measurements.quantile(quantile).item()

        return minimum, maximum

    # Create a dictionary that we can save later
    outlier_reference = {}

    # Start by tidying up patient weight - set to Null, we will forward fill these values later
    min_weight, max_weight = df.select([pl.col('patientweight').quantile(0.005).alias('min'),
                                        pl.col('patientweight').quantile(0.9995).alias('max')])
    df = df.with_columns([
        pl.when(pl.col('patientweight').is_between(min_weight, max_weight, closed="both"))
        .then(pl.col('patientweight'))
        .otherwise(pl.lit(None))
    ])

    outlier_reference['patientweight'] = {'min': min_weight, 'max': max_weight}

    # Get our variable names
    antibiotic_variables = get_antibiotic_names()
    drug_variables = get_drug_names()
    lab_variables = get_lab_names()
    nutrition_variables = get_nutrition_names()

    # Iterate through all the variables in the DataFrame
    for variable in df.select('label').unique().to_series().to_list():
        # Antibiotics are set to binary boluses, so we can skip these
        if variable in antibiotic_variables:
            continue

        matches_variable = pl.col('label') == variable
        is_drug = variable in drug_variables or variable in nutrition_variables
        is_lab = variable in lab_variables

        if not is_lab:
            if not is_drug:
                raise ValueError(f'Variable {variable} not found in known variables')
            # Unlikely to be relevant but filter out any 0 boluses
            else:
                df = (
                    df
                    .filter(matches_variable.not_() |
                            (matches_variable & (
                                    pl.col('rate').is_not_null() |
                                    (pl.col('bolus') > 0)
                            )))
                )
        # rate_min set to 0 by default (as rate can always be stopped!)
        _, rate_max = find_min_max('rate', variable, 0.995) if is_drug else (0, np.inf)
        bolus_min, bolus_max = find_min_max('bolus', variable, 0.995) if is_drug else (0, np.inf)
        lab_min, lab_max = find_min_max('valuenum', variable, 0.999) if is_lab else (None, None)

        if is_lab:
            outlier_reference[variable] = {'min': lab_min, 'max': lab_max}
        else:
            outlier_reference[variable] = {'bolus': {'min': bolus_min, 'max': bolus_max},
                                           'rate': {'min': 0, 'max': rate_max}}

        bolus_not_in_range = pl.col('bolus').is_between(bolus_min, bolus_max, closed="both").not_()
        rate_not_in_range = pl.col('rate').is_between(0, rate_max, closed="both").not_()
        lab_not_in_range = pl.col('valuenum').is_between(lab_min, lab_max, closed="both").not_()

        df = (
            df.with_columns([
                # Clip drugs to range - this is an intervention and shouldn't be ignored completely
                pl.when(matches_variable & is_drug & rate_not_in_range)
                .then(pl.col('rate').clip(0, rate_max))
                .otherwise(pl.col('rate')),

                pl.when(matches_variable & is_drug & bolus_not_in_range)
                .then(pl.col('bolus').clip(bolus_min, bolus_max))
                .otherwise(pl.col('bolus')),

                # Set labs to null if outside range i.e., ignore these values completely
                pl.when(matches_variable & is_lab & lab_not_in_range)
                .then(pl.lit(None).alias('valuenum'))
                .otherwise(pl.col('valuenum')),

                pl.when(matches_variable & is_lab & lab_not_in_range)
                .then(pl.lit(None).alias('valueuom'))
                .otherwise(pl.col('valueuom'))
            ])
        )

    # Remove all the 'non-existent' rows
    df = (
        df.filter(
            pl.col('rate').is_not_null() | pl.col('valuenum').is_not_null() | pl.col('bolus').is_not_null())
    )

    # Save our outlier reference file (pickled)
    with open('./data/outlier_reference.pkl', 'wb') as f:
        pickle.dump(outlier_reference, f)
        f.close()

    return df


def resize_hdf5_datasets(h5_array: h5py.File, next_size: int, label_features: list = None,
                         context_window: int = 400):
    for embedding in ['future_indices', 'features', 'timepoints', 'values', 'delta_time', 'delta_value']:
        for i in ['', '_next']:
            if i == '_next' and embedding == 'future_indices':
                continue
            h5_array[embedding+i].resize((next_size, context_window, 1))

    h5_array['labels'].resize((next_size, len(label_features)))

    return h5_array


def save_patient_ids(train_patient_ids, val_patient_ids, test_patient_ids):
    patient_ids_dir = f'./data/mimic/patient_ids'
    if not os.path.exists(patient_ids_dir):
        os.makedirs(patient_ids_dir, exist_ok=True)

    np.save(os.path.join(patient_ids_dir, 'train_patient_ids.npy'), train_patient_ids)
    np.save(os.path.join(patient_ids_dir, 'val_patient_ids.npy'), val_patient_ids)
    np.save(os.path.join(patient_ids_dir, 'test_patient_ids.npy'), test_patient_ids)

    return


def separate_basal_bolus_in_mimic(df):
    """
    We need to separate out 'rates' from 'boluses'.

    To do this, we follow the following rules:

       1) Remove any rows where the endtime is before / equal to the starttime
       2) For doses given over 1 minute, fill in any missing amount data, and remove rate
       3) For doses given over >1 minute, fill in any missing rate data, and then remove amount
       4) Rename amount to bolus

    """

    # 1) Remove rows where the endtime is before / equal to the starttime - hopefully shouldn't apply!
    df = df.filter((pl.col('endtime') > pl.col('starttime'))
                   | (pl.col('endtime').is_null()))

    df_no_rates = df.filter(pl.col('valuenum').is_not_null())
    df = df.filter(pl.col('valuenum').is_null())

    # 2) For all Med Boluses, remove the rate, and any rows where amount = 0
    criterion_1 = pl.col('ordercategorydescription').is_in(['Drug Push', 'Bolus'])
    criterion_2 = pl.col('amount') > 0

    df = (
        # Remove boluses equal to zero
        df.filter(
            # Is med bolus with amount over zero
            (criterion_1 & criterion_2)
            # Or is not med bolus i.e., is continuous rate
            | criterion_1.not_()
        )
        # And remove rate and original rate from the boluses
        .with_columns([
            pl.when(criterion_1)
            .then(pl.lit(None).alias('rate'))
            .otherwise(pl.col('rate')),

            pl.when(criterion_1)
            .then(pl.lit(None).alias('rateuom'))
            .otherwise(pl.col('rateuom'))
        ])
    )

    # 3) For all Continuous Meds, remove the amount and remove any rows where rate = 0 (should be none)
    criterion_1 = pl.col('ordercategorydescription').is_in(['Continuous Med', 'Continuous IV'])
    criterion_2 = pl.col('amount') > 0

    df = (
        # Remove infusions with rate equal to zero
        df.filter(
            # Is infusion with amount over zero
            (criterion_1 & criterion_2)
            # Or is not infusion i.e., is med bolus
            | criterion_1.not_()
        )
        # For infusions, convert amount to per-hour rate and then remove amount - currently independent of weight
        .with_columns([
            pl.when(criterion_1)
            .then(pl.col('amount').alias('rate') / ((pl.col('endtime') - pl.col('starttime')).dt.total_minutes() / 60))
            .otherwise(pl.col('rate')),

            pl.when(criterion_1)
            .then((pl.col('amountuom') + "/hour").alias('rateuom'))
            .otherwise(pl.col('rateuom'))
        ])

        # And remove amount/amountuom from the infusions
        .with_columns([
            pl.when(criterion_1)
            .then(pl.lit(None).alias('amount'))
            .otherwise(pl.col('amount'))])
        .unique()
    )

    # Merge back together
    df = pl.concat([df, df_no_rates])

    # 5) Rename 'amount' to 'bolus', and 'amountuom' to 'bolusuom'
    df = df.rename({'amount': 'bolus', 'amountuom': 'bolusuom'})

    return df


def split_labels_to_rate_and_bolus(df, *args):
    """
    We want a single 'value' column - to facilitate this for drugs, we create separate labels for 'rate' and 'bolus'
    """
    select_cols = ['subject_id', 'input_id_num', 'feature', 'starttime',
                    # Move 'bolus' into 'value' for boluses
                    pl.when(pl.col('bolus').is_not_null())
                   .then(pl.col('bolus').alias('value'))

                   # Convert 'rate' into 'value' for rates
                   .otherwise(pl.when(pl.col('rate').is_not_null())
                              .then(pl.col('rate').alias('value'))

                              # Keep 'valuenum' as 'value' for all other features
                              .otherwise(pl.col('valuenum')))] + [arg for arg in args]

    df = (
        df
        .with_columns([
            # Rename feature for boluses to feature + ' bolus'
            pl.when(pl.col('bolus').is_not_null())
            .then((pl.col('feature') + ' bolus').alias('feature'))

            # Rename feature for original rates to feature + ' rate'
            .otherwise(pl.when(pl.col('rate').is_not_null())
                       .then(
                (pl.col('feature') + ' rate').alias('feature'))

                       # Keep feature for valuenum as-is
                       .otherwise(pl.col('feature'))),
        ])
        .select(select_cols)
        .unique()
    )
    return df

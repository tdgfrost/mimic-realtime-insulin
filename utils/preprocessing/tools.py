import argparse
import os
import shutil
import sys
import time
import itertools
import numpy as np
import math
import polars as pl
import yaml
import tempfile
from safetensors.numpy import save_file

pl.Config.set_tbl_cols(10000)
pl.Config.set_tbl_width_chars(10000)
pl.Config.set_tbl_rows(50)
pl.Config.set_fmt_str_lengths(10000)


class StateNormalizer:
    def __init__(self, stats_yaml_path: str, mapping_yaml_path: str):
        with open(stats_yaml_path, 'r') as f:
            self.stats = yaml.safe_load(f)

        with open(mapping_yaml_path, 'r') as f:
            self.mapping = yaml.safe_load(f)

        self.num_features = max(self.mapping.keys()) + 1

        # Pre-allocate lookup arrays for the arcsinh stats
        self.val_arcsinh_mean = np.zeros(self.num_features, dtype=np.float32)
        self.val_arcsinh_std = np.ones(self.num_features, dtype=np.float32)

        # Populate the lookup arrays using the mapping
        for feature_idx, feature_name in self.mapping.items():
            if feature_name in self.stats:
                feat_stats = self.stats[feature_name]
                self.val_arcsinh_mean[feature_idx] = feat_stats['value_arcsinh_mean']
                # Prevent division by zero
                self.val_arcsinh_std[feature_idx] = feat_stats['value_arcsinh_std'] or 1.0

        # Extract time stats
        self.ft_log_mean = self.stats['time']['time_log_mean']
        self.ft_log_std = self.stats['time']['time_log_std']

    def transform(self, state_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Normalizes the state dictionary by key.
        """
        col_enc = 'feature'
        col_ft = 'time'
        col_val = 'value'

        # Isolate the encoded features and handle NaNs for safe array indexing
        features = state_dict[col_enc]
        nan_mask = np.isnan(features)
        safe_feat_idx = np.where(nan_mask, 0, features).astype(int)

        # 1. Normalize time (log1p-normalized)
        ft = state_dict[col_ft]
        state_dict[col_ft] = (np.log1p(ft) - self.ft_log_mean) / self.ft_log_std

        # 2. Normalize value (arcsinh-normalized)
        val = state_dict[col_val]
        v_mean = self.val_arcsinh_mean[safe_feat_idx]
        v_std = self.val_arcsinh_std[safe_feat_idx]
        state_dict[col_val] = (np.arcsinh(val) - v_mean) / v_std

        # Re-apply NaNs to padded rows where feature was NaN
        state_dict[col_ft][nan_mask] = np.nan
        state_dict[col_val][nan_mask] = np.nan

        return state_dict


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


def build_demographics():
    """
    The following can be used to generate some baseline demographics for the patients within Insulin4RL.

    Baseline Covariates:
    - Age <- patients
    - Sex <- patients
    - Race/ethnicity <- admissions
    - Admission type <- admissions
    - Insurance <- admissions
    - Language <- admissions
    - Marital status <- admissions
    - Comorbidities <- diagnoses_icd

    Paper reference for codes: "Coding algorithms for defining comorbidities in ICD-9-CM and ICD-10 administrative data"
    by Quan et al. (2005)
    """

    # Get our various indexes/codes
    comorbidity_index, admission_index, ethnicity_index = get_demographic_information_for_mimic()

    # Load our diagnoses - we use Enhanced ICD-9-CM (first row) + ICD-10 (second row) codes.
    diagnoses = load_files('diagnoses_icd')
    diagnoses = diagnoses.with_columns([pl.lit(None).alias('comorbidity')])
    for key, val in comorbidity_index.items():
        diagnoses = diagnoses.with_columns(
            [pl.when(pl.col('icd_code').str.contains(key)).then(pl.lit(val)).otherwise('comorbidity').alias(
                'comorbidity')])

    diagnoses = diagnoses.filter(pl.col('comorbidity').is_not_null()).collect()

    # Get our df
    admissions = load_files('admissions')
    df = pl.scan_parquet('./data/insulin4rl/all_data.parquet')

    demographic_info = (
        df
        .select('subject_id', 'episode_num', 'is_done', 'labeltime')
        .filter(pl.col('is_done').cast(pl.Boolean))
        .join(admissions.select('subject_id', 'hadm_id', 'admittime', 'dischtime', 'admission_type',
                                'insurance', 'language', 'marital_status', 'race'), on=['subject_id'], how='inner')
        .filter(pl.col('labeltime').is_between('admittime', 'dischtime'))
        .select('episode_num', 'hadm_id', 'admission_type', 'insurance', 'language',
                'marital_status', 'race')
        .collect()
    )

    hadm_ids = demographic_info.select('episode_num', 'hadm_id')

    # Join with our diagnoses as a one-hot wide table
    episode_comorbidities = (
        diagnoses
        .join(hadm_ids, on=['hadm_id'], how='inner')
        .select('episode_num', 'comorbidity')
        .unique()
        .with_columns([pl.lit(1).alias('flag')])
        # Pivot into wide one-hot encoded format
        .pivot(
            index="episode_num",
            on="comorbidity",
            values="flag"
        )
        .fill_null(0)
        .sort("episode_num")
    )

    # Get our age, sex, race/ethnicity, admission type, insurance, language, marital status
    patients = load_files('patients')

    age_sex_information = (
        df
        .select('subject_id', 'episode_num', 'step_num', 'labeltime')
        .filter((pl.col('step_num') == 1).cast(pl.Boolean))
        .join(patients.select('subject_id', 'gender', 'anchor_age', 'anchor_year'), on=['subject_id'], how='inner')
        .with_columns(((pl.col('labeltime').dt.year() - pl.col('anchor_year')) + pl.col('anchor_age')).alias('age'))
        .select('episode_num', 'age', 'gender')
        .collect()
    )

    # Join with our demographic info
    episode_demographics = (
        demographic_info
        .join(age_sex_information, on='episode_num', how='inner')
        # Set null for marital status as UNKNOWN
        .with_columns([
            pl.when(pl.col('marital_status').is_null())
            .then(pl.lit("UNKNOWN_MARITAL_STATUS"))
            .otherwise('marital_status').alias('marital_status')
        ])
    )
    # Group rarer ethnicity labels together
    episode_demographics = (
        episode_demographics
        .with_columns([
            pl.col('race').replace(ethnicity_index)
        ])
    )

    # Simplify admission types
    episode_demographics = (
        episode_demographics
        .with_columns([
            pl.col('admission_type').replace(admission_index)
        ])
    )

    # Convert insurance nulls to 'UNKNOWN'
    episode_demographics = (
        episode_demographics
        .with_columns([
            pl.when(pl.col('insurance').is_null())
            .then(pl.lit("UNKNOWN_INSURANCE"))
            .otherwise(pl.col('insurance').str.to_uppercase())
            .replace({'OTHER': 'OTHER_INSURANCE'})
            .alias('insurance')
        ])
    )

    # One-hot encode admission_type, marital_status, race, gender
    # (currently excluding language)
    episode_demographics = episode_demographics.with_columns([pl.lit(1).alias('flag')])
    one_hot_episode_demographics = episode_demographics.select('episode_num', 'hadm_id')
    for col in ['admission_type', 'marital_status', 'race', 'gender', 'insurance']:
        one_hot_episode_demographics = (
            episode_demographics
            .with_columns([pl.lit(1).alias('flag')])
            # Pivot into wide one-hot encoded format
            .pivot(
                index=["episode_num", "hadm_id"],
                on=col,
                values="flag"
            )
            .fill_null(0)
            .join(one_hot_episode_demographics, on=['episode_num', 'hadm_id'], how='inner')
        )

    # Join with our comorbidities
    one_hot_episode_demographics = (
        one_hot_episode_demographics
        .join(
            episode_comorbidities,
            on=['episode_num'],
            how='inner')
        # Join in age
        .join(
            episode_demographics.select('episode_num', 'age'),
            on=['episode_num'],
            how='inner'
        )
    )

    # Specifically, collect the following information
    # -28-day mortality from the final label
    # - total minutes duration of the episode
    # - number of steps per episode
    # - patient weight information
    # - number of hypoglycaemic events
    # - number of hyperglycaemic events (>10 mmol/L)
    with open('./data/insulin4rl/metadata/feature_mapping.yaml', 'r') as f:
        feature_mapping = {val: key for key, val in yaml.safe_load(f).items()}
        patientweight_feature_number = feature_mapping['patientweight']


    label_information = (
        df
        .select('episode_num', 'step_num', 'steps_per_episode', 'minutes_remaining', 'current_bm',
                '28-day-alive-final',

                # Get the patient weight
                pl.col('value')
                .explode()
                .filter(pl.col('feature').explode() == patientweight_feature_number)
                .alias('patientweight'))
        .group_by('episode_num')
        .agg([
            pl.col('steps_per_episode').max().alias('TOTAL_STEPS_PER_EPISODE'),
            (pl.col('minutes_remaining').max() / 60 / 24).alias('EPISODE_DURATION_DAYS'),
            (pl.col('current_bm') < 4.0).sum().alias('NUM_HYPOGLYCAEMIC_EVENTS'),
            (pl.col('current_bm') > 10.0).sum().alias('NUM_HYPERGLYCAEMIC_EVENTS'),
            (1 - pl.col('28-day-alive-final')).max().alias('28_DAY_MORTALITY'),
            pl.col('patientweight').alias('PATIENT_WEIGHT').mean()
        ])
        .collect()
    )

    # Join with our demographics
    one_hot_episode_demographics = (
        one_hot_episode_demographics
        .join(label_information, on='episode_num', how='inner')
    )

    one_hot_episode_demographics.write_parquet('./data/insulin4rl/metadata/demographics.parquet')


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
        .rename({'index': 'input_id'})
    )

    return combined_data


def convert_dataframe_to_mmap_safetensors(parquet_path: str, output_dir: str, chunk_size: int = 50_000):
    """
    Scans the aggregated parquet file, filters by train/val/test segments,
    processes the data in memory-safe chunks via temporary mmaps, and saves
    individual memory-mapped safetensors binaries for each category.
    """
    df = pl.scan_parquet(parquet_path)
    segments = ['train', 'val', 'test']
    context_length = df.head(1).select(pl.col('value').list.len()).collect().item()

    state_cols = ['feature', 'time', 'value']
    action_cols = ['insulin_maintain', 'insulin_change', 'insulin_stop', 'insulin_delta_change']
    reward_cols = [
        'next_bm', '1-day-alive', '3-day-alive', '7-day-alive', '14-day-alive', '28-day-alive',
        '1-day-alive-final', '3-day-alive-final', '7-day-alive-final', '14-day-alive-final',
        '28-day-alive-final'
    ]
    info_cols = [
        'label_id', 'episode_num', 'step_num', 'steps_per_episode', 'steps_remaining', 'minutes_remaining',
        'current_bm', 'prev_bm', 'time_since_prev_bm', 'time_until_next_bm', 'insulin_old_rate', 'insulin_new_rate'
    ]

    normalizer = StateNormalizer('./data/insulin4rl/metadata/feature_stats.yaml',
                                 './data/insulin4rl/metadata/feature_mapping.yaml')

    def get_segment_df(_df, _offset, _chunk_size):
        feature_map = {
            "value": ["feature", "time"],
            "value_next": ["feature_next", "time_next"],
        }
        return (
            _df
            .slice(_offset, _chunk_size)
            .with_row_index('tmp_idx')
            .with_columns([
                pl.when(pl.col(mask_col).explode().is_not_nan())
                .then(pl.col(target_cols).explode().cast(pl.Float32))
                .otherwise(pl.lit(np.nan))
                .implode(maintain_order=True)
                .over("tmp_idx")
                for mask_col, target_cols in feature_map.items()
            ])
            .drop('tmp_idx')
            .with_columns([
                pl.col(f'{col}{i}').cast(pl.List(pl.Float32))
                for col in state_cols for i in ['', '_next'] if col not in feature_map.values()
            ])
            .collect(engine='streaming')
        )

    for segment in segments:
        print(f"\nProcessing segment: {segment}")
        segment_lf = df.filter(pl.col('data_segment') == segment)
        total_rows = segment_lf.select(pl.len()).collect().item()

        if total_rows == 0:
            print(f"No data found for segment '{segment}'. Skipping.")
            continue

        segment_dir = os.path.join(output_dir, segment)
        os.makedirs(segment_dir, exist_ok=True)

        num_chunks = math.ceil(total_rows / chunk_size)
        print(f"Total rows: {total_rows}. Splitting into {num_chunks} chunks.")

        # Map each category to its columns and the expected shape of each individual tensor
        category_configs = {
            'states': (state_cols, (total_rows, context_length, 1), np.float32),
            'next_states': (state_cols, (total_rows, context_length, 1), np.float32),
            'actions': (action_cols, (total_rows, 1), np.float32),
            'next_actions': (action_cols, (total_rows, 1), np.float32),
            'reward_markers': (reward_cols, (total_rows, 1), np.float32),
            'infos': (info_cols, (total_rows, 1), np.float32),
            'dones': (['is_done'], (total_rows, 1), bool)
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize a nested dictionary to hold temporary memmaps for every single column
            category_mmaps = {cat: {} for cat in category_configs.keys()}

            for cat, (cols, shape, dtype) in category_configs.items():
                for col in cols:
                    temp_file_path = os.path.join(temp_dir, f'{cat}_{col}.dat')
                    category_mmaps[cat][col] = np.memmap(temp_file_path, mode='w+', dtype=dtype, shape=shape)

            for _, chunk_idx in progress_bar(range(num_chunks), with_time=True, time_unit='sec'):
                offset = chunk_idx * chunk_size
                chunk_df = get_segment_df(segment_lf, offset, chunk_size)
                end_idx = offset + len(chunk_df)

                # 1. States -> Extract dict, normalize, assign to mmap
                states_dict = {col: np.stack(chunk_df[col].to_numpy()) for col in state_cols}
                states_dict = normalizer.transform(states_dict)
                for col in state_cols:
                    # Append an empty dimension to reshape (N, L) to (N, L, 1)
                    category_mmaps['states'][col][offset:end_idx] = states_dict[col][:, :, None]

                # 2. Next States -> Extract dict, assign standard keys, normalize, and assign to mmap
                next_states_dict = {col: np.stack(chunk_df[f'{col}_next'].to_numpy()) for col in state_cols}
                next_states_dict = normalizer.transform(next_states_dict)
                for col in state_cols:
                    category_mmaps['next_states'][col][offset:end_idx] = next_states_dict[col][:, :, None]

                # 3-7. Actions, Rewards, Infos, Dones -> Extract and reshape to (N, 1)
                for col in action_cols:
                    category_mmaps['actions'][col][offset:end_idx] = chunk_df[col].to_numpy().astype(np.float32)[
                        :, None]

                for col in action_cols:
                    category_mmaps['next_actions'][col][offset:end_idx] = chunk_df[f'{col}_next'].to_numpy().astype(np.float32)[
                        :, None]

                category_mmaps['dones']['is_done'][offset:end_idx] = \
                chunk_df['is_done'].to_numpy().astype(bool)[:, None]

                for col in reward_cols:
                    category_mmaps['reward_markers'][col][offset:end_idx] = chunk_df[col].to_numpy().astype(np.float32)[
                        :, None]

                for col in info_cols:
                    category_mmaps['infos'][col][offset:end_idx] = chunk_df[col].to_numpy().astype(np.float32)[:, None]

            # Flush all temporary mmaps to disk
            for cat_dict in category_mmaps.values():
                for mmap_array in cat_dict.values():
                    mmap_array.flush()

            # Save individual safetensors files for each category
            for cat, cat_dict in category_mmaps.items():
                safetensors_path = os.path.join(segment_dir, f'{cat}.safetensors')
                save_file(cat_dict, safetensors_path)


def create_final_dataframe(encoded_input_data, labels, encodings, train_patient_ids, val_patient_ids, test_patient_ids,
                           sorting_columns, grouping_columns, context_length, low_memory=False):
    """
    This is the final pipeline for taking the encoded data and labels, and creating dedicated training data.


    N.B. We are creating a "sliding window" of data (with measurements being repeated across potentially
    many rows), which means the data is LARGE even with .parquet compression, and may be slow to complete.
    """
    # Convert encodings
    for key in ['age', 'gender', 'weight']:
        encodings[key] = np.int16(encodings[key])

    # Iterate through each data segment
    intermediate_path = './data/insulin4rl/_intermediate_data/'
    final_path = './data/insulin4rl'
    for patient_ids, group in [[train_patient_ids, 'train'], [val_patient_ids, 'val'], [test_patient_ids, 'test']]:
        if patient_ids is None:
            continue
        print(f'Processing {group}...')
        dataframe_path = os.path.join(intermediate_path, group)
        os.makedirs(dataframe_path, exist_ok=True)

        chunk_size = 100  # number of patient ids to process at once
        chunk_size = len(patient_ids) // chunk_size + 1

        join_cols = ['subject_id']

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
                .filter((pl.col('input_id_labels').list.contains(pl.col('input_id')).not_()
                         | (pl.col('input_id_labels').list.eval(pl.element().is_null().all()).explode())))
                .drop('input_id', 'input_id_labels')
                # Filter the measurements to the inclusion period for each label
                .filter(
                    (pl.col('time') >= pl.col('start_inclusion'))
                    & (pl.col('time') <= pl.col('end_inclusion')))
                .drop('start_inclusion')
                # Select the desired columns
                .select(sorting_columns + [
                    # Featuretime is now minutes UNTIL the end of the inclusion period
                    # (i.e., how many minutes ago, relative to right now)
                    (pl.col('end_inclusion') - pl.col('time')).dt.total_minutes().cast(pl.Int16).alias(
                        'time'),
                    'value', 'feature',
                    # For all our labels, just group these together into a Polars Struct for simplicity
                    pl.struct(pl.exclude(*sorting_columns, 'end_inclusion', 'time', 'value', 'feature')).alias('targets')
                ])
                .collect()
                .lazy()
            )

            # As we are limited to 400 measurements in a 7-day window, we want to make sure all current drug infusions
            # are included in the input data. The risk is that in edge cases where we have LOADS of recent measurements,
            # ongoing drug infusions might get missed out as a result.
            current_drug_infusions = (
                temp
                .filter(pl.col('feature').is_in(encodings['drug_names']))
                .rename({'feature': 'current_drug_feature', 'value': 'current_drug_value'})
                .sort(by=sorting_columns + ['time'])
                # Following is equivalent to group_by, but is MUCH faster than using group_by directly
                .select(pl.all().first().over(grouping_columns + ['current_drug_feature']))
                .unique()
                # Filter out drugs that have been stopped (i.e., value = 0)
                .filter(pl.col('current_drug_value') > 0)
                # Fill in the time column with 0
                .with_columns([pl.lit(0).cast(pl.Int16).alias('current_drug_time')])
                .drop('time')
                .group_by(grouping_columns)
                .agg(pl.all())
            )

            # Next, we will identify all historic drug rates and all other measurements.
            historic_events = (
                temp
                # Sorted from new to old (because increasing time = older measurement)
                .sort(by=sorting_columns + ['time', 'feature'])
                .with_columns([
                    # Get the 'rank' of each lab measurement
                    # i.e., the number of times the feature has appeared
                    # (we want to prioritise "unseen" lab features before repeat lab features)
                    pl.when(pl.col('feature').is_in(encodings['lab_names']))
                    .then(pl.col('feature').cum_count().over(sorting_columns + ['feature']))
                    .otherwise(pl.lit(1))
                    .alias('feature_rank')
                ])
                # Sort all historic lab measurements by 1) the feature rank, and then 2) the feature time
                .sort(sorting_columns + ['feature_rank', 'time'])
                .group_by(grouping_columns)
                .agg(pl.all())
            )

            # Add back in our "current drug" measurements
            target_data = (
                historic_events
                .join(current_drug_infusions, on=grouping_columns, how='full', coalesce=True)
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

                    for first_col, second_col in [['current_drug_time', 'time'],
                                                  ['current_drug_value', 'value'],
                                                  ['current_drug_feature', 'feature']]
                ])
                .drop('current_drug_feature', 'current_drug_value', 'current_drug_time', 'feature_rank')
            )

            input_feature_columns = ['time', 'feature', 'value']

            # Now for each label, filter to just 397 measurements (i.e., 400 - 3 for age/gender/weight),
            # with nulls added if necessary. Sort time from old -> new -> NaN right padding
            target_data = (
                target_data
                .with_columns([
                    pl.col(col)
                    # Isolate up to (max_context_window - 3) items, with NaN right-padding
                    .list.concat([None for _ in range(context_length - 3)])
                    .list.slice(0, context_length - 3)
                    for col in input_feature_columns
                ])
                .with_columns([
                    # Temporarily explode our feature columns
                    pl.col(input_feature_columns).explode()
                    # Sort by time (old to new)
                    .sort_by(pl.col('time').explode(), descending=True, maintain_order=True, nulls_last=True)
                    # Collapse back to lists again
                    .implode(maintain_order=True).over('episode_num', 'step_num')
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
                .drop('age', 'gender', 'patientweight')  # 'minutes_since_admission'
                # - concat [0, 0, 0] to the start of time
                .with_columns([pl.concat_list(pl.lit([np.int16(0) for _ in range(3)]),
                                              pl.col('time'))
                              .alias('time')])
                # - concat encoded feature values for [age, gender, patientweight] to the start of feature
                .with_columns([pl.concat_list(pl.lit([encodings['age'], encodings['gender'], encodings['weight']
                                                      ]),
                                              pl.col('feature'))
                              .alias('feature')])
            )

            # For dtype efficiency, change some of the nulls to -1 (and NaN for the rest)
            target_data = (
                target_data
                .with_columns([
                    pl.col('time').list.eval(pl.element().fill_null(-1)),
                    pl.col('value').list.eval(pl.element().fill_null(np.nan)),
                    pl.col('feature').list.eval(pl.element().fill_null(0)),
                ])
                .collect()
                .lazy()
            )

            # We need to then get our "next state", defined as 24hrs from now (same as our viewing window)
            has_next_state = target_data.filter(pl.col('label_id_next').is_not_null())

            has_next_state = (
                has_next_state
                .join(target_data.select(['labeltime', 'label_id'] + input_feature_columns),
                      left_on=['labeltime_next', 'label_id_next'],
                      right_on=['labeltime', 'label_id'], how='inner', suffix='_next')
            )

            no_next_state = target_data.filter(pl.col('label_id_next').is_null())

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
                    if col == 'time'
                ])

                .with_columns([
                    pl.when(pl.col(col + '_next').is_null())
                    .then(pl.lit([0 for _ in range(context_length)]).cast(pl.List(pl.Int16)).alias(col + '_next'))
                    .otherwise(col + '_next') for col in input_feature_columns
                    if 'feature' in col
                ])

                .with_columns([
                    pl.when(pl.col(col + '_next').is_null())
                    .then(pl.lit([np.nan for _ in range(context_length)]).alias(col + '_next'))
                    .otherwise(col + '_next') for col in input_feature_columns if 'value' in col
                ])
            )

            # Check our indexes are still in order
            index_checker = final_target_data.sort('label_id').select('episode_num', 'step_num', 'label_id', 'label_id_next')
            next_label_id_doesnt_match = pl.col('label_id_next') != pl.col('label_id').shift(-1).over('episode_num')
            not_last_label_id = ~(pl.col('label_id').is_last_distinct().over('episode_num'))
            if (
                index_checker
                .select((next_label_id_doesnt_match & not_last_label_id).any())
                .collect()
                .item()
            ):
                raise Exception("Label_id_next doesn't match the next sequential step - major error, please check data.")

            # Add our data segment name (train/val/test)
            final_target_data = (
                final_target_data
                .with_columns([
                    pl.lit(group).alias('data_segment')
                ])
            )

            # Check we have nulls in the correct places for 'next states' when current state is terminal
            final_target_data = (
                final_target_data
                .with_columns([
                    pl.when(pl.col('is_done') == 0)
                    .then(col)
                    .otherwise(pl.lit(None))

                    for col in ['labeltime_next', 'next_bm', 'time_until_next_bm']
                ])
                .with_columns([
                    pl.when(pl.col('step_num') > 1)
                    .then(col)
                    .otherwise(pl.lit(None))

                    for col in ['prev_bm', 'time_since_prev_bm']
                ])
            )

            # Here is our final column order
            columns = [
                # Unique identifiers:
                'data_segment',         # train/val/test
                'subject_id',           # unique patient identifier, same as MIMIC
                'label_id',             # unique label identifier
                'label_id_next',        # unique label identifier for next step (if any)
                'episode_num',          # unique episode number
                'step_num',             # unique step number for that episode

                # Temporal context
                'labeltime',            # MIMIC timestamp for label (rounded to nearest 5 minutes)
                'labeltime_next',       # MIMIC timestamp for next label (rounded to nearest 5 minutes)
                'steps_per_episode',    # Steps in the episode
                'steps_remaining',      # Steps remaining in the episode
                'minutes_remaining',    # Minutes remaining in the episode
                'is_done',              # Whether this is the final available state

                # Physiology
                'current_bm',           # Current blood glucose
                'prev_bm',              # Previous blood glucose (if available)
                'next_bm',              # Next blood glucose (if available)
                'time_since_prev_bm',   # Time since previous blood glucose (if available)
                'time_until_next_bm',   # Time until next blood glucose (if available)

                # Intervention (insulin)
                'insulin_changetime',   # MIMIC timestamp for insulin intervention, if any (rounded to 5 minutes)
                'insulin_old_rate',     # Current insulin rate (pre-action)
                'insulin_new_rate',     # New insulin rate (post-action)
                'insulin_maintain',     # (Binary) whether insulin rate is unchanged
                'insulin_change',       # (Binary) whether insulin rate is changed (but not stopped)
                'insulin_stop',         # (Binary) whether insulin rate is stopped
                'insulin_delta_change', # Change in insulin rate

                'insulin_maintain_prev', # Previous insulin action
                'insulin_change_prev',  # Previous insulin action
                'insulin_stop_prev',    # Previous insulin action
                'insulin_delta_change_prev',    # Previous insulin action
                'insulin_maintain_next', # Next insulin action
                'insulin_change_next',  # Next insulin action
                'insulin_stop_next',    # Next insulin action
                'insulin_delta_change_next',    # Next insulin action

                # Mortality
                '1-day-alive', '3-day-alive', '7-day-alive', '14-day-alive', '28-day-alive', # (Binary) survival from now
                
                '1-day-alive-final', '3-day-alive-final', '7-day-alive-final', '14-day-alive-final',
                '28-day-alive-final', # (Binary) survival relative to the terminal (last) state

                # Inputs
                'feature', 'time', 'value',
                'feature_next', 'time_next', 'value_next'
            ]

            (
                final_target_data
                # Sort, collect, and save the data
                .sort('label_id')
                .select(columns)
                .collect()
                .write_parquet(os.path.join(dataframe_path, f'dataframe{idx:03}.parquet'))
            )

    # Finally, group this all into a single dataframe
    print('\nWriting final dataframe...')
    complete_df = pl.concat([
        pl.scan_parquet(os.path.join(intermediate_path, group, '*.parquet'), low_memory=low_memory)
        for group in ['train', 'val', 'test']
    ], how='vertical')

    complete_df.sink_parquet(os.path.join(final_path, 'all_data.parquet'))

    # Delete the _intermediate dataframes
    for group in ['train', 'val', 'test']:
        shutil.rmtree(os.path.join(intermediate_path, group))


def create_glucose_labels_for_mimic(combined_data, admissions, patients, input_window_size, inclusion_hours=24,
                                    train_patient_ids=None, val_patient_ids=None, test_patient_ids=None):
    # Create our labels DataFrame - label is every new measurement starttime (the "state marker")
    labels = (
        combined_data
        .filter((pl.col('feature') == "Bedside Glucose") | ((pl.col('feature') == "Regular Insulin")
                                                            & (pl.col('rate').is_not_null())))
        .select(['input_id', 'subject_id', 'starttime', 'feature', 'valuenum', 'rate', 'patientweight'])
        .unique()
    )

    # Get the insulin labels - we can experiment with broader inclusion hours if desired later
    (labels,
     train_patient_ids,
     val_patient_ids,
     test_patient_ids) = get_insulin_labels_for_mimic(labels, inclusion_hours=inclusion_hours,
                                                      train_ids=train_patient_ids, val_ids=val_patient_ids,
                                                      test_ids=test_patient_ids)

    # Get the death labels
    labels = get_death_labels_for_mimic(labels, admissions, patients)

    label_columns = ['subject_id', 'episode_num', 'step_num', 'label_id', 'label_id_next',
                     'is_done', 'starttime', 'starttime_next', 'patientweight', 'current_bm', 'prev_bm',
                     'time_since_prev_bm', 'next_bm', 'time_until_next_bm', 'insulin_changetime',
                     'insulin_old_rate', 'insulin_new_rate', 'insulin_maintain', 'insulin_change', 'insulin_stop',
                     'insulin_delta_change', 'insulin_maintain_next', 'insulin_change_next', 'insulin_stop_next',
                     'insulin_delta_change_next', '1-day-alive', '1-day-alive-final', '3-day-alive',
                     '3-day-alive-final', '7-day-alive', '7-day-alive-final', '14-day-alive', '14-day-alive-final',
                     '28-day-alive', '28-day-alive-final', 'input_id']

    # Create our start/end inclusion times for input data (and save the DataFrame for the next step)
    labels_path = './data/insulin4rl/_intermediate_data/labels.parquet'
    (
        labels
        .select(label_columns +
                [(pl.col('starttime') - pl.duration(hours=input_window_size)).alias('start_inclusion'),
                 pl.col('starttime').alias('end_inclusion')])
        .unique()
        .collect()
        .write_parquet(labels_path)
    )

    labels = pl.scan_parquet(labels_path)

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

    # Set the column order
    label_col_order = ['subject_id', 'label_id', 'labeltime', 'label_id_next', 'labeltime_next',
                       'start_inclusion', 'end_inclusion', 'episode_num', 'step_num', 'steps_per_episode', 'steps_remaining',
                       'minutes_remaining', 'is_done', 'current_bm', 'prev_bm',
                       'time_since_prev_bm', 'next_bm', 'time_until_next_bm', 'insulin_changetime',
                       'insulin_old_rate', 'insulin_new_rate', 'insulin_maintain', 'insulin_change', 'insulin_stop',
                       'insulin_delta_change', 'insulin_maintain_prev', 'insulin_change_prev', 'insulin_stop_prev',
                       'insulin_delta_change_prev', 'insulin_maintain_next', 'insulin_change_next', 'insulin_stop_next',
                       'insulin_delta_change_next', '1-day-alive', '1-day-alive-final', '3-day-alive',
                       '3-day-alive-final', '7-day-alive', '7-day-alive-final', '14-day-alive', '14-day-alive-final',
                       '28-day-alive', '28-day-alive-final', 'age', 'gender', 'patientweight', 'input_id']

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


def create_scaling_dict():
    # Get our train data
    train_df = (
        pl.scan_parquet('./data/insulin4rl/all_data.parquet')
        .filter(pl.col('data_segment') == 'train')
        .select('feature', 'time', 'value')
        .explode(pl.all())
    )

    normalisation_dict = {'time': {}}

    # Helper function to handle transformations for negative and non-negative data
    def compute_stat(series, specific_stat_name, can_be_negative=False):
        if specific_stat_name in ['log_mean', 'arcsinh_mean']:
            transformed = series.arcsinh() if can_be_negative else series.log1p()
            transformed_val = transformed.mean()
            return transformed_val if transformed_val is not None else 0
        elif specific_stat_name in ['log_std', 'arcsinh_std']:
            transformed = series.arcsinh() if can_be_negative else series.log1p()
            transformed_val = transformed.std()
            return transformed_val if transformed_val is not None else 1
        else:
            stat_val = getattr(series, specific_stat_name)()
            if stat_val is None:
                if specific_stat_name in ['min', 'mean']:
                    stat_val = 0
                elif specific_stat_name in ['max', 'std']:
                    stat_val = 1
            return stat_val

    # Get our encodings
    feature_encoding_path = './data/insulin4rl/_intermediate_data/feature_encoding.parquet'
    feature_encodings_df = pl.read_parquet(feature_encoding_path)
    feature_encodings_dict = dict(zip(
        feature_encodings_df["str_feature"],
        feature_encodings_df["feature"]
    ))

    # Stats keys to track
    stats = ['max', 'min', 'mean', 'std', 'log_mean', 'log_std']

    # --- Section 1: Time Normalisation ---
    def get_time_df(df, col):
        return (
            df.filter(pl.col(col) >= 0)
            .select(pl.col(col).cast(pl.Int32)).collect().to_series()
        )

    print('\nCalculating time normalisation constants...')
    time_tasks = list(itertools.product(['time'], stats))

    for _, (time_category, stat_name) in progress_bar(time_tasks, with_time=True):
        category_df = get_time_df(train_df, time_category)
        # Time is never negative
        value = compute_stat(category_df, stat_name, can_be_negative=False)
        normalisation_dict['time'][f'{time_category}_{stat_name}'] = value

    # --- Section 2: Value Normalisation ---
    def get_value_df(df, col):
        return (
            df.filter(pl.col(col).is_not_nan())
            .select('feature', pl.col(col).cast(pl.Float32))
            .collect()
        )

    def get_feature_df(df, encoding):
        return (
            df.filter(pl.col('feature') == encoding)
            .drop('feature')
            .to_series()
        )

    print('\nCalculating value normalisation constants...')
    value_categories = ['value']
    stats = ['max', 'min', 'mean', 'std', 'arcsinh_mean', 'arcsinh_std']
    feature_items = list(feature_encodings_dict.items())
    value_tasks = list(itertools.product(value_categories, feature_items, stats))

    current_val_cat = None
    current_feat_label = None
    category_df = None
    feature_df = None

    for _, (v_cat, (f_label, f_encoding), stat) in progress_bar(value_tasks, with_time=True):
        if v_cat != current_val_cat:
            category_df = get_value_df(train_df, v_cat)
            current_val_cat = v_cat

        if f_label != current_feat_label:
            feature_df = get_feature_df(category_df, f_encoding)
            current_feat_label = f_label
            if f_label not in normalisation_dict:
                normalisation_dict[f_label] = {}

        # Values can be negative
        value = compute_stat(feature_df, stat, can_be_negative=True)
        normalisation_dict[f_label][f'{v_cat}_{stat}'] = value

    return normalisation_dict


def encode_combined_data_for_mimic(combined_data):
    """
    Create the encodings for our features
    """
    # First, create separate drug variables for rate and bolus, and create a single 'value' column
    encoded_input_data = split_labels_to_rate_and_bolus(df=combined_data)
    encoded_input_data = encoded_input_data.rename({'starttime': 'time'})

    # Convert our feature names to integer encodings
    features = encoded_input_data.select(pl.col('feature').unique()).to_series().sort().to_list()
    features.append('age')
    features.append('gender')
    features.append('patientweight')

    feature_encoding = (
        pl.DataFrame({'str_feature': features,
                      'feature': np.array([i for i in range(len(features))], dtype=np.int16)})
        .with_columns([pl.col('str_feature')])
    )

    # Encode the features in combined_data
    encoded_input_data = (
        encoded_input_data
        .rename({'feature': 'str_feature'})
        .join(feature_encoding, on='str_feature', how='inner')
    )

    # Keep track of the encodings for all our features (by category), for use later on
    age_encoding = feature_encoding.filter(pl.col('str_feature') == 'age').select('feature').item()
    gender_encoding = feature_encoding.filter(pl.col('str_feature') == 'gender').select('feature').item()
    weight_encoding = feature_encoding.filter(pl.col('str_feature') == 'patientweight').select('feature').item()

    str_features, all_features = (encoded_input_data.select('str_feature', 'feature')
                                          .unique().sort('str_feature'))
    drug_names_encoded, lab_names_encoded = [], []
    for str_feature, feature in zip(str_features, all_features):
        if 'rate' in str_feature:
            drug_names_encoded.extend([feature])
        elif 'bolus' in str_feature:
            continue
        else:
            lab_names_encoded.extend([feature])

    encodings = {'age': age_encoding, 'gender': gender_encoding, 'weight': weight_encoding,
                 'drug_names': drug_names_encoded, 'lab_names': lab_names_encoded,
                 'all_features': all_features.to_list()}

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
            pl.when((pl.col('step_num').shift(-1) == 1)
                    | (pl.col('step_num').shift(-1).is_null()))
            .then(pl.col(f'{day}-day-alive').alias(f'{day}-day-alive-final'))
            .otherwise(pl.lit(None)) for day in [1, 3, 7, 14, 28]
        ])
        .with_columns([
            pl.col(f'{day}-day-alive-final').fill_null(strategy='backward').over('subject_id')
            for day in [1, 3, 7, 14, 28]
        ])
    )

    return df





def get_insulin_labels_for_mimic(labels, inclusion_hours, train_ids, val_ids, test_ids):
    # Specify some parameters
    insulin_delta_change_outlier_threshold = 5.5  # Filter out insulin actions greater than this
    insulin_rate_floor = 0.1  # Insulin rates less than this are treated as 0.
    insulin_delta_floor = 0.25  # Insulin changes less than this count as no change

    min_interstate_delay = 5  # 5 minute delay between states

    valid_window_pre_bm = 5  # Allow insulin actions up to 5 minutes before the BM check
    valid_window_post_bm = 30  # Allow insulin actions up to 30 minutes after the BM check

    assert min_interstate_delay >= valid_window_pre_bm, \
        ("If you are going to change either of these variables, min_interstate_delay must be equal or greater than"
         "valid_window_pre_bm.")

    labels = labels.lazy()

    # Get our BM changes
    bm_changes = (
        labels
        .filter(pl.col('feature') == "Bedside Glucose")
        .drop('rate', 'patientweight', 'input_id')
        .unique()
        .sort('subject_id', 'starttime')
    )

    # Aggregate to nearest 5 minutes
    agg_bm_changes = (
        bm_changes
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
            # Take the latest BM/rate over the last 5 minutes (where multiple measurements exist)
            pl.col('current_bm').last()
        ])
        .sort('subject_id', 'starttime')
    )

    # Get our insulin rate changes
    insulin_rate_changes = (
        labels
        .filter(pl.col('feature') == "Regular Insulin")
        .drop('valuenum')
        .rename({'starttime': 'insulin_changetime'})
        .unique()
        .sort('subject_id', 'insulin_changetime')
    )

    # Identify the current insulin rate "visible" at each blood glucose check
    # - we will allow up to 5 minutes retrospective window to account for minor logging artefacts
    agg_bm_changes_with_rate = (
        agg_bm_changes
        .join_asof(
            insulin_rate_changes
            .select(['subject_id', (pl.col('insulin_changetime') + pl.duration(minutes=valid_window_pre_bm+1)), 'rate']),
            left_on='starttime', right_on='insulin_changetime', suffix='_right',
            by='subject_id',
            strategy='backward',
            check_sortedness=False,  # Cannot be True when using by columns
            allow_exact_matches=True,
        )
        .rename({'rate': 'insulin_old_rate'})
        .drop('insulin_changetime')
    )

    # Start filtering for our episode window of viable insulin actions
    # We only want labels when patients are getting IV insulin infusions.
    # Identify inclusion windows according to the following principles:
    # 1) Start_inclusion from when insulin is started for the first time (minus inclusion_hours),
    # or restarted after a gap of >2 * inclusion_hours (minus inclusion_hours)
    # 2) End_inclusion when insulin is stopped for 2 * inclusion_hours (i.e., at the point when rate is set to 0)

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
        insulin_rate_changes
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

    # Join with the insulin rate changes
    bm_changes_with_actions = (
        agg_bm_changes_with_rate
        .join_asof(
            # Find our next starttime label (with minimum interstate delay)
            agg_bm_changes.select(
                ['subject_id',
                 # Subtract our required delay to align our starttime and starttime_next close to one another
                 pl.col('starttime') - pl.duration(minutes=min_interstate_delay)
                 ]).rename({'starttime': 'starttime_next'}),
            left_on='starttime',
            right_on='starttime_next',
            by='subject_id',
            strategy='forward',
            check_sortedness=False,  # Cannot be True when using by columns
            allow_exact_matches=True,
        )
        # Move 'next_starttime' back to its correct timestamp
        .with_columns([pl.col('starttime_next') + pl.duration(minutes=min_interstate_delay)])
        # Join with our insulin rate changes
        .join(insulin_rate_changes.drop('feature'), on='subject_id', how='inner')
        # Make sure we only have BM labels in our overall valid inclusion window (see previous section)
        .filter(pl.col('starttime').is_between('start_inclusion', 'end_inclusion', closed="both"))
        # If the starttime_next is actually outside the end_inclusion range, change it to null
        # (because there won't be any starttime labels available to match with it!)
        .with_columns([
            pl.when(pl.col('starttime_next') > pl.col('end_inclusion'))
            .then(pl.lit(None))
            .otherwise('starttime_next').alias('starttime_next')
        ])
        .unique()
        .sort('subject_id', 'starttime', 'insulin_changetime')
    )

    # Filter to our eligible insulin actions (-5 to +30 minutes around BM)
    eligible_insulin_change = pl.col('insulin_changetime').is_between('valid_insulin_pre',
                                                                      'valid_insulin_post', closed="left")

    filtered_labels = (
        bm_changes_with_actions
        # Identify the "valid insulin" duration, which is the first 30 minutes after BM check
        .with_columns([
            (pl.col('starttime') - pl.duration(minutes=valid_window_pre_bm)).alias('valid_insulin_pre'),
            (pl.col('starttime') + pl.duration(minutes=valid_window_post_bm)).alias('valid_insulin_post')
        ])
        # When the next BM check happens sooner than 30 minutes, ignore insulin changes after this time
        # (offset by valid_window_pre_bm minutes, to avoid one action shared by multiple BM checks)
        .with_columns([
            pl.when(pl.col('starttime_next') - pl.duration(minutes=valid_window_pre_bm) < pl.col('valid_insulin_post'))
            .then((pl.col('starttime_next') - pl.duration(minutes=valid_window_pre_bm)).alias('valid_insulin_post'))
            .otherwise(pl.col('valid_insulin_post'))
        ])
        # Identify potential valid insulin changes
        .with_columns([
            pl.when(eligible_insulin_change)
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias('eligible_insulin_row')
        ])
        # Filter to only valid changes (or a "null" row if no valid changes")
        .filter(
            pl.col('eligible_insulin_row')
            |
            (pl.col('eligible_insulin_row').not_().all()
             & pl.col('eligible_insulin_row').is_last_distinct()).over('subject_id', 'starttime')
        )
        .with_columns([
            pl.when(pl.col('eligible_insulin_row').not_())
            .then(pl.col('insulin_old_rate').alias('rate'))
            .otherwise('rate'),
        ])
        .with_columns([
            pl.when(pl.col('eligible_insulin_row').not_())
            .then(pl.lit(None).alias(col))
            .otherwise(col)
            for col in ['insulin_changetime', 'input_id', 'patientweight']
        ])
    )

    # Note that there are roughly 430,000 valid decision markers, and of these, >98% have a single unique
    # insulin label. The following code is adjusting for the 1.5% that don't have a unique insulin label.
    aggregated_labels = (
        filtered_labels
        # For valid insulin rows, take the final insulin change
        .group_by([
            pl.exclude('input_id', 'insulin_changetime', 'rate', 'patientweight')
        ])
        .agg([
            # Keep a track of input_id so we can exclude any relevant insulins later from the inputs
            pl.col('input_id'),
            pl.col('insulin_changetime').last(),
            pl.col('rate').last().alias('insulin_new_rate'),
            pl.col('patientweight').last()
        ])
        .with_columns([
            (pl.col('insulin_new_rate') - pl.col('insulin_old_rate')).alias('insulin_delta_change')
        ])
        # Set our rules for insulin_maintain/stop/change
        .with_columns([
            # Insulin stop - going from above the rate floor to below the rate floor.
            pl.when(
                (pl.col("insulin_old_rate") >= insulin_rate_floor) &
                (pl.col("insulin_new_rate") < insulin_rate_floor)
            )
            .then(pl.lit(1)).otherwise(pl.lit(0)).alias("insulin_stop"),

            # Insulin maintain - staying below the rate floor, or having a change less than delta_floor
            pl.when(
                ((pl.col("insulin_old_rate") < insulin_rate_floor) & (
                            pl.col("insulin_new_rate") < insulin_rate_floor)) |
                ((pl.col("insulin_new_rate") >= insulin_rate_floor) & (pl.col("insulin_delta_change").abs() < insulin_delta_floor))
            )
            .then(pl.lit(1)).otherwise(pl.lit(0)).alias("insulin_maintain")
        ])
        .with_columns([
            # Insulin change (anything not maintenance or stopping)
            pl.when(pl.col("insulin_stop").cast(pl.Boolean).not_() & pl.col("insulin_maintain").cast(pl.Boolean).not_())
            .then(pl.lit(1)).otherwise(pl.lit(0)).alias("insulin_change")
        ])
        # Filter out any actions where the absolute insulin rate change is ≥5.5 (and not part of insulin_stop)
        .filter(
            pl.when(pl.col('insulin_change').cast(pl.Boolean))
            .then(pl.col('insulin_delta_change').abs() < insulin_delta_change_outlier_threshold)
            .otherwise(True)
        )
        # Because of the above filtering steps, we need to update our starttime_nexts in case they point to rows which
        # have been removed (maybe 0.5% of rows)
        .sort('subject_id', 'starttime')
        .with_columns([
            pl.when((pl.col('starttime_next') != pl.col('starttime').shift(-1))
                    & pl.col('starttime').shift(-1).is_not_null())
            .then(pl.col('starttime').shift(-1).alias('starttime_next'))
            .otherwise('starttime_next')
            .over('subject_id', 'start_inclusion', 'end_inclusion')
        ])
        .drop('eligible_insulin_row', 'valid_insulin_post')
        .collect()
    )

    # Check there are no insulin actions shared by multiple BM checks
    assert (
        aggregated_labels
        .select(
            (pl.struct('subject_id', 'insulin_changetime').is_duplicated()
             & pl.col('insulin_changetime').is_not_null()).any().not_()
        ).item()), "Insulin action shared by multiple BM checks - please investigate further!"

    # Convert the start_/end_inclusions to a unique episode number
    episode_nums = (
        aggregated_labels
        .select('subject_id', 'start_inclusion', 'end_inclusion')
        .unique()
        .sort('subject_id', 'start_inclusion', 'end_inclusion')
        .with_columns([
            pl.struct('subject_id', 'start_inclusion', 'end_inclusion')
            .rank(method='ordinal').alias('episode_num')
        ])
    )

    # Add label episode and id_nums
    aggregated_labels = (
        aggregated_labels
        # Add episode nums
        .join(episode_nums, on=['subject_id', 'start_inclusion', 'end_inclusion'], how='inner')
        .drop('start_inclusion', 'end_inclusion')
        # Get our id_num (unique ID for all rows)
        .sort('subject_id', 'starttime')
        .with_row_index().rename({'index': 'label_id'})
    )

    # Bring in the id_num and action labels for the next label
    aggregated_labels = (
        aggregated_labels
        .join(
            aggregated_labels.select(['subject_id', 'starttime', 'label_id', 'insulin_maintain', 'insulin_change',
                           'insulin_stop', 'insulin_delta_change']),
            left_on=['subject_id', 'starttime_next'],
            right_on=['subject_id', 'starttime'],
            how='left',
            suffix='_next'
        )
    )

    # Add in the step nums and first/last state columns
    aggregated_labels = (
        aggregated_labels
        # Get our step_nums
        .sort('subject_id', 'episode_num', 'starttime')
        .with_columns([
            pl.col('starttime').rank(method='ordinal').over('episode_num').alias('step_num')
        ])

        # Finally, identify our "terminal state"
        .with_columns([
            pl.when(pl.col('step_num') == pl.col('step_num').max().over('episode_num'))
            .then(pl.lit(1))
            .otherwise(pl.lit(0)).alias('is_done')
        ])
    )

    # Add in our previous and next BM values
    aggregated_labels = (
        aggregated_labels
        .sort('episode_num', 'step_num')
        .with_columns([
            pl.col('current_bm').shift(1).over('episode_num').alias('prev_bm'),
            (pl.col('starttime') - pl.col('starttime').shift(1).over('episode_num')).alias('time_since_prev_bm'),

            pl.col('current_bm').shift(-1).over('episode_num').alias('next_bm'),
            (pl.col('starttime').shift(-1).over('episode_num') - pl.col('starttime')).alias('time_until_next_bm'),
        ])
        .with_columns([
            # Convert to total_minutes (leave null as they are)
            pl.when(pl.col('time_since_prev_bm').is_not_null())
            .then(pl.col('time_since_prev_bm').dt.total_minutes())
            .otherwise(pl.duration(minutes=0)),

            pl.when(pl.col('time_until_next_bm').is_not_null())
            .then(pl.col('time_until_next_bm').dt.total_minutes())
            .otherwise(pl.duration(minutes=0)),
        ])
    )

    # N.B. we need to update our train/val/test ids to reflect the smaller labels DF
    train_ids, val_ids, test_ids = train_ids.tolist(), val_ids.tolist(), test_ids.tolist()
    filtered_ids = set(aggregated_labels.select('subject_id').unique().to_series())
    all_ids = set(train_ids + val_ids + test_ids)
    removed_ids = all_ids - filtered_ids

    new_train_ids = np.array(list(set(train_ids) - removed_ids))
    new_val_ids = np.array(list(set(val_ids) - removed_ids))
    new_test_ids = np.array(list(set(test_ids) - removed_ids))

    return aggregated_labels, new_train_ids, new_val_ids, new_test_ids


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
            # And our steroid drugs
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


def get_patient_ids(combined_data, patients, train_test_split: float = 0.8, force: bool = False):
    # Check if train/val/test ids already exist
    file_path = './data/insulin4rl/patient_ids/test_patient_ids.npy'
    if os.path.exists(file_path) and not force:
        segments = ['train', 'val', 'test']
        return tuple([np.load(f'./data/insulin4rl/patient_ids/{segment}_patient_ids.npy')
                      for segment in segments])

    # 1. Prepare the cohort with stratification labels - 12,238 patients
    cohort = (
        combined_data.lazy()
        .filter((pl.col('label') == "Regular Insulin") & pl.col('rate').is_not_null())
        .sort('subject_id', 'starttime')
        .filter(pl.col('label').is_last_distinct().over('subject_id'))
        .select('subject_id')
        .unique()
        .join(patients.select('subject_id', 'gender', 'dod'), on='subject_id', how='inner')
        .with_columns(
            is_dead=pl.col('dod').is_not_null()
        )
        .with_columns(
            # Combine gender and death status into one strata key
            strata=pl.format("{}_{}", pl.col("gender"), pl.col("is_dead"))
        )
        .collect()
    )

    train_list, val_list, test_list = [], [], []

    # 2. Split each stratum individually
    unique_strata = cohort["strata"].unique()

    for s in unique_strata:
        # Extract IDs for this specific stratum
        group_ids = cohort.filter(pl.col("strata") == s)["subject_id"].to_numpy()

        # Shuffle within the stratum
        np.random.seed(42)
        np.random.shuffle(group_ids)

        # Calculate split indices
        n = len(group_ids)
        idx_train = int(n * train_test_split)
        idx_val = int(n * (train_test_split + (1 - train_test_split) / 2))

        # Distribute IDs
        train_list.extend(group_ids[:idx_train])
        val_list.extend(group_ids[idx_train:idx_val])
        test_list.extend(group_ids[idx_val:])

    # 3. Finalize as numpy arrays
    train_patient_ids = np.array(train_list)
    val_patient_ids = np.array(val_list)
    test_patient_ids = np.array(test_list)

    return train_patient_ids, val_patient_ids, test_patient_ids


def get_nutrition_values_for_mimic():
    with open('./utils/yaml_files/nutrition_conversion_values.yaml', 'r') as f:
        nutrition_conversion_values = yaml.safe_load(f)
        f.close()
    return nutrition_conversion_values


def get_nutrition_variables_for_mimic():
    with open('./utils/yaml_files/nutrition_variables.yaml', 'r') as f:
        nutrition_variables = yaml.safe_load(f)
        f.close()
    nutrition_variables = {key: key for key in nutrition_variables}
    return nutrition_variables


def get_variable_names_for_mimic():
    with open('./utils/yaml_files/drug_lab_variable_names.yaml', 'r') as f:
        variable_names = yaml.safe_load(f)
        f.close()
    return variable_names


def get_demographic_information_for_mimic():
    with open('./utils/yaml_files/demographic_indexes.yaml', 'r') as f:
        demographic_indexes = yaml.safe_load(f)
    return (demographic_indexes['comorbidity_index'],
            demographic_indexes['admission_index'],
            demographic_indexes['ethnicity_index'])


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


def load_files(filename):
    filepath = os.path.join('./data/mimic', filename + '.parquet')
    return pl.scan_parquet(filepath)


def load_mimic(variable_names: dict = None, train_test_split: float = 0.8):
    admissions = load_files('admissions')
    chartevents = load_files('chartevents')
    d_items = load_files('d_items')
    emar = load_files('emar')
    emar_detail = load_files('emar_detail')
    prescriptions = load_files('prescriptions')
    inputevents = load_files('inputevents')
    patients = load_files('patients')

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
    train_patient_ids, val_patient_ids, test_patient_ids = get_patient_ids(combined_data, patients,
                                                                           train_test_split=train_test_split)

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


def progress_bar(iterable, with_time: bool = False, time_unit='min'):
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
            if time_unit == 'sec':
                remaining = int(remaining * 60)

        progress = step / total_length
        filled_length = int(bar_length * progress)
        bar = '=' * filled_length + '>' + ' ' * (bar_length - filled_length)
        if with_time:
            statement = '\r[' + bar + '] ' + str(int(progress * 100)) + '%, ETA: ' + str(remaining) + time_unit
        else:
            statement = '\r[' + bar + '] ' + str(int(progress * 100)) + '%'
        sys.stdout.write(statement.ljust(50))
        sys.stdout.flush()
    if total_length > 0:
        sys.stdout.write('\r[' + '=' * bar_length + '] 100%'.ljust(50) + '\n')
        sys.stdout.flush()


def rebalance_to_train(labels, train_ids, val_ids, test_ids):
    """
    Rebalances patient splits one-way (Val/Test -> Train) to ensure
    equal mean survival and gender ratios at the episode level.
    """

    # 1. Aggregate to subject level with episode counts and strata
    subject_data = (
        labels.filter(pl.col('is_done') == 1)
        .group_by('subject_id')
        .agg(
            pl.len().alias('episode_count'),
            pl.first('28-day-alive-final').alias('alive'),
            pl.first('gender').alias('gender')
        )
        .with_columns(
            strata=pl.format("{}_{}", pl.col("alive"), pl.col("gender"))
        )
    )

    df_dicts = subject_data.to_dicts()

    # Create O(1) lookup sets
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    # Assign current splits
    for d in df_dicts:
        if d['subject_id'] in train_set:
            d['current_split'] = 'train'
        elif d['subject_id'] in val_set:
            d['current_split'] = 'val'
        elif d['subject_id'] in test_set:
            d['current_split'] = 'test'
        else:
            raise Exception(f'Unknown split {d["subject_id"]}')

    # 2. Calculate global strata distribution ratios
    strata_totals = {}
    for d in df_dicts:
        s = d['strata']
        strata_totals[s] = strata_totals.get(s, 0) + d['episode_count']

    global_total = sum(strata_totals.values())
    global_ratios = {s: count / global_total for s, count in strata_totals.items()}

    new_train = set(train_ids)
    new_val = set(val_ids)
    new_test = set(test_ids)

    # 3. Helper function to process shifts for a specific split
    def process_split(split_name, current_set):
        # Tally current episodes and group subjects by stratum for this split
        split_counts = {s: 0 for s in global_ratios}
        split_subjects = {s: [] for s in global_ratios}

        for d in df_dicts:
            if d['current_split'] == split_name:
                s = d['strata']
                split_counts[s] += d['episode_count']
                split_subjects[s].append(d)

        # Find the limiting stratum to determine the maximum perfectly balanced size
        max_keep_total = min(
            (split_counts[s] / global_ratios[s])
            for s in global_ratios if global_ratios[s] > 0
        )

        for s in global_ratios:
            # Calculate exactly how many episodes need to be removed to match the global ratio
            target_keep = max_keep_total * global_ratios[s]
            target_remove = split_counts[s] - target_keep

            if target_remove <= 0:
                continue

            # Sort candidates by episode_count descending to minimize the number of patients moved
            cands = split_subjects[s]
            np.random.seed(42)
            np.random.shuffle(cands)
            cands.sort(key=lambda x: x['episode_count'], reverse=True)

            removed_episodes = 0
            for cand in cands:
                if removed_episodes >= target_remove:
                    break

                pid = cand['subject_id']

                # Execute the one-way move
                if pid in current_set:
                    current_set.remove(pid)
                    new_train.add(pid)
                    removed_episodes += cand['episode_count']

    # 4. Process Validation and Test independently
    process_split('val', new_val)
    process_split('test', new_test)

    return np.array(list(new_train)), np.array(list(new_val)), np.array(list(new_test))


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

    outlier_reference['patientweight'] = {'min': min_weight.item(), 'max': max_weight.item()}

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
    with open('./data/insulin4rl/metadata/outlier_thresholds.yaml', 'w') as file:
        yaml.safe_dump(
            outlier_reference,
            file,
            default_flow_style=False,
        )

    return df


def save_patient_ids(train_patient_ids, val_patient_ids, test_patient_ids):
    patient_ids_dir = f'./data/insulin4rl/patient_ids'
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

    # 4) Merge back together
    df = pl.concat([df, df_no_rates])

    # 5) Rename 'amount' to 'bolus', and 'amountuom' to 'bolusuom'
    df = df.rename({'amount': 'bolus', 'amountuom': 'bolusuom'})

    return df


def split_labels_to_rate_and_bolus(df, *args):
    """
    We want a single 'value' column - to facilitate this for drugs, we create separate labels for 'rate' and 'bolus'
    """
    select_cols = ['subject_id', 'input_id', 'feature', 'starttime',
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

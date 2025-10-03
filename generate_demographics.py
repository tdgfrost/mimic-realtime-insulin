from utils.preprocessing.tools import *


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
    diagnoses = load_files('mimic/parquet', 'diagnoses_icd')
    diagnoses = diagnoses.with_columns([pl.lit(None).alias('comorbidity')])
    for key, val in comorbidity_index.items():
        diagnoses = diagnoses.with_columns(
            [pl.when(pl.col('icd_code').str.contains(key)).then(pl.lit(val)).otherwise('comorbidity').alias(
                'comorbidity')])

    diagnoses = diagnoses.filter(pl.col('comorbidity').is_not_null()).collect()

    # Get our df
    admissions = load_files('mimic/parquet', 'admissions')
    df = pl.concat([
        load_files(f'mimic/{segment}/dataframe_{segment}', '*') for segment in ['train', 'val', 'test']
    ], how='vertical')

    demographic_info = (
        df
        .select('subject_id', 'episode_num', 'is_last_state', 'labeltime')
        .filter(pl.col('is_last_state').cast(pl.Boolean))
        .join(admissions.select('subject_id', 'hadm_id', 'admittime', 'dischtime', 'admission_type',
                                'insurance', 'language', 'marital_status', 'race'), on=['subject_id'], how='inner')
        .filter(pl.col('labeltime').is_between('admittime', 'dischtime'))
        .select('episode_num', 'hadm_id', 'admission_type', 'insurance', 'language',
                'marital_status', 'race')
        .collect()
    )

    hadm_ids = demographic_info.select('episode_num', 'hadm_id')

    # Get our age, sex, race/ethnicity, admission type, insurance, language, marital status
    patients = load_files('mimic/parquet', 'patients')

    age_sex_information = (
        df
        .select('subject_id', 'episode_num', 'is_first_state', 'labeltime')
        .filter(pl.col('is_first_state').cast(pl.Boolean))
        .join(patients.select('subject_id', 'gender', 'anchor_age', 'anchor_year'), on=['subject_id'], how='inner')
        .with_columns(((pl.col('labeltime').dt.year() - pl.col('anchor_year')) + pl.col('anchor_age')).alias('age'))
        .select('episode_num', 'age', 'gender')
        .collect()
    )

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
    patients = load_files('mimic/parquet', 'patients')

    age_sex_information = (
        df
        .select('subject_id', 'episode_num', 'is_first_state', 'labeltime')
        .filter(pl.col('is_first_state').cast(pl.Boolean))
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

    # Load our labels for mortality information
    labels = load_files('mimic', 'labels')

    # Specifically, collect the following information
    # -28-day mortality from the final label
    # - total minutes duration of the episode
    # - number of steps per episode
    # - patient weight information
    # - number of hypoglycaemic events
    # - number of hyperglycaemic events (>10 mmol/L)
    label_information = (
        labels
        .select('episode_num', 'step_num', 'steps_per_episode', 'minutes_remaining', 'current_bm', '28-day-alive-final',
                'patientweight')
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

    one_hot_episode_demographics.write_parquet('./data/mimic/demographics.parquet')

if __name__ == "__main__":
    build_demographics()

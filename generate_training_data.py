import json

from utils.preprocessing.tools import *

# Define the argument parser and possible arguments globally
parser = argparse.ArgumentParser()
parser.add_argument('--context', type=int, default=400,
                    help="Specify the context window size for the model - default is 400")


def build_mimic_data(context_length=400):
    # Load our input data, label, and scaling datasets
    input_file_path = './data/mimic/encoded_input_data.parquet'
    label_file_path = './data/mimic/labels.parquet'
    assert os.path.exists(input_file_path) & os.path.exists(label_file_path), (
        "No input_data / label data found - run generate_dataset.py first."
    )

    encoded_input_data = pl.scan_parquet(input_file_path).drop('str_feature')
    labels = pl.scan_parquet(label_file_path)

    # Load encodings
    with open('./data/encodings.json', 'r') as f:
        encodings = json.load(f)
        f.close()

    for key in ['age', 'gender', 'weight']:
        encodings[key] = np.int16(encodings[key])  # Maybe move this later on?

    # Load our patient ids
    train_patient_ids, val_patient_ids, test_patient_ids = get_patient_ids(None, force=True)

    # Create the finalised input data and label data
    announce_progress('Constructing and saving final dataframes...')
    """
    create_final_dataframe(encoded_input_data=encoded_input_data,
                           labels=labels,
                           encodings=encodings,
                           train_patient_ids=train_patient_ids,
                           val_patient_ids=val_patient_ids,
                           test_patient_ids=test_patient_ids,
                           sorting_columns=['subject_id', 'episode_num', 'step_num'],
                           grouping_columns=['subject_id', 'episode_num', 'step_num', 'targets'],
                           context_length=context_length)
    """
    # Convert these to .hdf5 binaries
    announce_progress('Converting the dataframes to .hdf5 compressed binaries...')
    convert_dataframe_to_hdf5(context_length=context_length)


if __name__ == "__main__":
    # Parse the command-line arguments inside the block
    args = parser.parse_args()

    # Execute the conversion function
    build_mimic_data(context_length=args.context)

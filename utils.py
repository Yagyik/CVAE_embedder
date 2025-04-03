import logging
import os

import pandas as pd
from sklearn.model_selection import train_test_split

def setup_logger(name, save_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(os.path.join(save_dir, 'info.log'))
    fh.setLevel(logging.INFO)

    logger.addHandler(fh)

    return logger

# def read_dataframe_list(file_list_path):
#     # Read the list of dataframe file names
#     with open(file_list_path, 'r') as file:
#         dataframe_files = file.read().splitlines()

#     return dataframe_files


def read_and_combine_dataframes(file_list_path):
    # Read the list of dataframe file names
    with open(file_list_path, 'r') as file:
        dataframe_files = file.read().splitlines()

    # Read and combine all dataframes
    combined_df = pd.concat([pd.read_csv(df_file) for df_file in dataframe_files], ignore_index=True)

    return combined_df

def format_fix_dataframe(df):
    # Replace NaN values with empty strings for each column
    new_df = df.fillna('')
    return new_df

def split_and_save_dataframe(df, train_file='train_dataset.csv', test_file='test_dataset.csv', test_size=0.2, random_state=42):
    # Perform train-test split
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['condition_int'])

    # Save the train and test datasets to CSV files
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
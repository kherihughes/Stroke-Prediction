# ================================================================================
# Author:      Kheri Hughes - 2023
# Description: This script contains the dataset partioning logic.
# ================================================================================
import argparse
import logging
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

pd.options.mode.chained_assignment = None  # default='warn'

BUCKET_NAME = "sagemaker-strokeprediction-mlops"
BUCKET = f's3://{BUCKET_NAME}'

WRANGLED_DATA_FOLDER = 'Dataset/wrangler/processed_whole/wranglejob-2023-07-23T13-42-12'
#enable retrieval of latested wrangled file eventually instead of hcing it
WRANGLED_DATA_FILE = 'part-00000-594c41f8-25b9-4c7a-a165-bb1d53dfebe5-c000.csv'
WRANGLED_DATA_PATH = os.path.join(BUCKET, WRANGLED_DATA_FOLDER, WRANGLED_DATA_FILE)

# Path where the processed objects will be stored
now = datetime.now() # get current time to ensure uniqueness of the output folders
PROCESSED_DATA_FOLDER = 'processed_splits/' + now.strftime("%Y-%m-%d_%H%M_%S%f")
PROCESSED_DATA_PATH = os.path.join(BUCKET, PROCESSED_DATA_FOLDER)

# Paths for model train, validation, test split
TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, 'train.csv')
TRAIN_DATA_PATH_W_HEADER= os.path.join(PROCESSED_DATA_PATH, 'train_w_header.csv')
VALIDATION_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, 'validation.csv')
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, 'test.csv')
TEST_DATA_PATH_W_HEADER = os.path.join(PROCESSED_DATA_PATH, 'test_w_header.csv')

TARGET_COLUMN = 'stroke'

def extract_features_types(df, unique_threshold=10):
    numerical_features = []
    categorical_features = []

    for col in df.columns:
        if col == TARGET_COLUMN:
            continue
        if df[col].nunique() <= unique_threshold:
            categorical_features.append(col)
        else:
            numerical_features.append(col)

    return numerical_features, categorical_features

def split_dataset(dataset, target_column, test_size=0.2, validation_size=0.2, random_state=None):
    X = dataset.drop(target_column, axis=1)
    y = dataset[target_column]
    
    # Split dataset into train and test sets using StratifiedShuffleSplit
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    for train_index, test_index in stratified_split.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Split the remaining data into validation and train sets using StratifiedShuffleSplit
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=validation_size, random_state=random_state)
    
    for train_index, val_index in stratified_split.split(X_train, y_train):
        X_train, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def process_target(df: pd.DataFrame, col_target: str) -> pd.DataFrame:
    
    # Make sure that the 0 error type is also mapped to 1 (we do a binary classification later)
    df.loc[df[col_target] == 0, col_target] = 1
    
    df = fill_nulls(df=df, col=col_target)
    
    # Reorder columns
    colnames = list(df.columns)
    colnames.insert(0, colnames.pop(colnames.index(col_target)))
    df = df[colnames]
    
    return df

if __name__ == '__main__':
    # install('fsspec')
    # Parse the SDK arguments that are passed when creating the SKlearn container
    parser = argparse.ArgumentParser()
    parser.add_argument("--vali_fraction", type=float, default=.2)
    parser.add_argument("--test_fraction", type=float, default=.1)
    args, _ = parser.parse_known_args()
    
    logger.info(f"Received arguments {args}.")

    input_data_path = os.path.join("/opt/ml/processing/input", WRANGLED_DATA_FILE)
    logger.info(f"Reading input data from {input_data_path}")
    # Read raw input data
    df = pd.read_csv(input_data_path)
    logger.info(f"Shape of data is: {df.shape}")
    
    logger.info("Splitting %d rows of data into train, validation, test datasets.", len(df))
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(df, TARGET_COLUMN, test_size=args.test_fraction, validation_size=args.vali_fraction)

    # Create local output directories. These directories live on the container that is spun up.
    try:
        os.makedirs("/opt/ml/processing/train")
        os.makedirs("/opt/ml/processing/validation")
        os.makedirs("/opt/ml/processing/test")
        print("Successfully created directories")
    except Exception as e:
        # if the Processing call already creates these directories (or directory otherwise cannot be created)
        logger.debug(e)
        logger.debug("Could Not Make Directories.")
        pass

    # Save data locally on the container that is spun up.
    try:
        pd.concat([y_train, X_train], axis=1).to_csv("/opt/ml/processing/train/train.csv", header=False, index=False)
        pd.concat([y_train, X_train], axis=1).to_csv("/opt/ml/processing/train/train_w_header.csv", header=True, index=False)
        pd.concat([y_val, X_val], axis=1).to_csv("/opt/ml/processing/validation/val.csv", header=False, index=False)
        pd.concat([y_val, X_val], axis=1).to_csv("/opt/ml/processing/validation/val_w_header.csv", header=True, index=False)
        pd.concat([y_test, X_test], axis=1).to_csv("/opt/ml/processing/test/test.csv", header=False, index=False)
        pd.concat([y_test, X_test], axis=1).to_csv("/opt/ml/processing/test/test_w_header.csv", header=True, index=False)
        logger.info("Files Successfully Written Locally")
    except Exception as e:
        logger.debug("Could Not Write the Files")
        logger.debug(e)
        pass
    
    logger.info("Finished running processing job")

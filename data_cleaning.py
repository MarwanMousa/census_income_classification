"""
This is a script for cleaning the raw input data. The script expect raw data to exist in a subdirectory called data

This script performs cleaning for this particular dataset only, and expect the data to be named
'census_income_learn.csv' and 'census_income_test.csv' for the training and test datasets respectively

The script then saves the cleaned datasets in the same data sub-directory
"""
import numpy as np
import pandas as pd

def clean_dataset(dataset: pd.DataFrame):
    pass
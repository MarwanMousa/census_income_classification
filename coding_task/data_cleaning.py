"""
This is a script for cleaning the raw input data. The script expect raw data to exist in a subdirectory called data

This script performs cleaning for this particular dataset only, and expect the data to be named
'census_income_learn.csv' and 'census_income_test.csv' for the training and test datasets respectively

The script then saves the cleaned datasets in the same data sub-directory
"""
from typing import Optional

import numpy as np
import pandas as pd


def clean_dataset(location: str, save_name: Optional[str] = None) -> pd.DataFrame:
    """

    Args:
        location: location of the data file relative to the directory the script runs in
        save_name: filename of the clean dataset if it is to be saved, it not saved by default
    Returns:

        A pandas Dataframe with the clean dataset

    """
    # Make sure the input is a string
    assert type(location) == str

    # Load in data
    df_raw = pd.read_csv(location)

    # First we need to add column names to the dataset as it doesn't have any column headings
    # The columns were extracted from census_income_metadata.txt
    columns = [
        'age', 'class of worker', 'detailed industry code', 'detailed occupation code', 'education',
        'wage per hour', 'enroll in edu inst last wk', 'marital stat', 'major industry code', 'major occupation code',
        'race', 'hispanic origin', 'sex', 'member of a labor union', 'reason for unemployment',
        'full or part time employment stat', 'capital gains', 'capital losses', 'dividends from stocks',
        'tax filer stat', 'region of previous residence', 'state of previous residence',
        'detailed household and family stat', 'detailed household summary in household', 'instance weight',
        'migration code-change in msa', 'migration code-change in reg', 'migration code-move within reg',
        'live in this house 1 year ago', 'migration prev res in sunbelt', 'num persons worked for employer',
        'family members under 18', 'country of birth father', 'country of birth mother', 'country of birth self',
        'citizenship', 'own business or self employed', 'fill inc questionnaire for veterans admin',
        'veterans benefits', 'weeks worked in year', 'year', 'income'
    ]

    df_raw.columns = columns

    # Remove columns that are potentially duplicated of other columns, or carry no beneficical information,
    # have too many nans,
    remove_columns = [
        'detailed industry code', 'detailed occupation code', 'enroll in edu inst last wk', 'major industry code',
        'major occupation code',
        'hispanic origin', 'member of a labor union', 'reason for unemployment', 'tax filer stat',
        'region of previous residence', 'state of previous residence',
        'detailed household and family stat', 'detailed household summary in household', 'migration code-change in msa',
        'migration code-change in reg',
        'migration code-move within reg', 'live in this house 1 year ago', 'migration prev res in sunbelt',
        'family members under 18',
        'country of birth father', 'country of birth mother', 'country of birth self',
        'fill inc questionnaire for veterans admin', 'veterans benefits',
        'year', 'instance weight', 'own business or self employed'
    ]

    df_train_raw = df_train_raw.drop(columns=remove_columns)


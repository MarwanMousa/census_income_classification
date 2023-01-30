"""
This is a script contains functions for processing the raw input data.

Three functions are defined, clean_dataset(), feature_engineering(), preprocess()

-The first function is in charge of cleaning the raw dataset by adding column headings, removing unnecessary columns and
removing any NaNs.

-The second performs some feature engineering, turning binary categorical variables into booleans and reducing some of
classes in some categorical variables (see Exploratory_Data_Analysis.ipynb for more details)

-The third pre-process the data (one hot encoding and scaling) in order for the data to be used to train classifiers.

This script has functionality for cleaning this particular dataset only and expec

"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def clean_dataset(location: str) -> pd.DataFrame:
    """

    Args:
        location: location of the data file relative to the directory the script runs in
    Returns:

        A pandas Dataframe with the clean dataset

    """
    # Make sure the input is a string
    assert isinstance(location, str)

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
        'year', 'instance weight', 'own business or self employed', 'wage per hour'
    ]

    df_raw = df_raw.drop(columns=remove_columns)

    df_raw = df_raw.replace(' ?', np.nan)
    df_raw = df_raw.replace(' Not in universe', np.nan)
    df_clean = df_raw.dropna()

    return df_clean


def feature_engineering(dataset: pd.DataFrame) -> pd.DataFrame:
    """

    Args:
        dataset:

    Returns:

    """

    # Outcome is binary so we convert it to 0, 1
    dataset['income'] = dataset['income'].map({' - 50000.': 0, ' 50000+.': 1})
    dataset['income'] = dataset['income'].astype(int)

    # ---- Education ---- #
    # It was found that all children in the dataset had an income of less than <50,000 which seems obvious
    # Therefore to lets take children explicitly since they do not require a classifier
    dataset = dataset[dataset['education'] != ' Children']

    # Education seems to have too many categories, so we can reduce group some of them
    education_mapping = {
        ' Less than 1st grade': 'No Education',
        ' 1st 2nd 3rd or 4th grade': 'Middle School',
        ' 5th or 6th grade': 'Middle School',
        ' 7th and 8th grade': 'High School',
        ' 9th grade': 'High School',
        ' 10th grade': 'High School',
        ' 11th grade': 'High School',
        ' 12th grade no diploma': 'High School',
        ' High school graduate': 'High School Graduate',
        ' Some college but no degree': 'High School Graduate',
        ' Associates degree-academic program': 'Associates Degree',
        ' Associates degree-occup /vocational': 'Associates Degree',
        ' Bachelors degree(BA AB BS)': 'Undergraduate Degree',
        ' Masters degree(MA MS MEng MEd MSW MBA)': 'Postgraduate Degree',
        ' Prof school degree (MD DDS DVM LLB JD)': 'Advanced Degree',
        ' Doctorate degree(PhD EdD)': 'Advanced Degree',
    }

    dataset['education'] = dataset['education'].map(education_mapping)

    # Since education levels have clear progression, they can be mapped to numerical features
    # to reduce number of dimensions
    education_numerical_mapping = {
        'No Education': 0,
        'Middle School': 1,
        'High School': 2,
        'High School Graduate': 3,
        'Associates Degree': 4,
        'Undergraduate Degree': 5,
        'Postgraduate Degree': 6,
        'Advanced Degree': 7,
    }

    dataset['education'] = dataset['education'].map(education_numerical_mapping)
    dataset['education'] = dataset['education'].astype(int)

    # ---- Marriage Status ---- #
    # we reduce the number of categories to a binary married vs single
    marriage_status_mapping = {
        ' Divorced': 'Single',
        ' Never married': 'Single',
        ' Widowed': 'Single',
        ' Married-civilian spouse present': 'Married',
        ' Married-A F spouse present': 'Married',
        ' Separated': 'Married',
        ' Married-spouse absent': 'Married',
    }

    dataset['marital stat'] = dataset['marital stat'].map(marriage_status_mapping)
    dataset['marital stat'] = dataset['marital stat'].map({"Married": 1, "Single": 0})
    dataset['marital stat'] = dataset['marital stat'].astype(int)

    # ---- Sex ---- #
    # Converting sex to a binary variable
    dataset['sex'] = dataset['sex'].map({' Male': 0, ' Female': 1})
    dataset['sex'] = dataset['sex'].astype(int)

    # ---- Employment Status ---- #
    # Many if the different groupings were similar both by their definition and the relative association with income
    # These were group to reduce number of dimensions
    employment_mapping = {
        ' Full-time schedules': 'Full Time',
        ' PT for econ reasons usually FT': 'Part Time',
        ' PT for econ reasons usually PT': 'Part Time',
        ' PT for non-econ reasons usually FT': 'Part Time',
        ' Not in labor force': 'Not in labor force',
        ' Unemployed part- time': 'Unemployed',
        ' Unemployed full-time': 'Unemployed',
        ' Children or Armed Forces': 'Armed Forces',
    }

    dataset['full or part time employment stat'] = dataset['full or part time employment stat'].map(employment_mapping)

    return dataset


def preprocess(train_dataset: pd.DataFrame, test_dataset: pd.DataFrame, target: str) -> (pd.DataFrame, pd.DataFrame):
    """
    This functions takes in the cleaned training and test datasets and preprocesses the data for any ML task
    It applied one-hot encoding to the categorical variables and scales the numerical variables using a min-max scaler

    Args:
        train_dataset: A dataframe containing the clean feature engineered training dataset
        test_dataset: A dataframe containing the clean feature engineered test dataset
        target: The name of the target variable

    Returns:
        A tuple of two pandas dataframes where the first is the pre-processed training dataset and the second is the
        the preprocessed test dataset

    """

    # Extract categorical variable names
    categorical_cols = train_dataset.columns[train_dataset.dtypes == object].tolist()

    # Get dummy variables for categorical data
    train_dataset_new = pd.get_dummies(train_dataset, columns=categorical_cols)
    test_dataset_new = pd.get_dummies(test_dataset, columns=categorical_cols)

    # Extract numerical variable names
    numerical_columns = [col for col in train_dataset if col not in categorical_cols + [target]]

    # Get min max-scaler
    mms = MinMaxScaler()

    # Fit scaler to training data and transform it
    min_max_scaled_columns_train = mms.fit_transform(train_dataset_new[numerical_columns])
    # Using fit to training data, transform the test data
    min_max_scaled_columns_test = mms.transform(test_dataset_new[numerical_columns])

    # Update the columns with their scaled versions
    for i, col in enumerate(numerical_columns):
        train_dataset_new[col] = min_max_scaled_columns_train[:, i]
        test_dataset_new[col] = min_max_scaled_columns_test[:, i]

    return train_dataset_new, test_dataset_new

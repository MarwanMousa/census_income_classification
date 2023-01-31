"""
A script with utility functions for getting data in the right format, training the model and getting performance metrics
"""
from typing import Optional, Union, List
import pandas
import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


def get_train_test_data(train_data: pd.DataFrame, test_data: pd.DataFrame, target: str, imbalance_adjustment: bool):
    """
    This is a utility function for creating the training and testing features and targets from their respective datasets
    to be used for training classifiers
    Args:
        train_data: a pandas dataframe with the training data
        test_data: a pandas dataframe with the test data
        target: a string with the target variable name
        imbalance_adjustment: a boolean variable describing whether to use the SMOTE minority up-sampling methods

    Returns:
        a tuple of four pandas dataframes with 1. training features 2. training targets 3. test features 4. test targets
    """
    assert isinstance(train_data, pandas.DataFrame), "Train data is expected as pandas dataframe"
    assert isinstance(train_data, pandas.DataFrame), "Test data is expected as pandas dataframe"
    assert tuple(list(train_data.columns)) == tuple(list(test_data.columns)), "Train and Test data columns must match"
    assert isinstance(target, str), "The target must be a string name of a column"
    assert target in list(train_data.columns), "Provided target must be a valid column name is the dataset"
    assert isinstance(imbalance_adjustment, bool), "imbalance_adjustment must be a boolean value"

    x_train = train_data.drop(target, axis=1)
    y_train = train_data[target]

    x_test = test_data.drop(target, axis=1)
    y_test = test_data[target]

    if imbalance_adjustment:
        x_train, y_train = SMOTE(sampling_strategy='minority').fit_resample(x_train, y_train)

    return x_train, y_train, x_test, y_test


def train_model(model, x_train: pd.DataFrame, y_train: pd.Series):
    """
    A utility function for training an untrained model on the training data
    Args:
        model: An instance of a class that implements the fit method (for training the model)
        x_train: A pandas dataframe of the features of the training dataset
        y_train: A pandas dataframe of the targets of the training dataset

    Returns:
        the trained model
    """
    assert isinstance(x_train, pandas.DataFrame), "Train features are expected as pandas dataframe"
    assert isinstance(y_train, pandas.Series), "Train targets are expected as pandas series"
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test: pandas.DataFrame, y_test: Optional[pandas.Series] = None):
    """
    A utility function for getting the predictions of a trained model
    Args:
        model: An instance of a class that implements the predict method
        x_test: A pandas dataframe of the features of the test dataset
        y_test: an optional dataframe with the test targets to be provided if the model's predict method requires it

    Returns:

    """
    assert isinstance(x_test, pandas.DataFrame), "Train features are expected as pandas dataframe"

    if y_test is None:
        predictions = model.predict(x_test)
    else:
        assert isinstance(y_test, pandas.Series), "Train targets are expected as pandas series"
        predictions = model.predict(x_test, y_test)

    return predictions


def get_metrics(labels: Union[List, pandas.DataFrame], predictions: Union[List, pandas.DataFrame]):
    """
    A utility function for getting various classification metrics based on model predictions and the true labels
    Args:
        labels: a list of the true labels/targets
        predictions: a list of the corresponding model predictions

    Returns:
        a tuple of metrics:
        1. accuracy(float), 2. balanced_accuracy(float), 3. recall([float]), 4. precision([float]) 5. f1(float)
        6. roc(float)

    """

    accuracy = accuracy_score(labels, predictions)
    balanced_accuracy = balanced_accuracy_score(labels, predictions)
    recall = recall_score(labels, predictions, labels=[0, 1], average=None)
    precision = precision_score(labels, predictions, labels=[0, 1], average=None)
    f1 = f1_score(labels, predictions)
    roc = roc_auc_score(labels, predictions)

    return accuracy, balanced_accuracy, recall, precision, f1, roc


def print_summary_metrics(model_name: str, labels: Union[List, pandas.DataFrame],
                          predictions: Union[List, pandas.DataFrame]):
    """
    A utility function for printing the metrics of a model
    Args:
        model_name: a string with the model name
        labels: a list of the true labels/targets
        predictions: a list of the corresponding model predictions

    Returns:

    """
    accuracy, balanced_accuracy, recall, precision, f1, roc = get_metrics(labels, predictions)

    print(
        f"{model_name:<20} {accuracy:<15.2f} {balanced_accuracy:<15.2f} {f1:<15.2f} {roc:<15.2f} {recall[0]:<5.2f},"
        f"{recall[1]:<8.2f}  {precision[0]:<5.2f},{precision[1]:<8.2f}"
    )

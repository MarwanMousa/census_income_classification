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

    x_train = train_data.drop(target, axis=1)
    y_train = train_data[target]

    x_test = test_data.drop(target, axis=1)
    y_test = test_data[target]

    if imbalance_adjustment:
        x_train, y_train = SMOTE(sampling_strategy='minority').fit_resample(x_train, y_train)

    return x_train, y_train, x_test, y_test


def train_model(model, x_train, y_train):
    """
    A utility function for training an untrained model on the training data
    Args:
        model: An instance of a class that implements the fit method (for training the model)
        x_train: A pandas dataframe of the features of the training dataset
        y_train: A pandas dataframe of the targets of the training dataset

    Returns:
        the trained model
    """
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test: pandas.DataFrame, y_test: Optional[pandas.DataFrame] = None):
    """
    A utility function for getting the predictions of a trained model
    Args:
        model: An instance of a class that implements the predict method
        x_test: A pandas dataframe of the features of the test dataset
        y_test: an optional dataframe with the test targets to be provided if the model's predict method requires it

    Returns:

    """
    if y_test is None:
        predictions = model.predict(x_test)
    else:
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

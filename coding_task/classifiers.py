"""
This script is used for training several classifiers and printing their performance for the census income dataset

We try a number of known algorithms methods:
1. Logistic Regression
2. Naive Bayes
3. Decision Tree
4. Random Forest
5. K-Nearest Neighbours
6. Gradient Boosting
7. Artificial Neural Network

The performance of all models is tested on variations of the training dataset. One with re-sampling of the minority
class (income > 50K) and one without any resampling.

This is due to the significant class imbalance present in the dataset

It is worth noting that:
The models are all used with some default parameters and are trained on the entirety of the training data
No hyper parameter tuning on validation data (or cross-validation) was performed
"""
import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from models.NN_classifier import NeuralNetworkModel, Network
from data_modification import clean_dataset, feature_engineering, preprocess
from utility import get_train_test_data, train_model, evaluate_model, print_summary_metrics


# ----- DATA PREPARATION ----- #

# First we extract the data from the csv file and clean it
train_data_clean = clean_dataset('data/census_income_learn.csv')
test_data_clean = clean_dataset('data/census_income_test.csv')

# Then we apply some feature engineering to the datasets
train_data_fe = feature_engineering(train_data_clean)
test_data_fe = feature_engineering(test_data_clean)

# Then we preprocess the data (one-hot encoding and scaling) to prepare it for training classifier models
train_data, test_data = preprocess(train_dataset=train_data_fe, test_dataset=test_data_fe, target='income')

# Printing dataset size and class imbalance for quick check
print(f'Train dataset has {len(train_data):,} data points with proportion of high income earners: '
      f'{train_data.income.mean()}')
print(f'Test dataset has {len(test_data):,} data points with proportion of high income earners: '
      f'{test_data.income.mean()}')


# We obtain the training and testing features and targets
# First we obtain the features as they are
x_train, y_train, x_test, y_test = get_train_test_data(train_data, test_data, 'income', imbalance_adjustment=False)
# Second we obtain the same training dataset but with the minority class oversampled using the "SMOTE" methods
x_train_adj, y_train_adj, _, _ = get_train_test_data(train_data, test_data, 'income', imbalance_adjustment=True)


# ----- MODEL TRAINING ----- #
# We train several models and use the trained models to get predictions on the test dataset
# We train two version of each model, one where the training dataset is left unaltered and the second where we
# up-sampled the minority class (income > 50k)

# 1 Logistic Regression
# First model
LR_model1 = train_model(LogisticRegression(solver='newton-cholesky'), x_train, y_train)
LR_model1_pred = evaluate_model(LR_model1, x_test)

# Second model
LR_model2 = train_model(LogisticRegression(solver='newton-cholesky'), x_train_adj, y_train_adj)
LR_model2_pred = evaluate_model(LR_model2, x_test)


# 2 Naive Bayes
# First model
NB_model1 = train_model(GaussianNB(), x_train, y_train)
NB_model1_pred = evaluate_model(NB_model1, x_test)

# Second model
NB_model2 = train_model(GaussianNB(), x_train_adj, y_train_adj)
NB_model2_pred = evaluate_model(NB_model2, x_test)


# Decision Tree
# First model
DT_model1 = train_model(DecisionTreeClassifier(criterion='entropy', min_samples_split=8, max_depth=10),
                        x_train, y_train)
DT_model1_pred = evaluate_model(DT_model1, x_test)

# Second model
DT_model2 = train_model(DecisionTreeClassifier(criterion='entropy', min_samples_split=8, max_depth=10),
                        x_train_adj, y_train_adj)
DT_model2_pred = evaluate_model(DT_model2, x_test)

# Random Forest
# First model
RF_model1 = train_model(RandomForestClassifier(n_estimators=100), x_train, y_train)
RF_model1_pred = evaluate_model(RF_model1, x_test)

# Second model
RF_model2 = train_model(RandomForestClassifier(n_estimators=100), x_train_adj, y_train_adj)
RF_model2_pred = evaluate_model(RF_model2, x_test)

# Gradient Boosting
# First model
GB_model1 = train_model(GradientBoostingClassifier(n_estimators=100), x_train, y_train)
GB_model1_pred = evaluate_model(GB_model1, x_test)

# Second model
GB_model2 = train_model(GradientBoostingClassifier(n_estimators=100), x_train_adj, y_train_adj)
GB_model2_pred = evaluate_model(GB_model2, x_test)


# Nearest Neighbour
# First model
KNN_model1 = train_model(KNeighborsClassifier(), x_train, y_train)
KNN_model1_pred = evaluate_model(KNN_model1, x_test)

# Second model
KNN_model2 = train_model(KNeighborsClassifier(), x_train_adj, y_train_adj)
KNN_model2_pred = evaluate_model(KNN_model2, x_test)


# Neural Network
# Getting the input dimension for the network
input_dim = x_train.shape[1]
# The size of the hidden dimension
hidden_dim = 64
# The batch size used
batch_size = 64
# The learning rate
lr = 0.1
# The number of epochs to train the neural network over
epochs = 10

np.random.seed(0)
torch.manual_seed(0)

# First model
NN_model1 = train_model(
      NeuralNetworkModel(input_dim, hidden_dim, Network, batch_size, lr, epochs),
      x_train, y_train
)
NN_model1_pred = evaluate_model(NN_model1, x_test, y_test)

# Second model
NN_model2 = train_model(
      NeuralNetworkModel(input_dim, hidden_dim, Network, batch_size, lr, epochs),
      x_train_adj, y_train_adj
)
NN_model2_pred = evaluate_model(NN_model2, x_test, y_test)

# ----- MODEL PERFORMANCE SUMMARY ----- #
# A list of the names of the models tested
models = [
      "Logistic Regression", "Naive Bayes", "Decision Tree", "Random Forest", "Gradient Boosting", "KNN",
      "Neural Network"
]
# A list of the models' predictions to the training dataset
predictions = [
      LR_model1_pred, NB_model1_pred, DT_model1_pred, RF_model1_pred, KNN_model1_pred, GB_model1_pred, NN_model1_pred
]
# A list of the models' predictions to the minority class up-sampled training dataset
predictions_adj = [
      LR_model2_pred, NB_model2_pred, DT_model2_pred, RF_model2_pred, KNN_model2_pred, GB_model2_pred, NN_model2_pred
]

print('Regular Training Dataset')
print(f"{'Model':20} {'Accuracy':15} {'Balance Acc':15} {'f1':15} {'AUC':15} {'Recall':15} {'Precision':15}")
for model, pred in zip(models, predictions):
    print_summary_metrics(model, y_test, pred)

print()
print('SMOTE Training Dataset')
print(f"{'Model':20} {'Accuracy':15} {'Balance Acc':15} {'f1':15} {'AUC':15} {'Recall':15} {'Precision':15}")
for model, pred in zip(models, predictions_adj):
    print_summary_metrics(model, y_test, pred)

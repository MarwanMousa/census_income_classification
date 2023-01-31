"""
This script defines a Neural Network Classifier
It also includes the architecture of the neural network and
"""
import itertools
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# A helper class to get the data in the right format for the NN classifier
class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


# The architecture of the neural network used for the classifier
class Network(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Network, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = nn.functional.relu(self.layer_1(x))
        x = nn.functional.relu(self.layer_2(x))
        # Apply sigmoid function to last layer
        x = torch.sigmoid(self.layer_3(x))

        return x


# A class defining the neural network classifier model
# It implements two methods, fit and predict to train and evaluate the model respectively
class NeuralNetworkModel:
    def __init__(self, input_dim: int, hidden_dim: int, network, batch_size: int = 64, lr: float = 0.1,
                 epochs: int = 10):
        # Instantiate neural network
        self.net = network(input_dim, hidden_dim)
        self.batch_size = batch_size

        # using binary cross entropy loss as it is a binary classification task
        self.loss_fn = nn.BCELoss()

        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)
        self.epochs = epochs

    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame):
        """

        Args:
            x_train: The training features as dataframe
            y_train: The training targets as dataframe

        Returns:

        """
        # Create a torch dataset from the features, targets
        train_data = Data(x_train, y_train)
        # Create Dataloader
        train_dataloader = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for X, y in train_dataloader:
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                pred = self.net(X)
                loss = self.loss_fn(pred, y.unsqueeze(-1))
                loss.backward()
                self.optimizer.step()

    def predict(self, x_test: pd.DataFrame, y_test: pd.DataFrame):
        """

        Args:
            x_test: The test features as dataframe
            y_test: The test targets as dataframe

        Returns:
            predictions in the form of a list

        """
        # Create a torch dataset from the features, targets
        test_data = Data(x_test, y_test)
        # Create Dataloader
        test_dataloader = DataLoader(dataset=test_data, batch_size=len(y_test), shuffle=False)

        # Get predictions
        with torch.no_grad():
            for X, y in test_dataloader:
                outputs = self.net(X)
                predicted = np.where(outputs < 0.5, 0, 1)
                predicted = list(itertools.chain(*predicted))

        return predicted

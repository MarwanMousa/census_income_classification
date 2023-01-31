This repo contains the code used to train the classification models as well as any data cleaning/pre-processing required.

The `Exploratory_Data_Analysis` and `Models` notebook contain preliminary work done to create the models and is essentially a notebook version of the rest of the directory.

There are three files in this directory.
1. `data_modification.py` has functionaities for preparing the data.
2. `utility.py` has some utility functions used to prepare the data in the right format and getting performance metrics.
3. `classifiers.py` is the main script used to train the models and compare their performance.

There are two sub-directories as well
1. data - contains the datasets used to train and evaluate the models (the data needs to be downloaded externally and added here - see below)
2. models -  contains the definition of the neural network model used

In order to run any of the above, the datasets ***need to be downloaded*** from [here](https://drive.google.com/drive/folders/1PPsjCoM130k3n3V4roq-yF74jkPjkVd7)
The datasets are called `census_income_learn.csv` and `census_income_test.csv` for the training and testing data respectively. 
These should be downloaded and added to the data sub-directory with their ***names unchanged*** in order to run the scripts/notebooks without any changes.

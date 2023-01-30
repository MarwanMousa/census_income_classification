# Dataiku Task

This repo contain all the files for the data modelling task of income classification problem for U.S. census data 

The repo contains the coding task as well as a CV and slides summarising the results of the coding task

The root contain the CV, the slides, a subdirectory with the coding task and poetry files for dependency management.

Poetry has been used for managing the dependencies used in this repo. 

In order to run the files the repo has to cloned. 

Once cloned the following steps need to be done first. (Ignore step 1 if poetry is already installed on the system)


1. Install Poetry (https://python-poetry.org/docs/) \
   osx / linux / bashonwindows install instructions 
    ```properties
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
    ```

    windows powershell install instructions 
    ```properties
    (Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -
    ```

2. In the root directory of the repo run
   ```properties
   poetry install
   ``` 
   
  Alternatively, the dependencies in the poetry toml can be manually installed in a virtual environment with Python (>= 3.8 < 4) using pip.

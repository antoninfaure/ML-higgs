# :mag: ML Higgs Challenge

This repository contains the code and documentation for the Machine Learning course Project 1 accomplished our team.

## Team:
This project was accomplished by:
- Antonin Faure: @antoninfaure
- Manon Dorster : @mdorster
- Alexandre Maillard : @AlexMlld

## Introduction to the project:

The aim of our project is to apply machine learning methods to CERN particle accelerator data to determine Higgs boson
generation across multiple proton collision events. From a dataset consisting of feature vectors representing the decay
signature of collision events, our goal was to predict whether the events consisted in a signal (a Higgs boson) or background

# Project structure: 

## Data:

The data sets required to train and test the prediction models we have implemented can be found on [AIcrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/dataset_files)

**Note**
To run our code, please download these files and put them in the same folder as our code files. 

## Implementations:
The six functions we were asked to implement can be found in the [implementations.py](implementations.py) file. 
- ridge_regression(y, tx, lambda_)
- least_squares(y, tx)
- mean_squared_error_gd(y, tx, w_init, max_iters, gamma)
- mean_squared_error_sgd(y, tx, w_init, max_iters, gamma, batch_size = 1)
- logistic_regression(y, tx, initial_w, max_iters, gamma)
- reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)

## Helpers file
The [helpers.py](helpers.py) contains all the functions used in our [run.py](run.py) in order to use the implemented algorithms.

## Hyperparameter optimization
The [hyperparameter_opti.py](hyperparameter_opti.py) contains two methods used to find the best gamma and lambda for the logistic regression using training data.
- best_lambda_selection(y, tx, max_iters, gamma)
- best_gamma_selection(y, tx, max_iters)

## Best model prediction:

The script [run.py](run.py) produces the same predictions which we used in our best submission on AIcrowd, that is logistic regression after degree 2 expansion of the features with 4-fold data splitting.
The csv file that is generated is called [Predictions_Logistics_degree2_split4.csv](Predictions_Logistics_degree2_split4.csv).
To output the same result just run :
```
python run.py
```
NB : it might take some time to run

## Other model prediction algorithms:
In order to run the other prediction algorithms we implemented, please follow the following instructions:
- to run least squares, ridge regression, logistic regression, and regularized logistic regression (without the 4-sets split) : run the script [run_all_models.ipynb](run_all_models.ipynb)
- to run logistic regression and regularized logistic regression with the 4-sets data splitting : run [ML_Project_1_2.ipynb](ML_Project_1_2.ipynb)

## Report:

A 2 page scientific report describes the most relevant feature engineering techniques and implementations that we worked on, and explains how and why these techniques improved our predictions. 

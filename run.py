from implementations import *
from helpers import *
import numpy as np


# Load train data
train_data, train_labels, ids = load_data_train('train.csv')

# Standardize
new_data = standardize(clean_data(train_data))
train_labels[train_labels==-1]=0

# Shuffle data
inds = np.random.permutation(new_data.shape[0])
new_data = new_data[inds]
train_labels = train_labels[inds] 

# Add bias to data
tx = np.c_[np.ones((train_labels.shape[0], 1)), train_data]

# Initialize the weights randomly according to a Gaussian distribution
initial_w = np.random.normal(0., 0.1, [tx.shape[1],])

# Train model
trained_weights, train_loss = logistic_regression(train_labels, tx, initial_w, max_iters=20000, gamma=0.01)

print(trained_weights)

# Validation data
tx_validation = np.c_[np.ones((train_labels.shape[0], 1)), new_data]
validation_predict = predict_logistic(tx_validation, trained_weights)
validation_labels = train_labels.copy()

print(f"train_accuracy = {accuracy(validation_predict, validation_labels)}")
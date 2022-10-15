from implementations import *
from helpers import *
import numpy as np



train_data, train_labels, ids = load_data_train('train.csv')

train_data = standardize(clean_data(train_data))

# Add bias to data
tx = np.c_[np.ones((train_labels.shape[0], 1)), train_data]

#initialize the weights randomly according to a Gaussian distribution
initial_w = np.random.normal(0., 0.1, [tx.shape[1],])

trained_weights, train_loss = logistic_regression(train_labels, tx, initial_w, max_iters=4000, gamma=0.01)
#trained_weights, train_loss = reg_logistic_regression(train_labels, tx, 1, initial_w, max_iters=4000, gamma=0.01)

print(trained_weights)

print(train_loss)
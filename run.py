import implementations as impl
import helpers as helpers
import numpy as np


# Load train data
train_data, train_labels, train_ids = helpers.load_data('train.csv')

# Clean and standardize train data
new_data = helpers.standardize(helpers.clean_data(train_data))
train_labels[train_labels==-1]=0

# Shuffle data
inds = np.random.permutation(new_data.shape[0])
new_data = new_data[inds]
train_labels = train_labels[inds] 

# Add bias to data
tx = np.c_[np.ones((train_labels.shape[0], 1)), new_data]

# Initialize the weights randomly according to a Gaussian distribution
initial_w = np.random.normal(0., 0.1, [tx.shape[1],])

# Train model
trained_weights, train_loss = impl.logistic_regression(train_labels, tx, initial_w, max_iters=10000, gamma=0.01)
print(train_loss)
print(trained_weights)

# Cross-validation
tx_validation = np.c_[np.ones((train_labels.shape[0], 1)), new_data]
validation_predict = helpers.predict_logistic(tx_validation, trained_weights)
validation_labels = train_labels.copy()
validation_labels[validation_labels == 0] = -1

# Compute training accuracy
train_accuracy = helpers.accuracy(validation_predict, validation_labels)
print(train_accuracy)

# Load test data
test_data, test_labels, test_ids = helpers.load_data('test.csv', train=False)

# Clean and standardize test data
test_data = helpers.standardize(helpers.clean_data(test_data))

# Add bias to data
tx_test = np.c_[np.ones((test_data.shape[0], 1)), test_data]

# Predict labels
predicted_labels = helpers.predict_logistic(tx_test, trained_weights)

# Output csv
helpers.create_csv_submission(test_ids, predicted_labels, 'Predictions_Logistics.csv')

exit()

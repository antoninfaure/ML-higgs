import implementations as impl
import helpers as helpers
import numpy as np


# Load train data
train_data, train_labels, train_ids = helpers.load_data('train.csv')

# Clean and standardize train data
train_data = helpers.standardize(helpers.clean_data(train_data))
train_labels[train_labels==-1]=0

# Shuffle data
inds = np.random.permutation(train_data.shape[0])
train_data = train_data[inds]
train_labels = train_labels[inds] 

# Add bias to data
tx_train = np.c_[np.ones((train_labels.shape[0], 1)), train_data]

# Split data into training and validation sets
slice_id = int(np.floor(train_labels.shape[0]*0.25))
validation_labels, train_labels = train_labels[:slice_id], train_labels[slice_id:]
tx_validation, tx_train = tx_train[:slice_id, :], tx_train[slice_id:, :]

# Initialize the weights randomly according to a Gaussian distribution
initial_w = np.random.normal(0., 0.1, [tx_train.shape[1],])

# Train model
trained_weights, train_loss = impl.logistic_regression(train_labels, tx_train, initial_w, max_iters=1000, gamma=0.1)
print(f"train_loss = {train_loss}")

# Cross validation
validation_predict = helpers.predict_logistic(tx_validation, trained_weights)
train_predict = helpers.predict_logistic(tx_train, trained_weights)
validation_predict[validation_predict == -1] = 0
train_predict[train_predict == -1] = 0
train_accuracy = helpers.accuracy(train_predict, train_labels)
validation_accuracy = helpers.accuracy(validation_predict, validation_labels)
print(f"train_accuracy = {train_accuracy}")
print(f"validation_accuracy = {validation_accuracy}")

print(trained_weights)

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

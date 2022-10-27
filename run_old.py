import implementations as impl
import helpers as helpers
import numpy as np


# Load train data
train_data, train_labels, train_ids = helpers.load_data('train.csv')

# Clean and standardize train data
train_data = helpers.standardize(helpers.clean_data(train_data))
train_labels[train_labels==-1]=0

# Shuffle data
train_labels, train_data = helpers.shuffle_data(train_labels, train_data)

# Slice into training and validation sets
y_validation, y_train, tx_validation, tx_train = helpers.slice_data(train_labels, train_data, 0.25)

# Expand degre
tx_train = helpers.build_poly(tx_train, 2)
tx_validation = helpers.build_poly(tx_validation, 2)

# Add bias to data
tx_train = np.c_[np.ones((tx_train.shape[0], 1)), tx_train]
tx_validation = np.c_[np.ones((tx_validation.shape[0], 1)), tx_validation]

# Initialize the weights randomly according to a Gaussian distribution
initial_w = np.random.normal(0., 0.1, [tx_train.shape[1],])

# Train model
trained_weights, train_loss = impl.logistic_regression(y_train, tx_train, initial_w, max_iters=10000, gamma=0.1)
print(f"train_loss = {train_loss}")

print(trained_weights)

# Cross validation
validation_predict = helpers.predict_logistic(tx_validation, trained_weights)
train_predict = helpers.predict_logistic(tx_train, trained_weights)

validation_predict[validation_predict == -1] = 0
train_predict[train_predict == -1] = 0

train_accuracy = helpers.accuracy(train_predict, y_train)
validation_accuracy = helpers.accuracy(validation_predict, y_validation)

print(f"train_accuracy = {train_accuracy}")
print(f"validation_accuracy = {validation_accuracy}")

# Load test data
tx_test, y_test, ids_test = helpers.load_data('test.csv', train=False)

# Clean and standardize test data
tx_test = helpers.standardize(helpers.clean_data(tx_test))

# Expand degree
tx_test = helpers.build_poly(tx_test, 2)

# Add bias to data
tx_test = np.c_[np.ones((tx_test.shape[0], 1)), tx_test]

# Predict labels
predicted_labels = helpers.predict_logistic(tx_test, trained_weights)

# Output csv
helpers.create_csv_submission(ids_test, predicted_labels, 'Predictions_Logistics_degree2.csv')

exit()

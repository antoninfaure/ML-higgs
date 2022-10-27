import implementations as impl
import helpers as helpers
import numpy as np


# Load train data
tx_train, y_train, ids_train = helpers.load_data('train.csv')

# Refactor the -1 in 0 value for logistic regression
y_train[y_train==-1]=0

# Shuffle data
y_train, tx_train = helpers.shuffle_data(y_train, tx_train)

# Split, clean and standardize data into 4 sets according to 22nd feature
tx_0, y_0, _, miss_col_0 = helpers.split_i(tx_train, y_train, ids_train, 0)
tx_1, y_1, _, miss_col_1 = helpers.split_i(tx_train, y_train, ids_train, 1)
tx_2, y_2, _, miss_col_2 = helpers.split_i(tx_train, y_train, ids_train, 2)
tx_3, y_3, _, miss_col_3 = helpers.split_i(tx_train, y_train, ids_train, 3)

print(tx_0.shape)
print(tx_1.shape)
print(tx_2.shape)
print(tx_3.shape)

# Expand to degree 2
tx_train_0 = helpers.build_poly_deg2(tx_0)
tx_train_1 = helpers.build_poly_deg2(tx_1)
tx_train_2 = helpers.build_poly_deg2(tx_2)
tx_train_3 = helpers.build_poly_deg2(tx_3)

print(tx_train_0.shape)
print(tx_train_1.shape)
print(tx_train_2.shape)
print(tx_train_3.shape)

# Add bias to data
tx_train_0 = np.c_[np.ones((tx_train_0.shape[0], 1)), tx_train_0]
tx_train_1 = np.c_[np.ones((tx_train_1.shape[0], 1)), tx_train_1]
tx_train_2 = np.c_[np.ones((tx_train_2.shape[0], 1)), tx_train_2]
tx_train_3 = np.c_[np.ones((tx_train_3.shape[0], 1)), tx_train_3]

# Initialize the weights randomly according to a Gaussian distribution
initial_w_0 = np.random.normal(0., 0.1, [tx_train_0.shape[1],])
initial_w_1 = np.random.normal(0., 0.1, [tx_train_1.shape[1],])
initial_w_2 = np.random.normal(0., 0.1, [tx_train_2.shape[1],])
initial_w_3 = np.random.normal(0., 0.1, [tx_train_3.shape[1],])

# Train models
w_0, train_loss_0 = impl.logistic_regression(y_0, tx_train_0, initial_w_0, max_iters=20000, gamma=0.01)
w_1, train_loss_1 = impl.logistic_regression(y_1, tx_train_1, initial_w_1, max_iters=20000, gamma=0.01)
w_2, train_loss_2 = impl.logistic_regression(y_2, tx_train_2, initial_w_2, max_iters=20000, gamma=0.01)
w_3, train_loss_3 = impl.logistic_regression(y_3, tx_train_3, initial_w_3, max_iters=20000, gamma=0.01)

print(f"train_loss_0 = {train_loss_0}")
print(f"train_loss_1 = {train_loss_1}")
print(f"train_loss_2 = {train_loss_2}")
print(f"train_loss_3 = {train_loss_3}")

# Compute training accuracies
predict_train_0 = helpers.predict_logistic(tx_train_0, w_0)
predict_train_1 = helpers.predict_logistic(tx_train_1, w_1)
predict_train_2 = helpers.predict_logistic(tx_train_2, w_2)
predict_train_3 = helpers.predict_logistic(tx_train_3, w_3)

predict_train_0[predict_train_0 == -1] = 0
predict_train_1[predict_train_1 == -1] = 0
predict_train_2[predict_train_2 == -1] = 0
predict_train_3[predict_train_3 == -1] = 0

train_accuracy_0 = helpers.accuracy(predict_train_0, y_0)
train_accuracy_1 = helpers.accuracy(predict_train_1, y_1)
train_accuracy_2 = helpers.accuracy(predict_train_2, y_2)
train_accuracy_3 = helpers.accuracy(predict_train_3, y_3)

print(f"train_accuracy_0 = {train_accuracy_0}")
print(f"train_accuracy_1 = {train_accuracy_1}")
print(f"train_accuracy_2 = {train_accuracy_2}")
print(f"train_accuracy_3 = {train_accuracy_3}")

# Load test data
tx_test, y_test, ids_test = helpers.load_data('test.csv', train=False)

# Split, clean and standardize data into 4 sets according to 22nd feature
tx_test_0, y_test_0, ids_test_0, _ = helpers.split_i(tx_test, y_test, ids_test, 0, miss_col_0)
tx_test_1, y_test_1, ids_test_1, _ = helpers.split_i(tx_test, y_test, ids_test, 1, miss_col_1)
tx_test_2, y_test_2, ids_test_2, _ = helpers.split_i(tx_test, y_test, ids_test, 2, miss_col_2)
tx_test_3, y_test_3, ids_test_3, _ = helpers.split_i(tx_test, y_test, ids_test, 3, miss_col_3)

# Expand to degree 2
tx_test_0 = helpers.build_poly_deg2(tx_test_0)
tx_test_1 = helpers.build_poly_deg2(tx_test_1)
tx_test_2 = helpers.build_poly_deg2(tx_test_2)
tx_test_3 = helpers.build_poly_deg2(tx_test_3)

# Add bias to data
tx_test_0 = np.c_[np.ones((tx_test_0.shape[0], 1)), tx_test_0]
tx_test_1 = np.c_[np.ones((tx_test_1.shape[0], 1)), tx_test_1]
tx_test_2 = np.c_[np.ones((tx_test_2.shape[0], 1)), tx_test_2]
tx_test_3 = np.c_[np.ones((tx_test_3.shape[0], 1)), tx_test_3]

# Predict labels
predict_test_0 = helpers.predict_logistic(tx_test_0, w_0)
predict_test_1 = helpers.predict_logistic(tx_test_1, w_1)
predict_test_2 = helpers.predict_logistic(tx_test_2, w_2)
predict_test_3 = helpers.predict_logistic(tx_test_3, w_3)

# Concatenate sets
predict_test = np.concatenate((predict_test_0, predict_test_1, predict_test_2, predict_test_3))
ids_test = np.concatenate((ids_test_0, ids_test_1, ids_test_2, ids_test_3))

# Output csv
helpers.create_csv_submission(ids_test, predicted_labels, 'Predictions_Logistics_degree2_split4.csv')

exit()

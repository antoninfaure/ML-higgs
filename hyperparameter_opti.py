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

# Slice data into training and validation sets
slice_id = int(np.floor(train_labels.shape[0]*0.25))
validation_labels, train_labels = train_labels[:slice_id], train_labels[slice_id:]
tx_validation, tx_train = tx_train[:slice_id, :], tx_train[slice_id:, :]

# Initialize the weights randomly according to a Gaussian distribution
initial_w = np.random.normal(0., 0.1, [tx_train.shape[1],])

# Train model
validation_accuracies = []
train_accuracies = []
gammas = np.power(10,np.arange(-5, 1, 1, dtype=float))
for gamma in gammas:
	trained_weights, train_loss = impl.logistic_regression(train_labels, tx_train, initial_w, max_iters=1000, gamma=gamma)
	print(f"train_loss = {train_loss}, gamma={gamma}")

	# Cross-validation
	validation_predict = helpers.predict_logistic(tx_validation, trained_weights)
	train_predict = helpers.predict_logistic(tx_train, trained_weights)
	validation_predict[validation_predict == -1] = 0
	train_predict[train_predict == -1] = 0
	train_accuracy = helpers.accuracy(train_predict, train_labels)
	validation_accuracy = helpers.accuracy(validation_predict, validation_labels)
	print(f"train_accuracy = {train_accuracy}")
	print(f"validation_accuracy = {validation_accuracy}")
	train_accuracies.append(train_accuracy)
	validation_accuracies.append(validation_accuracy)

results = zip(gammas, train_accuracies, validation_accuracies)
results = sorted(results, key = lambda t: t[2])
for a,b,c in results:
	print(f"gamma = {a}, train_accuracy = {b}, validation_accuracy = {c}")

print(trained_weights)

exit()

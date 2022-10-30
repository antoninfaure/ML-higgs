import implementations as impl
import helpers as helpers
import numpy as np

def best_lambda_selection(y, tx, max_iters, gamma):
	seed = 12
	lambdas = np.logspace(-6, -1, 3) #change last parameter to 10 if it works; don't go above 1!

	# Split data into training and validation sets
	y_va, y_tr, tx_va, tx_tr = helpers.slice_data(y, tx, 0.25, seed)

	# Add biases to data
	tx_va = np.c_[np.ones((y_va.shape[0], 1)), tx_va]
	tx_tr = np.c_[np.ones((y_tr.shape[0], 1)), tx_tr]

	initial_w = np.random.normal(0., 0.1, [tx_tr.shape[1],])

	# define lists to store the loss of training data and validation data
	losses_training = []
	losses_validation = []

	# cross validation
	for lambda_ in lambdas:
		print(f"Current lambda={lambda_}")

		# weights and training loss for logistic regression model:
		w, loss_training = impl.reg_logistic_regression(y_tr, tx_tr, lambda_, initial_w, max_iters, gamma) 

		# calculate the loss for validation data:
		loss_validation = (np.sum(np.logaddexp(0, tx_va.dot(w)) + y_va * tx_va.dot(w)) + lambda_*np.linalg.norm(w)**2)/y_va.shape[0]

		losses_training.append(loss_training)
		losses_validation.append(loss_validation)
		print(f"training_loss = {loss_training}, validation_loss = {loss_validation}")

	ind_best_lambda = np.argmin(losses_validation)
	best_lambda = lambdas[ind_best_lambda]
	print(f"Best lambda = {best_lambda}, training_loss = {losses_training[ind_best_lambda]}, validation_loss = {losses_validation[ind_best_lambda]}")
	return best_lambda

def best_gamma_selection(y, tx, max_iters):
	seed = 12
	gammas = np.logspace(-6, 0, 9) #use linspace maybe to go up to 2
    gammas.append(2) #works?

	# Split data into training and validation sets
	y_va, y_tr, tx_va, tx_tr = helpers.slice_data(y, tx, 0.25, seed)

	# Add biases to data
	tx_va = np.c_[np.ones((y_va.shape[0], 1)), tx_va]
	tx_tr = np.c_[np.ones((y_tr.shape[0], 1)), tx_tr]

	initial_w = np.random.normal(0., 0.1, [tx_tr.shape[1],])

	# define lists to store the loss of training data and validation data
	losses_training = []
	losses_validation = []

	for gamma in gammas:
		print(f"Current gamma={gamma}")

		w, loss_training = impl.logistic_regression(y_tr, tx_tr, initial_w, max_iters, gamma=gamma)

		# calculate the loss for validation data:
		loss_validation = np.sum(np.logaddexp(0, tx_va.dot(w)) + y_va * tx_va.dot(w))/y_va.shape[0]

		losses_training.append(loss_training)
		losses_validation.append(loss_validation)
		print(f"training_loss = {loss_training}, validation_loss = {loss_validation}")

	ind_best_gamma = np.argmin(losses_validation)
	best_gamma = gammas[ind_best_gamma]
	print(f"Best gamma = {best_gamma}, training_loss = {losses_training[ind_best_gamma]}, validation_loss = {losses_validation[ind_best_gamma]}")
	return best_gamma


# Load train data
#tx, y, ids = helpers.load_data('train.csv')

# Clean and standardize train data
#tx = helpers.standardize(helpers.clean_data(tx))
#y[y==-1]=0

# Shuffle data
#y, tx = helpers.shuffle_data(y, tx)

#best_gamma = best_gamma_selection(y, tx, 1000)

#best_lambda = best_lambda_selection(y, tx, 1000, best_gamma)

#exit()
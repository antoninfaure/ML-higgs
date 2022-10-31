import implementations as impl
import helpers as helpers
import numpy as np

def best_lambda_selection(y, tx, max_iters, gamma):
	""" Function to find best gamma for logistic regression.
    
    Args:
        y (np.array): Labels of shape (N, ).
        tx (np.array): Dataset of shape (N, D).
        max_iters (integer): Maximum number of iterations.
        gamma (float): Step size
    Returns:
        lambda (float): Best regularization factor
    """  

	seed = 12
	lambdas = np.logspace(-6, -1, 3) #change last parameter to 10 if it works; don't go above 1!

	# Split data into training and validation sets
	ratio = 0.25
	y_validation, y_train, tx_validation, tx_train = helpers.slice_data(y, tx, ratio, seed)

	# Add biases to data
	tx_validation = np.c_[np.ones((y_validation.shape[0], 1)), tx_validation]
	tx_train = np.c_[np.ones((y_train.shape[0], 1)), tx_train]

	initial_w = np.random.normal(0., 0.1, [tx_train.shape[1],])

	# Define lists to store the losses of training data and validation data
	losses_training = []
	losses_validation = []
	losses = []

	# Test every lambdas
	for lambda_ in lambdas:
		print(f"Current lambda={lambda_}")

		# Compute weights and training loss for regularized logistic regression model
		w, loss_training = impl.reg_logistic_regression(y_train, tx_train, lambda_, initial_w, max_iters, gamma)

		# Compute the loss (without regularization term) on validation data
		loss_validation = np.sum(np.logaddexp(0, tx_validation @ w) - y_validation * tx_validation.dot(w))/tx_validation.shape[0]

		# Store results
		losses_training.append(loss_training)
		losses_validation.append(loss_validation)
		losses.append(loss_training*(1-ratio)+loss_validation*ratio)
		print(f"training_loss = {loss_training}, validation_loss = {loss_validation}, total_loss = {loss_training*(1-ratio)+loss_validation*ratio}")

	# Select best lambda according to validation and training loss combined
	ind_best_lambda = np.argmin(losses)
	best_lambda = lambdas[ind_best_lambda]
	print(f"Best lambda = {best_lambda}, training_loss = {losses_training[ind_best_lambda]}, validation_loss = {losses_validation[ind_best_lambda]}, total_loss = {losses[ind_best_lambda]}")
	return best_lambda

def best_gamma_selection(y, tx, max_iters):
	""" Function to find best gamma for logistic regression.
    
    Args:
        y (np.array): Labels of shape (N, ).
        tx (np.array): Dataset of shape (N, D).
        max_iters (integer): Maximum number of iterations.
    Returns:
        gamma (float): Best step size
    """  

	seed = 12
	gammas = np.logspace(-6, -1, 6)

	# Split data into training and validation sets
	ratio = 0.25
	y_validation, y_train, tx_validation, tx_train = helpers.slice_data(y, tx, ratio, seed)

	# Add biases to data
	tx_validation = np.c_[np.ones((y_validation.shape[0], 1)), tx_validation]
	tx_train = np.c_[np.ones((y_train.shape[0], 1)), tx_train]

	initial_w = np.random.normal(0., 0.1, [tx_train.shape[1],])

	# Define lists to store the losses of training data and validation data
	losses_training = []
	losses_validation = []
	losses = []

	# Test every gamma
	for gamma in gammas:
		print(f"Current gamma={gamma}")

		# Compute weights and training loss for logistic regression model
		w, loss_training = impl.logistic_regression(y_train, tx_train, initial_w, max_iters, gamma=gamma)

		# Compute the loss on validation data:
		loss_validation = np.sum(np.logaddexp(0, tx_validation.dot(w)) + y_validation * tx_validation.dot(w))/y_validation.shape[0]

		losses_training.append(loss_training)
		losses_validation.append(loss_validation)
		losses.append(loss_training*(1-ratio)+loss_validation*ratio)
		print(f"training_loss = {loss_training}, validation_loss = {loss_validation}, total_loss = {loss_training*(1-ratio)+loss_validation*ratio}")

	# Select best gamma according to validation and training loss combined
	ind_best_gamma = np.argmin(losses)
	best_gamma = gammas[ind_best_gamma]
	print(f"Best gamma = {best_gamma}, training_loss = {losses_training[ind_best_gamma]}, validation_loss = {losses_validation[ind_best_gamma]}, total_loss = {losses[ind_best_gamma]}")
	return best_gamma


# Load train data
#tx, y, ids = helpers.load_data('train.csv')

# Refactor the -1 in 0 value for logistic regression
#y[y==-1]=0

# Shuffle data
#y, tx = helpers.shuffle_data(y, tx)

# Split and clean data into 4 sets according to 22nd feature
#tx_train_0, y_0, _, miss_col_0 = helpers.split_i(tx, y, ids, 0)

#Standardize the data
#tx_train_0, mean_0, std_0 = helpers.standardize(tx_train_0)

# Expand to degree 2
#tx_train_0 = helpers.build_poly_deg2(tx_train_0)

# Add bias to data
#tx_train_0 = np.c_[np.ones((tx_train_0.shape[0], 1)), tx_train_0]

#best_gamma = best_gamma_selection(y_0, tx_train_0, 3000)

#best_lambda = best_lambda_selection(y_0, tx_train_0, 3000, best_gamma)

#exit()
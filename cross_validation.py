import numpy as np
import implementations as impl

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # Set seed
    np.random.seed(seed)
    # Generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # Create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation_ridge(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    te_index = k_indices[k]
    tr_index = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_index = tr_index.reshape(-1)
    y_te = y[te_index]
    y_tr = y[tr_index]
    x_te = x[te_index]
    x_tr = x[tr_index]
    # form data with polynomial degree
    xpoly_tr = build_poly(x_tr, degree)
    xpoly_te = build_poly(x_te, degree)
    # weights and training loss for ridge regression model:
    w, loss_tr = impl.ridge_regression(y_tr, xpoly_tr, lambda_) 
    # calculate the loss for test data:
    loss_tr = np.sqrt(2 * impl.compute_mse(y_tr, xpoly_tr, w))
    loss_te = np.sqrt(2 * impl.compute_mse(y_te, xpoly_te, w))
    return loss_tr, loss_te, w

def cross_validation_logistic(y, x, k_indices, k, max_iters, gamma, degree):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    te_index = k_indices[k]
    tr_index = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_index = tr_index.reshape(-1)
    y_te = y[te_index]
    y_tr = y[tr_index]
    x_te = x[te_index]
    x_tr = x[tr_index]
    # form data with polynomial degree
    xpoly_tr = build_poly(x_tr, degree)
    xpoly_te = build_poly(x_te, degree)
    #initialize weight vector:
    initial_w = np.random.normal(0., 0.1, [xpoly_tr.shape[1],])
    # weights and training loss for ridge regression model:
    w, loss_tr = impl.logistic_regression(y_tr, xpoly_tr, initial_w, max_iters, gamma) 
    # calculate the loss for test data:
    loss_tr = np.sqrt(2 * impl.logistic_reg_loss(y_tr, xpoly_tr, w))
    loss_te = np.sqrt(2 * impl.logistic_reg_loss(y_te, xpoly_te, w))
    return loss_tr, loss_te, w

def best_degree_selection_ridge(y, x, degrees, k_fold, lambdas, seed = 1):
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    #for each degree, we compute the best lambdas and the associated rmse
    best_lambdas = []
    best_rmses = []
    #vary degree
    for degree in degrees:
        # cross validation
        rmse_te = []
        for lambda_ in lambdas:
            rmse_te_tmp = []
            for k in range(k_fold):
                _, loss_te,_ = cross_validation_ridge(y, x, k_indices, k, lambda_, degree)
                rmse_te_tmp.append(loss_te)
            rmse_te.append(np.mean(rmse_te_tmp))
        
        ind_lambda_opt = np.argmin(rmse_te)
        best_lambdas.append(lambdas[ind_lambda_opt])
        best_rmses.append(rmse_te[ind_lambda_opt])
        
    ind_best_degree =  np.argmin(best_rmses)      
        
    return degrees[ind_best_degree], lambdas[ind_lambda_opt]

def best_degree_selection_logistic(y, x, max_iters, gamma, degrees, k_fold, seed=1):
    #split data in k fold:
    k_indices = build_k_indices(y, k_fold, seed)
    
    for degree in degrees:
        losses_te = []
        for k in range(k_fold):
            _, loss_te,_ = cross_validation_logistic(y, x, k_indices, k, max_iters, gamma, degree)
            losses_te.append(loss_te)
            
    ind_degree_opt = np.argmin(losses_te)
    
    return degrees[ind_degree_opt]
    
    
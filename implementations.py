import numpy as np

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Training function for logistic regression. 
    
    Args:
        y (np.array): Labels of shape (N, ).
        tx (np.array): Dataset of shape (N, D).
        initial_w (np.array): Initial weights of shape (D,)
        max_iters (integer): Maximum number of iterations.
        gamma (float): Step size
    Returns:
        w (np.array): weights of shape(D, )
        loss (float): Final value of the cost function.
    """  
    def sigmoid(t):
        """apply sigmoid function on t."""
        return 1 / (1 + np.exp(-t))

    threshold = 1e-8
    losses = []
    
    w = initial_w.copy()
    for it in range(max_iters):
        # Compute cost function
        loss = np.sum(np.logaddexp(0, tx.dot(w)) - y * tx.dot(w))/tx.shape[0]
        # Compute gradient
        grad = tx.T.dot(sigmoid(tx.dot(w)) - y)/tx.shape[0]
        # Apply GD
        w -= gamma * grad
        # Log info
        if it % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=it, l=loss))
        # Converge criterion
        losses.append(loss)
        # If loss doesn't change more than threshold, break
        if len(losses) > 1 and (np.abs(losses[-1] - losses[-2]) < threshold) :
            print("breaking threshold")
            break
        # If loss is looping between two values, break
        if len(losses) > 3 and (losses[-1] == losses[-3]) :
            print("breaking looping")
            break
    # Compute final cost function
    loss = np.sum(np.logaddexp(0, tx.dot(w)) - y * tx.dot(w))/tx.shape[0]

    return (w, loss)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Training function for logistic regression with regularization. 
    
    Args:
        y (np.array): Labels of shape (N, ).
        tx (np.array): Dataset of shape (N, D).
        lambda_ (float): Regularization factor
        initial_w (np.array): Initial weights of shape (D,)
        max_iters (integer): Maximum number of iterations.
        gamma (float): Step size
    Returns:
        w (np.array): weights of shape(D, )
        loss (float): Final value of the cost function.
    """  
    def sigmoid(t):
        """apply sigmoid function on t."""
        return 1.0 / (1 + np.exp(-t))

    threshold = 1e-8
    losses = []
    
    w = initial_w.copy()
    for it in range(max_iters):
        # Compute loss without regularization term
        #loss = np.sum(np.logaddexp(0, tx @ w) - y * tx.dot(w))/tx.shape[0]
        # Compute gradient with regularization term
        grad = (tx.T.dot(sigmoid(tx.dot(w)) - y))/tx.shape[0] + 2*lambda_*w
        # Apply GD
        w -= gamma * grad
        # Log info
        if it % 100 == 0:
            #print("Current iteration={i}, loss={l}".format(i=it, l=loss))
            print(f"Current it = {it}")
        # Converge criterion
        #losses.append(loss)
        # If loss doesn't change more than threshold, break
        #if len(losses) > 1 and (np.abs(losses[-1] - losses[-2]) < threshold) :
           #print("breaking threshold")
            #break
        # If loss is looping between two values, break
        #if len(losses) > 3 and (losses[-1] == losses[-3]) :
            #print("breaking looping")
            #break
    # Compute final loss without regularization term
    loss = np.sum(np.logaddexp(0, tx @ w) - y * tx.dot(w))/tx.shape[0]

    return (w, loss)

def least_squares(y, tx):
    """Compute the least squares solution using closed form.
    
    Args:
        y (np.array): Labels of shape (N,), N is the number of samples.
        tx (np.array): Dataset of shape (N,D), D is the number of features.
    
    Returns:
        w (np.array): optimal weights, numpy array of shape(D,), D is the number of features.
        loss (float): Final value of the cost function.
    """
    #We solve the normal equations using QR decomposition, which is a computationally efficient method.
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    err = y - tx.dot(w)
    loss = np.mean(err**2)/2
    
    return w, loss

def ridge_regression(y, tx, lambda_):
    """Compute ridge regression using closed form solution
    
    Args:
        y (np.array): Labels of shape (N,), N is the number of samples.
        tx (np.array): Dataset of shape (N,D), D is the number of features.
        lambda_ (float): Regularization factor

    Returns:
        w (np.array): optimal weights, numpy array of shape(D,), D is the number of features.
        loss (float): Final value of the cost function.
    """
    I = np.eye(tx.shape[1])
    # Compute closed form weights
    w = np.linalg.inv(tx.T.dot(tx) + 2*lambda_*y.shape[0] * I).dot(tx.T @ y)
    # Compute loss without regularization term
    loss = np.sum((tx.dot(w) - y)**2)/(2*y.shape[0])

    return w, loss


def mean_squared_error_gd(y, tx, w_init, max_iters, gamma):
    """Training function to compute mean squared error solution using gradient descent
    
    Args:
        y (np.array): Labels of shape (N,), N is the number of samples.
        tx (np.array): Dataset of shape (N,D), D is the number of features.
        w_init (np.array): Initial weights of shape (D,)
        max_iters (integer): Maximum number of iterations.
        gamma (float): Step size
        
    Returns:
        w (np.array): optimal weights, numpy array of shape(D,), D is the number of features.
        loss (float): Final value of the cost function.
    """
    w = w_init.copy()
    for n in range (max_iters):
        gradient = -tx.T.dot(y - tx.dot(w))/tx.shape[0]
        w -= gamma * gradient
        # Log info
        if n % 100 == 0:
            print(f"Current iteration={n}")
    loss = 1/(2*tx.shape[0])*np.sum((y - tx.dot(w))**2)

    return w, loss

def mean_squared_error_sgd(y, tx, w_init, max_iters, gamma, batch_size = 1):
    """Training function to compute mean squared error solution using stochastic gradient descent, with batch_size=1
    
    Args:
        y (np.array): Labels of shape (N,), N is the number of samples.
        tx (np.array): Dataset of shape (N,D), D is the number of features.
        w_init (np.array): Initial weights of shape (D,)
        max_iters (integer): Maximum number of iterations.
        gamma (float): Step size
        batch_size (integer): Batch size
        
    Returns:
        w (np.array): optimal weights, numpy array of shape(D,), D is the number of features.
        loss (float): Final value of the cost function.
    """
    w = w_init.copy()
    for n in range (max_iters):
        for j in range(0,len(y),batch_size):
            tx_batch = tx[j:j+batch_size, :]
            y_batch = y[j:j+batch_size]
            w += (gamma/batch_size) * tx_batch.T.dot(y_batch - tx_batch.dot(w))
            # Log info
            if j % 100 == 0:
                print(f"Current iteration: n={n}, j={j}")
    loss =  1/(2*tx.shape[0])*np.sum((y - tx.dot(w))**2)

    return w, loss



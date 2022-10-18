import numpy as np

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Training function for binary class logistic regression. 
    
    Args:
        y (np.array): Labels of shape (N, ).
        tx (np.array): Dataset of shape (N, D).
        initial_w (np.array): Initial weights of shape (D,)
        max_iters (integer): Maximum number of iterations.
        gamma (integer): Step size
    Returns:
        np.array: weights of shape(D, )
    """  
    def sigmoid(t):
        """apply sigmoid function on t."""
        return 1 / (1 + np.exp(-t))

    threshold = 1e-8
    losses = []
    
    w = initial_w.copy()
    for it in range(max_iters):
        loss = np.sum(np.logaddexp(0, tx.dot(w)) - y * tx.dot(w))/tx.shape[0]
        grad = tx.T.dot(sigmoid(tx.dot(w)) - y)/tx.shape[0]
        w -= gamma * grad
        # log info
        if it % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=it, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and (np.abs(losses[-1] - losses[-2]) < threshold) :
            print("breaking threshold")
            break
        if len(losses) > 3 and (losses[-1] == losses[-3]) :
            print("breaking looping")
            break
    loss = np.sum(np.logaddexp(0, tx.dot(w)) - y * tx.dot(w))/tx.shape[0]
    return (w, loss)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Training function for binary class logistic regression. 
    
    Args:
        y (np.array): Labels of shape (N, ).
        tx (np.array): Dataset of shape (N, D).
        lambda_ (integer): Regularization factor
        initial_w (np.array): Initial weights of shape (D,)
        max_iters (integer): Maximum number of iterations.
        gamma (integer): Step size
    Returns:
        np.array: weights of shape(D, )
    """  
    def sigmoid(t):
        """apply sigmoid function on t."""
        return 1.0 / (1 + np.exp(-t))

    threshold = 1e-8
    losses = []
    
    w = initial_w.copy()
    for it in range(max_iters):
        loss = np.sum(np.logaddexp(0, tx @ w) - y * tx.dot(w))/tx.shape[0]
        grad = (tx.T.dot(sigmoid(tx.dot(w)) - y))/tx.shape[0] + 2*lambda_*w
        w -= gamma * grad
        # log info
        if it % 100 == 0:
            #print("Current iteration={i}, loss={l}".format(i=it, l=loss))
            print(f"Current iteration={it}")
        # converge criterion
        losses.append(loss)
        if len(losses) > 2 and ((np.abs(losses[-1] - losses[-2]) < threshold) or(losses[-1] == losses[-3])):
            print("breaking")
            break
    loss = np.sum(np.logaddexp(0, tx @ w) - y * tx.dot(w))/tx.shape[0]
    print("loss={l}".format(l=loss))
    return (w, loss)

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.

    """
    #We solve the normal equations using QR decomposition, which is a computationally efficient method.
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    err = y-tx.dot(w)
    loss =  1/2*np.mean(err**2)
    
    return w, loss

def ridge_regression(y, tx, lambda_):
    A = tx.T.dot(tx) + 2*tx.shape[0]*lambda_*np.identity(tx.shape[1])
    B = tx.T.dot(y)
    w = np.linalg.solve(A,B)
    loss = 1/(2*tx.shape[0])*(np.sum((y - tx.dot(w))**2)+lambda_*np.sum(w**2))

    print(w)
    print(loss)
    return w, loss


def mean_squared_error_gd(y, tx, w_init, max_iters, gamma):
    w = w_init.copy()
    for n in range (max_iters):
        gradient = -tx.T.dot(y - tx.dot(w))/tx.shape[0]
        w -= gamma * gradient
    loss = 1/(2*tx.shape[0])*np.sum((y - tx.dot(w))**2)
    return w, loss

def mean_squared_error_sgd(y, tx, w_init, max_iters, gamma):
    batch_size = 1
    w = w_init.copy()
    for n in range (max_iters):
        for j in range(0,len(y),batch_size):
            w -= (gamma/batch_size) * (tx.dot(w) - y).dot(tx).T
    loss =  1/(2*tx.shape[0])*np.sum((y - tx.dot(w))**2)
    print(loss)
    print(w)
    return w, loss
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
        grad = tx.T.dot(sigmoid(tx.dot(w)) - y) + 2*lambda_*w
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
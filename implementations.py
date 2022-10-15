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
        if (np.sum(t < 0)/t.shape[0] > 0.8):
            return np.exp(t) / (1 + np.exp(t))
        else:
            return 1 / (1 + np.exp(-t))

    threshold = 1e-10
    losses = []
    
    w = initial_w.copy()
    for it in range(max_iters):
        loss = np.sum(np.logaddexp(0, tx.dot(w)) + y * tx.dot(w))/tx.shape[0]
        #loss = -np.sum(y * np.log(sigmoid(tx @ w)) + (1 - y) * np.log(1 - sigmoid(tx @ w)))/tx.shape[0]
        grad = tx.T.dot(sigmoid(tx.dot(w)) - y)
        w -= gamma * grad
        # log info
        if it % 100 == 0:
            #print(f"Current iteration={it}")
            print("Current iteration={i}, loss={l}".format(i=it, l=loss))
            print(w)
        # converge criterion
        losses.append(loss)
        if len(losses) > 3 and ((np.abs(losses[-1] - losses[-2]) < threshold) or (losses[-3] == losses[-1])) :
            break
    loss = np.sum(np.logaddexp(0, tx.dot(w)) + y * tx.dot(w))/tx.shape[0]
    #loss = -np.sum(y * np.log(sigmoid(tx @ w)) + (1 - y) * np.log(1 - sigmoid(tx @ w)))/tx.shape[0]
    #print("loss={l}".format(l=loss))
    return (w, loss)
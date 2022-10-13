import numpy as np 
from numpy import linalg

def compute_gradient(y, tx, w):
    e = y - tx.dot(w)
    return (-1.0/len(y))*tx.T.dot(e)

def compute_loss_mse(y, tx, w):
    e = y - tx.dot(w)
    return (1.0/2*len(y))*np.sum(np.power(e, 2))

def compute_stochastic_gradient(y, tx, w):
    e = y _tx.dot(w)
    return [-1.0*tx*e].T

def batch_iter(y, tx, batch_size, nb_batch=1, shuffle=True):
    size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]

    else:
        shuffled_y = y
        shuffled_tx = tx
    for n in range (nb_batch):
        start_index = nb_batch * batch_size
        end_index = min((nb_batch + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], suffled_tx[start_index:end_index]

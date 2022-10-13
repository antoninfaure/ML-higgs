import numpy as np

def least_square_gd(y, x, w_init, max_iter, gamma):
    w = w_init
    loss = compute_loss_mse(y, x, w_init)
    for n in range (max_iter):
        gradient = compute_gradient(y, x, w)
        w = w - gamma * gradient
        loss = compute_loss_mse(y, x, w)
    return loss, w

def least_square_sgd(y, x, w_init, max_iter, gamma, batch_size=1):
    w = w_init
    loss = compute_loss_mse(y, x, w_init)
    for n in range (max_iter):
        i = np.random.choice(len(y))
        gradient = compute_stochastic_gradient(y[i], x[i], w)
        for m in range (batch_size - 1):
            i = np.random.choice(len(y))
            gradient = gradient + compute_stochastic_gradient(y[i], x[i], w)
        gradient = gradient / batch_size
        w = w - gamma * gradient
        loss =  compute_loss_mse(y, x, w)
    return loss, w

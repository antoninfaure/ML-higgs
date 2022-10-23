import numpy as np
import csv as csv

def load_data(path_dataset,sub_sample=True, add_outlier=False, train=True):
    """Load data and convert it to the metric system."""
    data = np.genfromtxt(
        path_dataset, delimiter=",", dtype=str,  skip_header=1)
    ids = data[:,0]
    labels = data[:,1]
    if train == True:
        labels[labels=='s']=1
        labels[labels=='b']=-1
        labels = np.asarray(labels, dtype=float)
    data = np.delete(data, [0,1], 1)
    data = np.asarray(data, dtype=float)
    return data, labels, ids

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


# Shuffle data
def shuffle_data(y, tx, seed=1):
    np.random.seed(seed)
    inds = np.random.permutation(tx.shape[0])
    tx = tx[inds]
    y = y[inds]
    return y, tx

# Slice data and labels into training and validation sets
def slice_data(y, tx, ratio, seed=1): 
    slice_id = int(np.floor(y.shape[0]*ratio))
    y_va, y_tr = y[:slice_id], y[slice_id:]
    tx_va, tx_tr = tx[:slice_id], tx[slice_id:]
    return y_va, y_tr, tx_va, tx_tr

def clean_data(data):
    # Replace -999 by nan
    data = np.where(data == -999, np.nan, data)
    # Compute the columns means without nan values 
    means = np.nanmean(data, axis=0)
    #Find indices that you need to replace
    inds = np.where(np.isnan(data))
    #Place column means in the indices. Align the arrays using take
    data[inds] = np.take(means, inds[1])
    return data

# Standardize the data
def standardize(x):
    #Standardize the original data set.
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x

def predict_logistic(tx, w):
    def sigmoid(t):
        return 1.0 / (1 + np.exp(-t))
    y = sigmoid(tx @ w)
    # s = 1 , b = -1
    y[y > 0.5] = 1
    y[y <= 0.5] = -1
    return y

def accuracy(a, b):
    return np.sum(a == b)/a.shape[0]

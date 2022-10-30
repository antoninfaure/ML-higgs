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
    #means = np.nanmean(data, axis=0)
    medians = np.nanmedian(data, axis=0)
    #Find indices that you need to replace
    inds = np.where(np.isnan(data))
    #Place column means in the indices. Align the arrays using take
    data[inds] = np.take(medians, inds[1])
    return data

def build_poly_corr(x):
    """multiply correlated features"""
    poly = x.copy()
    coeff = np.corrcoef(x.T)
    ind = np.where(np.absolute(coeff)>=.5)
    indices = list(zip(ind[0], ind[1]))
    for i,j in indices:
        poly = np.c_[poly, x[:,i]*x[:,j]]
    return poly

def build_poly_deg2(x):
    """polynomial basis functions for input data x, to degree 2"""
    poly = x.copy()
    n = x.shape[1]
    for i in range(n):
        for j in range(n):
            if (i <= j):
                poly = np.c_[poly, x[:,i]*x[:,j]]
    return poly

# Standardize the data by the mean and std
def standardize(x, mean_x =[], std_x =[]):
    #Standardize the original data set.
    if len(mean_x) == 0:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if len(std_x) == 0:
        std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

# Predict the labels of logistic regression
def predict_logistic(tx, w):
    def sigmoid(t):
        return 1.0 / (1 + np.exp(-t))
    y = sigmoid(tx @ w)
    y[y <= 0.5] = -1
    y[y > 0.5] = 1
    return y

# Predict the labels
def predict(tx, w):
    y = tx.dot(w)
    y[y < 0.5] = -1
    y[y >= 0.5] = 1
    return y

# Compute accuracy
def accuracy(a, b):
    return np.sum(a == b)/a.shape[0]


# Split the dataset according to the 22nd feature value (=i)
def split_i(tx, y, ids, i, miss_col=[]):
    """
    Split the dataset according to the 22nd feature value (=i)
    Arguments: 
        tx (np.array): dataset of shape (N,D), D is the number of features.
        y (np.array): labels of shape (N,), N is the number of samples.
        ids (np.array): ids of the samples (N,)
        i (integer): value of the 22nd feature that we are evaluating
        miss_col (np.array): indices of the features to remove (if provided)
    Returns:
        tx_i (np.array): subset of tx corresponding to i value
        y_i (np.array): subset of y corresponding to i value
        ids_i (np.array): subset of ids corresponding to i value
        miss_col (np.array): indices of the features removed
    """

    # Gets the rows indicies where the 22nd feature is equal to i
    rows_i = np.where(tx[:, 22]==i)[0]

    # Extract the set corresponding to i from the dataset 
    tx_i = tx[rows_i]
    y_i = y[rows_i]
    ids_i = ids[rows_i]
    
    # (for test data the miss_col is provided)
    if len(miss_col) == 0:
        # Remove features with all values equals to -999
        miss_col = np.where(np.sum(tx_i == -999, axis=0) == tx_i.shape[0])[0]

        # Remove 22nd feature
        miss_col = np.append(miss_col, 22)

        # If label is 0 then remove 2nd last feature
        if (0 == i):
            miss_col = np.append(miss_col,tx_i.shape[1] - 1)

    # Remove chosen features
    tx_i = np.delete(tx_i, miss_col, 1)

    # Replace -999 values by median of feature
    tx_i = clean_data(tx_i)
    
    return tx_i, y_i, ids_i, miss_col

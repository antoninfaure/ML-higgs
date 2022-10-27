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

def clean_data_medians(data):
    # Replace -999 by nan
    data = np.where(data == -999, np.nan, data)
    # Compute the columns medians without nan values 
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

# Standardize the data
def standardize(x):
    #Standardize the original data set.
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x

# Build polynomial expansion
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly[:,1:]

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


def split_i(tx, y, ids, i, miss_col=[]):
    rows_i = np.where(tx[:, 22]==i)[0]
    tx_i = tx[rows_i]
    y_i = y[rows_i]
    ids_i = ids[rows_i]
    
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

    #Standardize the data
    tx_i = standardize(tx_i)
    
    return tx_i, y_i, ids_i, miss_col

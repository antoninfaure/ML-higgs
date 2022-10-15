import numpy as np

def load_data_train(path_dataset,sub_sample=True, add_outlier=False):
    """Load data and convert it to the metric system."""
    data = np.genfromtxt(
        path_dataset, delimiter=",", dtype=str,  skip_header=1)
    ids = data[:,0]
    labels = data[:,1]
    labels[labels=='s']=0
    labels[labels=='b']=1
    labels = np.asarray(labels, dtype=float)
    data = np.delete(data, [0,1], 1)
    data = np.asarray(data, dtype=float)
    return data, labels, ids

def clean_data(data):
    # Remove columns with more than 50% of -999
    dirty_cols = np.where(np.sum(data == -999, axis=0)/data.shape[0] < 0.5, True, False)
    data = data[:, dirty_cols]
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
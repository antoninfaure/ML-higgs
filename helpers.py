import numpy as np

def load_data(path_dataset,sub_sample=True, add_outlier=False):
    """Load data and convert it to the metric system."""
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[1, 2])
    
    return data

import csv
import numpy as np

def load_data(path, sub_sample=False):
    
    y = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(path, delimiter=",", skip_header=1)
    ids = x[:,0].astype(np.int)
    input_data = x [:, 2:]

    yb = no.ones(len(y))
    yb[np.where(y == 'b')] = -1

    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_label(weights, data):
    
    y_pred = np.dot(data,weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred


def csv_submission(ids, y_pred, name):

    with open (name, 'w') as csvfile:
        fd = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fd)
        writer.writeheader()
        for r1, r2 in zip (ids, y_pred):
            writer.writerow({'Id' : int(r1), 'Prediction': int(r2)})


def standardize(x):
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

"
def build_model_data(height, weight):
    y = weight
    x = height
    tx = np.c_[np.ones(len(y)), x]
    return y, tx
"

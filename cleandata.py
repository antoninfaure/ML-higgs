import numpy as np

e=np.array([
[1, 0, 1, 4],
[3, 3, 3, 0],
[2, -999, 2, 4],
[2, -999, 2, 4],
[3, 3, 3, 4]
])
print(e)

def clean_data_v2(tx, y, ids, i):
	rows_i = np.where(tx[:, 22]==i)[0]
	tx_i = tx[rows_i]
	y_i = y[rows_i]
	ids_i = ids[rows_i]

	# Remove features with all values equals to -999
	miss_col = np.where(np.sum(tx_i == -999, axis=0) == tx_i.shape[0])[0]
	print(miss_col)
	
	# Remove 22nd feature
	miss_col = np.append(miss_col, 22)

	# If label is 0 then remove 2nd last feature
	if (0 == i):
		miss_col = np.append(miss_col,tx_i.shape[1] - 1)

	# Remove chosen features
	tx_i = np.delete(tx_i, miss_col, 1)

	# Replace -999 values by median of feature
	tx_i = np.where(tx_i == -999, np.nan, tx_i)
	medians = np.nanmedian(tx_i, axis=0)
	inds = np.where(np.isnan(tx_i))
	tx_i[inds] = np.take(medians, inds[1])

	#Standardize the data
    tx_i = (tx_i - np.mean(tx_i, axis=0))
    tx_i = tx_i/np.std(tx_i, axis=0)
    return tx_i, y_i, ids_i


print(clean_data_v2(e, 4))
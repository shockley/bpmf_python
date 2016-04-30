import pystan,pickle,matplotlib,warnings
import sys,time,os
import numpy as np
import pandas,pmf
from sklearn.datasets import *
from scipy.sparse import coo_matrix




# Define our evaluation function.
def rmse(test_data, predicted):
    """Calculate root mean squared error.
    Ignoring missing values in the test data.
    """
    I = ~np.isnan(test_data)   # indicator for missing values
    N = I.sum()                # number of non-missing values
    sqerror = abs(test_data - predicted) ** 2  # squared error array
    mse = sqerror[I].sum() / N                 # mean squared error
    return np.sqrt(mse)                        # RMSE

import hashlib


# Define a function for splitting train/test data.
def split_train_test(data, percent_test=10):
    """Split the data into train/test sets.
    :param int percent_test: Percentage of data to use for testing. Default 10.
    """
    n, m = data.shape             # # users, # jokes
    N = n * m                     # # cells in matrix
    test_size = N / percent_test  # use 10% of data as test set
    train_size = N - test_size    # and remainder for training

    # Prepare train/test ndarrays.
    train = data.copy().values
    test = np.ones(data.shape) * np.nan

    # Draw random sample of training data to use for testing.
    tosample = np.where(~np.isnan(train))       # ignore nan values in data
    idx_pairs = zip(tosample[0], tosample[1])   # tuples of row/col index pairs
    indices = np.arange(len(idx_pairs))         # indices of index pairs
    sample = np.random.choice(indices, replace=False, size=test_size)

    # Transfer random sample from train set to test set.
    for idx in sample:
        idx_pair = idx_pairs[idx]
        test[idx_pair] = train[idx_pair]  # transfer to test set
        train[idx_pair] = np.nan          # remove from train set

    # Verify everything worked properly
    assert(np.isnan(train).sum() == test_size)
    assert(np.isnan(test).sum() == train_size)
    
    # Finally, hash the indices and save the train/test sets.
    index_string = ''.join(map(str, np.sort(sample)))
    name = hashlib.sha1(index_string).hexdigest()
    savedir = os.path.join('data', name)
    save_np_vars({'train': train, 'test': test}, savedir)
    
    # Return train set, test set, and unique hash of indices.
    return train, test, name


def load_train_test(name):
    """Load the train/test sets."""
    savedir = os.path.join('data', name)
    vars = load_np_vars(savedir)
    return vars['train'], vars['test']

 



if __name__ == '__main__':
	raw_data='../data/ml1m_d.sparse'
	with warnings.catch_warnings():
	  warnings.simplefilter("ignore")
	  (x,y) = load_svmlight_file(raw_data, dtype=np.int32, zero_based=False)
	#x=coo_matrix(x)
	data = pandas.DataFrame(data=x.todense())
	train, test, name_hash = split_train_test(data, percent_test=10)

	#ratings = np.transpose(np.array([x.row,x.col,x.data]))
	# train, test, name = split_train_test(data)

	#train, test = load_train_test('6bb8d06c69c0666e6da14c094d4320d115f1ffc8')
	# Let's see the results:
	baselines = {}
	for name in baseline_methods:
	    Method = baseline_methods[name]
	    method = Method(train)
	    baselines[name] = method.rmse(test)
	    print '%s RMSE:\t%.5f' % (method, baselines[name])

	# We use a fixed precision for the likelihood.
	# This reflects uncertainty in the dot product.
	# We choose 2 in the footsteps Salakhutdinov
	# Mnihof.
	ALPHA = 2

	# The dimensionality D; the number of latent factors.
	# We can adjust this higher to try to capture more subtle
	# characteristics of each joke. However, the higher it is,
	# the more expensive our inference procedures will be.
	# Specifically, we have D(N + M) latent variables. For our
	# Jester dataset, this means we have D(1100), so for 5
	# dimensions, we are sampling 5500 latent variables.
	DIM = 10


	pmf = PMF(train, DIM, ALPHA, std=0.01)
	# Find MAP for PMF.
	#pmf.find_map()
	pmf.find_advi()
	# pmf.load_map()
	
	# Evaluate PMF MAP estimates.
	#pmf_rmse = pmf.eval_map(train, test)
	pmf_rmse = pmf.eval_advi(train, test)
	pmf_improvement = baselines['mom'] - pmf_rmse
	print 'PMF Improvement:   %.5f' % pmf_improvement

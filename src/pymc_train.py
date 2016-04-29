import pystan,pickle,matplotlib,warnings
import sys,time,os
import numpy as np
from sklearn.datasets import *
from scipy.sparse import coo_matrix


raw_data='../data/ml1m_d.sparse'
with warnings.catch_warnings():
  warnings.simplefilter("ignore")
  (x,y) = load_svmlight_file(raw_data, dtype=np.int32, zero_based=False)
x=coo_matrix(x)
#ratings = np.transpose(np.array([x.row,x.col,x.data]))
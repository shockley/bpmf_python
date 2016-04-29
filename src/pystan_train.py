import pystan,pickle,matplotlib,warnings
import sys,time,os
import numpy as np
from sklearn.datasets import *
from scipy.sparse import coo_matrix


raw_data='../data/ml1m_d.sparse'
save_to='../save'
with warnings.catch_warnings():
  warnings.simplefilter("ignore")
  (x,y) = load_svmlight_file(raw_data, dtype=np.int32, zero_based=False)
x=coo_matrix(x)
ratings = np.transpose(np.array([x.row,x.col,x.data]))
#ratings={'_ratings':ratings}

users = ratings[:, 0].astype(int) + 1
items = ratings[:, 1].astype(int) + 1

assert np.all(users - 1 == ratings[:, 0])
assert np.all(items - 1 == ratings[:, 1])

ranka = 10
rating_data = {
'ranka': ranka,
'n_users': users.max(),
'n_items': items.max(),
'n_obs': ratings.shape[0],
'obs_users': users,
'obs_items': items,
'obs_ratings': ratings[:, 2],
'rating_std': 2,
'mu_0': np.zeros(ranka),
'beta_0': 2,
'nu_0': ranka,
'w_0': np.eye(ranka),
}


codefile='bpmf_straightforward.stan'
# if binary==1:
#   codefile='binary_bpmf.stan'


start_time = time.clock()

with open(codefile, 'r') as code:
  #fit = pystan.stan(model_code=schools_code, data=a, iter=1000, chains=4)
  #fit = pystan.stan(file=code, data=a, iter=1000, sample_file='sample_file', chains=4)

  fit = pystan.stan(file=code, data=rating_data, iter=1000, chains=1, verbose = True)

  end_time = time.clock()
  print "Fitting data cost: " + str(end_time - start_time)+ " seconds."
  #print(fit)


  mu_u = fit.extract(permuted=True)['mu_u']
  mu_v = fit.extract(permuted=True)['mu_v']
  cov_u = fit.extract(permuted=True)['cov_u']
  cov_v = fit.extract(permuted=True)['cov_v']
  U = fit.extract(permuted=True)['U']
  V = fit.extract(permuted=True)['V']
  training_error = fit.extract(permuted=True)['training_error']
  predictions = fit.extract(permuted=True)['predictions']
  z = fit.extract(permuted=True)['z']

  print training_error
  print training_error.shape

  print 'mean training_error : ' + str(np.mean(training_error))

  start_time = time.clock()
  if not os.path.exists(prefix):
      os.makedirs(prefix)
  
  with open(save_to+'/training_error','w') as f:
    pickle.dump(training_error,f)
  with open(save_to+'/predictions','w') as f:
    print np.mean(predictions,0)
    pickle.dump(np.mean(predictions,0),f)
  # with open(prefix+'/mu_u','w') as f:
  #   pickle.dump(mu_u,f)
  # with open(prefix+'/mu_v','w') as f:
  #   pickle.dump(mu_v,f)
  # with open(prefix+'/cov_u','w') as f:
  #   pickle.dump(cov_u,f)
  # with open(prefix+'/cov_v','w') as f:
  #   pickle.dump(cov_v,f)
  # with open(prefix+'/U','w') as f:
  #   pickle.dump(np.mean(U,0),f)
  # with open(prefix+'/V','w') as f:
  #   pickle.dump(np.mean(V,0),f)
  # with open(prefix+'/z','w') as f:
  #   print 'z = '+str(z)
  #   pickle.dump(z,f)

  end_time = time.clock()
  print "Saving paras cost: " + str(end_time - start_time)+ " seconds."

  #fitfile = 'save/fit'
  # I don't know why but this takes time
  # start_time = time.clock()
  #with open(fitfile,'w') as f:
  #  f.write(str(fit))
  #  end_time = time.clock()
  # print "Printing fit cost: " + str(end_time - start_time)+ " seconds."

  # if matplotlib is installed (optional, not required), a visual summary and
  # traceplot are available
  #fit.plot(training_rmse)
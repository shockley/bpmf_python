import time
import logging
import pymc3 as pm
import theano
import scipy as sp


# Enable on-the-fly graph computations, but ignore 
# absence of intermediate test values.
theano.config.compute_test_value = 'ignore'

# Set up logging.
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class PMF(object):
    """Probabilistic Matrix Factorization model using pymc3."""

    def __init__(self, train, dim, alpha=2, std=0.01, bounds=(1, 5)):
        """Build the Probabilistic Matrix Factorization model using pymc3.

        :param np.ndarray train: The training data to use for learning the model.
        :param int dim: Dimensionality of the model; number of latent factors.
        :param int alpha: Fixed precision for the likelihood function.
        :param float std: Amount of noise to use for model initialization.
        :param (tuple of int) bounds: (lower, upper) bound of ratings.
            These bounds will simply be used to cap the estimates produced for R.
        """
        self.dim = dim
        self.alpha = alpha
        self.std = np.sqrt(1.0 / alpha)
        self.bounds = bounds
        self.data = train.copy()
        n, m = self.data.shape

        # Perform mean value imputation
        nan_mask = np.isnan(self.data)
        self.data[nan_mask] = self.data[~nan_mask].mean()

        # Low precision reflects uncertainty; prevents overfitting.
        # Set to the mean variance across users and items.
        self.alpha_u = 1 / self.data.var(axis=1).mean()
        self.alpha_v = 1 / self.data.var(axis=0).mean()

        # Specify the model.
        logging.info('building the PMF model')
        with pm.Model() as pmf:
            U = pm.MvNormal(
                'U', mu=0, tau=self.alpha_u * np.eye(dim),
                shape=(n, dim), testval=np.random.randn(n, dim) * std)
            V = pm.MvNormal(
                'V', mu=0, tau=self.alpha_v * np.eye(dim),
                shape=(m, dim), testval=np.random.randn(m, dim) * std)
            R = pm.Normal(
                'R', mu=theano.tensor.dot(U, V.T), tau=self.alpha * np.ones((n, m)),
                observed=self.data)

        logging.info('done building the PMF model') 
        self.model = pmf

    def __str__(self):
        return self.name

try:
    import ujson as json
except ImportError:
    import json


# First define functions to save our MAP estimate after it is found.
# We adapt these from `pymc3`'s `backends` module, where the original
# code is used to save the traces from MCMC samples.
def save_np_vars(vars, savedir):
    """Save a dictionary of numpy variables to `savedir`. We assume
    the directory does not exist; an OSError will be raised if it does.
    """
    logging.info('writing numpy vars to directory: %s' % savedir)
    os.mkdir(savedir)
    shapes = {}
    for varname in vars:
        data = vars[varname]
        var_file = os.path.join(savedir, varname + '.txt')
        np.savetxt(var_file, data.reshape(-1, data.size))
        shapes[varname] = data.shape

        ## Store shape information for reloading.
        shape_file = os.path.join(savedir, 'shapes.json')
        with open(shape_file, 'w') as sfh:
            json.dump(shapes, sfh)
            
            
def load_np_vars(savedir):
    """Load numpy variables saved with `save_np_vars`."""
    shape_file = os.path.join(savedir, 'shapes.json')
    with open(shape_file, 'r') as sfh:
        shapes = json.load(sfh)

    vars = {}
    for varname, shape in shapes.items():
        var_file = os.path.join(savedir, varname + '.txt')
        vars[varname] = np.loadtxt(var_file).reshape(shape)
        
    return vars


# Now define the MAP estimation infrastructure.
def _map_dir(self):
    basename = 'pmf-map-d%d' % self.dim
    return os.path.join('data', basename)
def _means_dir(self):
    basename = 'pmf-mean-d%d' % self.dim
    return os.path.join('data', basename)
def _sds_dir(self):
    basename = 'pmf-sd-d%d' % self.dim
    return os.path.join('data', basename)
def _elbos_dir(self):
    basename = 'pmf-elbo-d%d' % self.dim
    return os.path.join('data', basename)

def _find_map(self):
    """Find mode of posterior using Powell optimization."""
    tstart = time.time()
    with self.model:
        logging.info('finding PMF MAP using Powell optimization...')
        self._map = pm.find_MAP(fmin=sp.optimize.fmin_powell, disp=True)

    elapsed = int(time.time() - tstart)
    logging.info('found PMF MAP in %d seconds' % elapsed)
    
    # This is going to take a good deal of time to find, so let's save it.
    save_np_vars(self._map, self.map_dir)
    
def _load_map(self):
    self._map = load_np_vars(self.map_dir)
def _load_means(self):
    self._means = load_np_vars(self.means_dir)
def _load_sds(self):
    self._sds = load_np_vars(self.sds_dir)
def _load_elbos(self):
    self._elbos = load_np_vars(self.elbos_dir)

def _map(self):
    try:
        return self._map
    except:
        if os.path.isdir(self.map_dir):
            self.load_map()
        else:
            self.find_map()
        return self._map

def _means(self):
    try:
        return self._means
    except:
        if os.path.isdir(self.advi_dir):
            self.load_advi()
        else:
            self.find_advi()
        return self._means
def _sds(self):
    try:
        return self._sds
    except:
        if os.path.isdir(self.advi_dir):
            self.load_advi()
        else:
            self.find_advi()
        return self._sds
def _elbos(self):
    try:
        return self._elbos
    except:
        if os.path.isdir(self.advi_dir):
            self.load_elbos()
        else:
            self.find_advi()
        return self._elbos

def eval_advi(pmf_model, train, test):
    U = pmf_model.means['U']
    V = pmf_model.means['V']
    # Make predictions and calculate RMSE on train & test sets.
    predictions = pmf_model.predict(U, V)
    train_rmse = rmse(train, predictions)
    test_rmse = rmse(test, predictions)
    overfit = test_rmse - train_rmse
    # Print report.
    print 'PMF MAP training RMSE: %.5f' % train_rmse
    print 'PMF MAP testing RMSE:  %.5f' % test_rmse
    print 'Train/test difference: %.5f' % overfit
    return test_rmse

def eval_map(pmf_model, train, test):
    U = pmf_model.map['U']
    V = pmf_model.map['V']
    # Make predictions and calculate RMSE on train & test sets.
    predictions = pmf_model.predict(U, V)
    train_rmse = rmse(train, predictions)
    test_rmse = rmse(test, predictions)
    overfit = test_rmse - train_rmse
    # Print report.
    print 'PMF MAP training RMSE: %.5f' % train_rmse
    print 'PMF MAP testing RMSE:  %.5f' % test_rmse
    print 'Train/test difference: %.5f' % overfit
    return test_rmse


def _find_advi(self):
    """Find mode of posterior using Powell optimization."""
    tstart = time.time()
    with self.model:
        logging.info('finding PMF MAP using Powell optimization...')
        #self._map = pm.find_MAP(fmin=sp.optimize.fmin_powell, disp=True)
        self._means, self._sds, self._elbos = pm.variational.advi(model=self.model, n=20000, accurate_elbo=True)

    elapsed = int(time.time() - tstart)
    logging.info('found PMF MAP in %d seconds' % elapsed)
    
    # This is going to take a good deal of time to find, so let's save it.
    save_np_vars(self._means, self.means_dir)
    save_np_vars(self._sds, self.sds_dir)
    save_np_vars(self._elbos, self.elbos_dir)


def _predict(self, U, V):
    """Estimate R from the given values of U and V."""
    R = np.dot(U, V.T)
    n, m = R.shape
    sample_R = np.array([
        [np.random.normal(R[i,j], self.std) for j in xrange(m)]
        for i in xrange(n)
    ])

    # bound ratings
    low, high = self.bounds
    sample_R[sample_R < low] = low
    sample_R[sample_R > high] = high
    return sample_R




# Draw MCMC samples.
def _trace_dir(self):
    basename = 'pmf-mcmc-d%d' % self.dim
    return os.path.join('data', basename)

def _draw_samples(self, nsamples=1000, njobs=2):
    # First make sure the trace_dir does not already exist.
    if os.path.isdir(self.trace_dir):
        raise OSError(
            'trace directory %s already exists. Please move or delete.' % self.trace_dir)
    start = self.map  # use our MAP as the starting point
    with self.model:
        logging.info('drawing %d samples using %d jobs' % (nsamples, njobs))
        step = pm.NUTS(scaling=start)
        backend = pm.backends.Text(self.trace_dir)
        logging.info('backing up trace to directory: %s' % self.trace_dir)
        self.trace = pm.sample(nsamples, step, start=start, njobs=njobs, trace=backend)
        
def _load_trace(self):
    with self.model:
        self.trace = pm.backends.text.load(self.trace_dir)

PMF.predict = _predict

# Add eval function to PMF class.
PMF.eval_map = eval_map
    
# Update our class with the new MAP infrastructure.
PMF.find_map = _find_map
PMF.load_map = _load_map
PMF.map_dir = property(_map_dir)
PMF.map = property(_map)

# Update our class with the sampling infrastructure.
PMF.trace_dir = property(_trace_dir)
PMF.draw_samples = _draw_samples
PMF.load_trace = _load_trace
PMF.predict = _predict


PMF.find_advi = _find_advi
PMF.means_dir = property(_means_dir)
PMF.means = property(_means)
PMF.sds_dir = property(_sds_dir)
PMF.sds = property(_sds)
PMF.elbos_dir = property(_elbos_dir)
PMF.elbos = property(_elbos)

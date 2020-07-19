from __future__ import division
import openmdao.api as om 
import pickle
import numpy as np
import scipy.linalg as linalg
from scipy.optimize import minimize

from openmdao.surrogate_models.surrogate_model import SurrogateModel

MACHINE_EPSILON = np.finfo(np.double).eps


class KrigingSurrogate(om.KrigingSurrogate):

    def _declare_options(self):
        super(KrigingSurrogate, self)._declare_options()
        self.options.declare('cache_trained_model', default=False, types=bool,
                             desc='Cache the trained model to avoid repeat training')

        self.options.declare('cached_model_filename', default='krigemodel.pkl', types=str,
                             desc='Filename for cached trained model')

    def train(self, x, y):
        """
        Train the surrogate model with the given set of inputs and outputs.
        Parameters
        ----------
        x : array-like
            Training input locations
        y : array-like
            Model responses at given inputs.
        """
        cache_model = self.options['cache_trained_model']
        cached_model_filename = self.options['cached_model_filename']
        if cache_model:
            try:
                cached_model = pickle.load(open(cached_model_filename, 'rb'))
            except:
                cached_model = None
        else:
            cached_model = None
        
        super(om.KrigingSurrogate, self).train(x, y)

        if not cached_model:
            x, y = np.atleast_2d(x, y)

            self.n_samples, self.n_dims = x.shape

            if self.n_samples <= 1:
                self._raise('KrigingSurrogate requires at least 2 training points.',
                            exc_type=ValueError)

            # Normalize the data
            X_mean = np.mean(x, axis=0)
            X_std = np.std(x, axis=0)
            Y_mean = np.mean(y, axis=0)
            Y_std = np.std(y, axis=0)

            X_std[X_std == 0.] = 1.
            Y_std[Y_std == 0.] = 1.

            X = (x - X_mean) / X_std
            Y = (y - Y_mean) / Y_std

            self.X = X
            self.Y = Y
            self.X_mean, self.X_std = X_mean, X_std
            self.Y_mean, self.Y_std = Y_mean, Y_std

            def _calcll(thetas):
                """Calculate loglike (callback function)."""
                loglike = self._calculate_reduced_likelihood_params(np.exp(thetas))[0]
                return -loglike

            bounds = [(np.log(1e-5), np.log(1e5)) for _ in range(self.n_dims)]

            optResult = minimize(_calcll, 1e-1 * np.ones(self.n_dims), method='slsqp',
                                options={'eps': 1e-3},
                                bounds=bounds)

            if not optResult.success:
                msg = 'Kriging Hyper-parameter optimization failed: {0}'.format(optResult.message)
                self._raise(msg, exc_type=ValueError)

            self.thetas = np.exp(optResult.x)
            _, params = self._calculate_reduced_likelihood_params()
            self.alpha = params['alpha']
            self.U = params['U']
            self.S_inv = params['S_inv']
            self.Vh = params['Vh']
            self.sigma2 = params['sigma2']
            if cache_model:
                pickle.dump(self, open(cached_model_filename, "wb"))
        else:
            self.X = cached_model.X
            self.Y = cached_model.Y
            self.n_samples = cached_model.n_samples
            self.n_dims = cached_model.n_dims
            self.X_mean = cached_model.X_mean
            self.X_std = cached_model.X_std
            self.Y_mean = cached_model.Y_mean
            self.Y_std = cached_model.Y_std
            self.thetas = cached_model.thetas
            self.alpha = cached_model.alpha
            self.U = cached_model.U
            self.S_inv = cached_model.S_inv
            self.Vh = cached_model.Vh
            self.sigma2 = cached_model.sigma2

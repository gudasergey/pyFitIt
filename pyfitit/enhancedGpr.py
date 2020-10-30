from scipy.spatial import distance
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.utils import check_array
from scipy.linalg import cholesky, cho_solve, solve_triangular
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

import numpy as np

import warnings


class EnhancedGaussianProcessRegressor(GaussianProcessRegressor):

    def __init__(self, kernel=None, alpha=1e-10, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0, normalize_y=False,
                 copy_X_train=True, random_state=None):
        super().__init__(kernel, alpha, optimizer, n_restarts_optimizer, normalize_y, copy_X_train, random_state)



    def predict(self, X, return_std=False, return_cov=False):
        """Predict using the Gaussian process regression model

                We can also predict based on an unfitted model by using the GP prior.
                In addition to the mean of the predictive distribution, also its
                standard deviation (return_std=True) or covariance (return_cov=True).
                Note that at most one of the two can be requested.

                Parameters
                ----------
                X : array-like, shape = (n_samples, n_features)
                    Query points where the GP is evaluated

                return_std : bool, default: False
                    If True, the standard-deviation of the predictive distribution at
                    the query points is returned along with the mean.

                return_cov : bool, default: False
                    If True, the covariance of the joint predictive distribution at
                    the query points is returned along with the mean

                Returns
                -------
                y_mean : array, shape = (n_samples, [n_output_dims])
                    Mean of predictive distribution a query points

                y_std : array, shape = (n_samples,), optional
                    Standard deviation of predictive distribution at query points.
                    Only returned when return_std is True.

                y_cov : array, shape = (n_samples, n_samples), optional
                    Covariance of joint predictive distribution a query points.
                    Only returned when return_cov is True.
                """
        if return_std and return_cov:
            raise RuntimeError(
                "Not returning standard deviation of predictions when "
                "returning full covariance.")

        X = check_array(X)

        if not hasattr(self, "X_train_"):  # Unfitted;predict based on GP prior
            if self.kernel is None:
                kernel = (C(1.0, constant_value_bounds="fixed") *
                          RBF(1.0, length_scale_bounds="fixed"))
            else:
                kernel = self.kernel
            y_mean = np.zeros(X.shape[0])
            if return_cov:
                y_cov = kernel(X)
                return y_mean, y_cov
            elif return_std:
                y_var = kernel.diag(X)
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean
        else:  # Predict based on GP posterior
            K_trans = self.kernel_(X, self.X_train_)
            y_mean = K_trans.dot(self.alpha_)  # Line 4 (y_mean = f_star)
            y_mean = self._y_train_mean + y_mean  # undo normal.
            if return_cov:
                v = cho_solve((self.L_, True), K_trans.T)  # Line 5
                y_cov = self.kernel_(X) - K_trans.dot(v)  # Line 6
                return y_mean, y_cov
            elif return_std:
                # cache result of K_inv computation
                if self._K_inv is None:
                    # compute inverse K_inv of K based on its Cholesky
                    # decomposition L and its inverse L_inv
                    K = self.kernel_(self.X_train_)
                    K += np.linalg.norm(K)*1e-3*np.eye(K.shape[0])
                    L_inv = solve_triangular(self.L_.T,
                                             np.eye(self.L_.shape[0]))
                    self._K_inv = L_inv.dot(L_inv.T)
                    # self._K_inv = np.linalg.inv(K)

                # Compute variance of predictive distribution
                # y_var = self.kernel_.diag(X)
                # y_var -= np.einsum("ij,ij->i",
                #                    np.dot(K_trans, self._K_inv), K_trans)

                # enhancement 1
                Sarr = []
                index = 0
                for Xi in X:
                    d = distance.cdist([Xi], self.X_train_, 'euclidean')
                    ind = d.argmin()
                    Xm = self.X_train_[ind]
                    Km = self.kernel_([Xm], self.X_train_)
                    # print('111', np.diag(np.dot(np.dot(Km, self._K_inv), Km.T))-1)
                    # np.diag(np.dot(np.dot(Km, self._K_inv), Km.T)) +
                    # print(K_trans[index])
                    # print(Km)
                    S = 1 +\
                        np.diag(np.dot(np.dot(K_trans[index] - Km, self._K_inv), Km.T)) +\
                        np.diag(np.dot(np.dot(Km, self._K_inv), (K_trans[index] - Km).T)) + \
                        np.diag(np.dot(np.dot(K_trans[index] - Km, self._K_inv), (K_trans[index] - Km).T))
                    index += 1

                    Sarr.append(S[0])

                # l = np.linalg.eig(K)
                # print('Cond: ', np.linalg.cond(K), 'Min: ', np.min(np.abs(l)), 'Max: ', np.max(np.abs(l)))
                # print('222', self.kernel_.diag(X))
                # print('333', np.array(Sarr))
                y_var = self.kernel_.diag(X) - np.array(Sarr)
                # y_var = 2 - np.array(Sarr)


                # Check if any of the variances is negative because of
                # numerical issues. If yes: set the variance to 0.
                # y_var_negative = y_var < 0
                # if np.any(y_var_negative):
                #     warnings.warn("Predicted variances smaller than 0. "
                #                   "Setting those variances to 0.")
                #     y_var[y_var_negative] = 0.0
                return y_mean, y_var
            else:
                return y_mean

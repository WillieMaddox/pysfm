import numpy as np
import scipy.integrate
import itertools

from numpy_test import NumpyTestCase
import finite_differences
from algebra import dots

LOG_MIN_REPRESENTABLE_FLOAT = -300 * np.log(10)  # TODO: do this exactly


def is_positive_definite(A):
    evals = np.linalg.eigvals(A)
    return np.all(evals > 0)


################################################################################
def integrate2d(f, lower, upper):
    assert np.shape(lower) == (2,)
    assert np.shape(upper) == (2,)
    y, err = scipy.integrate.dblquad(lambda y, x: f(np.array([x, y])),
                                     lower[0],
                                     upper[0],
                                     lambda x: lower[1],
                                     lambda x: upper[1])
    return y


################################################################################
# Compute the logarithm of the determinant of a symmetric positive
# definite matrix in a numerically stable way
def log_det_covariance(cov):
    return 2. * np.sum(np.log(np.diag(np.linalg.cholesky(cov))))


################################################################################
# Compute the logarithm of the determinant of the inverse of a
# information matrix. (Actually is applicable to any symmetric positive definite matrix)
def log_det_inv_infomat(L):
    return 2. * np.sum(np.log(np.diag(np.linalg.inv(np.linalg.cholesky(L)))))


################################################################################
# Evaluate the normal distribution at x
def evaluate_normal(x, mean, cov):
    x = np.asarray(x)
    mean = np.asarray(mean)
    cov = np.asarray(cov)

    assert x.ndim == 1
    assert mean.ndim == 1
    assert cov.ndim == 2
    assert x.shape == mean.shape
    assert cov.shape == (x.size, x.size)

    k = np.size(x)
    expterm = -.5 * dots(x - mean, np.linalg.inv(cov), x - mean)
    if expterm < LOG_MIN_REPRESENTABLE_FLOAT:
        return 0.  # Cannot do better than this unfortunately

    # denom = (2. * pi) ** (k/2.) * sqrt(det(cov))
    # return np.exp(expterm) / denom

    log_denom = .5 * k * np.log(2. * np.pi) + .5 * log_det_covariance(cov)
    return np.exp(expterm - log_denom)


################################################################################
# Evaluate the inverse normal distribution at x
def evaluate_invnormal(x, v, L):
    x = np.asarray(x)
    v = np.asarray(v)
    L = np.asarray(L)

    assert x.ndim == 1
    assert v.ndim == 1
    assert L.ndim == 2
    assert x.shape == v.shape
    assert L.shape == (x.size, x.size)

    k = np.size(x)
    expterm_var = -.5 * dots(x, L, x) + np.dot(x, v)  # This term depends on x
    expterm_const = -.5 * dots(v, np.linalg.inv(L), v)  # This term does not depend on x
    expterm = expterm_var + expterm_const
    if expterm < LOG_MIN_REPRESENTABLE_FLOAT:
        return 0.  # Cannot do better than this unfortunately

    # denom = (2. * pi) ** (k/2.) * sqrt(det(inv(L)))
    # return np.exp(expterm) / denom

    log_denom = .5 * k * np.log(2. * np.pi) + .5 * log_det_inv_infomat(L)
    return np.exp(expterm - log_denom)


################################################################################
# Sample from the normal distrubtion
def sample_normal(mean, cov):
    return np.random.multivariate_normal(mean, cov)


################################################################################
# Sample from the inverse normal distrubtion
def sample_invnormal(info_vec, info_mat):
    cov = np.linalg.inv(info_mat)
    return np.random.multivariate_normal(np.dot(cov, info_vec), cov)


################################################################################
# Marginalize parameters out of the normal distribution
def marginalize_normal(mean, cov, params_to_keep):
    mean = np.asarray(mean)
    cov = np.asarray(cov)
    assert mean.ndim == 1
    assert cov.ndim == 2
    assert cov.shape == (mean.size, mean.size)

    # check the params - tuples are interpreted differently to lists
    params_to_keep = list(params_to_keep)

    # marginalize
    marg_mean = mean[params_to_keep]

    marg_cov = cov[params_to_keep].T[params_to_keep].T
    return marg_mean, marg_cov


################################################################################
# Marginalize parameters out of the inverse normal distribution
def marginalize_invnormal(v, L, params_to_keep):
    v = np.asarray(v)
    L = np.asarray(L)
    assert v.ndim == 1
    assert L.ndim == 2
    assert L.shape == (v.size, v.size)

    # make a mask
    paramset = set(params_to_keep)
    keep_mask = np.array([i in paramset for i in range(len(v))])

    # pull out the pieces
    Laa = L[keep_mask].T[keep_mask].T
    Lab = L[keep_mask].T[~keep_mask].T
    Lba = L[~keep_mask].T[keep_mask].T
    Lbb = L[~keep_mask].T[~keep_mask].T
    va = v[keep_mask]
    vb = v[~keep_mask]

    # this is simply the schur compliment
    Lbb_inv = np.linalg.inv(Lbb)
    marg_L = Laa - dots(Lab, Lbb_inv, Lba)
    marg_v = va - dots(Lab, Lbb_inv, vb)

    return marg_v, marg_L


################################################################################
# Convert from normal to inverse normal parametrisation
def normal_to_invnormal(mean, cov):
    return np.dot(np.linalg.inv(cov), mean), np.linalg.inv(cov)


################################################################################
# Convert from inverse normal to normal parametrisation
def invnormal_to_normal(v, L):
    return np.dot(np.linalg.inv(L), v), np.linalg.inv(L)


################################################################################
# Compute product of normal distributions in ordinary parametrisation
def product_of_normals(mean1, cov1, mean2, cov2):
    cov = np.linalg.inv(np.linalg.inv(cov1) + np.linalg.inv(cov2))
    mean = np.dot(cov, (np.linalg.solve(cov1, mean1) + np.linalg.solve(cov2, mean2)))
    return mean, cov


################################################################################
# Compute product of normal distributions in inverse parametrisation
def product_of_invnormals(v1, L1, v2, L2):
    return v1 + v2, L1 + L2


################################################################################
# Create a random (but valid) mean and covariance matrix for a k-dimensional Gaussian
def make_gaussian(k):
    mean = np.random.randn(k)
    A = np.random.rand(k, k)
    cov = np.dot(A.T, A)
    return mean, cov


################################################################################
# Compute the parameters for a Gaussian expressing the likelihood 
# P(z | x), where the output Gaussian is over x.
#
# The likelihood is p(z | x) = Normal(z ; mean, cov)
#    where mean = prediction
#           cov = inv(sensor_info)
#
# The parameters are:
#  - x0 is an arbitrary state about which the prediction function has
#    been linearized
#  - PREDICTION is the predicted measurement for the state x0 --
#    i.e. the mean of P(z|x0).
#  - JPREDICTION is the jacobian of the prediction function at x0 --
#    i.e. df/dx evaluated at x0
#  - MEASUREMENT is the actual observation
#  - SENSOR_INFO is the inverse of the covariance of the sensor model
#
# The function returns (v,L), which are the parameters for
# the distribution p(z | x) = Normal^-1(x ; v, L)
def compute_likelihood(x0,
                       prediction,
                       Jprediction,
                       measurement,
                       sensor_info):
    residual = measurement - prediction
    v = dots(Jprediction.T, sensor_info, residual + np.dot(Jprediction, x0))
    L = dots(Jprediction.T, sensor_info, Jprediction)
    return v, L


################################################################################
# Update a belief state given a new observation. The prior belief state is
#   P(x) = N^-1( x ; info_vec, invo_mat )
#
# The measurement is related to the state by a prediction
# function. The most likely measurement (i.e. "prediction") in any
# state x is given by: 
#   z = f(x)
# for some arbiatrary function f. The posterior is:
#   P(x | z) = N^-1( x ; posterior_info_vec , posterior_info_mat )
#
# where posterior_info_vec and posterior_info_mat are the values that
# this function returns.
#
# To do the update, the prediction function must be linearized at some
# point x0. This can theoretically be *any* state, so long as:
#  - the parameter "prediction" is the prediction *at this state*
#  - the parameter "Jprediction" is the jacobian of the prediction func *at this state*
#
# However, the choice of x0 is probably to the quality of the system
# overall, because it can lead to a better/worse approximation to the
# true posterior (we are linearizing, after all). It seems to me that
# it would be good to set this state to some state x0 such that:
#  - the likelihood for the actual observation at x0 is large, i.e. p(z | x0) >> 0
#  - x0 is close to the mean of the prior distribution, so that the
#    linearization is valid around the prior.
#  - perhaps the ideal would be to linearize about the mean of the
#    posterior distribution (yes!)
#  - this could also be chosen as the current best estimate for the
#    state (i.e. taking into account all the in-window data): this
#    would be especially neat since we already have all those values
#    handy following gradient descent!
#
# The parameters are:
#  - prior_v       : the information vector for the prior distribution
#  - prior_L       : the information matrix for the prior distribution
#  - x0            : the state about which to linearize the prediction function (see notes above)
#  - prediction    : the predicted measurement at x0
#  - Jprediction   : the jacobian of the prediction function at x0
#  - measurement   : the actual observation
#  - sensor_info   : the inverse of the sensor covariance
def compute_posterior(prior_v,
                      prior_L,
                      x0,
                      prediction,
                      Jprediction,
                      measurement,
                      sensor_info):
    likelihood_v, likelihood_L = compute_likelihood(x0,
                                                    prediction,
                                                    Jprediction,
                                                    measurement,
                                                    sensor_info)
    posterior_v = prior_v + likelihood_v
    posterior_L = prior_L + likelihood_L
    return posterior_v, posterior_L


################################################################################
class BeliefTest(NumpyTestCase):
    def setUp(self):
        # We really need the state to be overconstrained by the
        # measurements in order to do the numeric tests below. To
        # avoid the potential confusion of having size(measurement) ==
        # size(state) we choose a 2d state and 3d measurement

        self.predict = lambda x: np.array([x[0] ** 2,
                                        x[0] ** 2 + x[1] ** 2,
                                        3. * x[1]
                                        ])
        self.Jpredict = lambda x: np.array([[2. * x[0], 0.],
                                         [2. * x[0], 2. * x[1]],
                                         [0., 3.]])

    def test_jacobians(self):
        x0 = np.array([2., 0.])
        self.assertJacobian(self.predict, self.Jpredict, x0)

    def test_logdet(self):
        mean, cov = make_gaussian(10)
        self.assertAlmostEqual(np.log(np.linalg.det(cov)),
                               log_det_covariance(cov))

        v, L = normal_to_invnormal(mean, cov)
        self.assertAlmostEqual(np.log(np.linalg.det(np.linalg.inv(L))),
                               log_det_inv_infomat(L))

    def test_normal_distributions(self):
        x = np.array([1., 3.5])

        mean = np.array([2., 3.])
        cov = np.array([[4., 2.],
                     [2., 3.]])

        v = np.dot(np.linalg.inv(cov), mean)
        L = np.linalg.inv(cov)

        self.assertAlmostEqual(evaluate_normal(x, mean, cov),
                               evaluate_invnormal(x, v, L))

        f1 = lambda x: evaluate_normal(x, mean, cov)
        self.assertAlmostEqual(integrate2d(f1, mean - [10, 10], mean + [10, 10]), 1., places=5)

        f2 = lambda x: evaluate_invnormal(x, v, L)
        self.assertAlmostEqual(integrate2d(f2, mean - [10, 10], mean + [10, 10]), 1., places=5)

        self.assertFunctionsEqual(f1, f2, near=x, radius=5, nsamples=100)

    def test_product(self):
        mean1, cov1 = make_gaussian(5)
        mean2, cov2 = make_gaussian(5)

        mean, cov = product_of_normals(mean1, cov1, mean2, cov2)

        v1, L1 = normal_to_invnormal(mean1, cov1)
        v2, L2 = normal_to_invnormal(mean2, cov2)

        v, L = product_of_invnormals(v1, L1, v2, L2)
        mean_, cov_ = invnormal_to_normal(v, L)

        self.assertArrayEqual(mean, mean_)
        self.assertArrayEqual(cov, cov_)

    def test_marginalize(self):
        # Setup a Gaussian
        np.random.seed(245970)
        mean, cov = make_gaussian(5)  # sample a random mean and covariance matrix

        # Pick parameters to keep
        params_to_keep = (0, 2, 4)

        # Marginalize in ordinary parametrisation
        marg_mean, marg_cov = marginalize_normal(mean, cov, params_to_keep)
        self.assertShape(marg_mean, (3,))
        self.assertShape(marg_cov, (3, 3))

        # Marginalize in inverse parametrisation
        v, L = normal_to_invnormal(mean, cov)
        marg_v, marg_L = marginalize_invnormal(v, L, params_to_keep)
        marg_mean2, marg_cov2 = invnormal_to_normal(marg_v, marg_L)
        self.assertShape(marg_mean2, (3,))
        self.assertShape(marg_cov2, (3, 3))

        # Check that they wind up the same
        self.assertArrayEqual(marg_mean, marg_mean2)
        self.assertArrayEqual(marg_cov, marg_cov2)

    def test_posterior(self):
        A = np.arange(6).reshape((2, 3))
        b = np.array([-2., -1.])

        prior_cov = np.diag([5., 2., 1.5])
        prior_mean = np.array([2., 3., 2.1])
        prior_v, prior_L = normal_to_invnormal(prior_mean, prior_cov)

        predict = lambda x: dots(A, x) + b

        x0 = np.array([-5., 0., 3.])

        z = predict(x0) + np.array([.1, -.21])
        z_cov = np.diag([2., 4.]) + 1.
        z_info = np.linalg.inv(z_cov)

        posterior_v, posterior_L = compute_posterior(prior_v, prior_L, x0, predict(x0), A, z, z_info)

        P_prior = lambda x: evaluate_invnormal(x, prior_v, prior_L)
        P_lik = lambda x: evaluate_normal(z, predict(x), z_cov)
        P_post = lambda x: evaluate_invnormal(x, posterior_v, posterior_L)
        P_joint = lambda x: P_prior(x) * P_lik(x)

        nx = 10
        xs0 = np.linspace(x0[0] - 2., x0[0] + 2., nx)
        xs = np.vstack((xs0, np.ones(nx) * x0[1], np.ones(nx) * x0[2])).T

        posts = np.array([P_post(x) for x in xs])
        joints = np.array([P_joint(x) for x in xs])

        # The posterior should be proportional (not equal) to the joint
        self.assertArrayProportional(posts, joints)


if __name__ == '__main__':
    import unittest

    # seterr(all='raise')
    unittest.main()

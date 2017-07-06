import numpy as np

CAUCHY_SIGMA = .1  # determines the steepness of the Cauchy robustifier near zero
CAUCHY_SIGMA_SQR = CAUCHY_SIGMA * CAUCHY_SIGMA


################################################################################
# Multiply a list of matrices
def dots(*m):
    return reduce(np.dot, m)


################################################################################
# Divide the first n-1 elements in x by the last. If x is an array
# then do this row-wise.
def pr(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return x[:-1] / x[-1]
    elif x.ndim == 2:
        return x[:, :-1] / x[:, [-1]]
    else:
        raise Exception, 'Cannot pr() an array with %d dimensions' % x.ndim


################################################################################
# Append the number 1 to x. If x is an array then do this row-wise.
def unpr(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return np.hstack((x, 1.))
    elif x.ndim == 2:
        return np.hstack((x, np.ones((len(x), 1))))
    else:
        raise Exception, 'Cannot unpr() an array with %d dimensions' % x.ndim


################################################################################
#
# SO(3) stuff
#

# Get the skew-symmetric matrix for m
def skew(m):
    return np.array([[0, -m[2], m[1]],
                     [m[2], 0, -m[0]],
                     [-m[1], m[0], 0.]])


# Compute skew(m) * skew(m)
def skewsqr(m):
    a, b, c = m
    aa, bb, cc = np.square(m)  # element-wise square
    return np.array([[-bb - cc, a * b, a * c],
                     [a * b, -aa - cc, b * c],
                     [a * c, b * c, -aa - bb]])


# Compute matrix exponential from so(3) to SO(3) (i.e. evaluate Rodrigues)
def SO3_exp(m):
    t = np.linalg.norm(m)
    if t < 1e-8:
        return np.eye(3)  # exp(0) = I

    a = np.sin(t) / t
    b = (1. - np.cos(t)) / (t * t)
    return np.eye(3) + a * skew(m) + b * skewsqr(m)


################################################################################
#
# Robustifiers
#

# Given a residual vector, compute the square root of the
# Cauchy-robustified error response.
def cauchy_sqrtcost_from_residual(r):
    return np.sqrt(np.log(1. + r * r / CAUCHY_SIGMA_SQR))


# Given a residual vector and its gradient, compute the derivative of
# the square root of the Cauchy-robustified error response.
def Jcauchy_sqrtcost_from_residual(r, dr):
    assert np.isscalar(r)
    assert np.ndim(dr) == 1
    sqrtcost = cauchy_sqrtcost_from_residual(r)
    if sqrtcost < 1e-8:
        return 0.  # this is in fact the correct thing to do here
    return dr * r / (sqrtcost * (CAUCHY_SIGMA_SQR + r * r))


# Given a 2x1 reprojection error (in raw pixel coordinates), evaluate
# a robust error function and return a scalar cost.
def cauchy_cost_from_reprojection_error(x):
    x = np.asarray(x)
    return np.log(1. + (x[0] * x[0] + x[1] * x[1]) / CAUCHY_SIGMA_SQR)


# Given a 2x1 reprojection error (in raw pixel coordinates), evaluate
# a robust error function and return:
#   -- the 2x1 residual vector
#   -- the 2x2 Jacobian of that residual
def cauchy_residual_from_reprojection_error(x):
    x = np.asarray(x)
    rr = x[0] * x[0] + x[1] * x[1]
    if rr < 1e-8:  # within this window the residual is well-approximated as linear
        J_near_zero = np.eye(2) / CAUCHY_SIGMA
        r_near_zero = x / CAUCHY_SIGMA
        return r_near_zero, J_near_zero

    r = np.sqrt(rr)
    e = np.sqrt(np.log(1. + rr / CAUCHY_SIGMA_SQR))
    xx = np.outer(x, x)
    residual = x * e / r
    Jresidual = xx / (r * e * (rr + CAUCHY_SIGMA_SQR)) + (r * np.eye(2) - xx / r) * e / rr
    return residual, Jresidual


################################################################################
# Triangulate a 3D point from a set of observations by cameras with
# fixed parameters. Uses least squares on the algebraic error.
def triangulate_algebraic_lsq(K, Rs, ts, msms):
    A = np.empty((len(Rs) * 2, 3))
    b = np.empty(len(Rs) * 2)
    msms = np.asarray(list(msms))

    for i in range(len(Rs)):
        b[i * 2] = np.dot(msms[i, 0] * K[2] - K[0], ts[i])
        b[i * 2 + 1] = np.dot(msms[i, 1] * K[2] - K[1], ts[i])
        A[i * 2] = np.dot(K[0] - msms[i, 0] * K[2], Rs[i])
        A[i * 2 + 1] = np.dot(K[1] - msms[i, 1] * K[2], Rs[i])

    x, residuals, rank, sv = np.linalg.lstsq(A, b)
    return x


################################################################################
#
# Geometry
#

# Get a relative pose (R01,t01) that goes from the pose (R0,t0) to (R1,t1)
def relative_pose(R0, t0, R1, t1):
    R_delta = np.dot(R1, R0.T)
    t_delta = t1 - dots(R1, R0.T, t0)
    return R_delta, t_delta


def rotation_xy(th):
    return np.array([[np.cos(th), -np.sin(th), 0.],
                     [np.sin(th), np.cos(th), 0.],
                     [0., 0., 1.]])


def rotation_xz(th):
    return np.array([[np.cos(th), 0., -np.sin(th)],
                     [0., 1., 0.],
                     [np.sin(th), 0., np.cos(th), ]])


def rotation_yz(th):
    return np.array([[1., 0., 0.],
                     [0., np.cos(th), -np.sin(th)],
                     [0., np.sin(th), np.cos(th)]])

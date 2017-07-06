# from numpy import *
import unittest
import numpy as np
import fundamental as fund
from lie import SO3
from numpy_test import NumpyTestCase
import finite_differences
from algebra import pr, unpr, dots
from geometry import rotation_xy, rotation_xz, rotation_yz, relative_pose

K = fund.K

################################################################################
def setup_test_problem():
    noise = .1

    K = np.eye(3)
    K = np.array([[1.5, .2, -.4],
               [.25, .9, .18],
               [0., .2, 1.]])

    R0 = np.eye(3)
    t0 = np.zeros(3)
    P0 = np.hstack((R0, t0[:, np.newaxis]))

    R1 = dots(rotation_xz(-.2), rotation_xy(.1), rotation_yz(1.5))
    t1 = np.array([-1., .5, -2.])
    P1 = np.hstack((R1, t1[:, np.newaxis]))

    R, t = relative_pose(R0, t0, R1, t1)

    # Add noise
    np.random.seed(123)
    pts = np.random.randn(fund.NUM_CORRESPONDENCES, 3) + np.array([0., 0., -5.])
    xs0 = np.array([pr(np.dot(K, np.dot(R0, x) + t0)) for x in pts])
    xs1 = np.array([pr(np.dot(K, np.dot(R1, x) + t1)) for x in pts])
    xs0 += np.random.randn(*xs0.shape) * noise
    xs1 += np.random.randn(*xs1.shape) * noise

    # Perturb R and t so that we do not evaluate jacobians at the minima
    R = np.dot(R, SO3.exp(np.random.randn(3) * .05))
    t += np.random.randn(3) * .05

    return R, t, xs0, xs1


################################################################################
# Test jacobians of the sampson error
class FundamentalMatrixTest(NumpyTestCase):
    def setUp(self):
        self.R, self.t, self.xs0, self.xs1 = setup_test_problem()
        self.x0 = unpr(self.xs0[0])
        self.x1 = unpr(self.xs1[0])
        self.x = np.array([.6, -2., 1.])
        pass

    def test_cauchy(self):
        r = np.array([1., 3., -2.])
        J = np.eye(3)
        self.assertJacobian(fund.cauchy_sqrtcost_from_residual_multidimensional,
                            fund.Jcauchy_sqrtcost_from_residual_multidimensional(r, J),
                            r)

        x0 = np.arange(4) - 2.5
        f = lambda x: x ** 2 + x
        Jf = lambda x: np.diag(2. * x + 1)
        c = lambda x: fund.cauchy_sqrtcost_from_residual_multidimensional(f(x))
        Jc = lambda x: fund.Jcauchy_sqrtcost_from_residual_multidimensional(f(x), Jf(x))
        self.assertJacobian(f, Jf, x0)
        self.assertJacobian(c, Jc, x0)

    def test_cauchy_hessian(self):
        J = np.arange(6).reshape((3, 2)).astype(float)
        y = np.arange(3)[::-1].astype(float)
        f = lambda x: np.dot(J, x) - y
        E = lambda x: fund.cauchy_cost(f(x))

        x0 = np.arange(2) + 2.5
        r0 = f(x0)

        self.assertJacobian(f, J, x0)
        self.assertJacobian(f, J, x0)

        H_analytic = fund.cauchy_hessian_firstorder(r0, J)
        print H_analytic

        H_numeric = finite_differences.numeric_hessian(E, x0)[0]
        print H_numeric

        self.assertArrayEqual(H_analytic, H_numeric)

    def test_JF1x(self):
        R, t, x = self.R, self.t, self.x
        fR = lambda m: fund.F1x(K, np.dot(R, SO3.exp(m)), t, x)
        ft = lambda tt: fund.F1x(K, R, tt, x)
        self.assertJacobian(fR, fund.JF1x_R(K, R, t, x)[np.newaxis, :], np.zeros(3))
        self.assertJacobian(ft, fund.JF1x_t(K, R, t, x)[np.newaxis, :], t)

    def test_JFT1x(self):
        R, t, x = self.R, self.t, self.x
        fR = lambda m: fund.FT1x(K, np.dot(R, SO3.exp(m)), t, x)
        ft = lambda tt: fund.FT1x(K, R, tt, x)
        self.assertJacobian(fR, fund.JFT1x_R(K, R, t, x)[np.newaxis, :], np.zeros(3))
        self.assertJacobian(ft, fund.JFT1x_t(K, R, t, x)[np.newaxis, :], t)

    def test_JF2x(self):
        R, t, x = self.R, self.t, self.x
        fR = lambda m: fund.F2x(K, np.dot(R, SO3.exp(m)), t, x)
        ft = lambda tt: fund.F2x(K, R, tt, x)
        self.assertJacobian(fR, fund.JF2x_R(K, R, t, x)[np.newaxis, :], np.zeros(3))
        self.assertJacobian(ft, fund.JF2x_t(K, R, t, x)[np.newaxis, :], t)

    def test_JFT2x(self):
        R, t, x = self.R, self.t, self.x
        fR = lambda m: fund.FT2x(K, np.dot(R, SO3.exp(m)), t, x)
        ft = lambda tt: fund.FT2x(K, R, tt, x)
        self.assertJacobian(fR, fund.JFT2x_R(K, R, t, x)[np.newaxis, :], np.zeros(3))
        self.assertJacobian(ft, fund.JFT2x_t(K, R, t, x)[np.newaxis, :], t)

    def test_JxFx(self):
        R, t, x0, x1 = self.R, self.t, self.x0, self.x1
        fR = lambda m: fund.xFx(K, np.dot(R, SO3.exp(m)), t, x0, x1)
        ft = lambda tt: fund.xFx(K, R, tt, x0, x1)
        self.assertJacobian(fR, fund.JxFx_R(K, R, t, x0, x1)[np.newaxis, :], np.zeros(3))
        self.assertJacobian(ft, fund.JxFx_t(K, R, t, x0, x1)[np.newaxis, :], t)

    def test_Jresidual(self):
        R, t, x0, x1 = self.R, self.t, self.x0, self.x1
        f = lambda v: fund.residual(K, np.dot(R, SO3.exp(v[:3])), t + v[3:], x0, x1)
        self.assertJacobian(f, fund.Jresidual(K, R, t, x0, x1)[np.newaxis, :], np.zeros(6))

    def test_Jresidual_robust(self):
        R, t, x0, x1 = self.R, self.t, self.x0, self.x1
        f = lambda v: fund.residual_robust(K, np.dot(R, SO3.exp(v[:3])), t + v[3:], x0, x1)
        self.assertJacobian(f, fund.Jresidual_robust(K, R, t, x0, x1)[np.newaxis, :], np.zeros(6))

    def _test_normal_equations(self):
        R, t, xs0, xs1 = self.R, self.t, self.xs0, self.xs1
        xs0 = unpr(xs0)
        xs1 = unpr(xs1)
        c = lambda v: fund.cost(K, np.dot(R, SO3.exp(v[:3])), t + v[3:], xs0, xs1)
        JTJ, JTr = fund.compute_normal_equations(K, R, t, xs0, xs1, fund.residual, fund.Jresidual)
        H = finite_differences.numeric_hessian(c, np.zeros(6))
        print 'numeric hessian (least squares):'
        print H
        print '2*JTJ (least squares):'
        print 2. * JTJ
        self.assertJacobian(c, 2. * JTr, np.zeros(6))

    def _test_normal_equations_robust(self):
        R, t, xs0, xs1 = self.R, self.t, self.xs0, self.xs1
        xs0 = unpr(xs0)
        xs1 = unpr(xs1)
        c = lambda v: fund.cost_robust(K, np.dot(R, SO3.exp(v[:3])), t + v[3:], xs0, xs1)
        JTJ, JTr = fund.compute_normal_equations(K, R, t, xs0, xs1, fund.residual_robust, fund.Jresidual_robust)
        H = finite_differences.numeric_hessian(c, np.zeros(6))
        print 'numeric hessian (robust):'
        print H
        print '2*JTJ (robust):'
        print 2. * JTJ
        self.assertJacobian(c, 2. * JTr, np.zeros(6))


if __name__ == '__main__':
    np.seterr(all='raise')
    unittest.main()

import unittest
import numpy as np
import homography_estimator as he
from numpy_test import NumpyTestCase
from algebra import prdot
import lie


################################################################################
def setup_test_problem():
    NUM_CORRESPONDENCES = 12
    MEASUREMENT_NOISE = .2

    # Setup homography
    H = np.array([[1., 0., 2.],
                  [0., -.5, -1.5],
                  [.2, .1, 1.2]])

    # Sample points
    np.random.seed(123)
    xs0 = np.random.randn(NUM_CORRESPONDENCES, 2)
    xs1 = prdot(H, xs0)
    xs0 += np.random.randn(*xs0.shape) * MEASUREMENT_NOISE
    xs1 += np.random.randn(*xs1.shape) * MEASUREMENT_NOISE

    # Perturb H so that we do not evaluate jacobians at the minima
    H += np.random.randn(*H.shape) * .05

    return H, xs0, xs1


################################################################################
# Test jacobians of the sampson error
class HomographyEstimatorTest(NumpyTestCase):
    def setUp(self):
        self.H, self.xs0, self.xs1 = setup_test_problem()

    def test_algebraic_error_jacobian(self):
        x0, x1 = self.xs0[0], self.xs1[1]
        self.assertJacobian(lambda X: he.algebraic_error(self.H, X[0:2], X[2:4]),
                            he.algebraic_error_jacobian(self.H, x0, x1),
                            np.hstack((x0, x1)))

    def test_transfer_error_jacobian(self):
        x0, x1 = self.xs0[0], self.xs1[1]
        self.assertJacobian(lambda h: he.transfer_error(h.reshape((3, 3)), x0, x1),
                            he.transfer_error_jacobian(self.H, x0, x1),
                            self.H.flatten())

    def test_transfer_error_sl3_jacobian(self):
        x0, x1 = self.xs0[0], self.xs1[1]
        self.assertJacobian(lambda w: he.transfer_error(np.dot(self.H, lie.SL3_exp(w)), x0, x1),
                            he.transfer_error_sl3_jacobian(self.H, x0, x1),
                            np.zeros(8))

    # def test_symtransfer_error_jacobian(self):
    #    x0,x1 = self.xs0[0], self.xs1[1]
    #    self.assertJacobian(lambda h: he.symtransfer_error(h.reshape((3,3)), x0, x1),
    #                        he.symtransfer_error_jacobian(self.H, x0, x1),
    #                        self.H.flatten())

    def test_sampson_error_jacobian(self):
        x0, x1 = self.xs0[0], self.xs1[1]
        self.assertJacobian(lambda h: he.sampson_error(h.reshape((3, 3)), x0, x1),
                            he.sampson_error_jacobian(self.H, x0, x1),
                            self.H.flatten())

    def test_sampson_error_sl3_jacobian(self):
        x0, x1 = self.xs0[0], self.xs1[1]
        self.assertJacobian(lambda w: he.sampson_error(np.dot(self.H, lie.SL3_exp(w)), x0, x1),
                            he.sampson_error_sl3_jacobian(self.H, x0, x1),
                            np.zeros(8))


if __name__ == '__main__':
    np.seterr(all='raise')
    unittest.main()

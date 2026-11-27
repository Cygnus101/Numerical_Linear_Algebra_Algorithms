import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, diags

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import backward_substitution, lu_solve
from direct_methods.gaussian_elim import gaussian, gaussian_with_partial_pivoting
from direct_methods.lu_row import lu_row
from direct_methods.lu_column import lu_column
from direct_methods.lu_partial_pivoting import lu_partial_pivoting
from direct_methods.cholesky import cholesky
from Eigenvalue_Methods.power_method import power_method
from Eigenvalue_Methods.inverse_power_method import inverse_power_method
from Iterative_Methods.conjugate_gradient import conjugate_gradient
from Iterative_Methods.gradient_descent import gradient_descent
from Iterative_Methods.newton import newton
from Iterative_Methods.jacobi import jacobi
from Iterative_Methods.gauss_seidel import gauss_seidel
from Iterative_Methods.sor import sor
from Orthogonalization.householder_qr import householder_qr, apply_q
from Orthogonalization.gram_schmidt import gram_schmidt


class Algorithms(unittest.TestCase):
    def setUp(self):
        self.A = np.array([[4., 1., 0.], [1., 3., 1.], [0., 1., 2.]])
        self.x = np.array([1., -2., 3.])
        self.b = self.A @ self.x

    def test_lu_orders_and_permutation(self):
        rng = np.random.default_rng(7)
        A = rng.normal(size=(6, 6)) + 8*np.eye(6)
        for fn in [lu_row, lu_column]:
            with self.subTest(fn=fn.__name__):
                before = A.copy()
                L, U = fn(A)
                np.testing.assert_allclose(L @ U, A, atol=1e-12)
                np.testing.assert_allclose(lu_solve(L, U, A @ np.ones(6)), 1, atol=1e-12)
                np.testing.assert_array_equal(A, before)
        A = np.array([[0., 2., 1.], [1., 0., 3.], [4., 5., 6.]])
        L, U, p = lu_partial_pivoting(A)
        np.testing.assert_allclose(L @ U, A[p], atol=1e-12)
        np.testing.assert_allclose(lu_solve(L, U, A @ self.x, p), self.x, atol=1e-12)
        self.assertEqual(sorted(p), [0, 1, 2])
        for fn in [lu_row, lu_column, lu_partial_pivoting]:
            with self.assertRaises(ValueError):
                fn(np.array([[1., 2.], [2., 4.]]))

    def test_gaussian_variants(self):
        for fn in [gaussian, gaussian_with_partial_pivoting]:
            C, c = fn(self.A.copy(), self.b.copy())
            np.testing.assert_allclose(backward_substitution(C, c), self.x, atol=1e-12)
            for A in [np.array([[0., 1.], [0., 2.]]), np.array([[1., 2.], [2., 4.]])]:
                with self.assertRaises(ValueError):
                    fn(A.copy(), np.ones(2))
        A = np.array([[0., 2., 1.], [1., 0., 3.], [4., 5., 6.]])
        with self.assertRaises(ValueError):
            gaussian(A.copy(), np.ones(3))
        C, c = gaussian_with_partial_pivoting(A.copy(), A @ self.x)
        np.testing.assert_allclose(backward_substitution(C, c), self.x, atol=1e-12)

    def test_cholesky(self):
        L = cholesky(self.A)
        np.testing.assert_allclose(L @ L.T, self.A, atol=1e-12)
        np.testing.assert_allclose(lu_solve(L, L.T, self.b), self.x, atol=1e-12)
        for A in [np.array([[1., 2.], [2., 1.]]), np.array([[1., 2.], [0., 1.]])]:
            with self.assertRaises(ValueError):
                cholesky(A)

    def test_iterative_dense_and_sparse(self):
        functions = [conjugate_gradient, gradient_descent, newton, jacobi, gauss_seidel, sor]
        for fn in functions:
            for A in [self.A, csr_matrix(self.A)]:
                with self.subTest(fn=fn.__name__, sparse=isinstance(A, csr_matrix)):
                    x0 = np.array([3., 2., 1.])
                    saved = x0.copy()
                    x, info = fn(A, self.b, x0=x0, tol=1e-10)
                    self.assertTrue(info['converged'], info)
                    np.testing.assert_allclose(x, self.x, atol=1e-8)
                    np.testing.assert_array_equal(x0, saved)
                    x, info = fn(A, np.zeros(3))
                    self.assertTrue(info['converged'])
                    self.assertEqual(info['iterations'], 0)
                    _, info = fn(A, self.b, maxiter=0)
                    self.assertFalse(info['converged'])
        A = diags([-np.ones(199), 4*np.ones(200), -np.ones(199)], [-1, 0, 1], format='csr')
        x, info = conjugate_gradient(A, A @ np.ones(200), tol=1e-10)
        self.assertTrue(info['converged'])
        np.testing.assert_allclose(x, 1, atol=1e-8)

    def test_iterative_failures(self):
        A = np.array([[1., 2.], [2., 1.]])
        _, info = jacobi(A, np.ones(2), maxiter=50)
        self.assertFalse(info['converged'])
        for fn in [jacobi, gauss_seidel, sor]:
            with self.assertRaises(ValueError):
                fn(np.array([[0., 1.], [1., 2.]]), np.ones(2))
        for fn in [conjugate_gradient, gradient_descent]:
            with self.assertRaises(ValueError):
                fn(-np.eye(2), np.ones(2))
        with self.assertRaises(ValueError):
            sor(self.A, self.b, omega=2)

    def test_eigenmethods(self):
        for A in [np.diag([-7., 2., 4.]), diags([-7., 2., 4.], format='csr')]:
            lam, x, info = power_method(A, tol=1e-10)
            self.assertTrue(info['converged'])
            self.assertAlmostEqual(lam, -7., places=8)
            lam, x, info = inverse_power_method(A, tol=1e-10)
            self.assertTrue(info['converged'])
            self.assertAlmostEqual(lam, 2., places=8)
            lam, x, info = inverse_power_method(A, shift=3.5, tol=1e-10)
            self.assertTrue(info['converged'])
            self.assertAlmostEqual(lam, 4., places=8)
            _, _, info = power_method(A, maxiter=0)
            self.assertFalse(info['converged'])

    def test_orthogonalization(self):
        rng = np.random.default_rng(5)
        for shape in [(5, 3), (3, 3), (3, 5)]:
            A = rng.normal(size=shape)
            before = A.copy()
            V, R = householder_qr(A)
            Q = apply_q(V, np.eye(shape[0]))
            np.testing.assert_allclose(Q @ R, A, atol=1e-12)
            np.testing.assert_allclose(Q.T @ Q, np.eye(shape[0]), atol=1e-12)
            np.testing.assert_allclose(apply_q(V, A, transpose=True), R, atol=1e-12)
            np.testing.assert_array_equal(A, before)
            if shape[0] >= shape[1]:
                Q, R = gram_schmidt(A)
                np.testing.assert_allclose(Q @ R, A, atol=1e-12)
                np.testing.assert_allclose(Q.T @ Q, np.eye(shape[1]), atol=1e-12)
        A = np.array([[0., 1.], [0., 2.], [0., 3.]])
        V, R = householder_qr(A)
        np.testing.assert_allclose(apply_q(V, R), A, atol=1e-12)
        with self.assertRaises(ValueError):
            gram_schmidt(A)


if __name__ == '__main__':
    unittest.main()

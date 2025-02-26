import unittest
import numpy as np
from goph420_lab01.integration import integrate_newton

class TestNewtonCotes(unittest.TestCase):

    def test_trapezoidal_linear(self):
        x = np.linspace(0, 10, 11)
        f = 2 * x + 3
        result = integrate_newton(x, f, "trap")
        self.assertAlmostEqual(result, 130.0, places=1)

    def test_simpson_linear(self):
        x = np.linspace(0, 10, 11)
        f = 2 * x + 3
        result = integrate_newton(x, f, "simp")
        self.assertAlmostEqual(result, 130.0, places=1)

if __name__ == "__main__":
    unittest.main()

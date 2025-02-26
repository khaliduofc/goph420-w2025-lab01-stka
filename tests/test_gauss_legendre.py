import unittest
import numpy as np
from goph420_lab01.integration import integrate_gauss

class TestGaussLegendre(unittest.TestCase):

    def test_quadratic(self):
        f = lambda x: x**2
        result = integrate_gauss(f, [0, 1], 3)
        self.assertAlmostEqual(result, 1/3, places=3)

    def test_normal_pdf(self):
        from scipy.stats import norm
        result = integrate_gauss(norm.pdf, [-1, 1], 3)
        self.assertAlmostEqual(result, norm.cdf(1) - norm.cdf(-1), places=3)

if __name__ == "__main__":
    unittest.main()

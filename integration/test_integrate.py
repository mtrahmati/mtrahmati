import math
import unittest

from integrate import integrate, parse_function


class TestIntegrate(unittest.TestCase):
    CASES = [
        (lambda x: x**2, 0, 1, 1 / 3),
        (math.sin, 0, math.pi, 2.0),
        (math.exp, 0, 1, math.e - 1),
        (lambda x: 1 / x, 1, math.e, 1.0),
        (lambda x: math.exp(-x**2), -10, 10, math.sqrt(math.pi)),
    ]

    def test_methods_match_known_values(self):
        tolerances = {"trapezoid": 1e-5, "simpson": 1e-9, "adaptive": 1e-9}
        for method, tol in tolerances.items():
            for f, a, b, expected in self.CASES:
                with self.subTest(method=method, a=a, b=b):
                    self.assertAlmostEqual(integrate(f, a, b, method), expected, delta=tol)

    def test_reversed_limits_flip_sign(self):
        self.assertAlmostEqual(integrate(lambda x: x, 1, 0), -0.5)

    def test_zero_width_interval(self):
        self.assertEqual(integrate(math.sin, 2, 2), 0.0)

    def test_parse_function(self):
        f = parse_function("x**2 + sin(x)")
        self.assertAlmostEqual(f(1.0), 1 + math.sin(1.0))

    def test_parse_function_rejects_unsafe_names(self):
        with self.assertRaises(ValueError):
            parse_function("__import__('os').system('ls')")

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            integrate(math.sin, 0, 1, method="magic")


if __name__ == "__main__":
    unittest.main()

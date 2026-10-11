"""Numerical integration of a function f(x) over [a, b].

Usage as a library:
    from integrate import integrate
    integrate(lambda x: x**2, 0, 1)          # -> 0.3333333333333333

Usage from the command line:
    python integrate.py "sin(x)" 0 pi
    python integrate.py "exp(-x**2)" -5 5 --method simpson -n 1000
"""

import argparse
import math


def trapezoid(f, a, b, n=1000):
    """Composite trapezoidal rule with n subintervals."""
    if n < 1:
        raise ValueError("n must be at least 1")
    h = (b - a) / n
    total = (f(a) + f(b)) / 2
    for i in range(1, n):
        total += f(a + i * h)
    return total * h


def simpson(f, a, b, n=1000):
    """Composite Simpson's rule; n is rounded up to an even number."""
    if n < 2:
        n = 2
    if n % 2:
        n += 1
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        total += (4 if i % 2 else 2) * f(a + i * h)
    return total * h / 3


def adaptive_simpson(f, a, b, tol=1e-10, max_depth=50):
    """Adaptive Simpson's rule: refines only where the function needs it."""

    def simpson_once(fa, fm, fb, a, b):
        return (b - a) / 6 * (fa + 4 * fm + fb)

    def recurse(a, b, fa, fm, fb, whole, tol, depth):
        m = (a + b) / 2
        lm, rm = (a + m) / 2, (m + b) / 2
        flm, frm = f(lm), f(rm)
        left = simpson_once(fa, flm, fm, a, m)
        right = simpson_once(fm, frm, fb, m, b)
        delta = left + right - whole
        if depth <= 0 or abs(delta) <= 15 * tol:
            return left + right + delta / 15  # Richardson extrapolation
        return (recurse(a, m, fa, flm, fm, left, tol / 2, depth - 1)
                + recurse(m, b, fm, frm, fb, right, tol / 2, depth - 1))

    fa, fb, fm = f(a), f(b), f((a + b) / 2)
    whole = simpson_once(fa, fm, fb, a, b)
    return recurse(a, b, fa, fm, fb, whole, tol, max_depth)


METHODS = {
    "trapezoid": trapezoid,
    "simpson": simpson,
    "adaptive": adaptive_simpson,
}


def integrate(f, a, b, method="adaptive", **kwargs):
    """Integrate f from a to b. Swapping the limits flips the sign."""
    if a == b:
        return 0.0
    if a > b:
        return -integrate(f, b, a, method, **kwargs)
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {list(METHODS)}")
    return METHODS[method](f, a, b, **kwargs)


# Names allowed inside expressions typed on the command line.
_SAFE_NAMES = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
_SAFE_NAMES["abs"] = abs


def parse_function(expr):
    """Turn a string such as 'x**2 + sin(x)' into a Python function of x."""
    code = compile(expr, "<expr>", "eval")
    for name in code.co_names:
        if name != "x" and name not in _SAFE_NAMES:
            raise ValueError(f"unknown name {name!r} in expression")
    return lambda x: eval(code, {"__builtins__": {}}, {**_SAFE_NAMES, "x": x})


def parse_limit(text):
    return float(eval(compile(text, "<limit>", "eval"), {"__builtins__": {}}, _SAFE_NAMES))


def main():
    p = argparse.ArgumentParser(description="Numerically integrate f(x) from a to b.")
    p.add_argument("function", help='expression in x, e.g. "x**2" or "sin(x)"')
    p.add_argument("a", help="lower limit (e.g. 0, -pi, pi/2)")
    p.add_argument("b", help="upper limit")
    p.add_argument("--method", choices=METHODS, default="adaptive")
    p.add_argument("-n", type=int, default=1000, help="subintervals (trapezoid/simpson)")
    p.add_argument("--tol", type=float, default=1e-10, help="tolerance (adaptive)")
    args = p.parse_args()

    f = parse_function(args.function)
    a, b = parse_limit(args.a), parse_limit(args.b)
    kwargs = {"tol": args.tol} if args.method == "adaptive" else {"n": args.n}
    result = integrate(f, a, b, args.method, **kwargs)
    print(f"∫ {args.function} dx from {args.a} to {args.b} ≈ {result:.12g}  ({args.method})")


if __name__ == "__main__":
    main()

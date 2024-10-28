import numpy as np
from numba import njit
from numba.core.extending import overload, register_jitable
from numba.np.numpy_support import type_can_asarray


@njit
def secant(func, x0, args=(), tol=1.48e-8, maxiter=50, x1=None, rtol=0.0):
    """
    Numba friendly copy of scipy.optimize.newton that uses Secant method
    https://github.com/scipy/scipy/blob/v1.6.2/scipy/optimize/zeros.py
    """
    assert maxiter
    assert tol > 0

    # Convert to float (don't use float(x0); this works also for complex x0)
    p0 = 1.0 * x0
    funcalls = 0
    # Secant method
    if x1 is not None:
        if x1 == x0:
            raise ValueError("x1 and x0 must be different")
        p1 = x1
    else:
        eps = 1e-4
        p1 = x0 * (1 + eps)
        p1 += eps if p1 >= 0 else -eps
    q0 = func(p0, *args)
    funcalls += 1
    q1 = func(p1, *args)
    funcalls += 1
    if abs(q1) < abs(q0):
        p0, p1, q0, q1 = p1, p0, q1, q0
    for itr in range(maxiter):
        if q1 == q0:
            if p1 != p0:
                raise RuntimeError("Secant method failed to converge")
            p = (p1 + p0) / 2.0
            return p
        else:
            if abs(q1) > abs(q0):
                p = (-q0 / q1 * p1 + p0) / (1 - q0 / q1)
            else:
                p = (-q1 / q0 * p0 + p1) / (1 - q1 / q0)
        if np.isclose(p, p1, rtol=rtol, atol=tol):
            return p
        p0, q0 = p1, q1
        p1 = p
        q1 = func(p1, *args)
        funcalls += 1

    return p


# EC2021: to be removed when numba supports np.isclose (next release)
@register_jitable
def _within_tol(a, b, rtol, atol):
    return np.less_equal(np.abs(a - b), atol + rtol * np.abs(b))


@overload(np.isclose)
def np_isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    # https://github.com/numba/numba/pull/4610/commits/35d05ebd43c924cdd3d7574399cfe9329fbd8db0
    if not (type_can_asarray(a) and type_can_asarray(b)):
        raise RuntimeError("Inputs for `np.isclose` must be array-like.")

    def impl(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        xfin = np.asarray(np.isfinite(a))
        yfin = np.asarray(np.isfinite(b))
        x, y = np.asarray(a), np.asarray(b)
        if np.all(xfin) and np.all(yfin):
            return _within_tol(x, y, rtol, atol)
        else:
            finite = xfin & yfin
            r = np.zeros_like(finite)
            x = x * np.ones_like(r)
            y = y * np.ones_like(r)
            r = _within_tol(x, y, rtol, atol)
            # Negate every element that is not finite
            r &= xfin & yfin
            # Check for equality of infinite values
            r |= (x == np.asarray(np.inf)) & (y == np.asarray(np.inf))
            r |= (x == np.asarray(-np.inf)) & (y == np.asarray(-np.inf))
            if equal_nan:
                xnan = np.asarray(np.isnan(a))
                ynan = np.asarray(np.isnan(b))
                return r | (xnan & ynan)
            else:
                return r

    return impl

import numpy as np

def polynomial(xs, ys, degree, approximating_xs):
    """
    Computes and evaluates a polynomial function of a given degree based on the input data,
    using the method of fewest squares for polynomial regression. It returns the approximated
    y-values for the given x-values to be approximated.

    The function constructs a Vandermonde matrix to compute the coefficients of the polynomial
    that best fits the input data points. It then evaluates this polynomial at specified
    x-values for approximation.

    :param xs: List of x-coordinates corresponding to the data points for polynomial fitting.
    :type xs: list[float]
    :param ys: List of y-coordinates corresponding to the data points for polynomial fitting.
    :type ys: list[float]
    :param degree: Degree of the polynomial to fit.
    :type degree: int
    :param approximating_xs: List of x-values at which the polynomial function is evaluated to produce
                             approximated y-values.
    :type approximating_xs: list[float]
    :return: List of approximated y-values corresponding to the input approximating_xs.
    :rtype: list[float]
    :raises ValueError: If the number of x-values and y-values provided as inputs are not equal.
    """
    if len(xs) != len(ys):
        raise ValueError("A different number of x and y values were given")

    length = degree + 1

    raised_xs = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            for k in range(len(xs)):
                raised_xs[i][j] += xs[k]**(i + j)

    raised_ys = np.zeros(length)
    for i in range(length):
        for j in range(len(xs)):
            raised_ys[i] += ys[j] * xs[j]**i

    polynomials = np.linalg.solve(raised_xs, raised_ys)

    approximating_ys = np.zeros(len(approximating_xs))
    for i in range(len(approximating_xs)):
        for j in range(len(polynomials)):
            approximating_ys[i] +=  polynomials[j] * approximating_xs[i]**j

    return approximating_ys
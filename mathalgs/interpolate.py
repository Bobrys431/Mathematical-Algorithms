import numpy as np

def polynomial(xs, ys, interpolating_xs):
    """
    Computes interpolated y-values for given x-values using Newton's
    divided differences polynomial interpolation.

    This function calculates polynomial coefficients using Newton's
    divided differences method and evaluates the polynomial at the
    specified x-values for interpolation. It assumes that the input
    values `xs` and `ys` represent ordered sequential data and will
    raise an error if their lengths differ.

    :param xs: List of x-coordinates supplied for interpolation calculation
    :param ys: List of y-coordinates corresponding to `xs`
    :param interpolating_xs: List of x-coordinates where interpolated y-values
        will be calculated
    :return: List of interpolated y-values corresponding to `interpolating_xs`
    """
    if len(xs) != len(ys):
        raise ValueError("A different number of x and y values were given")

    length = len(xs)
    differences = [[0 for _ in range(length)] for _ in range(length)]
    differences[0][:length] = ys[:length]

    for i in range(1, length):
        for j in range(length - i):
            differences[i][j] = (differences[i - 1][j + 1] - differences[i - 1][j]) / (xs[j + i] - xs[j])

    interpolating_ys = np.zeros(len(interpolating_xs))

    for i in range(len(interpolating_xs)):
        interpolating_ys[i] = differences[0][0]
        mult = 1.0
        for j in range(1, length):
            mult *= (interpolating_xs[i] - xs[j - 1])
            interpolating_ys[i] += differences[j][0] * mult

    return interpolating_ys

def spline(xs, ys):
    pass
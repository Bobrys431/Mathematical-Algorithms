from mathalgs import statistics as st
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

def spline(xs, ys, interpolating_xs):
    """
    Calculates spline interpolation for a given set of points and interpolates values
    for specified x-coordinates. This function implements a cubic B-spline interpolation
    method. It ensures that x values are evenly spaced and that the input x and y data
    have the same length.

    :param xs: List or array of x-coordinates (evenly spaced).
    :param ys: List or array of y-coordinates corresponding to `xs`.
    :param interpolating_xs: List or array of x-coordinates at which to interpolate.
    :return: List or array of interpolated y-coordinates corresponding to
             `interpolating_xs`.

    :raises ValueError: If the number of x and y values are different.
    :raises ValueError: If the x values are not equally spaced.
    """
    if len(xs) != len(ys):
        raise ValueError("A different number of x and y values were given")
    if st.deviation(np.diff(xs)) > 0.01:
        raise ValueError("Given x values are not equally spaced")

    diff = st.average(np.diff(xs))
    length = len(xs) + 2

    splines = np.zeros((length, length))
    np.fill_diagonal(splines, 4)
    np.fill_diagonal(splines[1:], 1)
    np.fill_diagonal(splines[:, 1:], 1)

    splines[0, 0] = -3 / diff
    splines[0, 1] = 0
    splines[0, 2] = 3 / diff
    splines[-1, -3] = -3 / diff
    splines[-1, -2] = 0
    splines[-1, -1] = 3 / diff

    expanded_ys = np.insert(ys, 0, 0)
    expanded_ys = np.append(expanded_ys, 0)

    coefficients = np.linalg.solve(splines, expanded_ys)

    expanded_xs = np.copy(xs)
    for i in range(3):
        expanded_xs = np.insert(expanded_xs, 0, xs[0] - (i + 1) * diff)
        expanded_xs = np.append(expanded_xs, xs[-1] + (i + 1) * diff)

    def basis_value(index, x):
        x0 = expanded_xs[index - 2]
        x1 = expanded_xs[index - 1]
        x2 = expanded_xs[index]
        x3 = expanded_xs[index + 1]
        x4 = expanded_xs[index + 2]

        if x0 <= x < x1:
            return ((x - x0) ** 3) / diff ** 3
        elif x1 <= x < x2:
            dx = x - x1
            return (diff ** 3 + 3 * diff ** 2 * dx + 3 * diff * dx ** 2 - 3 * dx ** 3) / diff ** 3
        elif x2 <= x < x3:
            dx = x3 - x
            return (diff ** 3 + 3 * diff ** 2 * dx + 3 * diff * dx ** 2 - 3 * dx ** 3) / diff ** 3
        elif x3 <= x <= x4:
            return ((x4 - x) ** 3) / diff ** 3
        else:
            return 0

    interpolating_ys = np.zeros(len(interpolating_xs))

    for i in range(len(interpolating_xs)):
        for j in range(len(coefficients)):
            interpolating_ys[i] += coefficients[j] * basis_value(j + 2, interpolating_xs[i])

    return interpolating_ys
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
    Performs cubic spline interpolation on a set of data points and computes the interpolated
    values for given x-values. A spline is generated with added boundary conditions for smooth
    curve transitions. The input x-values must be equally spaced, otherwise an exception will
    be raised.

    :param xs: The x-coordinates of the data points, used for spline interpolation.
    :type xs: numpy.ndarray
    :param ys: The y-coordinates of the data points corresponding to `xs`.
    :type ys: numpy.ndarray
    :param interpolating_xs: The x-coordinates at which the interpolated y-values are to be computed.
    :type interpolating_xs: numpy.ndarray
    :return: The interpolated y-values corresponding to `interpolating_xs`.
    :rtype: numpy.ndarray
    :raises ValueError: If the sizes of `xs` and `ys` are unequal.
    :raises ValueError: If the x-coordinates in `xs` are not equally spaced.
    """
    if len(xs) != len(ys):
        raise ValueError("A different number of x and y values were given")
    if np.std(np.diff(xs)) > 0.01:
        raise ValueError("Given x values are not equally spaced")

    diff = np.average(np.diff(xs))
    length = len(xs) + 2

    splines = np.zeros((length, length))
    np.fill_diagonal(splines, 4)
    np.fill_diagonal(splines[1:], 1)
    np.fill_diagonal(splines[:,1:], 1)
    splines[0,0] = -3/diff
    splines[0,1] = 0
    splines[0,2] = 3/diff
    splines[length - 1, length - 3] = -3/diff
    splines[length - 1, length - 2] = 0
    splines[length - 1, length - 1] = 3/diff

    expanded_ys = np.insert(ys, 0, 0)
    expanded_ys = np.append(expanded_ys, 0)

    coefficients = np.linalg.solve(splines, expanded_ys)

    expanded_xs = np.copy(xs)
    for i in range(3):
        expanded_xs = np.insert(expanded_xs, 0, xs[0] - (i + 1) * diff)
        expanded_xs = np.append(expanded_xs, xs[-1] + (i + 1) * diff)

    interpolating_ys = np.zeros(len(interpolating_xs))
    for i in range(len(interpolating_xs)):
        for j in range(len(coefficients)):
            jx = j + 2
            if expanded_xs[jx - 2] <= interpolating_xs[i] <= expanded_xs[jx - 1]:
                interpolating_ys[i] += coefficients[j] * (interpolating_xs[i] - expanded_xs[jx - 2])**3 / diff**3
            elif expanded_xs[jx - 1] <= interpolating_xs[i] <= expanded_xs[jx]:
                interpolating_ys[i] += coefficients[j] * (diff**3 + 3 * diff**2 * (interpolating_xs[i] - expanded_xs[jx - 1]) + 3 * diff * (interpolating_xs[i] - expanded_xs[jx - 1])**2 - 3 * (interpolating_xs[i] - expanded_xs[jx - 1])**3) / diff**3
            elif expanded_xs[jx] <= interpolating_xs[i] <= expanded_xs[jx + 1]:
                interpolating_ys[i] += coefficients[j] * (diff**3 + 3 * diff**2 * (expanded_xs[jx + 1] - interpolating_xs[i]) + 3 * diff * (expanded_xs[jx + 1] - interpolating_xs[i])**2 - 3 * (expanded_xs[jx + 1] - interpolating_xs[i])**3) / diff**3
            elif expanded_xs[jx + 1] <= interpolating_xs[i] <= expanded_xs[jx + 2]:
                interpolating_ys[i] += coefficients[j] * (expanded_xs[jx + 2] - interpolating_xs[i])**3 / diff**3
            else:
                interpolating_ys[i] += 0

    return interpolating_ys
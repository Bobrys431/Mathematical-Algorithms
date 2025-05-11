import numpy as np

import mathalgs.statistics as st

def derivatives(xs, ys, degree=1):
    """
    Calculates the first or second numerical derivative of a set of y-values
    with respect to corresponding x-values. The function uses numerical
    differentiation methods based on finite differences. It ensures that the
    x-values provided are evenly spaced and of the same length as the y-values.
    Different calculations are made based on the specified derivative degree.

    :param xs: Sequence of x-values
    :type xs: list or numpy.ndarray
    :param ys: Sequence of y-values corresponding to the x-values
    :type ys: list or numpy.ndarray
    :param degree: Degree of the derivative to calculate (1 for first derivative,
                   2 for second derivative). Default is 1.
    :type degree: int
    :raises ValueError: If the lengths of xs and ys differ
    :raises ValueError: If xs are not equally spaced based on standard deviation
    :raises ValueError: If the degree is neither 1 nor 2
    :return: A numpy array containing the computed derivatives
    :rtype: numpy.ndarray
    """
    if len(xs) != len(ys):
        raise ValueError("A different number of x and y values were given")
    if st.deviation(np.diff(xs)) > 0.001:
        raise ValueError("Given x values are not equally spaced")

    length = len(xs)
    diff = st.average(np.diff(xs))
    result_derivatives = np.zeros(length)

    if degree == 1:
        result_derivatives[0] = (ys[1] - ys[0]) / diff
        for i in range(1, length - 1):
            result_derivatives[i] = (ys[i + 1] - ys[i - 1]) / (2 * diff)
        result_derivatives[-1] = (ys[-1] - ys[-2]) / diff

    elif degree == 2:
        result_derivatives[0] = (ys[2] - 2 * ys[1] + ys[0]) / (diff ** 2)
        for i in range(1, length - 1):
            result_derivatives[i] = (ys[i - 1] - 2 * ys[i] + ys[i + 1]) / (diff ** 2)
        result_derivatives[-1] = (ys[-1] - 2 * ys[-2] + ys[-3]) / (diff ** 2)

    else:
        raise ValueError("There are only first and second derivative able to be calculated")

    return result_derivatives
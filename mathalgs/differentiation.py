import numpy as np
import mathalgs.statistics as st

def differentiate(xs, ys, degree=1):
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

def monotonicity(xs, ys, derivatives=None):
    """
    Determines the monotonicity of a given set of data points by analyzing their derivatives.
    Calculates whether the function represented by the data points is increasing, decreasing,
    or steady over specific segments of the input range and organizes the results by monotonicity type.

    :param xs: A list of x-values representing the independent variable.
    :type xs: list
    :param ys: A list of y-values corresponding to the x-values, representing the dependent variable.
    :type ys: list
    :param derivatives: Optional. A list of pre-computed derivative values for the given points.
        If not provided, the derivatives will be calculated from the input data.
    :type derivatives: list, optional
    :return: A dictionary containing three keys: "increases", "decreases", and "steady". Each key maps
        to a list of tuples indicating the start and end indices for contiguous segments of the input
        data where the function is increasing, decreasing, or steady, respectively.
    :rtype: dict
    """
    if derivatives is None:
        derivatives = differentiate(xs, ys)
    if len(xs) != len(ys) != len(derivatives):
        raise ValueError("A different number of x, y and derivative values were given")

    result = dict()
    result["increases"] = []
    result["decreases"] = []
    result["steady"] = []

    def get_state(_derivative):
        if _derivative > 0:
            return "increases"
        elif _derivative < 0:
            return "decreases"
        else:
            return "steady"

    state = get_state(derivatives[0])
    segment_start_index = 0

    for segment_end_index, derivative in enumerate(derivatives):
        if get_state(derivative) == state:
            continue

        result[state].append((segment_start_index, segment_end_index))
        segment_start_index = segment_end_index - 1
        state = get_state(derivative)
    result[state].append((segment_start_index, len(xs)))

    return result
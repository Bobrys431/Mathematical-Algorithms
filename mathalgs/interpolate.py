def polynomial(xs, ys):
    """
    Computes the coefficients of a polynomial that passes through a given set of points
    using Newton's divided difference method.

    :param xs: List of x-coordinates of the points. All x-values must be distinct.
    :type xs: list[float]
    :param ys: List of y-coordinates of the points corresponding to the x-values.
    :type ys: list[float]
    :return: A list representing the coefficients of the polynomial in increasing order
        of powers (e.g., [c0, c1, c2, ...] represents c0 + c1*x + c2*x^2 + ...).
    :rtype: list[float]
    :raises ValueError: If the lengths of xs and ys do not match.
    """
    if len(xs) != len(ys):
        raise ValueError("A different number of x and y values were given")

    length = len(xs)
    differences = [[0 for _ in range(length)] for _ in range(length)]
    differences[0][:length] = ys[:length]

    for i in range(1, length):
        for j in range(length - i):
            differences[i][j] = (differences[i - 1][j + 1] - differences[i - 1][j]) / (xs[j + i] - xs[j])

    polynomials = [0 for _ in range(length)]
    polynomials[0] = differences[0][0]
    for i in range(1, length):
        coefficients = [0 for _ in range(length)]
        coefficients[i] = differences[i][0]
        for j in range(i-1, -1, -1):
            for k in range(i, 0, -1):
                coefficients[k] -= xs[j] * coefficients[k - 1]
        for k in range(i + 1):
            polynomials[k] += coefficients[k]

    return polynomials

def spline(xs, ys):
    pass
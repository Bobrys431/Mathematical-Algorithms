def average(sequence):
    """
    Calculates the average of a given sequence of numbers.

    This function takes a sequence of numeric values and computes their average.
    If the given sequence is empty, the function raises a ValueError.

    :param sequence: A sequence of numeric values (e.g., list, tuple).
    :return: The average of the numbers in the sequence if non-empty,
        otherwise raises ValueError.
    """
    if len(sequence) == 0: raise ValueError("Sequence has to be not empty.")

    return sum(sequence) / len(sequence)

def median(sequence):
    """
    Calculates the median value of a given sequence of numbers.

    The median is the value separating the higher half from the lower half of a
    data set. If the sequence has an even number of elements, the median is defined
    as the average of the two middle numbers. If it has an odd number of elements,
    the median is the middle value in the sorted sequence.

    :param sequence: A sequence of numeric values for which the median needs to
        be calculated.
    :type sequence: list[float] | list[int]
    :return: The median value of the sequence if it is non-empty.
    :rtype: float | int
    """
    length = len(sequence)
    if length == 0: raise ValueError("Sequence has to be not empty.")

    sequence = sorted(sequence)
    if (length % 2) == 0:
        return average(sequence[length // 2 - 1:length // 2 + 1])
    else:
        return sequence[length // 2]

def deviation(sequence):
    """
    Calculate the sample standard deviation of a given sequence.

    This function computes the sample standard deviation, which quantifies the
    amount of variation or dispersion in a given sequence of numeric values. The
    calculation uses the average (mean) of the sequence and ensures the result
    is unbiased by dividing by `(n - 1)` for a sequence of length `n`.

    :param sequence: A list or any iterable of numeric values.
    :return: The sample standard deviation if the sequence has at least two
        elements. Returns 0 if the sequence contains exactly one element. Raises a ValueError if the sequence is empty.
    """
    length = len(sequence)
    avg = average(sequence)
    if length == 0: raise ValueError("Sequence has to be not empty.")
    if length == 1: return 0

    return (sum((i - avg) ** 2 for i in sequence) / (length - 1)) ** 0.5
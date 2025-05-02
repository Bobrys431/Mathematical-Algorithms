import random
import pandas as pd


def extract_data(file_name, y=None):
    """
    Extracts a subset of data from a CSV file based on the specified or randomly
    chosen value of the 'y' column.

    This function reads the given CSV file containing data with space delimiters,
    filters the rows based on the provided or randomly selected value for the 'y'
    column, and returns the filtered dataset as a DataFrame.

    :param file_name: The path to the CSV file to read.
    :type file_name: str
    :param y: Optional. The specific value of the 'y' column to filter the data.
               If not provided, a random value from the unique 'y' column values
               will be used.
    :type y: int or str, optional
    :return: A filtered pandas DataFrame containing data where the 'y' column
             matches the specified or randomly selected value.
    :rtype: pandas.DataFrame
    """

    df = pd.read_csv(file_name, delimiter=r'\s+')

    if y is None:
        unique_values = df['y'].unique().tolist()
        y = random.choice(unique_values)

    df = df[df['y'] == y]

    return df
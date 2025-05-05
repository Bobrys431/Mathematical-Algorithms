import mathalgs as ma  # Import custom math algorithms module
import pandas as pd  # Import pandas for data manipulation
import matplotlib.pyplot as plt  # Import matplotlib for plotting



# Read data from the text file
df = pd.read_csv("data.txt", delimiter=r'\s+')



# Create a new figure
plt.figure("Chart F(x,y) for every y")
plt.title("Chart F(x,y) for every y")

# Group data by 'y' values and plot each group
for value, subdf in df.groupby('y'):
    plt.plot(subdf['x'], subdf['f(x,y)'], label="y = " + str(value))

# Set labels and show the plot
plt.xlabel("x")
plt.ylabel("F(x,y)")
plt.legend()
plt.show()
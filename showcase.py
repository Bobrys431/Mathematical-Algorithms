import mathalgs as ma  # Import custom math algorithms module
import pandas as pd  # Import pandas for data manipulation
import matplotlib.pyplot as plt  # Import matplotlib for plotting



# Read data from the text file
df = pd.read_csv("data.txt", delimiter=r'\s+')



# Create a new figure
plt.figure("Chart F(x,y) for every y")
plt.title("Chart F(x,y) for every y")

# Group data by 'y' values and plot each group
for y, subdf in df.groupby('y'):
    plt.plot(subdf['x'], subdf['f(x,y)'], label="y = " + str(y))

# Set labels and show the plot
plt.xlabel("x")
plt.ylabel("F(x,y)")
plt.legend()
plt.show()



# Create a figure for statistical calculations
plt.figure("Statistical calculations for F(x,y)")
plt.title("Statistical calculations for F(x,y)")

# Get unique y values
ys = df['y'].unique()

# Initialize lists to store statistical measures
averages = []
medians = []
deviations = []

# Calculate statistics for each y group
for _, subdf in df.groupby('y'):
    averages.append(ma.statistics.average(subdf['f(x,y)']))
    medians.append(ma.statistics.median(subdf['f(x,y)']))
    deviations.append(ma.statistics.deviation(subdf['f(x,y)']))

# Create a grouped bar chart with statistical measures
plt.bar(ys - 0.35, averages, 0.35, color="green", label="Average")
plt.bar(ys, medians, 0.35, color="blue", label="Median")
plt.bar(ys + 0.35, deviations, 0.35, color="red", label="Standard deviation")

# Customize plot appearance
plt.xticks(df['y'].unique())
plt.xlabel("y")
plt.ylabel("F(x,y)")
plt.legend()
plt.grid()
plt.show()
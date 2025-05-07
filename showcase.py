# Import required libraries
# mathalgs - custom mathematical algorithms library
# pandas - data manipulation and analysis
# matplotlib - plotting library
# numpy - numerical computing library
import mathalgs as ma
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



# Load data from space-separated text file into pandas DataFrame
df = pd.read_csv("data.txt", delimiter=r'\s+')



# Create first plot: F(x,y) values for each y
plt.figure("Chart F(x,y) for every y")
plt.title("Chart F(x,y) for every y")

# Plot F(x,y) for each unique y value
for y, subdf in df.groupby('y'):
    plt.plot(subdf['x'], subdf['f(x,y)'], label="y = " + str(y))

# Set labels and display the plot
plt.xlabel("x")
plt.ylabel("F(x,y)")
plt.legend()
plt.show()



# Create a second plot: Statistical calculations
plt.figure("Statistical calculations for F(x,y)")
plt.title("Statistical calculations for F(x,y)")

# Get unique y values
ys = df['y'].unique()

# Initialize lists for statistical measures
averages = []
medians = []
deviations = []

# Calculate statistics for each y value
for _, subdf in df.groupby('y'):
    averages.append(ma.statistics.average(subdf['f(x,y)']))
    medians.append(ma.statistics.median(subdf['f(x,y)']))
    deviations.append(ma.statistics.deviation(subdf['f(x,y)']))

# Create a grouped bar chart for statistical measures
plt.bar(ys - 0.35, averages, 0.35, color="green", label="Average")
plt.bar(ys, medians, 0.35, color="blue", label="Median")
plt.bar(ys + 0.35, deviations, 0.35, color="red", label="Standard deviation")

# Set labels and display the plot
plt.xticks(df['y'].unique())
plt.xlabel("y")
plt.ylabel("F(x,y)")
plt.legend()
plt.grid()
plt.show()



# Prepare data for interpolation
# Select random unique y value for interpolation
interpolating_y = np.random.choice(df['y'].unique())
# Get x and F(x,y) values for selected y
xs = df[df['y'] == interpolating_y]['x'].tolist()
ys = df[df['y'] == interpolating_y]['f(x,y)'].tolist()



# Create a third plot: Polynomial interpolation
plt.figure("Polynomial interpolation for y = " + str(interpolating_y))
plt.title("Polynomial interpolation for y = " + str(interpolating_y))

# Generate points for a smooth interpolation curve
interpolating_xs = np.linspace(min(xs), max(xs), 300)
interpolating_ys = ma.interpolate.polynomial(xs, ys, interpolating_xs)

# Plot both interpolated function and original points
plt.plot(interpolating_xs, interpolating_ys, label="Interpolated function")
plt.plot(xs, ys, 'o', label="Original points")

# Set labels and display the plot
plt.xticks(xs)
plt.yticks(ys)
plt.xlabel("x")
plt.ylabel("F(x,y)")
plt.legend()
plt.grid()
plt.show()
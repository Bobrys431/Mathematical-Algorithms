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
plt.figure("Statistical calculations of F(x,y)")
plt.title("Statistical calculations of F(x,y)")

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
plt.xticks(ys)
plt.xlabel("y")
plt.ylabel("F(x,y)")
plt.legend()
plt.grid()
plt.show()



# Prepare data for interpolation
# Select random unique y value for interpolation
chosen_y = np.random.choice(df['y'].unique())
# Get x and F(x,y) values for selected y
xs = df[df['y'] == chosen_y]['x'].tolist()
ys = df[df['y'] == chosen_y]['f(x,y)'].tolist()



# Create a third plot: Polynomial interpolation
plt.figure("Polynomial interpolation for y = " + str(chosen_y))
plt.title("Polynomial interpolation for y = " + str(chosen_y))

# Generate points for a smooth interpolation curve
interpolating_xs = np.linspace(min(xs), max(xs), 300)
interpolating_ys = ma.interpolate.polynomial(xs, ys, interpolating_xs)

# Plot both interpolated function and original points
plt.plot(interpolating_xs, interpolating_ys, label="Interpolated function")
plt.plot(xs, ys, 'o', label="Original points", markersize=4)

# Set labels and display the plot
plt.xlabel("x")
plt.ylabel("F(x,y)")
plt.legend()
plt.grid()
plt.show()



# Create a fourth plot: Spline interpolation
plt.figure("Spline interpolation for y = " + str(chosen_y))
plt.title("Spline interpolation for y = " + str(chosen_y))

# Generate points for a smooth interpolation curve
interpolating_xs = np.linspace(min(xs), max(xs), 300)
interpolating_ys = ma.interpolate.spline(xs, ys, interpolating_xs)

# Plot both interpolated function and original points 
plt.plot(interpolating_xs, interpolating_ys, label="Interpolated function")
plt.plot(xs, ys, 'o', label="Original points", markersize=4)

# Set labels and display the plot
plt.xlabel("x")
plt.ylabel("F(x,y)")
plt.legend()
plt.grid()
plt.show()



# Create a fifth plot: Comparison of both interpolation methods
plt.figure("Comparison of spline and polynomial interpolation for y = " + str(chosen_y))
plt.title("Comparison of spline and polynomial interpolation for y = " + str(chosen_y))

# Generate x values for interpolation
interpolating_xs = np.linspace(min(xs), max(xs), 300)

# Calculate interpolated y values using both methods
interpolating_ys_polynomial = ma.interpolate.polynomial(xs, ys, interpolating_xs)
interpolating_ys_spline = ma.interpolate.spline(xs, ys, interpolating_xs)

# Plot interpolated curves and original points
plt.plot(interpolating_xs, interpolating_ys_polynomial, label="Polynomial interpolation", linestyle="--")
plt.plot(interpolating_xs, interpolating_ys_spline, label="Spline interpolation", linestyle="--")
plt.plot(xs, ys, 'o', label="Original points", markersize=3)

# Add labels and display the plot
plt.xlabel("x")
plt.ylabel("F(x,y)")
plt.legend()
plt.grid()
plt.show()



# Create a sixth plot: Comparison of polynomial approximations with different degrees
plt.figure("Comparison of approximated functions of different degrees for y = " + str(chosen_y))
plt.title("Comparison of approximated functions of different degrees for y = " + str(chosen_y))

# Generate x values for approximation extending beyond original data points
approximating_xs = np.linspace(min(xs) - 3, max(xs) + 3, 400)

# Calculate approximated y values using polynomials of degree 1 and 3
approximating_ys_first = ma.approximate.polynomial(xs, ys, 1, approximating_xs)
approximating_ys_third = ma.approximate.polynomial(xs, ys, 3, approximating_xs)

# Calculate R-squared (coefficient of determination) for first-degree approximation
approximating_ys_first_metrics = ma.approximate.polynomial(xs, ys, 1, xs)
numerator = 0
denominator = 0
for i in range(len(ys)):
    numerator += (ys[i] - approximating_ys_first_metrics[i]) ** 2
    denominator += (ys[i] - ma.statistics.average(ys)) ** 2
error_determination_coefficient = 1 - (numerator / denominator)

# Calculate RMSE (Root Mean Square Error) for third-degree approximation
approximating_ys_third_metrics = ma.approximate.polynomial(xs, ys, 3, xs)
error_root_mean_square = 0
for i in range(len(ys)):
    error_root_mean_square += (ys[i] - approximating_ys_third_metrics[i]) ** 2
error_root_mean_square = np.sqrt(error_root_mean_square / len(ys))

# Plot approximated functions and original data points
plt.plot(approximating_xs, approximating_ys_first, label="Approximation with degree 1\nError(R^2): " + str(round(error_determination_coefficient,2)))
plt.plot(approximating_xs, approximating_ys_third, label="Approximation with degree 3\nError(RMSE): " + str(round(error_root_mean_square,2)))
plt.plot(xs, ys, 'o', label="Original points", markersize=4)

# Set labels and display the plot
plt.xlabel("x")
plt.ylabel("F(x,y)")
plt.legend()
plt.grid()
plt.show()



# Create a seventh plot: Integration visualization
plt.figure("Showing the impact of changing a step value while integrating for y = " + str(chosen_y))
plt.title("Showing the impact of changing a step value while integrating for y = " + str(chosen_y))

# Calculate integral with low number of points (5)
less_xs = np.linspace(min(xs), max(xs), 5)
less_ys = ma.interpolate.polynomial(xs, ys, less_xs)
less_accuracy_integral = ma.integrate.trapezoidal(less_xs, less_ys)

# Calculate integral with medium number of points (30) 
medium_xs = np.linspace(min(xs), max(xs), 30)
medium_ys = ma.interpolate.polynomial(xs, ys, medium_xs)
medium_accuracy_integral = ma.integrate.trapezoidal(medium_xs, medium_ys)

# Calculate integral with high number of points (300)
high_xs = np.linspace(min(xs), max(xs), 300)
high_ys = ma.interpolate.polynomial(xs, ys, high_xs)
high_accuracy_integral = ma.integrate.trapezoidal(high_xs, high_ys)

# Plot the original function and filled areas representing different integration accuracies
plt.plot(interpolating_xs, interpolating_ys_polynomial, linewidth=1, label="Original function")
plt.fill_between(less_xs, less_ys, alpha=0.1, label="Integrate by 5 points: " + str(round(less_accuracy_integral, 2)))
plt.fill_between(medium_xs, medium_ys, alpha=0.1, label="Integrate by 30 points: " + str(round(medium_accuracy_integral, 2)))
plt.fill_between(high_xs, high_ys, alpha=0.1, label="Integrate by 300 points: " + str(round(high_accuracy_integral, 2)))

# Set labels, add legend and grid
plt.xlabel("x")
plt.ylabel("F(x,y)")
plt.legend()
plt.grid()
plt.show()
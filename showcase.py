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
    plt.plot(subdf['x'], subdf['f(x,y)'], label=f"y = {y}")

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
plt.figure(f"Polynomial interpolation for y = {chosen_y}")
plt.title(f"Polynomial interpolation for y = {chosen_y}")

# Generate points for a smooth interpolation curve
interpolating_xs = np.linspace(min(xs), max(xs), 1000)
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
plt.figure(f"Spline interpolation for y = {chosen_y}")
plt.title(f"Spline interpolation for y = {chosen_y}")

# Generate points for a smooth interpolation curve
interpolating_xs = np.linspace(min(xs), max(xs), 1000)
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
plt.figure(f"Comparison of spline and polynomial interpolation for y = {chosen_y}")
plt.title(f"Comparison of spline and polynomial interpolation for y = {chosen_y}")

# Generate x values for interpolation
interpolating_xs = np.linspace(min(xs), max(xs), 1000)

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
plt.figure(f"Comparison of approximated functions of different degrees for y = {chosen_y}")
plt.title(f"Comparison of approximated functions of different degrees for y = {chosen_y}")

# Generate x values for approximation extending beyond original data points
approximating_xs = np.linspace(min(xs) - 3, max(xs) + 3, 1400)

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
plt.plot(approximating_xs, approximating_ys_first, label=f"Approximation with degree 1\nError(R^2): {round(error_determination_coefficient,2)}")
plt.plot(approximating_xs, approximating_ys_third, label=f"Approximation with degree 3\nError(RMSE): {round(error_root_mean_square,2)}")
plt.plot(xs, ys, 'o', label="Original points", markersize=4)

# Set labels and display the plot
plt.xlabel("x")
plt.ylabel("F(x,y)")
plt.legend()
plt.grid()
plt.show()



# Create a seventh plot: Integration visualization
plt.figure(f"Showing the impact of changing a step value while integrating for y = {chosen_y}")
plt.title(f"Showing the impact of changing a step value while integrating for y = {chosen_y}")

# Calculate integral with low number of points (5)
less_xs = np.linspace(min(xs), max(xs), 5)
less_ys = ma.interpolate.polynomial(xs, ys, less_xs)
less_accuracy_integral = ma.integrate.trapezoidal(less_xs, less_ys)

# Calculate integral with medium number of points (30) 
medium_xs = np.linspace(min(xs), max(xs), 30)
medium_ys = ma.interpolate.polynomial(xs, ys, medium_xs)
medium_accuracy_integral = ma.integrate.trapezoidal(medium_xs, medium_ys)

# Calculate integral with high number of points (1000)
high_xs = np.linspace(min(xs), max(xs), 1000)
high_ys = ma.interpolate.polynomial(xs, ys, high_xs)
high_accuracy_integral = ma.integrate.trapezoidal(high_xs, high_ys)

# Plot the original function and filled areas representing different integration accuracies
plt.plot(interpolating_xs, interpolating_ys_polynomial, linewidth=1, label="Original function")
plt.fill_between(less_xs, less_ys, alpha=0.1, label=f"Integrate by 5 points: {round(less_accuracy_integral, 2)}")
plt.fill_between(medium_xs, medium_ys, alpha=0.1, label=f"Integrate by 30 points: {round(medium_accuracy_integral, 2)}")
plt.fill_between(high_xs, high_ys, alpha=0.1, label=f"Integrate by 1000 points: {round(high_accuracy_integral, 2)}")

# Set labels, add legend and grid
plt.xlabel("x")
plt.ylabel("F(x,y)")
plt.legend()
plt.grid()
plt.show()



# Create an eighth plot: Differentiation visualization
plt.figure(f"Showing the impact of changing a step value while differentiate for y = {chosen_y}")
plt.title(f"Showing the impact of changing a step value while differentiate for y = {chosen_y}")

# Calculate derivatives with low number of points (5)
less_xs = np.linspace(min(xs), max(xs), 5)
less_ys = ma.interpolate.spline(xs, ys, less_xs)
less_accuracy_derivatives = ma.differentiation.differentiate(less_xs, less_ys)

# Calculate derivatives with medium number of points (30)
medium_xs = np.linspace(min(xs), max(xs), 30)
medium_ys = ma.interpolate.spline(xs, ys, medium_xs)
medium_accuracy_derivatives = ma.differentiation.differentiate(medium_xs, medium_ys)

# Calculate derivatives with high number of points (1000)
high_xs = np.linspace(min(xs), max(xs), 1000)
high_ys = ma.interpolate.spline(xs, ys, high_xs)
high_accuracy_derivatives = ma.differentiation.differentiate(high_xs, high_ys)

# Plot an original function and its derivative
plt.plot(interpolating_xs, interpolating_ys_spline, label="Function")
plt.plot(less_xs, less_accuracy_derivatives, label="Differentiate with 5 points", linestyle="--")
plt.plot(medium_xs, medium_accuracy_derivatives, label="Differentiate with 30 points", linestyle="--")
plt.plot(high_xs, high_accuracy_derivatives, label="Differentiate with 1000 points", linestyle="--")

# Set labels and display settings
plt.xlabel("x")
plt.ylabel("F(x,y)")
plt.legend()
plt.grid()
plt.show()



# Create a nine plot: Monotonicity analysis
plt.figure(f"Monotonicity for y = {chosen_y}")
plt.title(f"Monotonicity for y = {chosen_y}")

# Use high-resolution x and y values for monotonicity analysis
monotonic_xs = high_xs
monotonic_ys = high_ys
monotonicity = ma.differentiation.monotonicity(monotonic_xs, monotonic_ys, derivatives=high_accuracy_derivatives)

# Helper function to format interval segments for legend labels
def format_segments(segments):
    return ", ".join([f"[{round(monotonic_xs[k[0]], 2)}; {round(monotonic_xs[k[1] - 1], 2)})" for k in segments])

# Plot increasing segments in green
for up in monotonicity['increases']:
    label = f"Increasing: {format_segments(monotonicity['increases'])}" if up == monotonicity['increases'][0] else ""
    plt.plot(monotonic_xs[up[0]:up[1]], monotonic_ys[up[0]:up[1]], color="green", label=label)

# Plot decreasing segments in red  
for down in monotonicity['decreases']:
    label = f"Decreasing: {format_segments(monotonicity['decreases'])}" if down == monotonicity['decreases'][0] else ""
    plt.plot(monotonic_xs[down[0]:down[1]], monotonic_ys[down[0]:down[1]], color="red", label=label)

# Plot steady (constant) segments in blue
for flat in monotonicity['steady']:
    label = f"Steady: {format_segments(monotonicity['steady'])}" if flat == monotonicity['steady'][0] else ""
    plt.plot(monotonic_xs[flat[0]:flat[1]], monotonic_ys[flat[0]:flat[1]], color="blue", label=label)

# Add labels and display the plot
plt.xlabel("x")
plt.ylabel("F(x,y)")
plt.legend()
plt.grid()
plt.show()
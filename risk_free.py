import numpy as np
from scipy.interpolate import CubicSpline

# Load US discount rate data from a CSV file or other sources
us_discount_data = np.loadtxt('Data\YC_US.csv', delimiter=',', skiprows=1)

# Extract maturities and discount rate values from the loaded data
maturities = us_discount_data[:, 0]
discount_rates = us_discount_data[:, 1]

# Create a cubic spline curve for the US discount rate data
us_discount_spline = CubicSpline(maturities, discount_rates)

# Define the specific times for which you want to obtain the discount rates
times = [0.5, 1, 1.5, 2]

# Evaluate the cubic spline curve at the specified times to obtain discount rates
discount_rates = us_discount_spline(times)

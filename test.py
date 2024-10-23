import numpy as np

x = np.linspace(0, 10, 100)

print(x)


# Define a new fitting function with a shift parameter b
def fit_func(x, a, b):
    return a / (x ** 2 + b)


# Calculate weights that increase with x to emphasize fitting the tail
weights = var_values ** 2

# Fit the model to the data
popt, _ = curve_fit(fit_func, var_values, P_values, p0=[1.0, 1.0], sigma=weights)
# Generate fitted values
fitted_vals = fit_func(var_values, *popt)

plt.plot(var_values, fitted_vals, label=f'Fit: $\\frac{{{popt[0]:.2f}}}{{x^2 + {popt[1]:.2f}}}$', linestyle='--')
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

# Define the sine function to fit
def sine_function(x, amplitude, frequency, phase, offset):
    return amplitude * np.sin(frequency * x + phase) + offset


# Fit the sine curve
def fit_sine_curve(x, y):
    # Initial guesses for amplitude, frequency, phase, and offset
    initial_guess = [np.ptp(y) / 2, 2 * np.pi / (x[-1] - x[0]), 0, np.mean(y)]
    params, _ = curve_fit(sine_function, x, y, p0=initial_guess)
    return params

# Plot and save the results
def plot_and_save(x, y, params, output_file):
    plt.figure()
    plt.scatter(x, y, label='Data', color='blue')

    # Generate fitted data
    x_fit = np.linspace(min(x), max(x), 1000)
    fitted_y = sine_function(x_fit, *params)
    plt.plot(x_fit, fitted_y, label='Fitted Sine Curve', color='red')
    plt.legend()
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Singles')
    plt.title('Sine Curve Fitting')

    # Add the parameters as text on the plot
    textstr = '\n'.join((
        r'Amplitude=%.2f' % params[0],
        r'Frequency=%.2f' % params[1],
        r'Phase=%.2f' % params[2],
        r'Offset=%.2f' % params[3]))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
                   verticalalignment='top', bbox=props)

    plt.savefig(output_file)
    plt.close()

# Main function
def main():
    input_csv = './waveplates/data/ht/singles.csv'  # Replace with your CSV file path
    output_plot = './waveplates/data/ht/fitted_sine_curve.png'

    data_df = pd.read_csv(input_csv)
    x = data_df['angle'].values
    y = data_df['singles'].values

    # Fit sine curve
    params = fit_sine_curve(x, y)

    # Save plot
    plot_and_save(x, y, params, output_plot)

    # Print fitted parameters
    print("Fitted Parameters:")
    print(f"Amplitude: {params[0]}")
    print(f"Frequency: {params[1]}")
    print(f"Phase: {params[2]}")
    print(f"Offset: {params[3]}")

if __name__ == "__main__":
    main()
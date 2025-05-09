import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def sine_function(x, amplitude, phase, offset, frequency):
    # Convert x from degrees to radians for the sine function
    x = np.radians(x)
    return amplitude * np.sin(frequency * (x - phase)) + offset

# Fit the sine curve
def fit_sine_curve(x, y, waveplate_type='hwp'):
    # Determine frequency based on waveplate type
    if waveplate_type == 'hwp':
        frequency = 4
    elif waveplate_type == 'qwp':
        frequency = 8
    else:
        raise ValueError("Invalid waveplate type. Use 'hwp' or 'qwp'.")

    # Initial guesses for amplitude, phase, and offset
    initial_guess = [np.ptp(y) / 2, 0, np.mean(y)]
    params, _ = curve_fit(lambda x, amplitude, phase, offset: sine_function(x, amplitude, phase, offset, frequency), 
                          x, y, p0=initial_guess, bounds=(0, [np.inf, 2 * np.pi, np.inf]))

    params = np.append(params, frequency)

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
    phase_degrees = np.degrees(params[1])
    textstr = '\n'.join((
        r'Amplitude=%.2f' % params[0],
        r'Phase=%.2f' % phase_degrees,
        r'Offset=%.2f' % params[2]))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
                   verticalalignment='top', bbox=props)

    plt.savefig(output_file)
    plt.close()


def analyse_waveplate_data(waveplate_name, data_dir):
    """
    Function to analyse waveplate characterisation data.
    Args:
        waveplate_name: Name of the waveplate (e.g., 'ht', 'qt')

    Returns:
        tuple:
            singles_params: Fitted parameters for singles data
            0: Amplitude
            1: Phase (in radians)
            2: Offset
            coincidences_params: Fitted parameters for coincidences data
            0: Amplitude
            1: Phase (in radians)
            2: Offset
    """

    waveplate_type_initial = waveplate_name.split('/')[-1][0]
    waveplate_type = 'hwp' if waveplate_type_initial == 'h' else 'qwp'

    input_csv = f'{data_dir}/counts.csv'

    data_df = pd.read_csv(input_csv)
    angles = data_df['angle'].values
    singles = data_df['singles'].values
    coincidences = data_df['coincidences'].values

    singles_params = fit_sine_curve(angles, singles, waveplate_type)
    coincidences_params = fit_sine_curve(angles, coincidences, waveplate_type)

    print(f"singles_params: {singles_params}")
    print(f"coincidences_params: {coincidences_params}")

    singles_output_plot = f'{data_dir}/sine_fit.png'
    coincidences_output_plot = f'{data_dir}/sine_fit_coincidences.png'

    plot_and_save(angles, singles, singles_params, singles_output_plot)
    plot_and_save(angles, coincidences, coincidences_params, coincidences_output_plot)

    print(f"Waveplate Type: {waveplate_type}")
    print("Fitted Parameters:")
    print(f"Singles: Amplitude={singles_params[0]}, Phase={singles_params[1]}, Offset={singles_params[2]}")
    print(f"Coincidences: Amplitude={coincidences_params[0]}, Phase={coincidences_params[1]}, Offset={coincidences_params[2]}")

    return singles_params, coincidences_params

if __name__ == "__main__":
    analyse_waveplate_data('ht', './data/ht/2025-05-08_16-16-11')

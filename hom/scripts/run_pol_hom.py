import time
import numpy as np
import pandas as pd
import qutip as qt
import state_preparation
import state_preparation.waveplates
from set_waveplate_angles import *
from ttag_console import *
from pathos import multiprocessing as mp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

H = qt.basis(2, 0)  # |H>
V = qt.basis(2, 1)  # |V>

MEASUREMENT_TIME = 1  # seconds
COINCIDENCE_WINDOW = 1.0e-9  # seconds

DETECTOR_PATTERN = 'd4, d2 - d12, d10'
LABELS = ["TT", "RT", "TR", "RR"]

wp = load_waveplates_from_config('waveplates.json')

sys.path.append(os.environ["TTAG"])
from ttag import *

if getfreebuffer() == 0:
   	buf = TTBuffer(0)
else:
   	buf = TTBuffer(getfreebuffer() - 1)

if buf.getrunners() == 0:
	buf.start()

ccTwoFolds = nFold_create(DETECTOR_PATTERN,2)

keys = list(ccTwoFolds.keys())
opts = [(key, MEASUREMENT_TIME, COINCIDENCE_WINDOW, ccTwoFolds[key][0], ccTwoFolds[key][1]) for key in keys]

def run_pol_hom():
    # Prepare H in A and tomos in Z
    set_waveplate_angles(wp, 0)

    # Sit in the dip position
    # TODO

    # Sweep through V -> D -> H -> A -> V in B
    theta_samples = np.linspace(0, 2 * np.pi, 61)

    pool = mp.Pool(8)

    df = pd.DataFrame()
    try:
        for theta in theta_samples:
            # Rotate B
            alpha = np.cos(theta)
            beta = np.sin(theta)
            
            psi = alpha * H + beta * V
            psi_hwp_rad, psi_qwp_rad = state_preparation.waveplates.get_hwp_qwp_from_target_state(psi)
            psi_hwp = np.degrees(psi_hwp_rad)
            psi_qwp = np.degrees(psi_qwp_rad)

            set_waveplate_angles(wp, {
                'hla': 0,
                'qla': 0,
                'hlb': psi_hwp,
                'qlb': psi_qwp
            })

            time.sleep(0.5) # Wait for the waveplates to settle
            
            # Get the coincidences
            time.sleep(MEASUREMENT_TIME + 0.5) # Wait for the measurement to finish
            result = pool.map(lambda i: buf.multicoincidences(*opts[i][1::]), range(len(opts)))

            # Save the results to a csv
            new_row = {'theta': np.degrees(theta)}
            for i, key in enumerate(keys):
                new_row[key] = result[i]
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            df.to_csv('pol_hom.csv', index=False)

    finally:
        pool.close()

    # Plot the coincidences as a function of the angle


def fit_pol_hom(file):
    df = pd.read_csv(file)
    theta = df['theta'].values
    coincidences = [df[key].values for key in LABELS]

    # Fit the data to a sine function with 180 degree period

    def sine_func(x, a, b, c):
        return a * np.sin(2*np.deg2rad(x + b)) + c

    # Initial guess for the parameters
    initial_guess = [10000, 0, 0]

    popt_TT, _ = curve_fit(sine_func, theta, coincidences[0], p0=initial_guess)
    popt_RT, _ = curve_fit(sine_func, theta, coincidences[1], p0=initial_guess)
    popt_TR, _ = curve_fit(sine_func, theta, coincidences[2], p0=initial_guess)
    popt_RR, _ = curve_fit(sine_func, theta, coincidences[3], p0=initial_guess)

    # Plot the results
    plt.figure()

    plt.plot(df['theta'], df['TT'], 'o', label='TT')
    plt.plot(df['theta'], df['RT'], 'o', label='RT')
    plt.plot(df['theta'], df['TR'], 'o', label='TR')
    plt.plot(df['theta'], df['RR'], 'o', label='RR')
    plt.plot(df['theta'], sine_func(theta, *popt_TT), '-', label='TT fit')
    plt.plot(df['theta'], sine_func(theta, *popt_RT), '-', label='RT fit')
    plt.plot(df['theta'], sine_func(theta, *popt_TR), '-', label='TR fit')
    plt.plot(df['theta'], sine_func(theta, *popt_RR), '-', label='RR fit')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Coincidences')
    
    # add the fit parameters to the legend
    offset_TT = np.degrees(popt_TT[1]) % 360
    offset_RT = np.degrees(popt_RT[1]) % 360
    offset_TR = np.degrees(popt_TR[1]) % 360
    offset_RR = np.degrees(popt_RR[1]) % 360
    plt.legend([f'TT offset: {offset_TT:.2f} degrees',
                f'RT offset: {offset_RT:.2f} degrees',
                f'TR offset: {offset_TR:.2f} degrees',
                f'RR offset: {offset_RR:.2f} degrees'], loc='upper right')
    
    plt.savefig('pol_hom_fit.png')
    
    
if __name__ == "__main__":
    run_pol_hom()
    fit_pol_hom('pol_hom.csv')
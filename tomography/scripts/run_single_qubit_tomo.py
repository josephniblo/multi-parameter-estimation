import os
import sys
from datetime import datetime
import time
import qutip as qt

import pandas as pd
import pathos.multiprocessing as mp

from ttag_console import *
from move_plates import *
from set_waveplate_angles import *
from analyze_single_qubit_tomo import *

sys.path.append(os.environ["TTAG"])
from ttag import *

# Prepare the ttag buffer
if getfreebuffer() == 0:
    buf = TTBuffer(0)
else:
    buf = TTBuffer(getfreebuffer() - 1)
    
if buf.getrunners() == 0:
    buf.start()

# Constants
MEASUREMENT_BASES = ['H','D','R', 'V','A','L']

H = qt.basis(2, 0)  # |H>
V = qt.basis(2, 1)  # |V>

# File naming
SAGNAC = "2"
POWER = "40mW"

# Detector settings
MEASUREMENT_TIME = 1  # seconds
COINCIDENCE_WINDOW = 1.0e-9

DETECTOR_MAPPINGS = {
    'TT': 12,
    'TR': 10,
    'RT': 4, 
    'RR': 2 
}

def run_and_analyze_tomo(wp, nominal_state, target_density_matrix, launcher):
    repo_root = os.popen('git rev-parse --show-toplevel').read().strip()
    data_path = os.path.join(repo_root, 'tomography', 'data')

    start_time = datetime.now().strftime("%F--%Hh-%Mm")

    try:
        tomo_t = TomographyController(
            name='tomo_t',
            quarter_waveplate=wp['qt'],
            half_waveplate=wp['ht']
        )

        tomo_r = TomographyController(
            name='tomo_r',
            quarter_waveplate=wp['qr'],
            half_waveplate=wp['hr']
        )

        print("Starting single-qubit tomography...")

        pool = mp.Pool(8)

        df = pd.DataFrame(columns=['Measurement Basis'] + list(DETECTOR_MAPPINGS.keys()))

        for measurement_basis in MEASUREMENT_BASES:
            print(f"Setting measurement basis to {measurement_basis}")
           
            # Set the waveplate angles for the measurement basis
            with ThreadPoolExecutor(max_workers=2) as executor:
                executor.submit(tomo_t.set_label, measurement_basis)
                executor.submit(tomo_r.set_label, measurement_basis)
                
            time.sleep(2) # wait for waveplates to move

            time.sleep(1 + MEASUREMENT_TIME)
            singles = buf.singles(MEASUREMENT_TIME)
            
            relevant_singles = [singles[j-1] for j in DETECTOR_MAPPINGS.values()]
            new_row = pd.DataFrame([[measurement_basis] + relevant_singles], columns=['Measurement Basis'] + list(DETECTOR_MAPPINGS.keys()))

            df = pd.concat([df, new_row], ignore_index=True)

            # Save the data to a CSV file as we go
            df.to_csv('single_qubit_tomography_data_in_progress.csv')

    finally:
        try:
            st.closeDevice('Axis1')
            st.closeDevice('Axis2')
            st.closeDevice('Axis3')
            st.closeDevice('Axis4')
            st.closeDevice('Axis5')
            st.closeDevice('Axis6')
            st.closeDevice('Axis7')
            st.closeDevice('Axis8')
        except:
            print("Error closing waveplates. Please check the devices.")

        try:
            pool.close()
        except:
            print("Error closing the multiprocessing pool. Please check the pool.")

        try:
            end_time = datetime.now().strftime("%F--%Hh-%Mm")

            data_file_directory = os.path.join(data_path, f"{start_time}--{end_time}_1qb_tomo_sagnac{SAGNAC}_{POWER}_{launcher}_nominally_{nominal_state}")
            os.makedirs(data_file_directory, exist_ok=True)

            data_file_path = os.path.join(data_file_directory, 'tomography_data.csv')
            
            os.rename('single_qubit_tomography_data_in_progress.csv', data_file_path)
        except:
            print("Error renaming the data file.")
            print("Data file saved as 'single_qubit_tomography_data_in_progress.csv'. Please rename it manually.")

    # Transmitted arm
    transmitted_col = 'TT'
    reflected_col = 'TR'
    arm_name = 'T'

    rho, projs, fidelity, purity = plot_density_matrix_from_tomo_data(data_file_directory, target_density_matrix, transmitted_col, reflected_col, arm_name)
    
    print("Transmitted Arm Results:")
    print("    Reconstructed Density Matrix (Transmitted):")
    print("    ", rho)
    print("    Fidelity (Transmitted):", fidelity)
    print("    Purity (Transmitted):", purity)

    # Reflected arm
    transmitted_col = 'RT'
    reflected_col = 'RR'
    arm_name = 'R'

    rho, projs, fidelity, purity = plot_density_matrix_from_tomo_data(data_file_directory, target_density_matrix, transmitted_col, reflected_col, arm_name)
    print("Reflected Arm Results:")
    print("    Reconstructed Density Matrix (Reflected):")
    print("    ", rho)
    print("    Fidelity (Reflected):", fidelity)
    print("    Purity (Reflected):", purity)


if __name__ == "__main__":
    #----------------#
    # Options
    #----------------#
    NOMINAL_STATE = "H"
    LAUNCHER = "A" # 'A' or 'B'
    alpha_1 = 1
    beta_1 = 0

    psi = alpha_1 * H + beta_1 * V
    target_density_matrix = qt.ket2dm(psi).full()

    wp = load_waveplates_from_config('waveplates.json')

    run_and_analyze_tomo(wp, NOMINAL_STATE, target_density_matrix, LAUNCHER)
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
from analyze_two_qubit_tomo import *

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
MEASUREMENT_BASES = ['HH','HD','HR','DH','DD','DR','RH','RD','RR']

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

DETECTOR_PATTERN = 'd4, d2 - d12, d10'

# Linear stage settings
MOTORISED_LINEAR_STAGE_NAME = "dave"
STANDA_LINEAR_CONFIG_FILE_LOCATION = os.path.abspath("/home/jh115/emqTools/standa/8CMA06-25_15-Lin_G34.cfg")

def run_and_analyze_tomo(wp, nominal_state, target_density_matrix):
    repo_root = os.popen('git rev-parse --show-toplevel').read().strip()
    data_path = os.path.join(repo_root, 'tomography', 'data')

    start_time = datetime.now().strftime("%F--%Hh-%Mm")

    try:
        # Linear Stage
        with open('linear_stages.json') as f:
            linear_stages_config = json.load(f)

        linear_stage_config = linear_stages_config[MOTORISED_LINEAR_STAGE_NAME]
        linear_stage_dip_position = linear_stage_config['dip_position']
        linear_stage_dip_half_width = linear_stage_config['dip_half_width']

        usb_device = st.findDevices("usb")
        usb_name = usb_device[MOTORISED_LINEAR_STAGE_NAME]
        linear_stage = st.device(usb_name, STANDA_LINEAR_CONFIG_FILE_LOCATION)

        # Come out the dip so that we don't have HOM interference
        linear_stage.goTo(linear_stage_dip_position + linear_stage_dip_half_width)

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

        df = pd.DataFrame(columns = ['basis'] + [detector_name for detector_name in DETECTOR_MAPPINGS.keys()])

        twoFolds = nFold_create(DETECTOR_PATTERN, 2)
        keys = list(twoFolds.keys())
        opts = [(key, MEASUREMENT_TIME, COINCIDENCE_WINDOW, twoFolds[key][0], twoFolds[key][1]) for key in keys]
        optsSize = len(opts)

        print("Starting two-qubit tomography...")

        pool = mp.Pool(8)

        for measurement_basis in MEASUREMENT_BASES:
            print(f"Setting measurement basis to {measurement_basis}")
            tomo_t_basis = measurement_basis[0]
            tomo_r_basis = measurement_basis[1]
            
            tomo_t.set_label(tomo_t_basis)
            tomo_r.set_label(tomo_r_basis)
            time.sleep(2) # wait for waveplates to move

            time.sleep(1 + MEASUREMENT_TIME)
            singles = buf.singles(MEASUREMENT_TIME)
            
            # array with 4 positions for each outcome "TT", "TR", "RT", "RR"
            coincidences = pool.map(lambda i: buf.multicoincidences(*opts[i][1::]), range(len(opts)))
            
            new_row = pd.DataFrame([[measurement_basis] + coincidences])
            new_row.columns=df.columns

            df = pd.concat([df, new_row], ignore_index=True)

            # Save the data to a CSV file as we go
            df.to_csv('tomography_data_in_progress.csv')

    finally:
        try:
            linear_stage.goTo(linear_stage_dip_position)
        except:
            print("Error moving linear stage to dip position. Please check the stage.")

        try:
            linear_stage.closeDevice()
        except:
            print("Error closing linear stage. Please check the stage.")
            
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

            data_file_directory = os.path.join(data_path, f"{start_time}--{end_time}_2qb_tomo_sagnac{SAGNAC}_{POWER}_nominally_{nominal_state}")
            os.makedirs(data_file_directory, exist_ok=True)

            data_file_path = os.path.join(data_file_directory, 'tomography_data.csv')
            
            os.rename('tomography_data_in_progress.csv', data_file_path)
        except:
            print("Error renaming the data file.")
            print("Data file saved as 'tomography_data_in_progress.csv'. Please rename it manually.")

    rho, projs, fidelity, purity = plot_density_matrix_from_tomo_data(data_file_directory, target_density_matrix)

    print("Reconstructed Density Matrix (rho):")
    print(rho)

    print("Fidelity:", fidelity)
    print("Purity:", purity)


if __name__ == "__main__":
    #----------------#
    # Options
    #----------------#
    NOMINAL_STATE = "HH"
    alpha_1 = 1
    beta_1 = 0
    alpha_2 = 1
    beta_2 = 0

    psi_1 = alpha_1 * H + beta_1 * V
    psi_2 = alpha_2 * H + beta_2 * V

    target_pure_state = qt.tensor(psi_1, psi_2)
    target_density_matrix = qt.ket2dm(target_pure_state).full()

    wp = load_waveplates_from_config('waveplates.json')

    run_and_analyze_tomo(wp, NOMINAL_STATE, target_density_matrix)
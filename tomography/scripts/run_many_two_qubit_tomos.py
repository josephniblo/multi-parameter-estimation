import state_preparation.waveplates
from run_two_qubit_tomo import *
from set_waveplate_angles import *
import state_preparation
import qutip as qt

states = [
    {
        "name": "DH",
        "alpha_1": 1/np.sqrt(2),
        "beta_1": 1/np.sqrt(2),
        "alpha_2": 1,
        "beta_2": 0
    }
]

wp = load_waveplates_from_config('waveplates.json')

# Prepare the ttag buffer
if getfreebuffer() == 0:
    buf = TTBuffer(0)
else:
    buf = TTBuffer(getfreebuffer() - 1)
if buf.getrunners() == 0:
    buf.start()

for state in states:
    alpha_1 = state["alpha_1"]
    beta_1 = state["beta_1"]
    alpha_2 = state["alpha_2"]
    beta_2 = state["beta_2"]

    psi_1 = alpha_1 * H + beta_1 * V
    psi_2 = alpha_2 * H + beta_2 * V

    target_pure_state = qt.tensor(psi_1, psi_2)
    target_density_matrix = qt.ket2dm(target_pure_state).full()

    print(f"Running tomography for state: {state['name']}")
    print(f"Target Pure State:\n{target_pure_state}")

    # State Preparation
    psi_1_hwp_rad, psi_1_qwp_rad = state_preparation.waveplates.get_hwp_qwp_from_target_state(psi_1)
    psi_2_hwp_rad, psi_2_qwp_rad = state_preparation.waveplates.get_hwp_qwp_from_target_state(psi_2)

    # Convert angles to degrees
    psi_1_hwp = np.degrees(psi_1_hwp_rad)
    psi_1_qwp = np.degrees(psi_1_qwp_rad)
    psi_2_hwp = np.degrees(psi_2_hwp_rad)
    psi_2_qwp = np.degrees(psi_2_qwp_rad)

    # Set up the waveplates
    wp['hla'].set_angle(psi_1_hwp)
    wp['qla'].set_angle(psi_1_qwp)
    wp['hlb'].set_angle(psi_2_hwp)
    wp['qlb'].set_angle(psi_2_qwp)

    time.sleep(2) # wait for waveplates to move

    run_and_analyze_tomo(wp, state["name"], target_density_matrix)

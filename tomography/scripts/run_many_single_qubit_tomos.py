import state_preparation.waveplates
from run_single_qubit_tomo import *
from set_waveplate_angles import *
import state_preparation
import qutip as qt

states = [
    # {
    #     "name": "V",
    #     "alpha": 0,
    #     "beta": 1
    # },
    {
        "name": "D",
        "alpha": 1/np.sqrt(2),
        "beta": 1/np.sqrt(2)
    },
    {
        "name": "R",
        "alpha": 1/np.sqrt(2),
        "beta": 1j/np.sqrt(2)
    },
    {
        "name": "A",
        "alpha": 1/np.sqrt(2),
        "beta": -1/np.sqrt(2)
    },
    {
        "name": "L",
        "alpha": 1/np.sqrt(2),
        "beta": 1j/np.sqrt(2)
    }
]

wp = load_waveplates_from_config('waveplates.json')
# Set up the waveplates
for launcher_label in ['a', 'b']:
    # Wait for user to block the other, non specified launcher
    print(f"BLOCK THE OTHER LAUNCHER ({'a' if launcher_label == 'b' else 'b'}) AND PRESS ENTER TO CONTINUE")
    input()

    for state in states:
        alpha = state["alpha"]
        beta = state["beta"]

        psi = alpha * H + beta * V

        target_pure_state = psi
        target_density_matrix = qt.ket2dm(target_pure_state).full()

        print(f"Running tomography for state: {state['name']}")
        print(f"Target Pure State:\n{target_pure_state}")

        # State Preparation
        psi_hwp_rad, psi_qwp_rad = state_preparation.waveplates.get_hwp_qwp_from_target_state(psi)

        # Convert angles to degrees
        psi_hwp = np.degrees(psi_hwp_rad)
        psi_qwp = np.degrees(psi_qwp_rad)


        wp[f'hl{launcher_label}'].set_angle(psi_hwp)
        wp[f'ql{launcher_label}'].set_angle(psi_qwp)
        
        run_and_analyze_tomo(wp, state["name"], target_density_matrix, launcher_label)
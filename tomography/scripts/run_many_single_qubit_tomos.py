import state_preparation.waveplates
from run_single_qubit_tomo import *
from set_waveplate_angles import *
import state_preparation
import qutip as qt

states = [
    {
        "name": "H",
        "alpha": 1,
        "beta": 0
    },
    {
        "name": "D",
        "alpha": 1/np.sqrt(2),
        "beta": 1/np.sqrt(2),
        "hwp": 22.5,
        "qwp": 45
    },    
    {
        "name": "V",
        "alpha": 0,
        "beta": 1
    },
    {
        "name": "R",
        "alpha": 1/np.sqrt(2),
        "beta": 1j/np.sqrt(2),
        "hwp": 0,
        "qwp": 45
    },
    {
        "name": "A",
        "alpha": 1/np.sqrt(2),
        "beta": -1/np.sqrt(2),
        "hwp": -22.5,
        "qwp": 45
    },
    {
        "name": "L",
        "alpha": 1/np.sqrt(2),
        "beta": -1j/np.sqrt(2),
        "hwp": 45,
        "qwp": 45
    }
]

# delete the waveplate angles files if they exist
try:
    os.remove('waveplate_angles_a.txt')
    os.remove('waveplate_angles_b.txt')
except OSError:
    pass

wp = load_waveplates_from_config('waveplates.json')
# Set up the waveplates
for launcher_label in ['B','A']:
    # Wait for user to block the other, non specified launcher
    print(f"BLOCK LAUNCHER ({'A' if launcher_label == 'B' else 'B'}) AND PRESS ENTER TO CONTINUE")
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

        with open(f'waveplate_angles_{launcher_label.lower()}.txt', 'a') as f:
            f.write(f"State: {state['name']}\n")
            f.write(f"psi_hwp: {psi_hwp}\n")
            f.write(f"psi_qwp: {psi_qwp}\n")

        set_waveplate_angles(wp, {
            f'hl{launcher_label.lower()}': psi_hwp,
            f'ql{launcher_label.lower()}': psi_qwp,
            f'hr{launcher_label.lower()}': 0,
            f'qr{launcher_label.lower()}': 0,
            f'ht{launcher_label.lower()}': 0,
            f'qt{launcher_label.lower()}': 0
        })
        
        run_and_analyze_tomo(wp, state["name"], target_density_matrix, launcher_label)
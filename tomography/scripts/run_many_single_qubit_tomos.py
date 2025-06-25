import state_preparation.waveplates
from run_single_qubit_tomo import *
from set_waveplate_angles import *
import state_preparation
import qutip as qt


states = [
    {
        "name": f"theta={theta}deg-phi={phi}deg",
        "alpha": np.cos(np.deg2rad(theta / 2)),
        "beta": np.exp(1j * np.deg2rad(phi)) * np.sin(np.deg2rad(theta / 2))
    }
    for theta in np.linspace(0, 180, 9)
    for phi in [0, 22.5, 45, 67.5, 90]
]


# delete the waveplate angles files if they exist
try:
    os.remove('waveplate_angles_a.txt')
    os.remove('waveplate_angles_b.txt')
except OSError:
    pass

wp = load_waveplates_from_config('waveplates.json')

def pre_compensate_state(psi: qt.Qobj, launcher_label):
    """
    Pre-compensate the state for the specified launcher.
    Applies the phase from pre_comp.json to the state.
    """
    with open('pre_comp.json', 'r') as f:
        pre_comp = json.load(f)
    
    pre_comp_phase = pre_comp[launcher_label]
    pre_comp_phase = np.deg2rad(pre_comp_phase)
    
    compensation_matrix = (1j * pre_comp_phase / 2 * qt.sigmaz()).expm()

    return compensation_matrix * psi

# Set up the waveplates
for launcher_label in ['B', 'A']:
    # Wait for user to block the other, non specified launcher
    print(f"BLOCK LAUNCHER ({'A' if launcher_label == 'B' else 'B'}) AND PRESS ENTER TO CONTINUE")
    input()

    for state in states:
        alpha = state["alpha"]
        beta = state["beta"]

        psi = alpha * H + beta * V

        target_pure_state = pre_compensate_state(psi, launcher_label)
        print(f"Target Pure State (pre-compensated):\n{target_pure_state}")
        target_density_matrix = qt.ket2dm(target_pure_state).full()

        print(f"Running tomography for state: {state['name']}")
        print(f"Target Pure State:\n{target_pure_state}")

        # State Preparation
        psi_hwp_rad, psi_qwp_rad = state_preparation.waveplates.get_hwp_qwp_from_target_state(target_pure_state)

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
            f'hr': 0,
            f'qr': 0,
            f'ht': 0,
            f'qt': 0
        })

        run_and_analyze_tomo(wp, state["name"], target_density_matrix, launcher_label)
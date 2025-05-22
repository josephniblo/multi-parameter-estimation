import numpy as np
from set_waveplate_angles import *
from run_twofold_hom import *
import fit_hom 
import state_preparation.waveplates
import qutip as qt

TEMPERATURE = "42C"
POWER = 51

repo_root = os.popen('git rev-parse --show-toplevel').read().strip()

H = qt.basis(2, 0)  # |H>
V = qt.basis(2, 1)  # |V>

states = [
    {
        "name": "H",
        "tomo_basis": "V",
        "alpha": 1,
        "beta": 0
    },
    {
        "name": "D",
        "tomo_basis": "A",
        "alpha": 1/np.sqrt(2),
        "beta": 1/np.sqrt(2)
    },    
    {
        "name": "V",
        "tomo_basis": "H",
        "alpha": 0,
        "beta": 1
    },
    {
        "name": "R",
        "tomo_basis": "L",
        "alpha": 1/np.sqrt(2),
        "beta": 1j/np.sqrt(2)
    },
    {
        "name": "A",
        "tomo_basis": "R",
        "alpha": 1/np.sqrt(2),
        "beta": -1/np.sqrt(2)
    },
    {
        "name": "L",
        "tomo_basis": "A",
        "alpha": 1/np.sqrt(2),
        "beta": -1j/np.sqrt(2)
    }
]

wp = load_waveplates_from_config('waveplates.json')

tomo_t = TomographyController(
    "T",
    quarter_waveplate=wp["qt"],
    half_waveplate=wp["ht"]
)
tomo_r = TomographyController(
    name="R",
    quarter_waveplate=wp["qr"],
    half_waveplate=wp["hr"]
)

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

start_time = datetime.datetime.now().strftime("%F--%Hh-%Mm")

for state in states:
    # Prepare the state 
    alpha = state["alpha"]
    beta = state["beta"]
    psi = alpha * H + beta * V

    target_pure_state_a = pre_compensate_state(psi, "A")
    target_pure_state_b = pre_compensate_state(psi, "B")

    psi_hwp_rad_a, psi_qwp_rad_a = state_preparation.waveplates.get_hwp_qwp_from_target_state(target_pure_state_a)
    psi_hwp_rad_b, psi_qwp_rad_b = state_preparation.waveplates.get_hwp_qwp_from_target_state(target_pure_state_b)

    psi_hwp_a = np.degrees(psi_hwp_rad_a)
    psi_qwp_a = np.degrees(psi_qwp_rad_a)
    psi_hwp_b = np.degrees(psi_hwp_rad_b)
    psi_qwp_b = np.degrees(psi_qwp_rad_b)

    set_waveplate_angles(wp, {
            f'hla': psi_hwp_a,
            f'hlb': psi_hwp_b,
            f'qla': psi_qwp_a,
            f'qlb': psi_qwp_b,
        })
    
    # Set the tomos to the state preparation basis (to ensure we have phase sensitivity)
    with ThreadPoolExecutor() as executor:
        executor.submit(tomo_t.set_label, state["tomo_basis"])
        executor.submit(tomo_r.set_label, state["tomo_basis"])
    
    # Save the state parameters to a json file
    with open('state_parameters.json', 'w') as f:
        json.dump({
            "alpha": {
                "real": np.real(alpha),
                "imag": np.imag(alpha)
            },
            "beta": {
                "real": np.real(beta),
                "imag": np.imag(beta)
            },
            "psi_compensated_a": {
                "real": np.real(target_pure_state_a.full().tolist()).tolist(),
                "imag": np.imag(target_pure_state_a.full().tolist()).tolist()
            },
            "psi_compensated_b": {
                "real": np.real(target_pure_state_b.full().tolist()).tolist(),
                "imag": np.imag(target_pure_state_b.full().tolist()).tolist()
            },
            "psi_hwp_a": np.real(psi_hwp_a),
            "psi_qwp_a": np.real(psi_qwp_a),
            "psi_hwp_b": np.real(psi_hwp_b),
            "psi_qwp_b": np.real(psi_qwp_b)
        }, f, indent=4)

    # Run the twofold hom
    data_file = run_twofold_hom(TEMPERATURE, POWER)

    # Analyse the twofold hom
    fit_hom.fit_from_file(
        file_name=data_file,
        output_dir=os.path.join(repo_root, 'hom', 'plots', 'state-preparation', start_time, state['name']),
    )

    # Move all the files to the correct directory
    os.makedirs(f"data/state-preparation/{start_time}/{state['name']}", exist_ok=True)
    
    # data file
    os.rename(data_file, f"data/state-preparation/{start_time}/{state['name']}/{os.path.basename(data_file)}")
    
    # state parameters
    os.rename('state_parameters.json', f"data/state-preparation/{start_time}/{state['name']}/state_parameters.json")

import datetime
import os
import time
import numpy as np
import pandas as pd
import qutip as qt
from ttag_console import *
import state_preparation.waveplates
from set_waveplate_angles import *
from pathos import multiprocessing as mp

# check timetagger is running
sys.path.append(os.environ["TTAG"])
from ttag import *

# White and blue output ports of the in-fibre beam splitters
DETECTORS = {
    9: {"arm": "TT", "color": "white"},
    12: {"arm": "TT", "color": "blue"},
    11: {"arm": "TR", "color": "white"},
    10: {"arm": "TR", "color": "blue"},
    1: {"arm": "RT", "color": "white"},
    4: {"arm": "RT", "color": "blue"},
    7: {"arm": "RR", "color": "white"},
    2: {"arm": "RR", "color": "blue"},
}

# Get coincidences across the 8 detectors
MEASUREMENT_TIME = 1e-3 / 20
REPETITIONS = 30000
COINCIDENCE_WINDOW = 1.0e-9

sample_range = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.375, 0.5, 0.625, 0.75, 0.8, 0.85, 0.9, 0.95, 1])

theta_range = np.pi * sample_range
theta_states = [{"theta": theta, "delta_phi": 0} for theta in theta_range] + [{"theta": theta, "delta_phi": np.pi / 4} for theta in theta_range] + [{"theta": theta, "delta_phi": np.pi / 2} for theta in theta_range]

delta_phi_range_theta_pi_by_4 = np.pi / 2 * sample_range
delta_phi_states_theta_pi_by_4 = [{"theta": np.pi/4, "delta_phi": delta_phi} for delta_phi in delta_phi_range_theta_pi_by_4]

delta_phi_range_theta_pi_by_2 = np.pi / 2 * sample_range
delta_phi_states_theta_pi_by_2 = [{"theta": np.pi/2, "delta_phi": delta_phi} for delta_phi in delta_phi_range_theta_pi_by_2]
delta_phi_states = (
    delta_phi_states_theta_pi_by_4 + delta_phi_states_theta_pi_by_2
)

states = theta_states + delta_phi_states
print(states)

delays = pd.read_json("detChannels.json").transpose().reset_index()
delays.columns = ["detector", "det_index", "det_delay", "todo"]
delays["det_name"] = delays["detector"].apply(lambda x: int(x[1:]))

repo_root = os.popen("git rev-parse --show-toplevel").read().strip()

H = qt.basis(2, 0)  # |H>
V = qt.basis(2, 1)  # |V>

wp = load_waveplates_from_config("waveplates.json")

tomo_t = TomographyController("T", quarter_waveplate=wp["qt"], half_waveplate=wp["ht"])
tomo_r = TomographyController(
    name="R", quarter_waveplate=wp["qr"], half_waveplate=wp["hr"]
)

def pre_compensate_state(psi: qt.Qobj, launcher_label):
    """
    Pre-compensate the state for the specified launcher.
    Applies the phase from pre_comp.json to the state.
    """
    with open("pre_comp.json", "r") as f:
        pre_comp = json.load(f)

    pre_comp_phase = pre_comp[launcher_label]
    pre_comp_phase = np.deg2rad(pre_comp_phase)

    compensation_matrix = (1j * pre_comp_phase / 2 * qt.sigmaz()).expm()

    return compensation_matrix * psi


if getfreebuffer() == 0:
    buf = TTBuffer(0)
else:
    buf = TTBuffer(getfreebuffer() - 1)

if buf.getrunners() == 0:
    buf.start()

for i, state in enumerate(states):
    start_time = datetime.datetime.now().strftime("%F--%Hh-%Mm-%Ss")

    alpha_a_ket = np.cos(state["theta"] / 2) * H + np.sin(state["theta"] / 2) * V
    # Note that in the paper, the delta phi is given as HALF the difference in phase between the two arms
    alpha_b_ket = (
        np.cos(state["theta"] / 2) * H
        + np.exp(1j * (2 * state["delta_phi"])) * np.sin(state["theta"] / 2) * V
    )

    target_pure_state_a = pre_compensate_state(alpha_a_ket, "A")
    target_pure_state_b = pre_compensate_state(alpha_b_ket, "B")

    psi_hwp_rad_a, psi_qwp_rad_a = (
        state_preparation.waveplates.get_hwp_qwp_from_target_state(target_pure_state_a)
    )
    psi_hwp_rad_b, psi_qwp_rad_b = (
        state_preparation.waveplates.get_hwp_qwp_from_target_state(target_pure_state_b)
    )

    psi_hwp_a = np.degrees(psi_hwp_rad_a)
    psi_qwp_a = np.degrees(psi_qwp_rad_a)
    psi_hwp_b = np.degrees(psi_hwp_rad_b)
    psi_qwp_b = np.degrees(psi_qwp_rad_b)

    set_waveplate_angles(
        wp,
        {
            f"hla": psi_hwp_a,
            f"hlb": psi_hwp_b,
            f"qla": psi_qwp_a,
            f"qlb": psi_qwp_b,
        },
    )

    # UNORDERED pairs of detectors
    coincidence_pairs = pd.DataFrame(
        [
            (
                i,
                j
            )
            for i in DETECTORS.keys()
            for j in DETECTORS.keys()
            if i < j
        ],
        columns=[
            "detector_a_name",
            "detector_b_name",
        ],
    )

    pool = mp.Pool(len(coincidence_pairs))

    all_coincidence_pairs = pd.DataFrame(
        columns=[
            "detector_a_name",
            "detector_b_name",
            "tomography_setting_t",
            "tomography_setting_r",
            "repetition",
            "coincidences",
        ],
    )
    try:
        for tomography_setting in [("H", "H"), ("H", "V"), ("V", "V"), ("V", "H")]:
            with ThreadPoolExecutor() as executor:
                executor.submit(tomo_t.set_label, tomography_setting[0])
                executor.submit(tomo_r.set_label, tomography_setting[1])
            time.sleep(0.1)

            for rep in range(round(REPETITIONS / 4)):
                rep_coincidence_pairs = coincidence_pairs.copy()
                rep_coincidence_pairs["repetition"] = rep
                rep_coincidence_pairs["coincidences"] = None
                rep_coincidence_pairs["timestamp"] = datetime.datetime.now().strftime(
                    "%F--%Hh-%Mm-%Ss-%f"
                )

                time.sleep(MEASUREMENT_TIME + 0.1e-3)

                coincidences = pool.map(
                    lambda i: buf.multicoincidences(
                        MEASUREMENT_TIME,
                        COINCIDENCE_WINDOW,
                        [
                            coincidence_pairs["detector_a_name"][i] - 1,
                            coincidence_pairs["detector_b_name"][i] - 1,
                        ],
                        [
                            delays[
                                delays["det_name"]
                                == coincidence_pairs["detector_a_name"][i]
                            ]["det_delay"].values[0],
                            delays[
                                delays["det_name"]
                                == coincidence_pairs["detector_b_name"][i]
                            ]["det_delay"].values[0],
                        ],
                    ),
                    range(len(coincidence_pairs)),
                )

                rep_coincidence_pairs["coincidences"] = coincidences

                rep_coincidence_pairs["tomography_setting_t"] = tomography_setting[0]
                rep_coincidence_pairs["tomography_setting_r"] = tomography_setting[1]

                all_coincidence_pairs = pd.concat(
                    [all_coincidence_pairs, rep_coincidence_pairs], ignore_index=True
                )

        # Throw away any repetitions with no coincidences anywhere in the repetition
        # This keeps the data volumes lower
        valid_reps = (
            all_coincidence_pairs.groupby("repetition")["coincidences"]
            .sum()
            .loc[lambda x: x > 0]
            .index
        )
        all_coincidence_pairs = all_coincidence_pairs[
            all_coincidence_pairs["repetition"].isin(valid_reps)
        ].reset_index(drop=True)

        output_dir = os.path.join(
            repo_root, "multi-parameter-estimation", "data", start_time
        )
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(
            repo_root,
            "multi-parameter-estimation",
            "data",
            start_time,
            f"coincidences.csv",
        )

        # save the parameters to a CSV file
        params_file = os.path.join(
            repo_root, "multi-parameter-estimation", "data", start_time, f"params.csv"
        )
        params_df = pd.DataFrame(
            {
                "theta": state["theta"],
                "delta_phi": state["delta_phi"],
                "psi_hwp_a": psi_hwp_a,
                "psi_qwp_a": psi_qwp_a,
                "psi_hwp_b": psi_hwp_b,
                "psi_qwp_b": psi_qwp_b,
                "measurement_time": MEASUREMENT_TIME,
                "repetitions": REPETITIONS,
            },
            index=[0],
        )
        params_df.to_csv(params_file, index=False)

        # save the results to a CSV file
        all_coincidence_pairs.to_csv(output_file, index=False)

    finally:
        pool.close()

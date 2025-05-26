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


def get_estimation_label(arm_a, arm_b):
    # Double bunched
    if arm_a == arm_b:
        # DB_H
        if arm_a[1] == "T":
            return "DB_H"
        # DB_V
        elif arm_a[1] == "R":
            return "DB_V"
        else:
            raise ValueError(f"Unknown arm: {arm_a}")
    # Coincidence
    elif arm_a[0] != arm_b[0]:
        return "C"
    # Single bunched
    elif arm_a[1] != arm_b[1]:
        return "SB"
    else:
        raise ValueError(f"Unknown arm combination: {arm_a}, {arm_b}")

delays = pd.read_json("detChannels.json").transpose().reset_index()
delays.columns = ["detector", "det_index", "det_delay", "todo"]
delays["det_name"] = delays["detector"].apply(lambda x: int(x[1:]))

repo_root = os.popen("git rev-parse --show-toplevel").read().strip()

H = qt.basis(2, 0)  # |H>
V = qt.basis(2, 1)  # |V>


# Define the states to be prepared
theta_range = np.linspace(0, np.pi, 60)
# delta_phi_range = np.linspace(0, np.pi / 2, 60)
states = [{"theta": theta, "delta_phi": 0} for theta in theta_range]

wp = load_waveplates_from_config("waveplates.json")

tomo_t = TomographyController("T", quarter_waveplate=wp["qt"], half_waveplate=wp["ht"])
tomo_r = TomographyController(
    name="R", quarter_waveplate=wp["qr"], half_waveplate=wp["hr"]
)

with ThreadPoolExecutor() as executor:
    executor.submit(tomo_t.set_label, "H")
    executor.submit(tomo_r.set_label, "H")


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


# Get coincidences across the 8 detectors
MEASUREMENT_TIME = 0.1e-3
REPETITIONS = 5000
COINCIDENCE_WINDOW = 1.0e-9

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
                j,
                get_estimation_label(DETECTORS[i]["arm"], DETECTORS[j]["arm"]),
            )
            for i in DETECTORS.keys()
            for j in DETECTORS.keys()
            if i < j
        ],
        columns=[
            "detector_a_name",
            "detector_b_name",
            "estimation_label",
        ],
    )

    pool = mp.Pool(len(coincidence_pairs))

    all_coincidence_pairs = pd.DataFrame(
        columns=[
            "detector_a_name",
            "detector_b_name",
            "estimation_label",
            "repetition",
            "coincidences",
        ],
    )
    try:
        for rep in range(REPETITIONS):
            rep_coincidence_pairs = coincidence_pairs.copy()
            rep_coincidence_pairs["repetition"] = rep
            rep_coincidence_pairs["coincidences"] = None
            rep_coincidence_pairs["timestamp"] = datetime.datetime.now().strftime(
                "%F--%Hh-%Mm-%Ss-%f"
            )

            time.sleep(MEASUREMENT_TIME + 1e-2)

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

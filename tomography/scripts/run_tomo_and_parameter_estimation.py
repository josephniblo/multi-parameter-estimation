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
import tomtag
from run_single_qubit_tomo import run_and_analyze_tomo

# check timetagger is running
sys.path.append(os.environ["TTAG"])
from ttag import *

DETECTOR_SECONDS_PER_UNIT = (
    156.5 / 1e12
)  # each unit of time on the timetagger corresponds to this many seconds

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
MEASUREMENT_TIME = 2  # seconds
COINCIDENCE_WINDOW = 1.0e-9  # 1 ns

POWER = 10  # mW
TEMPERATEURE = "32.7C"  # Of the crystal


states = [
    # {"theta": 0, "delta_phi": 0},
    # {"theta": np.pi / 4, "delta_phi": 0},
    # {"theta": np.pi / 2, "delta_phi": 0},
    # {"theta": 0, "delta_phi": np.pi / 4},
    # {"theta": np.pi / 4, "delta_phi": np.pi / 4},
    # {"theta": np.pi, "delta_phi": 0},
    # {"theta": np.pi / 2, "delta_phi": np.pi / 4},
    # {"theta": 0, "delta_phi": np.pi / 2},
    # {"theta": np.pi / 4, "delta_phi": np.pi / 2},
    # {"theta": np.pi / 2, "delta_phi": np.pi / 2},
    # {"theta": np.pi / 2, "delta_phi": np.pi / 8},
    # {"theta": np.pi / 4, "delta_phi": np.pi / 8},
    # {"theta": 3 * np.pi / 4, "delta_phi": np.pi / 8},
    # {"theta": 3 * np.pi / 4, "delta_phi": 0},
    # {"theta": 3 * np.pi / 4, "delta_phi": 3 * np.pi / 8},
    # {"theta": 3 * np.pi / 4, "delta_phi": np.pi / 2},
    # {"theta": np.pi / 4, "delta_phi": 3 * np.pi / 4},
    # {"theta": 2 * np.pi / 4, "delta_phi": 3 * np.pi / 4},
    # {"theta": 3 * np.pi / 4, "delta_phi": np.pi / 2},
    {"theta": 3 * np.pi / 4, "delta_phi": np.pi / 4},
    {"theta": 2 * np.pi / 4, "delta_phi": 3 * np.pi / 8},
    {"theta": np.pi / 4, "delta_phi": 3 * np.pi / 8},
]

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

for i, state in enumerate(states):
    print(f"UNBLOCK BOTH LAUNCHERS AND PRESS ENTER TO CONTINUE")
    input()

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
        [(i, j) for i in DETECTORS.keys() for j in DETECTORS.keys() if i < j],
        columns=[
            "detector_a_name",
            "detector_b_name",
        ],
    )

    try:
        state_coincidences = pd.DataFrame(
            columns=[
                "detector_a_name",
                "detector_b_name",
                "detector_a_time_tag",
                "detector_b_time_tag",
                "timestamp",
                "tomography_setting_t",
                "tomography_setting_r",
            ]
        )

        for tomography_setting in [("H", "H"), ("H", "V"), ("V", "V"), ("V", "H")]:
            with ThreadPoolExecutor() as executor:
                executor.submit(tomo_t.set_label, tomography_setting[0])
                executor.submit(tomo_r.set_label, tomography_setting[1])
            time.sleep(0.1)

            buf.start()
            time.sleep(MEASUREMENT_TIME)
            buf.stop()

            time_tags = np.array(buf.rawtags, dtype=np.int64)
            tag_channels = np.array(buf.rawchannels, dtype=np.int64)

            last_tag = max(time_tags)
            earliest_allowed_tag = (
                last_tag - MEASUREMENT_TIME / DETECTOR_SECONDS_PER_UNIT
            )

            # filter out tags that are too old
            tag_channels = tag_channels[time_tags >= earliest_allowed_tag]
            time_tags = time_tags[time_tags >= earliest_allowed_tag]

            # Get the coincidences for each pair of detectors using tomtag
            for i in range(len(coincidence_pairs)):
                print(
                    f"Processing coincidence pair {i + 1}/{len(coincidence_pairs)}: {coincidence_pairs['detector_a_name'][i]} and {coincidence_pairs['detector_b_name'][i]}"
                )
                detector_coincidences = pd.DataFrame(
                    columns=[
                        "detector_a_name",
                        "detector_b_name",
                        "detector_a_time_tag",
                        "detector_b_time_tag",
                        "timestamp",
                        "tomography_setting_t",
                        "tomography_setting_r",
                    ]
                )

                detector_a_index = int(coincidence_pairs["detector_a_name"][i]) - 1
                detector_b_index = int(coincidence_pairs["detector_b_name"][i]) - 1
                detector_a_tags = time_tags[tag_channels == detector_a_index]
                detector_b_tags = time_tags[tag_channels == detector_b_index]

                if len(detector_a_tags) == 0 or len(detector_b_tags) == 0:
                    print(
                        f"No tags found for detectors {coincidence_pairs['detector_a_name'][i]} and {coincidence_pairs['detector_b_name'][i]}"
                    )
                    continue

                # Get the coincidences using tomtag
                # Note have to add the offsets from the delays to the time tags
                detector_a_delay = delays[
                    delays["det_name"] == coincidence_pairs["detector_a_name"][i]
                ]["det_delay"].values[0]
                detector_b_delay = delays[
                    delays["det_name"] == coincidence_pairs["detector_b_name"][i]
                ]["det_delay"].values[0]

                coincidences = tomtag.get_twofold_tags(
                    np.array(
                        detector_a_tags - detector_a_delay / DETECTOR_SECONDS_PER_UNIT,
                        dtype=np.int64,
                    ),
                    np.array(
                        detector_b_tags - detector_b_delay / DETECTOR_SECONDS_PER_UNIT,
                        dtype=np.int64,
                    ),
                    len(detector_a_tags),
                    len(detector_b_tags),
                    int(COINCIDENCE_WINDOW / DETECTOR_SECONDS_PER_UNIT),
                )

                if len(coincidences) == 0:
                    print(
                        f"No coincidences found for detectors {coincidence_pairs['detector_a_name'][i]} and {coincidence_pairs['detector_b_name'][i]}"
                    )
                    continue

                detector_coincidences["detector_a_time_tag"] = coincidences[0]
                detector_coincidences["detector_b_time_tag"] = coincidences[1]

                detector_coincidences["detector_a_name"] = coincidence_pairs[
                    "detector_a_name"
                ][i]
                detector_coincidences["detector_b_name"] = coincidence_pairs[
                    "detector_b_name"
                ][i]

                detector_coincidences["timestamp"] = datetime.datetime.now().strftime(
                    "%F--%Hh-%Mm-%Ss"
                )
                detector_coincidences["tomography_setting_t"] = tomography_setting[0]
                detector_coincidences["tomography_setting_r"] = tomography_setting[1]

                state_coincidences = pd.concat(
                    [state_coincidences, detector_coincidences],
                    ignore_index=True,
                )

                output_dir = os.path.join(
                    repo_root, "multi-parameter-estimation", "data", start_time
                )
                os.makedirs(output_dir, exist_ok=True)

                # save the coincidences to a CSV file
                output_file = os.path.join(
                    repo_root,
                    "multi-parameter-estimation",
                    "data",
                    start_time,
                    f"coincidences.csv",
                )

                detector_coincidences.to_csv(
                    output_file,
                    mode="a",
                    header=not os.path.exists(output_file),
                    index=False,
                )
                print(
                    f"Saved coincidences for tomography setting {tomography_setting} to {output_file}"
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
                "power": POWER,
                "temperature": TEMPERATEURE,
            },
            index=[0],
        )
        params_df.to_csv(params_file, index=False)

        # Now run single qubit tomography 
        state_name = f"theta={state['theta']:.2f}rad-delta_phi={state['delta_phi']:.2f}rad"

        for launcher_label in ['B', 'A']:
            # Wait for user to block the other, non specified launcher
            print(f"BLOCK LAUNCHER ({'A' if launcher_label == 'B' else 'B'}) AND PRESS ENTER TO CONTINUE")
            input()

            target_state_matrix = qt.ket2dm(target_pure_state_a if launcher_label == 'A' else target_pure_state_b).full()

            run_and_analyze_tomo(wp, state_name, target_state_matrix, launcher_label)

    except Exception as e:
        print(f"Error processing state {i}: {e}")
        # rethrow
        raise e

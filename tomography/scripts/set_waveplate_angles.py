from concurrent.futures import ThreadPoolExecutor
import json
from move_plates import *

class WaveplateController:
    def __init__(self, name, type, fast_axis_angle):
        self.name = name
        self.type = type
        self.fast_axis_angle = fast_axis_angle

    def set_angle(self, target_angle):
        """
        Set the angle of the waveplate, with respect to its fast axis.
        """
        angleMove([target_angle + self.fast_axis_angle], [self.name])
        print(f"{self.name} set to {target_angle} degrees with respect to its fast axis.")
        # Here you would add the actual code to set the angle of the waveplate
        # For example: move_waveplate_to_angle(self.name, angle)

class TomographyController:
    # Projection labels and their corresponding angles for quarter and half waveplates
    BASIS_ANGLES = {
        'H': (0, 0),
        'V': (0, 45),
        'D': (45, 22.5),
        'A': (45, -22.5),
        'R': (45, 45),
        'L': (45, 0),
    }

    def __init__(self, name, quarter_waveplate, half_waveplate):
        self.name = name
        self.quarter_waveplate = quarter_waveplate
        self.half_waveplate = half_waveplate

    def set_label(self, label):
        """
        Set the waveplate angles based on the provided label.
        """
        if label not in self.BASIS_ANGLES:
            raise ValueError(f"Invalid label: {label}. Valid labels are: {list(self.BASIS_ANGLES.keys())}")

        qwp_angle, hwp_angle = self.BASIS_ANGLES[label]
        with ThreadPoolExecutor() as executor:
            executor.submit(self.quarter_waveplate.set_angle, qwp_angle)
            executor.submit(self.half_waveplate.set_angle, hwp_angle)
        print(f"{self.name} set to label {label}: QWP angle {qwp_angle}, HWP angle {hwp_angle}")
        

def load_waveplates_from_config(filepath) -> dict:
    """
    Load the waveplate configuration from a file.
    Returns:
        dict: A dictionary containing waveplates from the configuration file.
    """
    with open(filepath, 'r') as f:
        waveplates_json = json.load(f)
    
    # make a dictionary of WaveplateControllers
    waveplates_dict = {}
    for waveplate in waveplates_json:
        waveplate_config = waveplates_json[waveplate]
        waveplate_config['name'] = waveplate
        
        name = waveplate_config['name']
        type = waveplate_config['type']
        fast_axis_angle = waveplate_config['fast_axis']

        controller = WaveplateController(name, type, fast_axis_angle)
        waveplates_dict[name] = controller

    return waveplates_dict

def set_waveplate_angles(waveplates, angles):
    """
    Set the angles of the waveplates.
    Args:
        waveplates (dict): Dictionary of waveplate controllers.
        angles (dict): Dictionary of angles to set for each waveplate.
    """
    # spawn a thread for each waveplate
    with ThreadPoolExecutor() as executor:
        for name, angle in angles.items():
            if name in waveplates:
                executor.submit(waveplates[name].set_angle, angle)
            else:
                print(f"Waveplate {name} not found in the configuration.")

        # wait for all threads to finish
        executor.shutdown(wait=True)
        print("All waveplates set to their respective angles.")
    
def set_tomo_labels(tomos, labels):
    """
    Set the angles of the waveplates based on the provided labels.
    Args:
        tomos (dict): Dictionary of tomography controllers.
        labels (dict): Dictionary of labels to set for each tomography controller.
    """
    # spawn a thread for each tomography controller
    with ThreadPoolExecutor() as executor:
        for name, label in labels.items():
            if name in tomos:
                executor.submit(tomos[name].set_label, label)
            else:
                print(f"Tomography controller {name} not found in the configuration.")

        # wait for all threads to finish
        executor.shutdown(wait=True)
        print("All tomography controllers set to their respective labels.")


if __name__ == "__main__":
    import code
    import argparse

    parser = argparse.ArgumentParser(description="Interactive Waveplate Shell")
    parser.add_argument("config", help="Path to waveplate config JSON")
    args = parser.parse_args()

    wp = load_waveplates_from_config(args.config)

    banner = (
        "Waveplate interactive shell\n"
        f"Loaded waveplates from: {args.config}\n"
        "Use `wp['name'].set_angle(deg)` to control waveplates.\n"
        "Example: wp['qwp1'].set_angle(45)\n"
        "Type Ctrl-D or 'exit()' to exit.\n"
    )

    # Launch interactive shell with `wp` in the local namespace
    code.interact(local={"wp": wp}, banner=banner)
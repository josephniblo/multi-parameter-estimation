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
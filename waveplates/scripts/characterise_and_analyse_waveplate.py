import datetime
from characterise_waveplate import *
from analyse_waveplate_characterisation_data import *
import numpy as np

if __name__ == "__main__":
	if len(sys.argv) == 4:
		waveplate_name = sys.argv[1]
		singles_detector_name = int(sys.argv[2])
		coincidence_detector_name = int(sys.argv[3])
	else:
		print("Error: waveplate name, singles detector name and coincidence detector name not specified correctly.")
		print("Usage: python characterise_waveplate.py <waveplate_name> <singles_detector_name> <coincidence_detector_name>")
		print("eg: python characterise_waveplate.py ht 12 2")
		sys.exit(1)

	# create a timestamped directory for the waveplate
	repo_root = os.popen('git rev-parse --show-toplevel').read().strip()
	timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	
	waveplate_dir = os.path.join(repo_root, "waveplates/data", waveplate_name, timestamp)

	if not os.path.exists(waveplate_dir):
		os.makedirs(waveplate_dir)

	waveplate_type_initial = waveplate_name.split('/')[-1][0]
	waveplate_type = 'hwp' if waveplate_type_initial == 'h' else 'qwp'

	# coarse
	angles = [i for i in np.linspace(0, 180, 21)]

	run_characterisation(waveplate_name, singles_detector_name, coincidence_detector_name, angles, waveplate_dir)
	singles_fit_params, coincidences_fit_params = analyse_waveplate_data(waveplate_name, waveplate_dir)

	# fine
	best_guess_max_singles = (singles_fit_params[1] * 180 / np.pi) + 22.5 % 90 
	best_guess_max_coincidence = (coincidences_fit_params[1] * 180 / np.pi) + 22.5 % 90

	best_guess_min_singles = (best_guess_max_singles - 45) % 90
	best_guess_min_coincidence = (best_guess_max_coincidence - 45) % 90

	best_guess_midpoint_singles = (best_guess_max_singles + best_guess_min_singles) / 2
	best_guess_midpoint_coincidence = (best_guess_max_coincidence + best_guess_min_coincidence) / 2

	fine_scan_half_range = 5


	# skip this check for now
	if False:
	# if abs(best_guess_min_singles - best_guess_min_coincidence) > 10:
		print("Error: The best guess for the minimum singles and coincidences are too far apart.")
		print(f"Best guess min singles: {best_guess_min_singles}")
		print(f"Best guess min coincidences: {best_guess_min_coincidence}")
		sys.exit(1)

	min_coincidence_angles = [i for i in np.linspace(best_guess_min_coincidence - fine_scan_half_range, best_guess_min_coincidence + fine_scan_half_range, 11)]
	max_coincidence_angles = [i for i in np.linspace(best_guess_max_coincidence - fine_scan_half_range, best_guess_max_coincidence + fine_scan_half_range, 7)]
	midpoint_coincidence_angles = [i for i in np.linspace(best_guess_midpoint_coincidence - fine_scan_half_range, best_guess_midpoint_singles + fine_scan_half_range, 5)]

	min_singles_angles = [i for i in np.linspace(best_guess_min_singles - fine_scan_half_range, best_guess_min_singles + fine_scan_half_range, 11)]
	max_singles_angles = [i for i in np.linspace(best_guess_max_singles - fine_scan_half_range, best_guess_max_singles + fine_scan_half_range, 7)]
	midpoint_singles_angles = [i for i in np.linspace(best_guess_midpoint_singles - fine_scan_half_range, best_guess_midpoint_singles + fine_scan_half_range, 5)]

	# angles = min_coincidence_angles + max_coincidence_angles + midpoint_coincidence_angles
	angles = min_singles_angles + max_singles_angles + midpoint_singles_angles

	run_characterisation(waveplate_name, singles_detector_name, coincidence_detector_name, angles, waveplate_dir)
	singles_fit_params, coincidences_fit_params = analyse_waveplate_data(waveplate_name, waveplate_dir)

	best_guess_max_coincidence = (coincidences_fit_params[1] * 180 / np.pi) + 22.5 % 90
	best_guess_max_coincidence = best_guess_max_coincidence.round(2)

	best_guess_max_singles = (singles_fit_params[1] * 180 / np.pi) + 22.5 % 90
	best_guess_max_singles = best_guess_max_singles.round(2)

	# update the value in the waveplate.json file
	waveplate_json_path = os.path.join(repo_root, "waveplates", "waveplates.json")
	with open(waveplate_json_path, 'r') as f:
		waveplates = json.load(f)

	if waveplate_name not in waveplates:
		waveplates[waveplate_name] = {}

	# waveplates[waveplate_name]['fast_axis'] = best_guess_max_coincidence
	waveplates[waveplate_name]['fast_axis'] = best_guess_max_singles

	with open(waveplate_json_path, 'w') as f:
		json.dump(waveplates, f, indent=4)

	# put the waveplate at the fast axis
	angleMove([best_guess_max_singles], [waveplate_name])

	print(f"Found fast axis at: {best_guess_max_singles}")
	print("Waveplate moved to fast axis")
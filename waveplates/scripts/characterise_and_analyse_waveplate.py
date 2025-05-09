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
	angles = [i for i in np.linspace(0, 180, 20)]

	run_characterisation(waveplate_name, singles_detector_name, coincidence_detector_name, angles, waveplate_dir)
	singles_fit_params, coincidences_fit_params = analyse_waveplate_data(waveplate_name, waveplate_dir)

	# fine
	if waveplate_type == 'hwp':
		print(singles_fit_params[1] * 180 / np.pi)
		best_guess_max_singles = (singles_fit_params[1] * 180 / np.pi) + 22.5 % 90 
		best_guess_max_coincidence = (coincidences_fit_params[1] * 180 / np.pi) + 22.5 % 90

		best_guess_min_singles = (best_guess_max_singles - 45) % 90
		best_guess_min_coincidence = (best_guess_max_coincidence - 45) % 90
	elif waveplate_type == 'qwp':
		best_guess_max_singles = (singles_fit_params[1] * 180 / np.pi) + 12.25 % 45
		best_guess_max_coincidence = (coincidences_fit_params[1] * 180 / np.pi) + 12.25 % 45

		best_guess_min_singles = (best_guess_max_singles - 22.5) % 45
		best_guess_min_coincidence = (best_guess_max_coincidence - 22.5) % 45

	if abs(best_guess_min_singles - best_guess_min_coincidence) > 10:
		print("Error: The best guess for the minimum singles and coincidences are too far apart.")
		print(f"Best guess min singles: {best_guess_min_singles}")
		print(f"Best guess min coincidences: {best_guess_min_coincidence}")
		sys.exit(1)

	min_singles_angles = [i for i in np.linspace(best_guess_min_singles - 5, best_guess_min_singles + 5, 10)]
	max_singles_angles = [i for i in np.linspace(best_guess_max_singles - 5, best_guess_max_singles + 5, 6)]
	angles = min_singles_angles + max_singles_angles

	run_characterisation(waveplate_name, singles_detector_name, coincidence_detector_name, angles, waveplate_dir)
	singles_fit_params, coincidences_fit_params = analyse_waveplate_data(waveplate_name, waveplate_dir)
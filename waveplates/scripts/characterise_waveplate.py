import time
import pandas as pd
from ttag_console import *
from move_plates import *
from pathos import multiprocessing as mp

sys.path.append(os.environ["TTAG"])
from ttag import *

sys.path.append(os.environ["STANDA"])
import standa as st

if getfreebuffer() == 0:
   	buf = TTBuffer(0)
else:
   	buf = TTBuffer(getfreebuffer() - 1)

if buf.getrunners() == 0:
	buf.start()

#----------------#
# --- Options -- #
#----------------#
STANDA_CONFIG_FILE_LOCATION = os.path.abspath("/home/jh115/emqTools/standa/8MPR16-1.cfg")
MEASUREMENT_TIME = 1 # seconds

COINCIDENCE_WINDOW = 1.0e-9 # seconds

def run_characterisation(waveplate_name, singles_detector_name, coincidence_detector_name, angles, data_dir):
	counts_df = pd.DataFrame(columns = ["angle", "singles", "coincidences"])

	data_file_name = "counts.csv"
	data_file_path = os.path.join(data_dir, data_file_name)

	if os.path.exists(data_file_path):
		print("Loading existing data.")
		counts_df = pd.read_csv(os.path.join(data_file_path))

	singles_detector_index = singles_detector_name - 1
	coincidence_detector_index = coincidence_detector_name - 1

	# Set up the coincidences
	detector_pattern = f"d{singles_detector_name} - d{coincidence_detector_name}"
	print("Detector pattern: %s" % detector_pattern)
	twofolds = nFold_create(detector_pattern, 2)

	keys = list(twofolds.keys())
	opts = [(key, MEASUREMENT_TIME, COINCIDENCE_WINDOW, twofolds[key][0], twofolds[key][1]) for key in keys]
	opts_size = len(opts)

	pool = mp.Pool(8)
	print('Multiprocessing started successfully.')

	for target_angle in angles:
		angleMove([target_angle], [waveplate_name])
		time.sleep(1 + MEASUREMENT_TIME)

		singles = buf.singles(MEASUREMENT_TIME)[singles_detector_index]
		coincidences = pool.map(lambda i: buf.multicoincidences(*opts[i][1::]), range(len(opts)))

		print("Singles: %s" % singles)
		print("Coincidences: %s" % coincidences)

		counts_df = pd.concat([counts_df, pd.DataFrame({
			"angle": [target_angle],
			"singles": [singles],
			"coincidences": [sum(coincidences)]
		})], ignore_index=True)

		counts_df.to_csv(data_file_path, index=False)

		print("Angle: %d" % (target_angle))
		print("Singles: %s" % singles)
		print("Coincidences: %s" % coincidences)
		print("")


if __name__ == "__main__":
	# get the wavdeplate name from the command line
	# get the main detector from the command line
	# get the coincidences detector from the command line
	if len(sys.argv) == 4:
		waveplate_name = sys.argv[1]
		singles_detector_name = int(sys.argv[2])
		coincidence_detector_name = int(sys.argv[3])
	else:
		print("Error: waveplate name, singles detector name and coincidence detector name not specified correctly.")
		print("Usage: python characterise_waveplate.py <waveplate_name> <singles_detector_name> <coincidence_detector_name>")
		print("eg: python characterise_waveplate.py ht 12 2")
		sys.exit(1)
	# ensure the required data directory exists
	repo_root = os.popen('git rev-parse --show-toplevel').read().strip()
	data_dir = os.path.join(repo_root, "waveplates", "data", waveplate_name)
	
	angles = [i for i in range(0, 180, 20)]
	run_characterisation(waveplate_name, singles_detector_name, coincidence_detector_name, angles, data_dir)
import time
import pandas as pd
from ttag_console import *
from move_plates import *
import matplotlib.pyplot as plt

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
DETECTOR_PATTERN = 'd12 - d2' # TODO: T - R
STANDA_CONFIG_FILE_LOCATION = os.path.abspath("/home/jh115/emqTools/standa/8MPR16-1.cfg")

MEASUREMENT_TIME = 1 # seconds
WAVEPLATE_NAME = "ht"

# ensure the required data directory exists
repo_root = os.popen('git rev-parse --show-toplevel').read().strip()
data_dir = os.path.join(repo_root, "waveplates", "data", WAVEPLATE_NAME)
if not os.path.exists(data_dir):
	os.makedirs(data_dir)

singles_df = pd.DataFrame(columns = ["angle", "singles"])

for i in range(9):
	angleMove([10*i], [WAVEPLATE_NAME])
	time.sleep(1 + MEASUREMENT_TIME)

	singles = buf.singles(MEASUREMENT_TIME)[11]

	singles_df = pd.concat([singles_df, pd.DataFrame({"angle": [10*i], "singles": [singles]})], ignore_index=True)	

	# output to csv as we go
	out_file_name = "singles.csv"
	out_file_path = os.path.join(data_dir, out_file_name)
	singles_df.to_csv(out_file_path, index=False)

	print("Angle: %d" % (10*i))
	print("Singles: %s" % singles)
	print("")

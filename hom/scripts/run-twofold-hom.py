'''Using S3 through the multi-param HOM setup'''

import os, sys, time, datetime
import numpy as np
import pandas as pd
from ttag_console import *
from move_plates import *
from pathos import multiprocessing as mp


#----------------#
# --- Options -- #
#----------------#
MOTORISED_STAGE_NAME = "dave"
DETECTOR_PATTERN = 'd4 - d12'
LABELS = ["TT"]

# Scan between DIP_POSITION - TRANSLATION_HALF_RANGE and DIP_POSITION + TRANSLATION_HALF_RANGE in steps of size STEP_SIZE
# and between DIP_POSITION - DETAIL_HALF_RANGE and DIP_POSITION + DETAIL_HALF_RANGE with additional accuracy 
# in steps of size DETAIL_STEP_SIZE
DIP_POSITION = 9.94 # mm
TRANSLATION_HALF_RANGE = 6 # mm
STEP_SIZE = 0.2 # mm
DETAIL_HALF_RANGE = 0.5 # mm
DETAIL_STEP_SIZE = 0.05 # mm

MEASUREMENT_TIME = 4 # seconds
COINCIDENCE_WINDOW = 1.0e-9 # seconds

STANDA_CONFIG_FILE_LOCATION = os.path.abspath("/home/jh115/emqTools/standa/8CMA06-25_15-Lin_G34.cfg")

#----------------#
# Run parameters #
#----------------#
def throw_parameter_error():
    print("Error: power not specified correctly.")
    print("Usage: python run-twofold-hom.py <temperature> <power>")
    print("eg: python run-twofold-hom.py 23.2C 101mW")
    sys.exit(1)

# get the run parameters from the command line
if len(sys.argv) > 1:
    try:
        temperature = sys.argv[1]
        power = sys.argv[2]
    except:
        throw_parameter_error()
else:
    throw_parameter_error()

# Validate the run parameters
# expect temperature in degrees Celsius like "20.0C"
if temperature[-1] != "C":
    throw_parameter_error()
# expect power in mW like "10mW"
if power[-2:] != "mW":
    throw_parameter_error()

try:
    temperature = float(temperature[:-1])
    power = float(power[:-2])
except ValueError:
     throw_parameter_error()

# check timetagger is running
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

# find motorised linear stage
usb_device = st.findDevices("usb")
usb_name = usb_device[MOTORISED_STAGE_NAME]
linear_stage = st.device(usb_name, STANDA_CONFIG_FILE_LOCATION)

start_time = datetime.datetime.now().strftime("%F--%Hh-%Mm")

scan_range = [DIP_POSITION - TRANSLATION_HALF_RANGE, DIP_POSITION + TRANSLATION_HALF_RANGE]
detail_range = [DIP_POSITION - DETAIL_HALF_RANGE, DIP_POSITION + DETAIL_HALF_RANGE]

# inclusive
coarse_points = np.arange(scan_range[0],scan_range[1] + STEP_SIZE, STEP_SIZE, )
detail_points = np.arange(detail_range[0],detail_range[1] + DETAIL_STEP_SIZE, DETAIL_STEP_SIZE)

scan_points = np.sort(np.unique(np.around(np.concatenate((coarse_points, detail_points)), decimals=2)))

linear_stage.goTo(scan_points[0])
time.sleep(1)

ccTwoFolds = nFold_create(DETECTOR_PATTERN,2)

keys = list(ccTwoFolds.keys())
opts = [(key, MEASUREMENT_TIME, COINCIDENCE_WINDOW, ccTwoFolds[key][0], ccTwoFolds[key][1]) for key in keys]
optsSize = len(opts)
phase = 0

pool = mp.Pool(8)
print('Multiprocessing started successfully.')

# TODO: pre-alloc array structure - numpy doesnt index zero to create this array?
data = np.zeros((len(scan_points), 16 + len(LABELS) + 2), dtype = 'object')

########## Data Aquisition ##########
for i, j in enumerate(scan_points):

    linear_stage.goTo(j)
    time.sleep(1 + MEASUREMENT_TIME)
    singles = buf.singles(MEASUREMENT_TIME)
    
    # stage position
    data[i, 0] = j 

    # singles are next 8, only the first 8 channels from tagger used
    data[i, 1:17] = singles[0:16]
    # acquire coincidences
    result = pool.map(lambda i: buf.multicoincidences(*opts[i][1::]), range(len(opts)))
    
    # all coincidences are the next 16 - this is not true for us. I think this element should just span 2 colums
    data[i, 17:(17 + len(LABELS))] = result

    # read time is last element
    data[i, (17 + len(LABELS))] = MEASUREMENT_TIME

df_labels = ['position','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16'] + LABELS + ['read_time']
df = pd.DataFrame(data, columns = df_labels)

end_time = datetime.datetime.now().strftime("%F--%Hh-%Mm")
repo_root = os.popen('git rev-parse --show-toplevel').read().strip()

file_name = "Sag3_" + start_time + "_" + end_time + "_%dmW_%sdeg.csv" %(power, temperature)
out_file_path = os.path.join(repo_root, "hom", "data", file_name)

print("Writing data to file: %s" % out_file_path)
df.to_csv(out_file_path)

linear_stage.goTo(DIP_POSITION)
print("Moved to dip pos")

# close all devices
linear_stage.closeDevice()

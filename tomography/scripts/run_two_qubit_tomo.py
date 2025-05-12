import os
import sys
import time

import pandas as pd
from ttag_console import *
from move_plates import *
from set_waveplate_angles import *

sys.path.append(os.environ["TTAG"])
from ttag import *

if getfreebuffer() == 0:
   	buf = TTBuffer(0)
else:
   	buf = TTBuffer(getfreebuffer() - 1)
	
if buf.getrunners() == 0:
	buf.start()

MEASUREMENT_TIME = 1  # seconds

DETECTOR_MAPPINGS = {
	'TT': 12,
	'RT': 4,
	'TR': 10,
	'RR': 2
}

wp = load_waveplates_from_config('waveplates.json')

tomo_t = TomographyController(
    name='tomo_t',
    quarter_waveplate=wp['qt'],
    half_waveplate=wp['ht']
)

tomo_r = TomographyController(
    name='tomo_r',
    quarter_waveplate=wp['qr'],
    half_waveplate=wp['hr']
)

print(['basis'] + [detector_name for detector_name in DETECTOR_MAPPINGS.keys()])

df = pd.DataFrame(columns = ['basis'] + [detector_name for detector_name in DETECTOR_MAPPINGS.keys()])

print("Starting two-qubit tomography...")

print("Setting tomography T to H")
tomo_t.set_label('H')
time.sleep(2) # wait for waveplates to move

time.sleep(1 + MEASUREMENT_TIME)
singles = buf.singles(MEASUREMENT_TIME)

new_row = pd.DataFrame([
	['H'] + [singles[DETECTOR_MAPPINGS[detector_name]-1] for detector_name in DETECTOR_MAPPINGS.keys()]
])
new_row.columns=df.columns
df = pd.concat([df, new_row], ignore_index=True)

print('Setting tomography T to V')
tomo_t.set_label('V')
time.sleep(2) # wait for waveplates to move

time.sleep(1 + MEASUREMENT_TIME)
singles = buf.singles(MEASUREMENT_TIME)

new_row = pd.DataFrame([
	['V'] + [singles[DETECTOR_MAPPINGS[detector_name]-1] for detector_name in DETECTOR_MAPPINGS.keys()]
])
new_row.columns=df.columns
df = pd.concat([df, new_row], ignore_index=True)

print(df)
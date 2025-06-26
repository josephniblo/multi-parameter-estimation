import time
from set_waveplate_angles import *

from ttag_console import *
from move_plates import *
from set_waveplate_angles import *
from analyze_single_qubit_tomo import *

sys.path.append(os.environ["TTAG"])
from ttag import *

# Prepare the ttag buffer
if getfreebuffer() == 0:
    buf = TTBuffer(0)
else:
    buf = TTBuffer(getfreebuffer() - 1)
    
if buf.getrunners() == 0:
    buf.start()


DETECTOR_MAPPINGS = {
    'TT': 12,
    'TR': 10,
    'RT': 4, 
    'RR': 2 
}

MEASUREMENT_TIME = 5 # seconds

wp = load_waveplates_from_config('waveplates.json')

df = pd.DataFrame()

for qwp in np.linspace(-3, 1, 5):
    for hwp in np.linspace(-5, 1, 5):
        wp['qr'].set_angle(qwp)
        wp['hr'].set_angle(hwp)

        # Wait for the angles to be set
        time.sleep(5.1)

        singles = buf.singles(MEASUREMENT_TIME)

        df = pd.concat([df, pd.DataFrame({
            'qwp': [qwp],
            'hwp': [hwp],
            'singles': [singles]
        })], ignore_index=True)

# Save the results to a CSV file
repo_root = os.popen('git rev-parse --show-toplevel').read().strip()
data_path = os.path.join(repo_root, 'tomography', 'analysis')
df.to_csv(os.path.join(data_path, 'tomo_t_extinction.csv'), index=False)
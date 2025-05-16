from concurrent.futures import ThreadPoolExecutor
from set_waveplate_angles import TomographyController, load_waveplates_from_config, set_tomo_labels, set_waveplate_angles

wp = load_waveplates_from_config('waveplates.json')

set_waveplate_angles(wp, {
    'hla': 0,
    'hlb': 0,
    'qla': 0,
    'qlb': 0,
    'hr': 0,
    'qr': 0,
    'ht': 0,
    'qt': 0
})

print("Waveplates set to 0 degrees.")

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

with ThreadPoolExecutor(max_workers=2) as executor:
    executor.submit(tomo_t.set_label, 'V')
    executor.submit(tomo_r.set_label, 'V')
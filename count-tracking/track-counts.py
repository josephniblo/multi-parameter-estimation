import sys
import os
import time
import matplotlib
matplotlib.use('TkAgg')  # Must be set before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd
import pathos.multiprocessing as mp
from ttag_console import *

sys.path.append(os.environ["TTAG"])
from ttag import *

# Prepare the ttag buffer
if getfreebuffer() == 0:
    buf = TTBuffer(0)
else:
    buf = TTBuffer(getfreebuffer() - 1)
    
if buf.getrunners() == 0:
    buf.start()

MEASUREMENT_TIME = 1  # seconds

DETECTOR_MAPPINGS = {
    'TT': 12,
    'TR': 10,
    'RT': 4,
    'RR': 2
}

DETECTOR_PATTERN = 'd4, d2 - d12, d10'
COINCIDENCE_WINDOW = 1.0e-9

twoFolds = nFold_create(DETECTOR_PATTERN, 2)
keys = list(twoFolds.keys())
opts = [(key, MEASUREMENT_TIME, COINCIDENCE_WINDOW, twoFolds[key][0], twoFolds[key][1]) for key in keys]

current_time = time.time()
file_name = f"singles_{current_time}.csv"

df = pd.DataFrame(columns=['timestamp', '2', '4', '10', '12', 'TT', 'TR','RT','RR'])

# Setup plot
plt.ion()
# 2 subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 8))
axs[0].set_title('Singles over time')
axs[1].set_title('Coincidences over time')
ax = axs[0]
ax.set_xlabel('timestamp')
ax.set_ylabel('Counts')

pool = mp.Pool(processes=4)
try:
    while True:
        time.sleep(MEASUREMENT_TIME + 0.1)

        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

        singles = buf.singles(MEASUREMENT_TIME)
        relevant_singles = [singles[i] for i in [1,3,9,11]]

        coincidences = pool.map(lambda i: buf.multicoincidences(*opts[i][1::]), range(len(opts)))

        # Append the singles to the dataframe
        new_row = pd.DataFrame([[t] + relevant_singles + coincidences], columns=df.columns)
        df = pd.concat([df, new_row], ignore_index=True)

        print(relevant_singles + coincidences)


        # Save the dataframe to a CSV file
        repo_root = os.popen('git rev-parse --show-toplevel').read().strip()
        data_path = os.path.join(repo_root, 'count-tracking', 'data')
        filename = os.path.join(data_path, file_name)
        df.to_csv(filename, index=False)

        # Plot only the last 120 seconds of data
        plot_df = df.copy()
        if len(plot_df) > 120:
            plot_df = df.iloc[-120:]

        # Update plot
        ax.clear()
        axs[1].clear()
        df.plot(x='timestamp', y=['2', '4', '10', '12'], ax=ax, title='Singles over time')
        df.plot(x='timestamp', y=['TT', 'TR', 'RT', 'RR'], ax=axs[1], title='Coincidences over time')
        plt.draw()
        plt.pause(0.1)
finally: 
    pool.close()
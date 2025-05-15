import sys
import os
import time
import matplotlib
matplotlib.use('TkAgg')  # Must be set before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd

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

current_time = time.time()
file_name = f"singles_{current_time}.csv"

df = pd.DataFrame(columns=['timestamp', '2', '4', '10', '12'])

# Setup plot
plt.ion()
fig, ax = plt.subplots()

while True:
    time.sleep(MEASUREMENT_TIME + 0.1)

    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

    singles = buf.singles(MEASUREMENT_TIME)
    relevant_singles = [singles[i] for i in [1,3,9,11]]

    # Append the singles to the dataframe
    new_row = pd.DataFrame([[t] + relevant_singles], columns=df.columns)
    df = pd.concat([df, new_row], ignore_index=True)
    print(relevant_singles)

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
    df.plot(x='timestamp', y=['2', '4', '10', '12'], ax=ax, title='Singles over time')
    plt.draw()
    plt.pause(0.1)

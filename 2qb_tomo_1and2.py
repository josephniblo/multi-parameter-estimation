
# import a bunch of things
import os
import sys
import time
import datetime
import itertools
import numpy as np
import pandas as pd 

from ttag_console import *
from movePlates import *
import pathos.multiprocessing as mp

# define basis labels 
basis = ['Z', 'X', 'Y']

# define the number of settings for each qubit state (i.e., 3 for MUB)
n_sets = [3, 3]

# define measurement setting labels (as used in allPlatesMove.py)
proj_a = ['h', 'd', 'r'] # standard projective measurement settings for MUB tomography
#proj_b = ['v', 'a', 'l'] # Flipped basis

# iterate measurement settings for tomo, and produce
meas_a = [','.join(ii) for ii in itertools.product(*[proj_a for i in n_sets])]
basis_lbl = [''.join(ii) for ii in itertools.product(*[basis for i in n_sets])]

# measurement parameters
t_meas = 3
samples = 1

## the follow snippit was used to flip detectors for half measurement time
# t_acquire = t_meas / samples / 2

t_acquire = t_meas / samples
t_window = 1.0e-9
repetitions = 1
power = 20
# define singles channels ordering, and labels for upkeep of file
singles_lbls = ['d2', 'd4', 'd10', 'd12']
# grab channel values, remember python indices have -1
singles_order = np.array([0, 1, 2, 3]) 

# generate keys to iterate over list of detector patterns
twoFolds = nFold_create('d4, d2 - d12, d10', 2)
keys = list(twoFolds.keys())
opts = [(key, t_acquire, t_window, twoFolds[key][0], twoFolds[key][1]) for key in keys]
optsSize = len(opts)

# once all parameters are defined, start an instance of multiprocessor
pool = mp.Pool(8)
print('Multiprocessing started successfully.')

for rep in range(repetitions):
    # initialise empty arry to hold result
    resultA = np.zeros([9, 11], dtype = object)

    # grab timestamp for filename
    startTime = datetime.datetime.now().strftime("%F--%Hh-%Mm")
    
    for m in range(len(meas_a)):

        print('- - - - - + + + + + - - - - -')
        print('Measuring:', basis_lbl[m])
        print('- - - - - + + + + + - - - - -')
        
        # move waveplates, e.g., using labelMove('h,h', '3,4') to set HH on waveplate stages 3 and 4
        labelMove(meas_a[m], '1, 2')

        # acquire data in loops, initialise arrays for holding data
        t_spent = 0
        
        singl_a = np.zeros(16)
        
        coinc_a = np.zeros(optsSize)
        
        
        # for first loop, wait for acquisition time, with 2 seconds buffer for plate rotation
        time.sleep(t_acquire + 2.0)

        for ii in range(samples):
            # get current time at start of loop
            t_ini = time.time()
            
            # collect data (by adding newly acquired data with existing array)
            singles = buf.singles(t_acquire)

            coinc = pool.map(lambda i: buf.multicoincidences(*opts[i][1::]), range(len(opts)))
            
            singl_a += singles
            
            coinc_a += coinc # coincs is an array with 4 positions for each outcome "TT", "TR", "RT", "RR"

            #When using proj_b need to arrange TT<->RR, TR,<->RT 

            print(coinc_a) # for debugging purposes
            
            # check how long data collection took to determine how much time remaining for sleep
            t_tot = time.time() + t_ini
            t_spent += t_tot

            # sleep for remaining time unless exceeds nominal acquisition time (print warning)
            if t_tot > t_acquire:
                print('Warning, acquisition time took longer than sampling time, consider reducing sampling time or reducing rates.')
                continue
            else:
                time.sleep(t_acquire - t_tot)

        
        resultA[m, ::] = [basis_lbl[m][::]] + list(singl_a[singles_order]) + list(coinc_a[::]) + [t_meas] + [t_spent]

        
        print(resultA[m])
        
    
    # prepare dataframe
    df_lbls = ['basis'] + list(singles_lbls) + keys + ['t_meas'] + ['t_spent']
    df_A = pd.DataFrame(resultA, columns = df_lbls)
    
    # grab timestamp for filename
    endTime = datetime.datetime.now().strftime('%d--%Hh-%Mm')
    
    # format file name

    cwd = os.getcwd()
    fName = "/" + startTime + "_" + endTime + "_2qb_tomo_sagnac2_%dmW" %(power)
    
    # export dataframe to csv file
    df_A.to_csv(cwd + fName + '_A.csv')
 
# clean up
st.closeDevice('Axis1')
st.closeDevice('Axis2')
st.closeDevice('Axis3')
st.closeDevice('Axis4')
st.closeDevice('Axis5')
st.closeDevice('Axis6')
st.closeDevice('Axis7')
st.closeDevice('Axis8')
pool.close()

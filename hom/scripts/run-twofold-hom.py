
'''Using S3 through the multi-param HOM setup'''

# import a bunch of things
import os, sys, time, datetime
import numpy as np
import pandas as pd
from ttag_console import *
from movePlates import *
from pathos import multiprocessing as mp

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

########## Rotation & Linear Stage ##########

# first set projection stages after fusion gate to DD 
# 3 and 4 are in the tomo/measuremetn stage
#labelMove('d d', '1 2')

# find motorised linear stage
usbDevice = st.findDevices("usb")
usbName = usbDevice["dave"]

# assign linear stage
linearStage = st.device(usbName, os.path.abspath("/home/jh115/emqTools/standa/8CMA06-25_15-Lin_G34.cfg"))

# file creation parameters
startTime = datetime.datetime.now().strftime("%F--%Hh-%Mm")
temperature = '20.5'
power = 100

t_meas = 2
# first need to find the rough dip position  
#dipPos = 10.46 #S3
#dipPos = 7.3 #S2
dipPos = 12 #S1
shift = 11.4
# scan parameters
scanRange = [dipPos - shift, dipPos + shift]
detail2 = dipPos + shift
detail1 = dipPos - shift
detail = [detail1, detail2]

# scan parameters for detailed range
# scanRange = [dipPos - 4, dipPos + 4]
#detail2 = dipPos + 0.31
#detail1 = dipPos - 0.31
# detail = [detail1, detail2]

stepSize = 0.2
#scanRange=[1.0,24.0]
outPoints = np.arange(scanRange[0],scanRange[1], stepSize)
detailPoints = np.arange(detail[0],detail[1], stepSize)

scanPoints = np.sort(np.unique(np.around(np.concatenate((outPoints, detailPoints)), decimals=2)))

linearStage.goTo(scanPoints[0])
time.sleep(1)

# define detection pattens

#ccTwoFolds = nFold_create('d4, d2 - d14, d11',2)
ccTwoFolds = nFold_create('d4 - d12',2)
# grab labels generated for the two-fold patterns
labels = ['TT'] #[str(i) for i in ccTwoFolds.keys()]

# define measurement parameters
t_window = 1.0e-9

keys = list(ccTwoFolds.keys())
opts = [(key, t_meas, t_window, ccTwoFolds[key][0], ccTwoFolds[key][1]) for key in keys]
optsSize = len(opts)
phase = 0

pool = mp.Pool(8)
print('Multiprocessing started successfully.')

# pre-alloc array structure - numpy doesnt index zero to create this array?
data = np.zeros((len(scanPoints), 16 + 1 + 2), dtype = 'object')

# adding the phase due to PBS

########## Data Aquisition ##########

for i, j in enumerate(scanPoints):

    linearStage.goTo(j)
    time.sleep(1 + t_meas)
    singles = buf.singles(t_meas)
    
    # stage position
    data[i, 0] = j 

   # singles are next 8, only the first 8 channels from tagger used
    #data[i, 1:9] = singles[0:16] #data[i: 1:17]
    data[i, 1:17] = singles[0:16] #data[i: 1:17]
    # acquire coincidences
    result = pool.map(lambda i: buf.multicoincidences(*opts[i][1::]), range(len(opts)))
    
    # all coincidences are the next 16 - this is not true for us. I think this element should just span 2 colums
    data[i, 17:18] = result

    # read time is last element
    data[i, 18] = t_meas

# formatting output filename
workingDir = os.getcwd()

# file column header labels - labels is automatically generated
dfLabels = ['position','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16'] + labels + ['read_time']

# create Pandas dataframe to hold data
df = pd.DataFrame(data, columns = dfLabels)

# grab end time to format file name
endTime = datetime.datetime.now().strftime("%F--%Hh-%Mm")
fName = "/Sag3_" + startTime + "_" + endTime + "_%dmW_%sdeg.csv" %(power, temperature)

# write data to file
df.to_csv(workingDir + fName)


linearStage.goTo(dipPos)
print("Moved to dip pos")


# close all devices
linearStage.closeDevice()


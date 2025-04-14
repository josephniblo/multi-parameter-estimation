import os, sys, time, datetime,serial
import numpy as np
import pandas as pd
#import serial.tools.list_ports
import subprocess as sp
import matplotlib.pyplot as plt

sys.path.append(os.environ["TTAG"])
from ttag import *


if getfreebuffer() == 0:
   	buf = TTBuffer(0)
else:
   	buf = TTBuffer(getfreebuffer() - 1)

if buf.getrunners() == 0:
	buf.start()

#scanRange = [-1060e-9, 1060e-9]
scanRange = [-10e-9, 10e-9]
#scanRange = [1490e-9, 1515e-9]
step = buf.resolution*1

scanPoints = np.arange(scanRange[0],scanRange[1],step)

channels = [1, 9]

#channels = [6, 8]
cc = []

rt = 1

# 47.187 8maX
# -53.00
#-12.188e-9 possible 8
x=[]
for i in scanPoints:
    x.append(i)
    #cc.append(buf.multicoincidences(rt,step,channels,[0,47.969e-9, 12.812e-9,
	#					      i]))
    cc.append(buf.multicoincidences(rt,step,channels,[0,i]))
    print('%.3f - %d'%(i*10e8,cc[-1]))
delay = np.round(scanPoints[np.argmax(cc)]*1e9,2)
plt.plot(scanPoints, cc)
plt.title(f"Channels: {channels}, Delay: {delay} ns" )
plt.show()

print("Delay position", delay)

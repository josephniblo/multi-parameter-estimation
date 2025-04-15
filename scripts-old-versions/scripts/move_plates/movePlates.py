'''

Python script for terminal/console use: controlling waveplates.

Dictionaries to allay confusion by tying stage address to a waveplate reference key, along with 
information such as: stage type, i.e., HWP or QWP; and zero angles.

This requires upkeep of dictionary:
   If more information is to be contained within dictionary in future, it should be added to back
   of entries to minimise disruption of codes.

==================================================================================================
STANDARD USE CASE
==================================================================================================
$ labelMove(<SETTINGS>, <STAGES>)
<SETTINGS> - list projections using standard polarisation labels (h,v,d,etc.).
<STAGES> - list the stages to match number of settings, also stage order can be set in this way.

To measure in H/V moving HWP and QWP of stage 1, type
$ labelMove('h', '1')

To measure in H/V of stage 3, type
$ labelMove('h', '3')

To measure in H/V in all bases, type
$ labelMove('h, h, h, h', '1, 2, 3, 4')
==== OR ====
$ labelMove('h, h, h, h')
    NB: default value if second parameter not specified is ordered 1,2,3,4 (to be revised in future)

To measure in D/A in the fusion gate, but H/V in tomo stages, type
$ labelMove('d, d, h, h', '2, 1, 3, 4')
    NB: in strongBellTest experiments the stages were mixed up hence 1-2 are swapped

==================================================================================================
REMAINING ITEMS (FOR FUTURE UPDATES):
==================================================================================================
 X[partial completion] allow arbitrary calls to specify different stage combinations in code, without hard coding
 - allow inport/export of plate dictionary
 - create a class for modularity in future codes
 - include if __name__ == '__main__' for direct use in console with loops
 - update to allow access to second mainframe
 - look into proper method for closing/re-opening connection to Standa mainframe
 - (optional) verify the SIC basis
 

'''

# ================================================================================================
# import a bunch of things
# ================================================================================================

import sys
import os
import time
import numpy as np
import re

# ================================================================================================
# when run directly, call Standa mainframe(s), get list of available stages
# ================================================================================================
# if __name__ == '__main__':

try:
    # os.environ is a dict, get STANDA library path (assumed to be set correctly)
    standaPath = os.environ['STANDA']
except KeyError:
    print('STANDA is not a valid entry in os.environ, check system environment variables has entry for STANDA')
#standaPath = '~/emqTools/standa'

# using the path stored in os.environ, load standa module
sys.path.append(standaPath)

# import Standa module
import standa as st
print('import successful')
mFrame1 = st.findDevices('10.42.0.20', st.current_mainframe_config['10.42.0.20']) 

devList1 = [str(i) for i in mFrame1.keys()]
print('Devices found on mainframe 1: ' + str(devList1))

# define configuration files for rotation stages
confPath = os.path.abspath('/home/jh115/emqTools/standa/8MPR16-1.cfg')
print('confPath', confPath)
# inialise dictionaries for plateList and qubitList
plateList = {}
qubitList = {}


if len(mFrame1) == 8:
    # assuming 8 axes available, assign waveplate reference with device address, config file and zeros 
    #+90 degrees to align fast and slow axis
    plateList['h1'] =  ['hwp', st.device(mFrame1['Axis2'], confPath), 67.55]
    plateList['h2'] =  ['hwp', st.device(mFrame1['Axis4'], confPath), 72.54]
    plateList['h3'] =  ['hwp', st.device(mFrame1['Axis6'], confPath), 2.9]
    plateList['h4'] =  ['hwp', st.device(mFrame1['Axis8'], confPath), 15.30]
    plateList['q1'] =  ['qwp', st.device(mFrame1['Axis1'], confPath), 91.06]
    plateList['q2'] =  ['qwp', st.device(mFrame1['Axis3'], confPath), 64.05+90] 
    plateList['q3'] =  ['qwp', st.device(mFrame1['Axis5'], confPath), 57.79]
    plateList['q4'] =  ['qwp', st.device(mFrame1['Axis7'], confPath), 27.27+90]

    # dictionary to assign tomography waveplate sets following the qubit ordering
    qubitList['1'] = [plateList['h1'], plateList['q1']]
    qubitList['2'] = [plateList['h2'], plateList['q2']]
    qubitList['3'] = [plateList['h3'], plateList['q3']]
    qubitList['4'] = [plateList['h4'], plateList['q4']]

else:
    print('unable to assign stages of mainframe 1 (currently hard-coded to need all 8 axes available).')



# build dictionary for generalised waveplate angles, HWP setting first, then QWP setting
plateSettings = {
    'h' : [0, 0],
    'v' : [45, 0],
    'd' : [22.5, 45],
    'a' : [-22.5, 45],
    'r' : [45, 45],
    'l' : [0, 45],
    'sic1' : [20.0661, 22.50],
    'sic2' : [42.5661, -22.50],
    'sic3' : [-42.5661, 22.50],
    'sic4' : [-20.0661, -22.50]
    }

# ================================================================================================
# code snippit by PaulMcG, iterative checking for duplicates in list, aborts if duplicate found
# ================================================================================================
def allUnique(x):
    # define an empty list to store elements as checks performed
    seen = list()
    return not any(i in seen or seen.append(i) for i in x)


# ================================================================================================
# main code: move waveplates in sets to specific settings
# ================================================================================================
def labelMove(m, n = '1,2,3,4'):
    # prepare string inputs into correct format, splitting elements by delimiters
    # m-string for projection assignment
    strProj = re.split(',|;|:|, |; |: | |\n', m)
    # n-string for stages (can be partially filled)
    strStage = re.split(',|;|:|, |; |: | |\n', n)

    # get rid of null elements due to excessive spacing from string
    strProj = list(filter(None, strProj))
    strStage = list(filter(None, strStage))

    # create empty dictionary called stageList
    stageList = {}
    # check number of stages in n-string is less than the number of stages present
    if len(strStage) <= len(qubitList.keys()):
        # if so, proceed to building dictionary of stages for run
        for ii in strStage:
            # note that elements in strStage are strings, and we need integers for iterables
            stageList[ii] = qubitList[ii]

    # check the input string length matches number of qubit stages
    if len(strProj) == len(stageList.keys()):
        pass
    elif len(strProj) < len(stageList.keys()):
        print('\n Insufficient settings listed, only first n-specified settings will be applied.\n')
    elif len(strProj) > len(stageList.keys()):
        strProj = strProj[0:len(stageList.keys())]
        print('\n Too many settings specified, only first n-specified settings will be used.\n')
    
    # check the qubitList uniquely defines waveplates
    if allUnique([stageList[i] for i in stageList.keys()]) == True:
        print('device assignment successful.')
    elif allUnique([stageList[i] for i in stageList.keys()]) == False:
        print('\n \n WAVEPLATE ASSIGNMENT DUPLICATION DETECTED, CODE MAY NOT WORK AS INTENDED. \n \n')

    # check each input string corresponds to a valid key in plateSetting dictionary
    for el in range(len(strProj)):
        if strProj[el] in plateSettings:
            print('Setting command recognised')
            pass
        else:
            print('Warning! Setting command <<' + strProj[el] + '>> not recognised, defaulting to H')
            strProj[el] = 'h'

    # create empty arrays for the plate address list, and angle list
    angles = []
    plates = []

    # enumerate up to the number of settings specified
    for m in range(len(strProj)):
        # grab stage handle from qubit list, starting from first key
        # stage = stageList[list(qubitList.keys())[m]]
        stage = stageList[strStage[m]]

        for n in range(len(stage)):
            # update the plate list with current stage address
            plates.append(stage[n][1])

            # to update angle list, first grab zero angle for stage
            wpZero = stage[n][2]

            # next, determine if stage is a HWP or QWP
            if stage[n][0] == 'hwp':
                # grab first angle setting from plateSettings dictionary
                offsetAngle = plateSettings[strProj[m]][0]
            elif stage[n][0] == 'qwp':
                # grab second angle setting from plateSettings dictionary
                offsetAngle = plateSettings[strProj[m]][1]

            # update angle list for current stage, with HWP or QWP setting according to basis
            angles.append(wpZero + offsetAngle)
            
            # print stage assignment, setting and movement status
            print('Tomo[' + str(m+1) + '], ' + stage[n][0] + ' moving to ' + 
            strProj[m] + ' [ZeroAngle: ' + str(wpZero) + ', SetAngle: ' + '%.2f' %(wpZero + offsetAngle) +']')
    
    # move waveplates in one go, using list of stage addresses and their relative angles
    st.goToMulti(plates, angles)

# ==================================================================================================================
# direct assignment of waveplate and angle values, using tuples only
# ==================================================================================================================
def angleMove(m=[0, 0], n = ['h1', 'q1']):
    # create empty arrays for the plate address list, and angle list
    angles = []
    plates = []

    # need to convert the list of stages from n, into the stage addresses
    # similarly, add the list of angls offsets from m, w.r.t. zero angles
    for i in range(len(n)):
        plates.append(plateList[n[i]][1])
        angles.append(m[i] + plateList[n[i]][2])

        # print stage assignment, setting and movement status
        print('Stage: [' + str(n[i]) + '], ' + ' moving to [ ' + '%.2f' %(m[i] + plateList[n[i]][2]) + ']')
    
    # # debugging purposes
    print(plates)
    print(angles)
    
    # move waveplates in one go, using list of stage addresses and their relative angles
    st.goToMulti(plates, angles)
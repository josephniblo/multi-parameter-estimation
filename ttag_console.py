'''

Python script for terminal/console use.
Use WHILE loops and CATCH-EXCEPTION to enable continuous operation.

To generate N-fold channel and delay patterns, use dictionaries to
numerate all entries.

REMAINING ITEMS:
 [-] integrate find delays subroutine to update delays in dictionary automatically
 [-] allow import/export of an external dictionary for holding channel number and delays
     this will cater custom delays set by experimental configurations
     For example definitions: nFold_export() and nFold_import()
 [X] include if __name__ == '__main__' for direct use
 [-] create a class structure for modular use: ttagConsole should return an array with all
     counting information; this should include the count rates (coincidences, and singles)
     as well as the setup of the detection pattersn for future verification of the results
 [-] add a method for detecting multi-allocation of detection
 [-] add sum of all rows after each print (in either two-folds, or four-fold)
 [-] 


The script requires the termcolor library, it might need additional conda install termcolor
when run on new machines

'''



# import a bunch of stuff
import sys, os, time, numpy, re
import itertools
import pathos.multiprocessing as mp
from termcolor import colored

# load Ttag library
sys.path.append(os.environ['TTAG'])
from ttag import *

# initiate Ttag buffer
if getfreebuffer() == 0:
    buf = TTBuffer(0)
else:
    buf = TTBuffer(getfreebuffer() - 1)

# check if Ttag started
if buf.getrunners() == 0:
    buf.start()

# ================================================================================================
# create standard dictionaries to reference counting electronics by channel
# each dict. element is identifiable as { 'name' : [ttag ch. number, delay time (secs)] }
# ================================================================================================

ch = {
    'd01':  [0,   0e-9,   0.10],
    'd02':  [1,   0e-9,   0.30],
    'd03':  [2,   0e-9,   0.10],
    'd04':  [3,   0e-9,   0.10],
    'd05':  [4,   0e-9,   0.10],
    'd06':  [5,   3.4e-9,   0.10],
    'd07':  [6,   3.4e-9,   0.10],
    'd08':  [7,   1.9e-9,   0.10],
    'd09':  [8,   0.00e-9,   0.10],
    'd10':  [9,   50.3e-9,   0.10],
    'd11':  [10,  64.4e-9,   -0.03],
    'd12':  [11,  7.5e-9,  -0.05],
    'd13':  [12,  0.00e-9,  -0.01],
    'd14':  [13,  47.2e-9,   0.50],
    'd15':  [14,  0.00e-9,   0.10],
    'd16':  [15,  0.00e-9,   0.10]
    } # 11 April 2025



#================================================================================================
# allow string input to call elements from the main dictionary and return a tuple with the 
# channel address and delays to be used with buf.multicoincidences(t_meas, t_wind, [channels], [delays])
# ================================================================================================
def interpLine(input):   
    # split input string into a list delimitered by comma, semicolon, colon, or space
    input = re.split(',|;|:|, |; |: | |\n', input)
    
    # eliminate null elements from list
    input = list(filter(None, input))
    
    # for each element in list, split string on integer: likely the channel number user is trying specify
    el = [re.split('(\d+)', i) for i in input]
    
    # ensure formating: 'd##' where ## is a 2 digit padded integer
    input = ['d'+i[1].rjust(2, '0') for i in el]

    # initialise variables for returned channel tuple
    channels = []
    delays = []

    # for each element in user specified list, read channel information
    for item in input:
        channels.append(ch[item][0])
        delays.append(ch[item][1])

    # create channel information tuple suitable for ttag multicoincidence code
    output = [channels, delays]
    
    return output

# ================================================================================================
# generate dictionary for two-folds, user specifies detectors for each stage separated by dash
# NB: superceded by nFold_create (see below)
# ================================================================================================
def twoFold_create(input):
    # split string on dash
    input = re.split('-', input)
    
    # check there are two elements after split
    if len(input) == 2:
        # process stage 1, split list of detector(s) on delimeter
        s1 = re.split(',|;|:|, |; |: | |\n', input[0])

        # process stage 2
        s2 = re.split(',|;|:|, |; |: | |\n', input[1])

        # get rid of empty elements
        s1 = list(filter(None, s1))
        s2 = list(filter(None, s2))

        # create empty dictionary
        twoFoldDict = {}

        # create labels for dictionary convention
        lbls = ['T', 'R']

        for ii in range(len(s1)):
            for ij in range(len(s2)):
                twoFoldDict[str(lbls[ii]+lbls[ij])] = interpLine(str(s1[ii] + ', ' + s2[ij]))

        # return populated dicationary
        return twoFoldDict

    # if number of stages was incorrectly specified, do nothing and print warning. Either too many stages;
    elif len(input) > 2:
        print('Too many stages indicated for a two fold, check string again. Stages are separated by dash, channels by comma.')
    # Or not enough stages.
    elif len(input) < 2:
        print('Not enough stages indicated for two fold, check string again. Remember to separate stages by dash.')


# ================================================================================================
# generate an n-fold detector pattern dictionary, if n unspecified, assumes n=2
# ================================================================================================
def nFold_create(input, n = 2):
    # split string on dash
    input = re.split('-', input)

    if len(input) == n:
        # process stages, split list of detector(s) on delimeter
        s = [list(filter(None, re.split(',|;|:|, |; |: | |\n', i))) for i in input]         
        
        # create empty dictionary
        nFoldDict = {}
    
        # create labels for dictionary convention
        lbls = ['T', 'R']
        
        # list of the number of detectors per qubit (each element can be either 1 or 2)
        lengths = list(map(len, s))
        
        # generate the keys of the dictionary, in the form of TTT, TTR etc...
        # the inner list comprehension is used for replacing 1 with T and 2 with T,R
        # itertools product creates all the combinations of detectors
        # the outer list comprehension format the strings correctly
        keys = [''.join(ii) for ii in itertools.product(*[lbls[0:1] if i==1 else lbls for i in lengths])]

        # generate the values of the dictionary applying interpLine to each combination of stages
        values = list(map(interpLine, [' , '.join(i) for i in list(itertools.product(*s))]))
        
        # generate populated dicationary        
        nFoldDict = dict(zip(keys, values))

        # return populated dicationary
        return nFoldDict

    # if number of stages was incorrectly specified, do nothing and print warning. Either too many stages;
    elif len(input) > n:
        print('Too many stages indicated for the n-fold, check string again. Stages are separated by dash, channels by comma.')
    # Or not enough stages.
    elif len(input) < n:
        print('Not enough stages indicated for n-fold, check string again. Remember to separate stages by dash.')


# def coincidences(input, readTime = 1.00):
# 	# grab coincidences specified by channel combo and readtime, if no readtime specified default to 1.0 sec
# 	coincidences = buf.multicoincidences(t_meas, t_window, input[0], input[1])


# ====================================================================================================
# if script is run directly, use while-loop and catch-exception to enable continuous operation
# ====================================================================================================
if __name__ == '__main__':

    # default parameters for TTag operation
    t_meas = 1.0
    t_window = 1.00e-9

    # outer loop; if counter is interupted go back to user selection
    while True:
        
        # ============================================================================================
        # user selection
        # ============================================================================================
        try:
            opt = input('||To begin counting enter an option, i.e., number. ' + 
                'If invalid option entered, will default to [1]. ||\n' + 
                '||[1] Singles, all channels [1], [2], [3], [4] - [5], [6], [7], [8] - [...]                        ||\n' + 
                '||[2] Full four-fold display, for each fusion gate, FG1 then FG2                                   ||\n' +
                '||[3] Custom input via string (further details when prompted)                                      ||\n' +
                '||[4] Two-folds for Sagnac 1, 2 through fusion X                    ||\n' +
                '||[5] Two-folds hard-coded for: Sagnacs 1 + 2, TT+RR, TR-RT                                        ||\n' +
                '||[6] Four-folds of a specialised kind: TBA                                                        ||\n \n' + 
                'Enter a number: ')
            
            # check user selection was valid, default to singles otherwise
        except SyntaxError:
            print('Really? Too lazy to enter some-41.0thing... defaulting to [1]')
            opt = '1'
        except NameError:
            print("Please don't enter garbage, it confuses Python... defaulting to [1]")
            opt = '1'

        # parse user selection
        if opt == '1':
            # for singles, no further action requires
            pass
        elif opt == '2':
            # for four-folds, create the dictionary
            #fourFolds = nFold_create('d7, d6 - d8, d5 - d1, d2 - d3, d4', 4)
            # fourFolds = nFold_create('d2, d13 - d4, d12 - d3, d10 - d1, d7', 4)
            # fourFolds = nFold_create('d2, d13 - d4, d12 - d3, d10 - d5, d6', 4)
            fg1 = nFold_create('d4, d2 - d12, d10 - d1, d7 - d8, d6', 4)
            #fg2 = nFold_create('d4, d10 - d9, d6 - d5, d8 - d1, d7', 4)

            # for each two-fold dictionary update keys to something meaningful
            for keys in list(fg1.keys()):
                fg1['1' + str(keys)] = fg1.pop(keys)
                #fg2['2' + str(keys)] = fg2.pop(keys)
                
            fourFolds = {}
            #fourFolds = {**fg1, **fg2}
            fourFolds = {**fg1}
            print('Channel allocation successful')
            
            # enumerate keys in dictionary
            keys = list(fourFolds.keys())
            opts = [(key, t_meas, t_window, fourFolds[key][0], fourFolds[key][1]) for key in keys]
            optsSize = len(opts)
            
            # start multiprocessor for fast acquisition of multifold coincidences
            pool = mp.Pool(8)
            
            print('Multiprocessing started successfully.')

            # prepare labels for printout
            labels = [str(i).rjust(5, ' ') for i in fourFolds.keys()]

            # prepare structure for terminal printout: this gets updated in main code so labels do not move
            # allResults = [[0] * 16] * 20
            allResults = [[0] * (16+16)] * 20
            
        elif opt =='3':
            # for custom string to be entered by user
            userInput = input('||Enter detector pattern as string.                                         ||\n' +
            '||Separate each channel by delimiter {, ; or :}.                            ||\n' +
            '||Separate a group of channels by a dash {-}                                ||\n' +
            '||E.g., two-folds for channels 3 and 4, type: d3, d4                        ||\n' +
            '||E.g., four-folds for channels 5, 6, 7, and 8, type: d5, d6, d7, d8        ||\n' +
            '||E.g., two-folds for channel pair [1, 2] and [3, 4], type: d1, d2 - d3, d4 ||\n \n' +
            'Enter string option here: ')
            
            # split user input string on dash
            userList = userInput.split('-')
            
            # obtain channel number and delays in tuples
            try:
                channels = [interpLine(i) for i in userList]
            except KeyError:
                print('One or more channel number exceeds 1-16. Defaulting to [1].')
                opt = '1'
            except IndexError:
                print('One or more channel entry was not recognised. ', 
                'Accepted characters are {d, D, 0-9} and delimiters {, ; :}. Defaulting to [1].')
                opt = '1'

        elif opt =='4':
            # for hard-coded two-folds need dictionaries with detector combos

            # for finding the 2-folds in 6-GHZ state, NB: s1, s2, then s3 for the biSep
            #sagnac1  = nFold_create('d1, d7 - d3, d2', 2)
            sagnac1  = nFold_create('d4, d2 - d12, d10',2)
            sagnac2  = nFold_create('d4, d2 - d14, d11', 2)
            sagnac3  = nFold_create('d4 - d12', 2)
            sagnac1x = nFold_create('d4, d2 - d1, d10', 2)
            sagnac2x = nFold_create('d8, d6 - d12, d7', 2)
            # for each two-fold dictionary update keys to something meaningful
            for keys in list(sagnac1.keys()):
                sagnac1['s1' + str(keys)] = sagnac1.pop(keys)
                sagnac2['s2' + str(keys)] = sagnac2.pop(keys)
                sagnac1x['s1X' + str(keys)] = sagnac1x.pop(keys)
                sagnac2x['s2x' + str(keys)] = sagnac2x.pop(keys)
            
            # create empty two-fold dictionary
            twoFolds = {}

            # populate final two-fold dictionary
            twoFolds = {**sagnac3}
            #twoFolds = {**sagnac1, **sagnac2}

            print('Channel allocation successful')
            
            # generate keys for enumerating the dictionary of four-fold coincidences
            keys = list(twoFolds.keys())
            opts = [(key, t_meas, t_window, twoFolds[key][0], twoFolds[key][1]) for key in keys]
            optsSize = len(opts)
            
            # start multiprocessor for fast acquisition of multifold coincidences
            pool = mp.Pool(8)			
            print('Multiprocessing started successfully.')

            # prepare labels for printout
            labels = [str(i).rjust(6, ' ') for i in twoFolds.keys()]

            # prepare structure for terminal printout: this gets updated so labels do not move along screen
            allResults = [[0] * 8] * 20

        elif opt =='5':
               # for hard-coded two-folds need dictionaries with detector combos

            # for finding the 2-folds in 6-GHZ state, NB: s1, s2, then s3 for the biSep
            #sagnac1  = nFold_create('d1, d7 - d3, d2', 2)
            sagnac1  = nFold_create('d4, d2 - d12, d10',2)
            sagnac2  = nFold_create('d4, d2 - d14, d11', 2)
            sagnac1x = nFold_create('d1, d7 - d14, d2', 2)
            sagnac2x = nFold_create('d8, d6 - d4, d11', 2)
            # for each two-fold dictionary update keys to something meaningful
            for keys in list(sagnac1.keys()):
                sagnac1['s1' + str(keys)] = sagnac1.pop(keys)
                sagnac2['s2' + str(keys)] = sagnac2.pop(keys)
                sagnac1x['s1X' + str(keys)] = sagnac1x.pop(keys)
                sagnac2x['s2x' + str(keys)] = sagnac2x.pop(keys)
            
            # create empty two-fold dictionary
            twoFolds = {}

            # populate final two-fold dictionary
            #twoFolds = {**sagnac1x, **sagnac2x}
            twoFolds = {**sagnac1}

            print('Channel allocation successful')
            
            # generate keys for enumerating the dictionary of four-fold coincidences
            keys = list(twoFolds.keys())
            opts = [(key, t_meas, t_window, twoFolds[key][0], twoFolds[key][1]) for key in keys]
            optsSize = len(opts)
            
            # start multiprocessor for fast acquisition of multifold coincidences
            pool = mp.Pool(8)			
            print('Multiprocessing started successfully.')

            # prepare labels for printout
            labels = [str(i).rjust(6, ' ') for i in twoFolds.keys()]

            # prepare structure for terminal printout: this gets updated so labels do not move along screen
            allResults = [[0] * 8] * 20

        elif opt =='6':
            # for hard-coded two-folds need dictionaries with detector combos

            # specify two-fold combinations
            fourFolds = nFold_create('d6 - d5, d8 - d1, d2 - d3, d4', 4)

            print('Channel allocation successful')
            
            # generate keys for enumerating the dictionary of four-fold coincidences
            keys = list(fourFolds.keys())
            opts = [(key, t_meas, t_window, fourFolds[key][0], fourFolds[key][1]) for key in keys]
            optsSize = len(opts)
            
            # start multiprocessor for fast acquisition of multifold coincidences
            pool = mp.Pool(8)			
            print('Multiprocessing started successfully.')

            # prepare labels for printout
            labels = [str(i).rjust(6, ' ') for i in fourFolds.keys()]
            
            # prepare structure for terminal printout: this gets updated so labels do not move along screen
            allResults = [[0] * len(labels)] * 20

        else:
            opt = '1'
            print('Invalid option detected, defaulting to [1]:')
        

        try:
            while True:
                # inner loop, updates counters based on option selected by user

                # delay to slow printouts on terminal
                time.sleep(0.10)

                # get all singles and divide by 1000 to truncate least significant figures in counts
                singles = buf.singles(t_meas)
                sumSing = numpy.sum(singles)

                singles = [(i//100)/10 for i in singles]
                singles = [str(i).rjust(6, ' ') for i in singles]

                sumSing = (sumSing//100)/10
                sumSing = str(sumSing).rjust(7, ' ')

                if opt == '1':
                    # print singles, formated with a dash after every 4 channel printout
                    print(singles[0], singles[1], singles[2], singles[3], " - ",
                    singles[4], singles[5], singles[6], singles[7], " - ",
                    singles[8], singles[9], singles[10], singles[11], " - ",
                    singles[12], singles[13], singles[14], singles[15], "  --  ", sumSing
                    )

                elif opt == '2':
                    time.sleep(0.1)
                    # grab coincidences, hard-coded (to be updated and removed)
                    result = pool.map(lambda i: buf.multicoincidences(*opts[i][1::]), range(len(opts)))
                    # sum all the coincidences
                    sumRes1 = numpy.sum(result[0:16])
                    sumRes2 = numpy.sum(result[16:32])
                    
                    # string formatting to ensure numbers are padded
                    result = [str(i).rjust(5, ' ') for i in result]
                    sumRes1 = str(sumRes1).rjust(5, ' ')
                    sumRes2 = str(sumRes2).rjust(5, ' ')
                    
                    # additional formatting to include braces around each four values
                    strResult = '[' + ', '.join(result[0:4]) + ' ] - [' + ', '.join(result[4:8]) + ' ] - [' + ', '.join(result[8:12]) + ' ] - [' + ', '.join(result[12:16])+' ], '+str(sumRes1) + '[' + ', '.join(result[16:20]) + ' ] - [' + ', '.join(result[20:24]) + ' ] - [' + ', '.join(result[24:28]) + ' ] - [' + ', '.join(result[28:32])+' ], '+str(sumRes2)
                    allResults.pop(0)
                    allResults.append(strResult)

                    # include labels in printout, and using termcolor to change text colour
                    print(colored('[' + ', '.join(labels[0:4]) + ' ] - [' + ', '.join(labels[4:8]) + ' ] - [' + ', '.join(labels[8:12]) + ' ] - [' + ', '.join(labels[12:16])+' ]       '+'[' + ', '.join(labels[16:20]) + ' ] - [' + ', '.join(labels[20:24]) + ' ] - [' + ', '.join(labels[24:28]) + ' ] - [' + ', '.join(labels[28:32])+' ]', 'green'))
                    
                    for i in allResults:
                        print(i)

                elif opt =='3':
                    for ch in channels:
                        singles = [singles[i] for i in ch[0]]
                        coincidences = buf.multicoincidences(t_meas, t_window, ch[0], ch[1])
                    
                        coincidences = str(coincidences).rjust(5, ' ')
                        print(coincidences, " -- ", singles)
                        #print(coincidences, " -- ", singles)

                elif opt =='4':
                    # grab coincidences, hard-coded (to be updated and removed)
                    result = pool.map(lambda i: buf.multicoincidences(*opts[i][1::]), range(len(opts)))
                    sumRes = numpy.sum(result)
                    result = [str(i).rjust(6, ' ') for i in result]
                    sumRes = str(sumRes).rjust(6, ' ')
                    strResult = '[' + ', '.join(result[0:4]) + ' ] - [' + ', '.join(result[4:8]) + ' ] - [' + ', '.join(result[8:12]) + ' ]' +', '+str(sumRes)
                    allResults.pop(0)
                    allResults.append(strResult)

                    print(colored('[' + ', '.join(labels[0:4]) + ' ] - [' + ', '.join(labels[4:8]) + ' ] - [' + ', '.join(labels[8:12]) + ' ]', 'green'))
                    
                    for i in allResults:
                        print(i)

                elif opt == '5':
                    # grab coincidences, hard-coded (to be updated and removed)
                    result = pool.map(lambda i: buf.multicoincidences(*opts[i][1::]), range(len(opts)))
                    sumRes = numpy.sum(result)
                    result = [str(i).rjust(6, ' ') for i in result]
                    sumRes = str(sumRes).rjust(6, ' ')
                    strResult = '[' + ', '.join(result[0:4]) + ' ] - [' + ', '.join(result[4:8]) + ' ] - [' + ', '.join(result[8:12]) + ' ]' +', '+str(sumRes)
                    allResults.pop(0)
                    allResults.append(strResult)

                    print(colored('[' + ', '.join(labels[0:4]) + ' ] - [' + ', '.join(labels[4:8]) + ' ] - [' + ', '.join(labels[8:12]) + ' ]', 'green'))
                    
                    for i in allResults:
                        print(i)

                elif opt == '6':
                    time.sleep(2)
                    # grab coincidences, hard-coded (to be updated and removed)
                    result = pool.map(lambda i: buf.multicoincidences(*opts[i][1::]), range(len(opts)))
                    # sum all the coincidences
                    sumRes = numpy.sum(result)
                    
                    # string formatting to ensure numbers are padded
                    result = [str(i).rjust(6, ' ') for i in result]
                    sumRes = str(sumRes).rjust(6, ' ')
                    
                    # additional formatting to include braces around each four values
                    strResult = '[' + ', '.join(result[0:4]) + ' ] - [' + ', '.join(result[4:8]) + ' ]' + str(sumRes)
                    allResults.pop(0)
                    allResults.append(strResult)

                    # include labels in printout, and using termcolor to change text colour
                    print(colored('[' + ', '.join(labels[0:4]) + ' ] - [' + ', '.join(labels[4:8]) + ' ]', 'green'))
                    
                    for i in allResults:
                        print(i)

        except KeyboardInterrupt:
                # catch break command <CTRL+C> and return to outter loop: to fully exit script user needs to repeat break command
                print('Counters interupted, to exit code use <ctrl> + <c> then <return>.')
                pass

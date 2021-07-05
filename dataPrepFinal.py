import numpy
import random
import re
 ## start by prosessing data for the desired time span########

# Open all pre formatted stock data including: Open, High, Low, Close in that order 
data_file = open("ppb1412.csv", 'r')
# Read all of the lines from the file into memory
data_list = data_file.readlines()
# Close the file (we are done with it)
data_file.close()
#print(data_list)


datalength = len(data_list)
#print(data_list[1])
# function returns set of test data given the number of days to be investigated and the number of test sets to be included
def dataPrep (testNum, nDays,data_list):
    target = []
    loopResults = []
    training = numpy.empty((testNum,nDays,4))
    datalength = len(data_list)
    i =1
    for  i in range(0,testNum):
# randomly select a target day in training set 
        ind = random.randrange(0,datalength-1) # change 0 to 1 for test against data in sheet
        if nDays >= datalength - 2:
            ind  = 1
            #print(ind)
            print(datalength)
        while ind>(datalength - (nDays+1)): # make sure a point is chosen with sufficient days 
            ind = random.randrange(0,datalength-1) 
            #print(ind)
            pass
        row = data_list[ind]
        #print(ind)
        #convert row from a script into a list of floats
        rowInt = [float(s) for s in re.findall('[-+]?[.]?[\d]+(?:\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?',row)]  
        target.append(rowInt[3]) #target list containing the target for every randomly selected iteration of training data
        for n in range(0,nDays): # days before the target day, used to give the pattern spotted
            trow = data_list[ind + 1 + n] # select row from list n days before the randomly selected target
            trowint = [float(s) for s in re.findall('[-+]?[.]?[\d]+(?:\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?',trow)] # convert to a list of floating point numbers
            loopResults.append(trowint[:4]) # add sub-list for row to the list for the training set 
            looptr = numpy.array(loopResults) # convert to array
            pass
        loopResults = [] # reset loop results to empty list
        training[i,:,:] = looptr # 3D array containing each randomly selected training set
        i += 1
        pass
    #print(training)
    target = numpy.array(target)
    #print(training.shape)
    return (target, training,)


def colTestData (nDays,data_list):
    loopResults = []
    testData = numpy.empty((1,nDays,4))
    datalength = len(data_list)
    for n in range(0,nDays): # days before target day 
        trow = data_list[n]     #data_list[1+n] for testing with current data available data_list[n] to predict tomorrow 
        trowint = [float(s) for s in re.findall('[-+]?[.]?[\d]+(?:\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?',trow)]
        loopResults.append(trowint[:4])
        looptr = numpy.array(loopResults)
        pass
    loopResults = [] # reset loop results to empty list
    testData[0,:,:] = looptr
    #print (testtarget)
    #print(testData)
    return testData


def maxMin (data_list):
    datalength = len(data_list)
    loopResults = []
    for n in range(datalength-1):
        trow = data_list[n]
        trowint = [float(s) for s in re.findall('[-+]?[.]?[\d]+(?:\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?',trow)]
        loopResults.append(trowint[:4])
        pass
    aRes = numpy.array(loopResults)
    dataMax = numpy.amax(aRes)
    dataMin = numpy.amin(aRes)
    return dataMax, dataMin
        
def dataPrep2 (nDays,data_list):
    target = []
    loopResults = []
    datalength = len(data_list)
    training = numpy.empty((datalength-nDays,nDays,4))
    for  i in range(0,datalength-nDays):
        ind = i; 
        row = data_list[ind]
        #convert row from a script into a list of floats
        rowInt = [float(s) for s in re.findall('[-+]?[.]?[\d]+(?:\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?',row)]  
        target.append(rowInt[3]) #target list containing the target for every randomly selected iteration of training data
        for n in range(0,nDays): # days before the target day, used to give the pattern spotted
            trow = data_list[ind + 1 + n] # select row from list n days before the randomly selected target
            trowint = [float(s) for s in re.findall('[-+]?[.]?[\d]+(?:\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?',trow)] # convert to a list of floating point numbers
            loopResults.append(trowint[:4]) # add sub-list for row to the list for the training set 
            looptr = numpy.array(loopResults) # convert to array
            pass
        loopResults = [] # reset loop results to empty list
        training[i,:,:] = looptr
        pass
    target = numpy.array(target)
    #print(training.shape)
    return (target, training,)

[a,b] = dataPrep2(100 ,data_list)
t = colTestData(60, data_list)
#[mx,mn] = maxMin (data_list)
#print(mx,mn)
#print(t)
#f1 = 'Outputs.txt'
#print(b,file=open("output.txt", "a"))

print(t)




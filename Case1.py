"""
Class: Operations Research II
Case: 1
Date: 07/02/21
Author: Ryan Burruss (Group 5)
"""

### Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st                             # is this redundant w/sp above?

import pickle                                        # used to load data just once

import statsmodels.api as sm

np.set_printoptions(precision = 2)
np.set_printoptions(formatter={'float_kind':'{:.2f}'.format})


### File Initialization Functions 
def fileInit1():
    """ Only Run Once In Console, Then Comment Out """ 
    
    sData1  = '/Users/rtburruss/Desktop/Jupyter Notebook Files/Data/ccarrdata.txt'
    
    mData1 = np.loadtxt(sData1)
    
    pickleFile1 = open("mData1", "wb")
    pickle.dump(mData1, pickleFile1)
    pickleFile1.close()
   

def fileLoad1():
    """ Only Run After fileInit() Has Created Data File """

    pickleFile1 = open("mData1", "rb")
    mData1 = pickle.load(pickleFile1)
    pickleFile1.close()
    
    return mData1


def fileInit2():
    """ Only Run Once In Console, Then Comment Out """ 
    
    sData2  = '/Users/rtburruss/Desktop/Jupyter Notebook Files/Data/ccserdata.txt'
    
    mData2 = np.loadtxt(sData2)
    
    pickleFile2 = open("mData2", "wb")
    pickle.dump(mData2, pickleFile2)
    pickleFile2.close()
    

def fileLoad2():
    """ Only Run After fileInit() Has Created Data File """

    pickleFile2 = open("mData2", "rb")
    mData2 = pickle.load(pickleFile2)
    pickleFile2.close()
    
    return mData2


### General Functions
def floatArrayToInts(vX):
    """Converts an Array of Floats to Ints"""
    
    vI = np.array([0]*len(vX))
    for i in range(len(vX)):
        vI[i] = vX[int(i)]
        
    return vI


def floatMatrixToInts(mX):
    """Converts a Matrix of Floats to Ints"""
    
    iRows, iCols = mX.shape
    
    mI = np.zeros((iRows, iCols), dtype = int)
    
    for i in range(iRows):
        mI[i,:] = floatArrayToInts(mX[i,:])
    
    return mI


def deleteLastArrayElm(vOld):
    """Creates a New Vector without Last Index of Old Vector"""
    
    vNew = np.zeros(len(vOld) - 1)
    
    for i in range(len(vNew)):
        vNew[i] = vOld[i]
    
    return vNew


def lowValScan(vX, dLowerBound):
    """scans a given vector for items below a given lower bound"""
    
    iCount = int(0)
    
    for i in range(len(vX)):
        if vX[i] < dLowerBound:
            iCount += 1
    
    print("Vector Has %d Term(s) Below Theshold of %.2f" %(iCount, dLowerBound))
    
    if iCount > 0:
        print("Raise Binning Theshold or Re-Bin\n")
    else:
        print("")


### Distribution Functions
def poissonPMF(dLambda, vK):
    """(UNUSED) Create a Vector of Poisson Probabilities Given Lambda and a Max K for the x-axis"""
    
    vProb = np.zeros(len(vK))
    
    for i in range(len(vK)):
        # vProb[i] = (dLambda ** vK[i]) * (np.exp(-dLambda)) / (np.math.factorial(vK[i]))
        vProb[i] = st.poisson.pmf(vK[i], dLambda)
    
    return vProb


def expPDF(vX, dLambda):
    """Create a Vector of Exponential Probabilities Given Mu = dLambda and vector of inputs vX"""
    
    vProb = np.zeros(len(vX))
    
    for i in range(len(vX)):
        vProb[i] = (dLambda * np.exp(-dLambda * vX[i]))
    
    return vProb


### Binning Functions
def binCustom(vX, vBins, bPrint = False):
    """(UNUSED)creates vector of bins counts given a data vector vX and a vector of custom bin locations"""
    
    vY = np.zeros(len(vBins))                           # initialize vector to store counts per bin

    for i in range(len(vX)):                            # loop through each data point in vX
        for j in range(len(vBins) - 1):
            
            if bPrint == True:
                print("vX[%d]:" % i, vX[i])
                print("Current Bin [%d, %d)" %(vBins[j], vBins[j+1]), "\n")
            
            if vX[i] >= vBins[j] and vX[i] < vBins[j+1]:
                vY[j] += 1
                break
            
        if vX[i] == vBins[-1]:                          # catches case where last datapoint = upper bound
            vY[-1] += 1
            
            if bPrint == True:
                print("DataPoint = %d Matches Upper Bound" % vX[i])
    
    return vY


def identifyBins(vX, dMinBin, bPrint = False):
    """
    Purpose: group the elements of an array into bins of dMinBin size
    Notes: in current iteration, the last two bins might still be under represented. note error message.
    """
    
    if bPrint == True:
        print("***** Begin Identify Bins Process *****\n")
    
    vBins = np.array([0])  # vector where numbers entered are the first excluded index of each bin
    dSum = 0.0
    
    for i in range(len(vX)):
        if dSum + vX[i] >= dMinBin: 
            vBins = np.append(vBins, i + 1)
            
            if bPrint == True:
                print("dSum (up to and including) index %d: %.2f" % (i, dSum + vX[i]))
                
            dSum = 0.0
            
        else:
            dSum += vX[i]
    
    vBins    = deleteLastArrayElm(vBins)
    vBins    = floatArrayToInts(vBins)
    vBinSums = binSumsByK(vX, vBins)
    
    if bPrint == True:
        print("\ndSum for remaining terms added to last result (index %d to %d): %.2f" 
              %(vBins[-1], len(vX) - 1, dSum), "\n")
        print("Start Points of All %d Bins:\n" % len(vBins), vBins, "\n")
        print("Bins Sums:", vBinSums, "\n")
        lowValScan(vBinSums, dMinBin)
        print("****** End Identify Bins Process ******\n")
    

    
    return vBins


def binSumsByK(vX, vBins):
    """custom bin summing by vector vBins, representing the last K to include in the previous bin"""
    
    vSums = np.zeros(len(vBins))                    # initialize vector for storing sums of bins
    dSum  = 0.0                                     # initialize temporary sum variable
    
    for i in range(len(vBins) - 1):                 # ranging through each bin marker
        for j in range(vBins[i], vBins[i+1]):       # ranging through adjacent bin locations
            dSum += vX[j]
            
        vSums[i] = dSum                             # store final sum val in vector
        dSum = 0.0                                  # reset temporary sum variable
    
    for j in range(vBins[-1], len(vX)):             # repeating process for last bin
        dSum += vX[j]
    
    vSums[-1] = dSum
            
    return vSums


def binCounts(vX, vBins, bPrint = False):
    """"given vector vX and bin partitions, count up number of elements per bin"""
    
    vCounts = np.zeros(len(vBins))                      # initialize vector to store counts in bins
    iCount  = int(0)                                    # initialize temporary count variable
    iCurr   = int(0)                                    # initialize progressive vX index tracker
    vXsort  = np.sort(vX)                               # sort Observed vals


    for i in range(0, len(vBins) - 1):                  # loop through all bin markers but the first/last
        for j in range(iCurr, len(vXsort)):             # loop through all unchecked sorted
           
            if bPrint == True:
                print("vXsort[%d] at i = %d:" %(j, i), vXsort[j])  
  
        
            if vXsort[j] >= vBins[i+1]:                 # check for vX val being out of bounds of bin
                vCounts[i] = iCount                     # store current count in relevant bin location
                iCurr      = j                          # set current starting point to current vX index
                iCount     = int(0)                     # reset count variable
                
                if bPrint == True:
                    print("\nStop Condition Triggered\n")
                    
                break                                   # break inner loop
            
            else:
                iCount += 1                             # otherwise, increment count of vals in bin
                
                if bPrint == True:
                    print("iCount:", iCount)
    
    for j in range(iCurr, len(vXsort)):                 # repeating process for last bin
        iCount += 1                                     # increment count of vals in bin
    
    vCounts[-1] = iCount                                # adding last results to last counts index
    vCounts     = floatArrayToInts(vCounts)             # convert array to integers
    
    return vCounts


### Chi-Squared Functions
def chiSqTval(vObsBin, vExpBin):
    """returns a t-value for a ChiSq Test"""
    
    if len(vObsBin) != len(vExpBin):
        print("Vector Lengths Don't Match")
        return
    
    dT = 0.0
    
    for i in range(len(vObsBin)):
        dT += (vObsBin[i] - vExpBin[i]) ** 2 / vExpBin[i]
        
    return dT


### Main
def main():     
    ### Running File Initialization Steps
    
    """ only run fileInit() on first run """
    # fileInit1()
    # fileInit2()
    
    """ run only fileLoad() on subsequent steps after running fileInit() """
    mArr = floatMatrixToInts(fileLoad1())
    mSer = floatMatrixToInts(fileLoad2())
    
    
    ### Magics
    print("")
    iArrRows, iArrCols = mArr.shape
    iSerRows, iSerCosl = mSer.shape
    
    
    ### Initial Visualizations
    bViz = 0                            # boolean switch for visualizations
    
    ## Arrivals
    sHours     = ('7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20')
    sHourRange = ('7-8', '8-9', '9-10', '10-11', '11-12', '12-13', '13-14', '14-15', '15-16', '16-17', 
                  '17-18', '18-19', '19-20', '20-21')
    
    # Total Number of Calls Per Day
    vArrRowSum = mArr.sum(axis = 1)
    
    if bViz == 1:
        plt.plot(vArrRowSum, color = "darkred")
        plt.title("Total Calls Per Day over 223 Working Days")
        plt.xlabel("Workday Index")
        plt.ylabel("Total Number of Calls")
        plt.grid()
        plt.tight_layout()
        plt.show()
    
    # Average Number of Calls Per Day
    vArrRowAvg = mArr.mean(axis = 1)
    
    if bViz == 1:
        plt.plot(vArrRowAvg, color = "darkred")
        plt.title("Average Number of Calls Per Hour Per Day")
        plt.xlabel("Workday Index")
        plt.ylabel("Average Number of Calls")
        plt.grid()
        plt.tight_layout()
        plt.show()
    
    # Average Number of Calls Per Hour
    vArrColAvg = mArr.mean(axis = 0)
    
    if bViz == 1:
        plt.plot(vArrColAvg, color = "darkred")
        plt.title("Average Number of Calls Per Hour over 223 Working Days")
        plt.xlabel("(Starting) Hour of Workday")
        plt.ylabel("Average Number of Calls")
        plt.grid()
        plt.xticks(np.arange(0, 14), sHours)
        plt.tight_layout()
        plt.show()
    
    
    ## Service Durations
    sMonths = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')
    
    # Total Length of Calls Per Month
    vSerColSum = mSer.sum(axis = 0) / 360    # convert to hours
    
    if bViz == 1:
        plt.plot(vSerColSum, color = "darkred")
        plt.title("Total Call Duration Per Month over 1 Year")
        plt.xlabel("Month")
        plt.ylabel("Total Call Duration (Hours)")
        plt.xticks(np.arange(0, 12), sMonths)
        plt.grid()
        plt.tight_layout()
        plt.show()
    
    # Average Length of Calls Per Month
    vSerColAvg = mSer.mean(axis = 0) / 60     # convert to minutes
    
    if bViz == 1:
        plt.plot(vSerColAvg, color = "darkred")
        plt.title("Average Call Duration Per Month over 1 Year")
        plt.xlabel("Month")
        plt.ylabel("Average Call Duration (Minutes)")
        plt.xticks(np.arange(0, 12), sMonths)
        plt.grid()
        plt.tight_layout()
        plt.show()
    
    
    #### Assignment A: Modelling
    
    ### Part I: Arrival Data
    bPoisPlot = 0                                               # turn on Poisson Plot with a '1'
    bQQplot   = 0                                               # turn on QQ Plot with a '1'
    
    iInd    = 0                                                 # set index of hour to get its lambda
    dLambda = vArrColAvg[iInd]                                  # grab ML lambda from that index
    
    ## Plot Poisson Distributions for Given Lambda's
    if bPoisPlot == 1:
        vK1       = np.arange(1, (np.floor(dLambda * 2) + 1))   # set vector of suitable K's
        vPoisson  = st.poisson.pmf(vK1, dLambda)                # obtain Poisson probabilities
        vObserved = mArr[:,iInd]
        iBins     = max(vObserved) - min(vObserved) + 1
        
        plt.plot(vK1, vPoisson, color = "darkred")
        plt.title("Poisson Dist vs. Observed Data of Arrivals in Hour %s" %(sHourRange[iInd]))
        plt.hist(mArr[:,iInd], density = True, bins = iBins, color = "blue", alpha = .60)
        plt.show()
    
    
    ## Q-Q Plots Over Each Hour Timeframe
    if bQQplot == 1:
        for i in range(len(vArrColAvg)):
            iInd      = i                                       # resets iInd to run through each column
            vObserved = mArr[:,iInd]                            # resets vObserved for each column
            dLambda   = vArrColAvg[iInd]                        # resets dLambda for each column
            
            st.probplot(vObserved, dist = 'poisson', sparams = (dLambda), plot = plt)
            plt.title("QQ Plot: Poisson Arrivals in Hour %s" %(sHourRange[iInd]))
            plt.grid()
            plt.tight_layout()
            plt.show()
    
    
    ## ChiSquared Test
    vObserved = mArr[:, iInd]                                   # retrieve Observed column data
    vObsSort = np.sort(vObserved)                               # sort Observed vals
    iObsMin = min(vObserved)                                    # min of Observed vals
    iObsMax = max(vObserved)                                    # max of Observed vals
    
    vK2       = np.arange(iObsMin, iObsMax + 1)                 # vector of k-vals for expected Poisson
    vExpected = iArrRows * st.poisson.pmf(vK2, dLambda)         # Poisson probs scaled by num of data pts
    
    dBinMin = 15.0                                              # set desired min expected bin frequency
    vBins   = identifyBins(vExpected, dBinMin, bPrint = True)   # create large enough bins from vExpected
    iBins   = len(vBins)                                        # number of bins generated
    
    vExpSums     = binSumsByK(vExpected, vBins)                 # sum up expected frequencies per bin
    dSumExpSums  = sum(vExpSums)                                # total number of frequencies 
    vObsBinCount = binCounts(vObserved, vBins)                  # counts of observed data in bins
    dObsBinSums  = sum(vObsBinCount)                            # sum up all bin counts


    print("Lambda at Index %d: %.2f" % (iInd, dLambda), "\n")
    print("Observed Values (Unsorted):\n", vObserved, "\n")
    print("Observed Values (Sorted):\n", vObsSort, "\n")
    print("Vector of K-vals (Given by Range of Observed Vals):\n", vK2, "\n")
    print("Expected Frequencies (Poisson Prob * Number of Observations):\n", vExpected, "\n")
    print("Observed Values Min/Max: (%d, %d)" %(iObsMin, iObsMax))
    print("Expected Values Min/Max: (%.3f, %.3f)" %(min(vExpected), max(vExpected)), "\n")
    print("Expected Sums by Bin:\n", vExpSums, "\n")
    print("Sum of Expected Sums:", dSumExpSums, "\n")
    print("Observed Bin Counts:\n", vObsBinCount, "\n")
    lowValScan(vObsBinCount, 5.0)
    print("Sum of Observed Counts:", dObsBinSums, "\n")
    
    
    # Run Chi-Squared Tests
    print(st.chisquare(vObsBinCount, vExpSums, ddof = iBins - 2))   # chiSq results
    print(chiSqTval(vObsBinCount, vExpSums))                        # double-checking t-value
    
    
    """
    ### Previous Ideas, Now Unused
    
    np.random.seed(1)                                           # set seed for random Poisson
    vExpected = np.random.poisson(dLambda, iArrRows)            # generate simulated Poisson data
    
    vQcutE    = pd.qcut(vExpected, q = iBins, retbins = True)   # qcut to obtain balanced expected bins
    vBinsE    = vQcutE[1]                                       # store bin partition results
    
    vQcutO     = pd.qcut(vObserved, q = iBins, retbins = True)  # qcut to obtain balanced observed bins
    vBinsO     = vQcutO[1]                                      # store bin partition results
    
    vBins = (vBinsE + vBinsO) / 2                               # take average of bin results
    
    
    # Method to Ensure Entire Range of Observed Values is Represented
    if vBins[0] > min(vObserved):
        vBins[0] = min(vObserved)
    if vBins[-1] < max(vObserved):
        vBins[-1] = max(vObserved)
    

    vExpBin = binCustom(vExpected, vBins)                       # counts of expected dist in bins
    vObsBin = binCustom(vObserved, vBins)                       # counts of observed dist in bins
    """
    
    # ### Part II: Service Time Data
    
    # bExpPlot = 1                                                # turn on Exponential Plot with a '1'
    # bQQplot  = 0                                                # turn on QQ Plot with a '1'
    
    # iMonth = 0                                                  # set index of month
    # dMu    = 1 / (mSer.mean() / 60)                                     # grab ML mu from whole service data set
    # # dMu = mSer[:,0].mean()
    
    # print("Mean Service Time Over All Data:", dMu)
    
    
    # ## Plot Exponential Distributions for Given Mu
    # if bExpPlot == 1:
    #     vX1       = np.arange(0, 1500 / 60, .01)      # set vector of suitable x's
    #     print(vX1)
    #     vExp      = expPDF(vX1, dMu)         # obtain Exponential probabilities
    #     print(vExp)
    #     # iBins     = max(mSer) - min(mSer) + 1
        
    #     plt.plot(vX1, vExp, color = "darkred")
    #     plt.title("Exponential Dist vs. Observed Data of Yearly Service Times")
    #     # plt.hist(mSer, density = True, color = "blue", alpha = .60)
    #     plt.show()
    
    
    # ## Q-Q Plots Over Each Hour Timeframe
    # if bQQplot == 1:
    #     for i in range(len(vArrColAvg)):
    #         iInd      = i                                       # resets iInd to run through each column
    #         vObserved = mArr[:,iInd]                            # resets vObserved for each column
    #         dLambda   = vArrColAvg[iInd]                        # resets dLambda for each column
            
    #         st.probplot(vObserved, dist = 'poisson', sparams = (dLambda), plot = plt)
    #         plt.title("QQ Plot: Poisson Arrivals in Hour %s" %(sHourRange[iInd]))
    #         plt.grid()
    #         plt.tight_layout()
    #         plt.show()

     
 
### Start Main
if __name__ == "__main__":
    main()
    
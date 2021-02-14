"""
Class: Operations Research II
Case: 1
Date: 14/02/21
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
        

def matrixToVector(mX, bCols = True):
    """Convert a Matrix to a Single Vector One Column (default) at a Time"""
    
    iRows, iCols = mX.shape             # obtain dimensions of matrix
    
    vX = np.empty(0)
    
    for i in range(iCols):
        for j in range(iRows):
            vX = np.append(vX, mX[j,i])
    
    return vX


### Distribution Functions
def expPDF(vX, dLambda):
    """Create a Vector of Exponential Probabilities Given Mu = dLambda and vector of inputs vX"""
    
    vProb = np.zeros(len(vX))
    
    for i in range(len(vX)):
        vProb[i] = (dLambda * np.exp(-dLambda * vX[i]))
    
    return vProb


### Binning Functions
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
            vBins = np.append(vBins, i+1)
            
            if bPrint == True:
                print("dSum (up to and including) index %d: %.2f" % (i, dSum + vX[i]))
                
            dSum = 0.0
            
        else:
            dSum += vX[i]
    
    vBinsDel = deleteLastArrayElm(vBins)
    vBinsDel = floatArrayToInts(vBinsDel)
    vBinSums = binSumsByK(vX, vBinsDel)
    
    if bPrint == True:
        print("\ndSum remaining terms (index %d to %d): %.2f" 
              %(vBins[-1], len(vX) - 1, dSum), "\n")
        print("Start Points of All %d Bins:\n" % len(vBinsDel), vBinsDel, "\n")
        print("Bins Sums:", vBinSums, "\n")
        lowValScan(vBinSums, dMinBin)
        print("****** End Identify Bins Process ******\n")
    
    
    return vBinsDel


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
    fileInit1()
    fileInit2()
    
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
    bPoisPlot  = 1                                              # turn on Poisson Plot with a '1'
    bPoisQQ    = 1                                              # turn on QQ Plot with a '1'
    bPoisChiSq = 1                                              # turn on ChiSquared with a '1'
    
    
    ## Plot Poisson Distributions for Given Lambda's
    if bPoisPlot == 1:
        for i in range(iArrCols):
            iInd    = i                                             # resets iInd to run through each col
            vObserved = mArr[:,iInd]                                # resets vObserved for each column
            dLambda = vArrColAvg[iInd]                              # resets dLambda for each column
            
            vK1       = np.arange(1, (np.floor(dLambda * 2) + 1))   # set vector of suitable K's
            vPoisson  = st.poisson.pmf(vK1, dLambda)                # obtain Poisson probabilities
   
            iBins     = max(vObserved) - min(vObserved) + 1
            
            plt.plot(vK1, vPoisson, color = "darkred")
            plt.title("Poisson Dist vs. Observed Data of Arrivals in Hour %s" %(sHourRange[iInd]))
            plt.hist(mArr[:,iInd], density = True, bins = iBins, color = "blue", alpha = .60)
            plt.xlabel("Data Values")
            plt.ylabel("Densities")
            plt.grid()
            plt.tight_layout()
            plt.show()
    
    
    ## Q-Q Plots Over Each Hour Timeframe
    if bPoisQQ == 1:
        for i in range(iArrCols):
            iInd      = i                                       # resets iInd to run through each column
            vObserved = mArr[:,iInd]                            # resets vObserved for each column
            dLambda   = vArrColAvg[iInd]                        # resets dLambda for each column
            
            st.probplot(vObserved, dist = st.poisson(dLambda), plot = plt)
            plt.title("QQ Plot: Poisson Arrivals in Hour %s" %(sHourRange[iInd]))
            plt.grid()
            plt.tight_layout()
            plt.show()
            
            vQpois    = np.arange(0.002, 1, (1 / iArrRows))
            vExpected = st.poisson.ppf(vQpois, dLambda)
            
            print("For Hour %s:" % sHourRange[iInd])
            print("Observed Min/Max: (%.2f, %.2f)" % (min(vObserved), max(vObserved)))
            print("Expected Min/Max: (%.2f, %.2f)" % (min(vExpected), max(vExpected)), "\n")
    
    
    ## ChiSquared Test
    if bPoisChiSq == 1:
        iInd    = 0                                                 # choose hour index
        dBinMin = 44.0                                              # set desired min expected bin frequency
        dLambda = vArrColAvg[iInd]                                  # grab ML lambda from that index
        
        vObserved = mArr[:, iInd]                                   # retrieve Observed column data
        iObsMin   = min(vObserved)                                  # min of Observed vals
        iObsMax   = max(vObserved)                                  # max of Observed vals
        
        vK2       = np.arange(iObsMin, iObsMax + 1)                 # vector of k-vals for expected Poisson
        vExpected = iArrRows * st.poisson.pmf(vK2, dLambda)         # Poisson probs scaled by num of data pts
        vBins     = identifyBins(vExpected, dBinMin, bPrint = True) # create large enough bins from vExpected
        iBins     = len(vBins)                                      # number of bins generated
        
        vExpSums     = binSumsByK(vExpected, vBins)                 # sum up expected frequencies per bin
        dSumExpSums  = sum(vExpSums)                                # total number of frequencies 
        vObsBinCount = binCounts(vObserved, vBins)                  # counts of observed data in bins
        dObsBinSums  = sum(vObsBinCount)                            # sum up all bin counts
    
        print(vBins)
        
        # Print Statements
        print("ML Lambda (Sample Mean) at Index %d: %.2f" % (iInd, dLambda), "\n")
        print("Observed Values:\n", vObserved, "\n")
        print("Vector of K-vals (Given by Range of Observed Vals):\n", vK2, "\n")
        print("Expected Frequencies (Poisson Prob * Number of Observations):\n", vExpected, "\n")
        print("Observed Values Min/Max: (%d, %d)" %(iObsMin, iObsMax))
        print("Expected Values Min/Max: (%.2f, %.2f)" %(min(vExpected), max(vExpected)), "\n")
        print("Expected Sums by Bin:\n", vExpSums, "\n")
        print("Sum of Expected Sums:", dSumExpSums, "\n")
        print("Observed Bin Counts:\n", vObsBinCount, "\n")
        lowValScan(vObsBinCount, 5.0)
        print("Sum of Observed Counts:", dObsBinSums, "\n")
        
        
        # Run Chi-Squared Tests
        print(st.chisquare(vObsBinCount, vExpSums, ddof = iBins - 2))   # chiSq results
        print("Via Self-Made T-Val Algorithm:  ", chiSqTval(vObsBinCount, vExpSums))
    
    
    ### Part II: Service Time Data (Exponential)
    bExpPlot  = 1                                               # turn on Exponential Plot with a '1'
    bExpQQ    = 1                                               # turn on QQ Plot with a '1'
    bExpChiSq = 1                                               # turn on ChiSq with a '1'
    
    vSerSec   = floatArrayToInts(matrixToVector(mSer))          # convert mSer to one vector (seconds)
    vSerMin   = vSerSec / 60                                    # vSer in minutes
    dXbarSec  = vSerSec.mean()                                  # xBar in seconds of whole service dataset
    dXbarMin  = vSerMin.mean()                                  # xBar in minutes of whole service dataset
    dMuSec    = 1 / dXbarSec                                    # ML of mu (lambda) in sec for Exp Dist
    dMuMin    = 1 / dXbarMin                                    # ML of mu (lambda) in min for Exp Dist
    

    ## Plot Exponential Distributions
    if bExpPlot == 1:
        print("Average Service Call Time: %.2f Seconds (%.2f Minutes)" %(dXbarSec, dXbarMin))
        print("ML Estimate for Mu (Lambda): %.3f Seconds (%.2f Minutes):" %(dMuSec, dMuMin))
        
        # in Seconds
        vXsec   = np.arange(0, max(vSerSec), .01)      # set vector of suitable x's
        vExpSec = expPDF(vXsec, dMuSec)                # obtain Exponential probabilities
        
        plt.plot(vXsec, vExpSec, color = "darkred")
        plt.title("Exponential Distribution vs. Observed Service Time Data")
        plt.xlabel("Service Time in Seconds")
        plt.ylabel("Density")
        plt.hist(vSerSec, density = True, bins = 100, color = "blue", alpha = .60)
        plt.grid()
        plt.tight_layout()
        plt.show()
        
        # in Minutes
        vXmin   = np.arange(0, max(vSerMin), .01)       # set vector of suitable x's
        vExpMin = expPDF(vXmin, dMuMin)                 # obtain Exponential probabilities
        
        plt.plot(vXmin, vExpMin, color = "darkred")
        plt.title("Exponential Distribution vs. Observed Service Time Data")
        plt.xlabel("Service Time in Minutes")
        plt.ylabel("Density")
        plt.hist(vSerMin, density = True, bins = 100, color = "blue", alpha = .60)
        plt.grid()
        plt.tight_layout()
        plt.show()
        

    ## Q-Q Plot of Whole Matrix
    if bExpQQ == 1:
        vQ        = np.arange(0, 1, 1/100000)
        
        # in Seconds
        vExpected = st.expon.ppf(vQ, scale = dXbarSec)
            
        print("Observed Min/Max: (%.2f, %.2f)" % (min(vSerSec), max(vSerSec)))
        print("Expected Min/Max: (%.2f, %.2f)" % (min(vExpected), max(vExpected)), "\n")
        
        st.probplot(vSerSec, dist = st.expon(scale = dXbarSec), plot = plt)
        plt.title("QQ Plot: Exponential Service Times Over Whole Year")
        plt.ylabel("Observed Values (Seconds)")
        plt.grid()
        plt.tight_layout()
        plt.show()
        
        
        # in Minutes
        vExpected = st.expon.ppf(vQ, scale = dXbarMin)
            
        print("Observed Min/Max: (%.2f, %.2f)" % (min(vSerMin), max(vSerMin)))
        print("Expected Min/Max: (%.2f, %.2f)" % (min(vExpected), max(vExpected)), "\n")
        
        st.probplot(vSerMin, dist = st.expon(scale = dXbarMin), plot = plt)
        plt.title("QQ Plot: Exponential Service Times Over Whole Year")
        plt.ylabel("Observed Values (Minutes)")
        plt.grid()
        plt.tight_layout()
        plt.show()
        

    ## ChiSquared Test
    if bExpChiSq == 1:
        iN          = len(vSerSec)                                  # number of obs (60k)
        iSerSecMin  = min(vSerSec)                                  # min of Observed vals
        iSerSecMax  = max(vSerSec)                                  # max of Observed vals
        
        vX2sec  = np.arange(iSerSecMin, iSerSecMax * 2)             # vector of x-vals for expected ExpDist
        vExpExp = iN * st.expon.pdf(vX2sec, scale = dXbarSec)       # Exp probs scaled by num of data pts
        
        dBinMin = 5000.0                                            # set desired min expected bin frequency
        vBins   = identifyBins(vExpExp, dBinMin, bPrint = True)     # create large enough bins from vExpected
        iBins   = len(vBins)                                        # number of bins generated
        
        vExpSums     = binSumsByK(vExpExp, vBins)                   # sum up expected frequencies per bin
        dSumExpSums  = sum(vExpSums)                                # total number of frequencies 
        vObsBinCount = binCounts(vSerSec, vBins)                    # counts of observed data in bins
        dObsBinSums  = sum(vObsBinCount)                            # sum up all bin counts
    
    
        print("Observed Values:\n", vSerSec, "\n")
        print("Vector of X-vals (Given by Range of Observed Vals):\n", vX2sec, "\n")
        print("Expected Frequencies (Exponential Prob * Number of Observations):\n", vExpExp, "\n")
        print("Observed Values Min/Max: (%d, %d)" %(iSerSecMin, iSerSecMax))
        print("Expected Values Min/Max: (%.3f, %.3f)" %(min(vExpExp), max(vExpExp)), "\n")
        print("Expected Sums by Bin:\n", vExpSums, "\n")
        print("Sum of Expected Sums:", dSumExpSums, "\n")
        print("Observed Bin Counts:\n", vObsBinCount, "\n")
        lowValScan(vObsBinCount, 5.0)
        print("Sum of Observed Counts:", dObsBinSums, "\n")
        
        
        # Run Chi-Squared Tests
        print(st.chisquare(vObsBinCount, vExpSums, ddof = iBins - 2))   # chiSq results
        print(chiSqTval(vObsBinCount, vExpSums), "\n")                  # double-checking t-value
        
        
    ### Part III: Checking Alternate Distributions
    
    bWeibull = 1
    bLogNorm = 1                # this method had issues, but the operable elements were left in
    
    ## Trying Weibull function
    if bWeibull == 1:
        vWeibullSec = st.weibull_min.fit(vSerSec)
        vWeibullMin = st.weibull_min.fit(vSerMin)
        
        print("Weibull (in sec) Shape:    %.2f" % vWeibullSec[0])
        print("Weibull (in sec) Location: %.2f" % vWeibullSec[1])
        print("Weibull (in sec) Scale:    %.2f" % vWeibullSec[2], "\n")
        
        print("Weibull (in min) Shape:    %.2f" % vWeibullMin[0])
        print("Weibull (in min) Location: %.2f" % vWeibullMin[1])
        print("Weibull (in min) Scale:    %.2f" % vWeibullMin[2])
    
    ## Plot Weibull Distribution
       
        # in Seconds
        vXsec    = np.arange(0, 100, .01)                       # set vector of suitable x's
        vWeibSec = st.weibull_min.pdf(vXsec, vWeibullSec[0], 
                                      loc = vWeibullSec[1], 
                                      scale = vWeibullSec[2])
        
        plt.plot(vXsec, vWeibSec, color = "darkred")
        plt.title("Weibull Distribution vs. Observed Service Time Data")
        plt.xlabel("Service Time in Seconds")
        plt.ylabel("Density")
        plt.hist(vSerSec, density = True, bins = 400, color = "blue", alpha = .60)
        plt.grid()
        plt.tight_layout()
        plt.show()
        
        # in Minutes
        vXmin    = np.arange(0, max(vSerMin), .01)              # set vector of suitable x's
        vWeibMin = st.weibull_min.pdf(vXmin, vWeibullMin[0], 
                                      loc = vWeibullMin[1], 
                                      scale = vWeibullMin[2])
        
        plt.plot(vXmin, vWeibMin, color = "darkred")
        plt.title("Weibull Distribution vs. Observed Service Time Data")
        plt.xlabel("Service Time in Minutes")
        plt.ylabel("Density")
        plt.hist(vSerMin, density = True, bins = 400, color = "blue", alpha = .60)
        plt.grid()
        plt.tight_layout()
        plt.show()
        

    ## Q-Q Plot of Whole Matrix
        vQ        = np.arange(0, 1, 1/100000)
        
        # in Seconds
        vExpected = st.weibull_min.ppf(vQ, vWeibullSec[0], 
                                       loc = vWeibullSec[1], 
                                       scale = vWeibullSec[2])
            
        print("Observed Min/Max: (%.2f, %.2f)" % (min(vSerSec), max(vSerSec)))
        print("Expected Min/Max: (%.2f, %.2f)" % (min(vExpected), max(vExpected)), "\n")
        
        st.probplot(vSerSec, dist = st.weibull_min(vWeibullSec[0], 
                                                   loc = vWeibullSec[1], 
                                                   scale = vWeibullSec[2]), plot = plt)
        plt.title("QQ Plot: Weibull Service Times Over Whole Year")
        plt.ylabel("Observed Values (Seconds)")
        plt.grid()
        plt.tight_layout()
        plt.show()
        
        
        # in Minutes
        vExpected = st.weibull_min.ppf(vQ, vWeibullMin[0],
                                       loc = vWeibullMin[1],
                                       scale = vWeibullMin[2])
            
        print("Observed Min/Max: (%.2f, %.2f)" % (min(vSerMin), max(vSerMin)))
        print("Expected Min/Max: (%.2f, %.2f)" % (min(vExpected), max(vExpected)), "\n")
        
        st.probplot(vSerMin, dist = st.weibull_min(vWeibullMin[0], 
                                                   loc = vWeibullMin[1],
                                                   scale = vWeibullMin[2]), plot = plt)
        plt.title("QQ Plot: Weibull Service Times Over Whole Year")
        plt.ylabel("Observed Values (Minutes)")
        plt.grid()
        plt.tight_layout()
        plt.show()
        

    ## ChiSquared Test
        iN          = len(vSerSec)                                  # number of obs (60k)
        iSerSecMin  = min(vSerSec)                                  # min of Observed vals
        iSerSecMax  = max(vSerSec)                                  # max of Observed vals
        
        vX2sec   = np.arange(iSerSecMin, iSerSecMax * 2)             # vector of x-vals for expected WeibDist
        vExpWeib = iN * st.weibull_min.pdf(vX2sec, vWeibullSec[0],
                                          loc = vWeibullSec[1],
                                          scale = vWeibullSec[2])       
        
        dBinMin = 5000.0                                            # set desired min expected bin frequency
        vBins   = identifyBins(vExpWeib, dBinMin, bPrint = True)    # create large enough bins from vExpected
        iBins   = len(vBins)                                        # number of bins generated
        
        vWeibSums     = binSumsByK(vExpWeib, vBins)                 # sum up expected frequencies per bin
        dSumWeibSums  = sum(vWeibSums)                               # total number of frequencies 
        vObsBinCount = binCounts(vSerSec, vBins)                    # counts of observed data in bins
        dObsBinSums  = sum(vObsBinCount)                            # sum up all bin counts
    
        print("Observed Values:\n", vSerSec, "\n")
        print("Vector of X-vals (Given by Range of Observed Vals):\n", vX2sec, "\n")
        print("Expected Frequencies (Exponential Prob * Number of Observations):\n", vExpWeib, "\n")
        print("Observed Values Min/Max: (%d, %d)" %(iSerSecMin, iSerSecMax))
        print("Expected Values Min/Max: (%.3f, %.3f)" %(min(vExpWeib), max(vExpWeib)), "\n")
        print("Expected Sums by Bin:\n", vWeibSums, "\n")
        print("Sum of Expected Sums:", dSumWeibSums, "\n")
        print("Observed Bin Counts:\n", vObsBinCount, "\n")
        lowValScan(vObsBinCount, 5.0)
        print("Sum of Observed Counts:", dObsBinSums, "\n")
        
        
        # Run Chi-Squared Tests
        print(st.chisquare(vObsBinCount, vWeibSums, ddof = iBins - 4))  # chiSq results
        print(chiSqTval(vObsBinCount, vWeibSums), "\n")                  # double-checking t-value
 
   
    ## Trying Log Normal function
    if bLogNorm == 1:
        vLogNormSec = st.lognorm.fit(vSerSec)
        vLogNormMin = st.lognorm.fit(vSerMin)
        
        print("Log Normal (in sec) Shape:    %.2f" % vLogNormSec[0])
        print("Log Normal (in sec) Location: %.2f" % vLogNormSec[1])
        print("Log Normal (in sec) Scale:    %.2f" % vLogNormSec[2], "\n")
        
        print("Log Normal (in min) Shape:    %.2f" % vLogNormMin[0])
        print("Log Normal (in min) Location: %.2f" % vLogNormMin[1])
        print("Log Normal (in min) Scale:    %.2f" % vLogNormMin[2], "\n")
    
    ## Plot Log Normal Distribution
        
        # in Minutes
        vXmin    = np.arange(0, max(vSerMin), .01)              # set vector of suitable x's
        vLogNormMin = st.lognorm.pdf(vXmin, vLogNormMin[0], 
                                      loc = vLogNormMin[1], 
                                      scale = vLogNormMin[2])
        
        plt.plot(vXmin, vLogNormMin, color = "darkred")
        plt.title("Log Normal Distribution vs. Observed Service Time Data")
        plt.xlabel("Service Time in Minutes")
        plt.ylabel("Density")
        plt.hist(vSerMin, density = True, bins = 400, color = "blue", alpha = .60)
        plt.grid()
        plt.tight_layout()
        plt.show()
        

    ## Q-Q Plot of Whole Matrix
        vQ        = np.arange(0, 1, 1/100000)
        
        # in Minutes
        vExpected = st.lognorm.ppf(vQ, vLogNormMin[0],
                                       loc = vLogNormMin[1],
                                       scale = vLogNormMin[2])
            
        print("Observed Min/Max: (%.2f, %.2f)" % (min(vSerMin), max(vSerMin)))
        print("Expected Min/Max: (%.2f, %.2f)" % (min(vExpected), max(vExpected)), "\n")
        
        st.probplot(vSerMin, dist = st.lognorm(vLogNormMin[0], 
                                               loc = vLogNormMin[1],
                                               scale = vLogNormMin[2]), plot = plt)
        plt.title("QQ Plot: Log Normal Service Times Over Whole Year")
        plt.ylabel("Observed Values (Minutes)")
        plt.grid()
        plt.tight_layout()
        plt.show()
        

### Start Main
if __name__ == "__main__":
    main()
    
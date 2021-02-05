"""
Class: Operations Research II
Case: 1
Date: 05/02/21
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
    jjjjj

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


### My Functions
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


def poissonPMF(dLambda, vK):
    """Create a Vector of Poisson Probabilities Given Lambda and a Max K for the x-axis"""
    
    vProb = np.zeros(len(vK))
    
    for i in range(len(vK)):
        # vProb[i] = (dLambda ** vK[i]) * (np.exp(-dLambda)) / (np.math.factorial(vK[i]))
        vProb[i] = st.poisson.pmf(vK[i], dLambda)
    
    return vProb


def binCustom(vX, vBins):
    """creates vector of bins counts given a data vector vX and a vector of custom bin locations"""
    
    vY = np.zeros(len(vBins) - 1)

    for i in range(len(vX)):
        for j in range(len(vBins) - 1):
            # print("vX[i]:", vX[i])
            # print("Current Bin [%d, %d)" %(vBins[j], vBins[j+1]))
            
            if vX[i] >= vBins[j] and vX[i] < vBins[j+1]:
                vY[j] += 1
                break
            
        if vX[i] == vBins[-1]:                          # catches case where last datapoint = upper bound
            vY[-1] += 1
            # print("DataPoint = %d Matches Upper Bound" % vX[i])
    
    return vY


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
    
    
    ### Boolean Switches
    bViz = 0
    
    
    ### Magics
    iArrRows, iArrCols = mArr.shape
    iSerRows, iSerCosl = mSer.shape
    
    
    ### Initial Visualizations

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
    
    
    ### Assignment A: Modelling
    
    ## Part I
    
    # Plot Poisson Distributions for Given Lambda's
 
    iInd     = 1                                                # set index of hour to get its lambda
    dLambda  = vArrColAvg[iInd]                                 # grab ML lambda from that index   
    vK       = np.arange(1, (np.floor(dLambda * 2) + 1))        # set vector of suitable K's
    vPoisson = poissonPMF(dLambda, vK)                          # obtain Poisson probabilities
    iBins    = 10                                               # set number of bins for ChiSq binning
    
    plt.plot(vK, vPoisson, color = "darkred")
    plt.title("Poisson Distribution of Arrivals in Hour: %s" %(sHourRange[iInd]))
    plt.show()
    
    
    # # Q-Q Plots Over Each Hour Timeframe
    # for i in range(len(vArrColAvg)):
    #     iInd      = i                                           # resets iInd to run through each column
    #     vObserved = mArr[:,iInd]                                # resets vObserved for each column
    #     dLambda   = vArrColAvg[iInd]                            # resets dLambda for each column
        
    #     st.probplot(vObserved, dist = 'poisson', sparams = (dLambda), plot = plt)
    #     plt.title("QQ Plot: Poisson Arrivals in Hour %s" %(sHourRange[iInd]))
    #     plt.grid()
    #     plt.tight_layout()
    #     plt.show()
    
    
    # ChiSquared Test
    vObserved = mArr[:, iInd]                                   # retrieve observed column data
    np.random.seed(1)                                           # set seed for random Poisson
    vK = np.arange(0, max(vObserved))
    vExpected = np.random.poisson(dLambda, iArrRows)            # generate simulated Poisson data
    vQcut     = pd.qcut(vExpected, q = iBins, retbins = True)   # use qcut to obtain balanced bins
    vBins     = vQcut[1]                                        # store bins partition results
    
    print(vBins)

    vExpBin = binCustom(vExpected, vBins)                       # counts of expected dist in bins
    vObsBin = binCustom(vObserved, vBins)                       # counts of observed dist in bins
    
    print(vExpBin)
    print(vObsBin)    
    
    print(st.chisquare(vObsBin, vExpBin, ddof = iBins - 2))
    print(chiSqTval(vObsBin, vExpBin))
    
    
    
    
    
    
        

        
    
    

    

    
    
 
### Start Main
if __name__ == "__main__":
    main()
    

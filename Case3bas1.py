# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 11:09:27 2021

@author: basbr
"""

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import scipy.optimize as opt


def error(dLoc, dSigma, dTarget):
    """
    Purpose: to compute the squared error between log-normal's mean with target mean
    
    Input:
        dLoc    location parameter
        dSigma  sigma parameter
        dTarget target mean
    
    Output:
        dError  squared error
    """
    dMean  = st.lognorm(dSigma, loc=dLoc).mean()[0]
    dError = (dMean - dTarget)**2
    
    return dError


def simulateLogNormal(iN, dLoc, dSigma, iSeed):
    """
    Purpose: to simulate data from the log-normal distribution
    
    Input:
        iN      number of realisations to compute 
        dMean   mu parameter for the log-normal distribution
        dSigma  sigma parameter for the log-normal distribution
        
    Output:
        vY      a vector (size iN, type floats) of realisations from the log-normal distribution
    """
    #dScale = np.exp(dMean)  # set mu parameter
    vX     = np.zeros(iN)
    
    np.random.seed(iSeed)
    
    for i in range(iN):
        iProb = np.random.rand()
        dX    = st.lognorm.ppf(iProb, loc=dLoc, s=dSigma)
        vX[i] = dX
    
    return vX


def waitingTimes(vDurations, vSchedule):
    """
    Purpose: to compute the waiting times given the consult durations and appointment schedule
        
    Input:
        vDurations  vector of consult durations (integers)
        vSchedule   vector of appoint times (integers from zero)
        
    Output:
        vWaitingTimes   vector of waiting times (integers) 
    """
    iN            = len(vDurations) 
    vWaitingTimes = np.zeros(iN)
    
    for i in range(1, iN):
        iStartPrev  = vSchedule[i-1] + vWaitingTimes[i-1]   # starting time of previous appointment
        iFinishPrev = iStartPrev + vDurations[i-1]          # ending time of previous appointment
        
        iAppTime         = vSchedule[i]                     # appointment time
        iStart           = max(iFinishPrev, iAppTime)       # determine starting time of current appointment
        vWaitingTimes[i] = iStart - iAppTime                # compute waiting time
    
    return vWaitingTimes
    

def variance(vX):
    """
    Purpose: to compute the standard deviation for a vector of data
        
    Input:
        vX      vector of data
        
    Output:
        dVar    float, variance   
    """
    
    iN    = len(vX)
    dMean = np.mean(vX) 
    dSum  = 0
    
    for i in range(iN):
        dDiff = vX[i] - dMean
        dSum += dDiff**2
    
    dSTD = dSum / (iN - 1)
    
    return dSTD
    

def partA(iPatients, dMeanTime, iDueTime, iN=100):
    """
    Purpose: 
        
    Input:
        iPatients   number of patients
        dMeanTime   average consult time
        iDueTime    time (in minutes from zero) we want the shift to be over
        iN          number of times to run the simulation
        
    Output:
        We print some results       
    """
    
    # set the parameters
    dTheta = 1.5
    dSigma = 1.0
    
    # find location parameter required to get a mean of dMeanTime given sigma 
    vOpt   = opt.minimize(error, x0 = 0, args = (dSigma, dMeanTime), method = 'BFGS')
    dLoc   = vOpt.x[0]
    
    # set appointment times
    dTimeBtwnApps = dTheta * dMeanTime  # time between appointments
    vSchedule     = np.zeros(iPatients) # vector to store appointment times (in minutes starting at 0)
    
    for i in range(1, iPatients):
        dPrevious    = vSchedule[i-1]
        dCurrent     = dPrevious + dTimeBtwnApps
        vSchedule[i] = dCurrent 
    
    # run simulations
    vY = np.zeros(iN)   # vector to store outcomes (i.e. waiting times + overtime)
    
    for i in range(iN):
        iSeed         = i                                                   # set seed
        vDurations    = simulateLogNormal(iPatients, dLoc, dSigma, iSeed)   # get random sample of consult durations
        vWaitingTimes = waitingTimes(vDurations, vSchedule)                 # compute patients' waiting times
        
        # compute doctor's overtime
        iStartLast  = vSchedule[-1] + vWaitingTimes[-1]                     # starting time of last patient
        iFinishLast = iStartLast + vDurations[-1]                           # ending time of last patient
        iOverTime   =  max((0,iFinishLast-iDueTime))                        # overtime
        
        vY[i] = sum(vWaitingTimes) + iOverTime                              # compute and store outcome
        
    
    # compute findings
    dEST = vY.mean()
    dVar = variance(vY)
    dSE  = np.sqrt(dVar/iN)
    dX   = dSE/dEST
    
    # report findings and parameters
    print('We ran some simulations')
    print('We chose a theta parameter of:', dTheta)
    print('The distribution we used for the consult times was log-normal')
    print('We set sigma to be %.1f, which led to a location parameter of %.2f' %(dSigma, dLoc))
    print('The average outcome over %i simulations, was %.2f minutes' %(iN, dEST))
    print('The standard error was %.2f' %dSE)
    print('SE/EST came out at %.3f' %dX)
    print('Finally, the 95% confidence interval:')
    
    
def main():
    # input
    iPatients  = 18
    dMeanTime  = 10  # average consulting time in minutes
    iStartTime = 9
    iDueTime   = 13
    
    iDueTimeMins = (iDueTime - iStartTime) * 60
    
    partA(iPatients,dMeanTime, iDueTimeMins)
    
    vX = np.arange(0,20,0.01)
    vLogN = st.lognorm.pdf(vX,loc=dMeanTime, s=1)
    plt.plot(vX, vLogN)
    plt.show()
    
    
    
if __name__ == "__main__":
    main()
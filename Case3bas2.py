# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 17:20:41 2021

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

def getSchedule(iPatients, dMeanTime, dTheta):
    """
    Purpose: to set up the patients' appointments time from zero
        
    Input:
        iPatients   number of patients
        dMeanTime   average consult time
        dTheta      spacing parameter (float) for setting up the schedule
        
    Output:
        vSchedule   vector of appoint times (integers from zero)
    """
    dTimeBtwnApps = dTheta * dMeanTime  # time between appointments
    vSchedule     = np.zeros(iPatients) # vector to store appointment times (in minutes starting at 0)
    
    for i in range(1, iPatients):
        dPrevious    = vSchedule[i-1]
        dCurrent     = dPrevious + dTimeBtwnApps
        vSchedule[i] = dCurrent 
        
    return vSchedule

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

def getWaitingTimes(vSchedule, vDurations):
    """
    Purpose: to compute the waiting times given the consult durations and appointment schedule
        
    Input:
        vSchedule   vector of appoint times (integers from zero)
        vDurations  vector of consult durations (integers)
        
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

def getYVariable(vSchedule, vDurations, iDueTime):
    """
    Purpose: to compute the Y variable (patients' waiting times + doctor's overtime)
        
    Input:
        vSchedule   vector of appoint times (integers from zero)
        vDurations  vector of consult durations (integers)
        iDueTime    time (in minutes from zero) we want the shift to be over      
        
    Output:
        dY          Y variable (float) 
    """
    vWaitingTimes = getWaitingTimes(vSchedule, vDurations)
    
    # compute doctor's overtime
    iStartLast  = vSchedule[-1] + vWaitingTimes[-1]     # starting time of last patient
    iFinishLast = iStartLast + vDurations[-1]           # ending time of last patient
    iOverTime   =  max((0,iFinishLast-iDueTime))        # overtime
        
    dY = sum(vWaitingTimes) + iOverTime
    
    return dY

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

def reportFindings(vY, dSigma, dLoc, dTheta):
    """
    Purpose: to compute the standard deviation for a vector of data
        
    Input:
        vY      vector of Y variables (floats)
        
    Output:
        We print several results   
    """
    iN = len(vY)
    
    # compute findings
    dEST = vY.mean()
    dVar = variance(vY)
    dSE  = np.sqrt(dVar/iN)
    dX   = dSE/dEST
    dLB  = dEST - 1.96 * dSE    # lower bound of the 95% confidence interval
    dUB  = dEST + 1.96 * dSE    # upper bound of the 95% confidence interval
    
    # report findings and parameters
    print('We ran some simulations')
    print('We chose a theta parameter of: %.2f' %dTheta)
    print('The distribution we used for the consult times was log-normal')
    print('We set sigma to be %.1f, which led to a location parameter of %.2f' %(dSigma, dLoc))
    print('The average outcome over %i simulations, was %.2f minutes' %(iN, dEST))
    print('The standard error was %.2f' %dSE)
    print('SE/EST came out at %.3f' %dX)
    print('Finally, the 95%% confidence interval:(%.2f, %.2f)' %(dLB,dUB) ,'\n')


def partA(iPatients, dMeanTime, iDueTime, iN=1000, dTheta=1.5, dSigma=1.0):
    """
    Purpose: 
        
    Input:
        iPatients   number of patients
        dMeanTime   average consult time
        iDueTime    time (in minutes from zero) we want the shift to be over
        iN          number of times to run the simulation
        dTheta      spacing parameter (float) for setting up the schedule
        dSigma      sigma parameter (float) for the log-normal distribution 
        
    Output:
        We print some results
    """
    # find location parameter required to get a mean of dMeanTime given sigma 
    vOpt   = opt.minimize(error, x0 = 0, args = (dSigma, dMeanTime), method = 'BFGS')
    dLoc   = vOpt.x[0]
    
    vSchedule = getSchedule(iPatients, dMeanTime, dTheta)   # set appointment times
    
    # run simulations
    vY = np.zeros(iN)   # vector to store outcomes (i.e. waiting times + overtime)
    
    for i in range(iN):
        iSeed      = i                                                   # set seed
        vDurations = simulateLogNormal(iPatients, dLoc, dSigma, iSeed)   # get random sample of consult durations
        
        dY    = getYVariable(vSchedule, vDurations, iDueTime)
        vY[i] = dY
    
    reportFindings(vY,dSigma,dLoc,dTheta)
    
    """
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
    print('Finally, the 95% confidence interval:', '\n')
    """

def partB(iPatients, dMeanTime, iDueTime, iN=100, dSigma=1.0):
    """
    Purpose: to run the simulations of part A for different levels of theta and see which schedule works best
    
    Input:
        iPatients   number of patients
        dMeanTime   average consult time
        iDueTime    time (in minutes from zero) we want the shift to be over   
        iN          number of times to run the simulation
        dSigma      sigma parameter (float) for the log-normal distribution 
        
    Output:
        
    """
    vTheta = np.arange(1,2,0.1)     # vector of theta's to be used in simulation
    
    # create (seeded) random consult times
    vOpt       = opt.minimize(error, x0 = 0, args = (dSigma, dMeanTime), method = 'BFGS')
    dLoc       = vOpt.x[0]                                                  
    mDurations = np.zeros((iN,iPatients))
    
    for i in range(iN):
        iSeed         = i
        vDurations    = simulateLogNormal(iPatients, dLoc, dSigma, iSeed)   # get random sample of consult durations
        mDurations[i] = vDurations
    
    # for each theta and each set of consult durations (iN times iPatients numbers), compute y's and plot them
    iM = len(vTheta)
    vY = np.zeros(iM)   # vector to store y's   
    
    for i in range(iM):
        dTheta    = vTheta[i]
        vSchedule = getSchedule(iPatients, dMeanTime, dTheta)   # set appointment times
        vObs      = np.zeros(iN)
        
        for j in range(iN):
            vDurations = mDurations[j]
            dY         = getYVariable(vSchedule, vDurations, iDueTime)
            vObs[j]    = dY
        
        reportFindings(vObs,dSigma,dLoc,dTheta)
        vY[i] = vObs.mean()
    
    # plot results
    plt.plot(vTheta,vY)
    plt.plot(vTheta, vY, 'o')
    plt.xlabel('Theta')
    plt.ylabel('Waiting time + overtime')
    plt.show()
    

def main():
    # input
    iPatients  = 18
    dMeanTime  = 10  # average consulting time in minutes
    iStartTime = 9
    iDueTime   = 13
    
    iDueTimeMins = (iDueTime - iStartTime) * 60
    
    partA(iPatients, dMeanTime, iDueTimeMins, dTheta=1.5)
    partB(iPatients, dMeanTime, iDueTimeMins)
    
    
if __name__ == "__main__":
    main()
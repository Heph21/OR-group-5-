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
    Purpose: to compute the variation for a vector of data
        
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
    
    dVar = dSum / (iN - 1)
    
    return dVar

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
    
    #ddof(1)
    dVar = variance(vY)
    dSE  = np.sqrt(dVar/iN)
    dX   = dSE/dEST
    dLB  = dEST - 1.96 * dSE    # lower bound of the 95% confidence interval
    dUB  = dEST + 1.96 * dSE    # upper bound of the 95% confidence interval
    
    # report findings and parameters
    print('Distribution: log-normal')
    print('Theta: %.2f' %dTheta)
    print('Sigma: %.1f' %dSigma)
    print('Location: %.2f' %dLoc)
    print('Number of simulations: %i' %iN)
    print('Outcome(minutes): %.2f' %dEST)
    print('Standard error: %.2f' %dSE)
    print('SE/EST: %.3f' %dX)
    print('95%% confidence interval:(%.2f, %.2f)' %(dLB,dUB) ,'\n')


def partA(iPatients, dMeanTime, iDueTime, iN=1000, dTheta=1.5, dSigma=1.0):
    """
    Purpose: to run simulations of consults and see how well a chosen schedule perfoms
        
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


def partB(iPatients, dMeanTime, iDueTime, iN=250, dSigma=.8):
    """
    Purpose: to run the simulations of part A for different levels of theta and see which schedule works best
    
    Input:
        iPatients   number of patients
        dMeanTime   average consult time
        iDueTime    time (in minutes from zero) we want the shift to be over   
        iN          number of times to run the simulation
        dSigma      sigma parameter (float) for the log-normal distribution 
        
    Output:
        We print some results for different levels of theta
        We plot the Y variable for different levels of theta
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
    sLabel = 'Sigma: %.2f' %dSigma
    
    plt.plot(vTheta,vY)
    plt.plot(vTheta, vY, 'o', label=sLabel)
    plt.xlabel('Theta')
    plt.ylabel('Waiting time + overtime')
    plt.legend()
    plt.show()
    
    return vY
    

def partC(iPatients, dMeanTime, iDueTime):
    """
    Purpose: to run the simulations of part A for different levels of sigma and see what happens as the variance increases
        
    Input:
        iPatients   number of patients
        dMeanTime   average consult time
        iDueTime    time (in minutes from zero) we want the shift to be over
        iN          number of times to run the simulation
        dTheta      spacing parameter (float) for setting up the schedule
        
    Output:
        We print some results for different levels of sigma
        We plot the Y variable for different levels of sigma  
    """
    vSigma = np.arange(.5,2,0.05)     # vector of sigma's to be used in simulation
    
    # for each sigma and each set of consult durations (iN times iPatients numbers), compute y's and plot them
    iM = len(vSigma)
    vY = np.zeros(iM)   # vector to store y's   
    
    for i in range(iM):
        dSigma = vSigma[i]
        vY[i]  = partB(iPatients, dMeanTime, iDueTime, dSigma=dSigma).min()
    
    # plot results
    plt.plot(vSigma,vY)
    plt.plot(vSigma, vY, 'o')
    plt.xlabel('Sigma')
    plt.ylabel('Waiting time + overtime')
    plt.show()


def partD(iPatients, dMeanTime, iDueTime, iN=1000, dTheta=1.5, dSigma=1.0):
    """
    Purpose: to run simulations of consults for an alternative schedule
        
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
    vSchedule1 = [0,13,26,39,60,73,86,99,120,133,146,159,180,193,206,219,240,253]
    vSchedule2 = [0,10,20,30,60,70,80,90,120,130,140,150,180,190,200,210,240,250]
    vSchedule3 = [0,15,25,40,60,75,85,100,120,135,145,160,180,195,205,220,240,255]
    vSchedule4 = [0,12,24,36,48,60,72,84,96,108,120,132,144,156,168,180,192,204]
    vSchedule5 = [0,12,30,42,60,72,90,102,120,132,150,162,180,192,210,222,240,252]
    vSchedule6 = [0,10,25,35,50,60,70,85,95,110,120,130,145,155,170,180,190,205]
    vSchedule7 = [0,10,25,35,50,60,75,85,100,110,125,135,150,160,175,185,200,210]
    
    vSchedule = vSchedule6
    
    # find location parameter required to get a mean of dMeanTime given sigma 
    vOpt   = opt.minimize(error, x0 = 0, args = (dSigma, dMeanTime), method = 'BFGS')
    dLoc   = vOpt.x[0]
    
    # run simulations
    vY = np.zeros(iN)   # vector to store outcomes (i.e. waiting times + overtime)
    
    for i in range(iN):
        iSeed      = i                                                   # set seed
        vDurations = simulateLogNormal(iPatients, dLoc, dSigma, iSeed)   # get random sample of consult durations
        
        dY    = getYVariable(vSchedule, vDurations, iDueTime)
        vY[i] = dY
    
    reportFindings(vY,dSigma,dLoc,dTheta)
    

def main():
    # input
    iPatients  = 18
    dMeanTime  = 10  # average consulting time in minutes
    iStartTime = 9
    iDueTime   = 13
    
    iDueTimeMins = (iDueTime - iStartTime) * 60
    
    print('PART A:\n')
    partA(iPatients, dMeanTime, iDueTimeMins, dTheta=1.5)
    
    print('PART B:\n')
    vY = partB(iPatients, dMeanTime, iDueTimeMins)
    
    print('PART C:\n')
    partC(iPatients, dMeanTime, iDueTimeMins)
    
    print('PART D:\n')
    partD(iPatients, dMeanTime, iDueTimeMins, dSigma=.8, dTheta=1.3)
    
    
if __name__ == "__main__":
    main()
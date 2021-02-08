# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 11:07:43 2021

@author: basbr
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st

### Auxiliary functions ###
def plotDistribution(vData, vDistribution, sDistribution):
    #plt.figure()
    plt.title('Distribution')
    if(sDistribution == 'Poisson'):
        plt.xlabel('Number of calls')
    else:
        plt.xlabel('Call duration')
    plt.ylabel('Probability / number of observations')
    plt.hist(vData, density=True, rwidth=.8, color='k')
    plt.plot(vDistribution, label=sDistribution, color='r')
    plt.legend()
    #plt.show()

def checkFitCalls(mData, vParameter, sDistribution):
    """
    Purpose: to check the fit of the Poisson distribution to the observed data
        
    Input:
        -mCalls, matrix of integers, represents calls received per hour (columns) and day (rows)
        -vL, vector of doubles, represents lambda parameter needed for the Poisson distribution for each hour
    
    Output:
        -plots for each hour, containing 
            in the first subplot: the Poisson distribution given the lambda and a histogram of the observed amount of calls in that hour
            in the second subplot: a QQ plot showing the goodness of fit
        -results from a chi-squared test
    """
    vLabels= ['7-8','8-9','9-10','10-11','11-12','12-13','13-14','14-15','15-16','16-17','17-18','18-19','19-20','20-21']
    iM= len(mData[:,0])
    iN= len(mData[0])
        
    for i in range(iN):
        vData= mData[:,i]
        dParameter= vParameter[i]
        
        dMin= np.min(vData)
        dMax= np.max(vData)
        dStepSize= dMax/iM
        vK= np.arange(dMax)
        #vK= np.linspace(dMin, dMax, iM)
        vDistribution= st.poisson.pmf(vK,dParameter)
        sLabel= vLabels[i]
        
        plt.figure()
        plt.subplot(1,2,1)
        plotDistribution(vData, vDistribution, sDistribution)
        plt.subplot(1,2,2)
        st.probplot(vData, dist=st.poisson(dParameter), plot=plt)
        plt.suptitle(sLabel)
        plt.show()
               
        #vDataBinned= st.binned_statistic(vData,vData, bins=5)
        #vExpObs= st.binned_statistic(vDistribution * iM, vDistribution * iM, bins=5)
        #print(vDataBinned)
        #print(vExpObs)
        #print(st.chisquare(vDataBinned,vExpObs), '\n')

def checkFitService(vData, dMu, sDistribution):
    sLabel= 'Call Duration'
    
    dMax= np.max(vData)
    vK= np.arange(dMax)
    iScale= 1/dMu
    
    vDistribution= st.expon.pdf(vK, scale=iScale)
    
    plt.figure()
    plt.subplot(1,2,1)
    plotDistribution(vData, vDistribution, sDistribution)
    plt.subplot(1,2,2)
    st.probplot(vData, dist=st.expon(scale=iScale), plot=plt)
    plt.show()

def plotQQBivariate(vData, dPar1, dPar2, sDistribution):
    if(sDistribution == 'Weibull'):
        st.probplot(vData, dist=st.weibull_min(dPar1,dPar2), plot=plt)
    elif(sDistribution == 'Gamma'):
        st.probplot(vData, dist=st.weibull_min(dPar1,dPar2), plot=plt)
    elif(sDistribution == 'Lognormal'):
        st.probplot(vData, dist=st.lognorm(dPar2,scale=np.exp(dPar1)), plot=plt)

def checkFitBivariate(mData, mParameter, sDistribution):
    """
    Purpose: to check the fit of the Poisson distribution to the observed data
        
    Input:
        -mCalls, matrix of integers, represents calls received per hour (columns) and day (rows)
        -vL, vector of doubles, represents lambda parameter needed for the Poisson distribution for each hour
    
    Output:
        -plots for each hour, containing 
            in the first subplot: the Poisson distribution given the lambda and a histogram of the observed amount of calls in that hour
            in the second subplot: a QQ plot showing the goodness of fit
        -results from a chi-squared test
    """
    vLabels= ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    #iM= len(mData[:,0])
    iN= len(mData[0])
        
    for i in range(iN):
        vData= mData[:,i]
        dPar1= mParameter[i,0]
        dPar2= mParameter[i,1]
        sLabel= vLabels[i]
        
        dMax= np.max(vData)
        #dStepSize= dMax/iM
        vK= np.arange(dMax)
        
        if(sDistribution == 'Weibull'):
            vDistribution= st.weibull.pdf(vK,dPar1,dPar2)
        elif(sDistribution == 'Gamma'):
            dShape, dLoc, dScale= st.gamma.fit(vData)
            vDistribution= st.gamma.pdf(vData, dShape, loc=dLoc, scale=dScale)
            #dLoc, dScale= st.expon.fit(vData)
            #vDistribution= st.expon.pdf(vData, loc=dLoc, scale=dScale)
        elif(sDistribution == 'Lognormal'):
            dMu= dPar1
            dSigma= dPar2
            vDistribution= st.lognorm.pdf(vK,dSigma,scale=np.exp(dMu))
        else:
            print('Something went wrong')
            return
        
        plt.figure()
        plt.subplot(1,2,1)
        plotDistribution(vData, vDistribution, sDistribution)
        plt.subplot(1,2,2)
        plotQQBivariate(vData,dPar1, dPar2, sDistribution)
        plt.suptitle(sLabel)
        plt.show()
        
        #print(st.chisquare(vCalls,vPoisson), '\n')    

def waitingTime(dMu, dRho, iAgents, t=1/120):
    dSumThingy= 0
    for i in range(iAgents):
        dSumThingy+= (dRho**i) / np.math.factorial(i)
    
    dFirstFactor= dRho**iAgents / np.math.factorial(iAgents)
    dSecondFactor= (1 - dRho/iAgents) * dSumThingy + dFirstFactor

    dPi= dFirstFactor / dSecondFactor
    
    dW= 1 - dPi * np.exp(-dMu * (iAgents - dRho) * t)
    
    return dW

def getTimesAndAgents(dMu, dRho, dSL, sLabel, iN):
    a= np.fix(dRho+1).astype(int)
    b= np.fix(dRho+1+iN).astype(int)
        
    vAgents= np.arange(a,b)
    vWaitingTime= np.zeros(b-a)
       
    for i in range(a,b):
        dWaitingTime= waitingTime(dMu, dRho, i) 
        vWaitingTime[i-a]= dWaitingTime
            
    plt.figure()
    plt.title('SL = W(30 secs)')
    plt.scatter(vAgents, vWaitingTime, label=sLabel, color='k')
    plt.hlines(dSL, a, b, color='r')
    plt.legend()
    plt.show()
    
    return vWaitingTime, vAgents

### The three parts ###
def partA(sCalls, sService):
    
    # first part: incoming calls
    mCalls= np.genfromtxt(sCalls)
    iHours= len(mCalls[0])
    vL= np.zeros(iHours)    # vector of lambdas per hour
    
    ## estimate lambdas (max. likelihood estimator is mean of observations)
    for i in range(iHours):
        vL[i]= np.mean(mCalls[:,i])
    
    checkFitCalls(mCalls, vL, 'Poisson')
    
    # second part: service time
    mService= np.genfromtxt(sService)
    iMonths= len(mService[0])
    vM= np.zeros(iMonths)   # vector of mu's per month
    vService= mService[:,0]
    
    for i in range(1,iMonths):
        vService= np.hstack((vService, mService[:,i]))
    
    ## estimate mu's (ML estimator is 1/mean of observations)
    dMean= np.mean(vService) 
    dMu= 1/dMean
    
    checkFitService(vService, dMu, 'Exponential')
    
    # third part: try some other distributions to model the service time
    mParametersLognormal= np.zeros((iMonths, 2)) 
    
    ## mean and standard deviations; parameters for log-normal
    for i in range(iMonths):
        dMu= np.mean(np.log(mService[:,i]))
        dSigma= np.mean(((np.log(mService[:,i]) - dMu))**2)
        mParametersLognormal[i,0]= dMu
        mParametersLognormal[i,1]= dSigma
    
    #checkFitBivariate(mService, mParametersLognormal, 'Lognormal')
    #checkFitBivariate(mService, mParametersLognormal, 'Gamma')
            
    #checkFitBivariate(mService, vM, 'Weibull')
    #checkFitBivariate(mService, vM, 'Gamma')
    
    return vL, dMu

def partB(vLambda, dMu, iN= 10):
    vLabels= ['7-8','8-9','9-10','10-11','11-12','12-13','13-14','14-15','15-16','16-17','17-18','18-19','19-20','20-21']
    dSL= 0.8
    iM= len(vLambda)
    mWaitingTime= np.zeros((iM,iN))
    mAgents= np.zeros((iM,iN))
    vAgentsSL= np.zeros(iM)
    
    for i in range(iM):
        dRho= vLambda[i]/dMu
        sLabel= vLabels[i]
        
        vWaitingTime, vAgents= getTimesAndAgents(dMu, dRho, dSL, sLabel, iN)
        mWaitingTime[i]= vWaitingTime
        mAgents[i]= vAgents
    
        for j in range(iN):
            if(vWaitingTime[j] >= dSL):
                vAgentsSL[i]= vAgents[j]  
                break
            #print('This service level seems infeasible')
    
    print('\nThe amount of agents to meet the SL at each hour, is:', vAgentsSL)
    
def main():
    sCalls= 'ccarrdata.txt'
    sService= 'ccserdata.txt'
    
    vLambda, vMu= partA(sCalls, sService)
    dMu= np.mean(vMu)
    partB(vLambda, dMu)

if __name__ == "__main__":
    main()
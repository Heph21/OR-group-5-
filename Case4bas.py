# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 14:15:11 2021

@author: basbr
"""
import numpy as np
import scipy.stats as st

# numbers that determine the game (global)
iCost  = 5
iLow   = 0
iHigh  = 10
iTurns = 5

def computeCombinations(vI):
    """
    Purpose: to compute the maximal expected profit given the state vector vI 
    
    Input: 
        vI, state vector containing integers (both positive and negative), the numbers left to play with
        
    Output:
        dRes, maximal expected profit
    """
    iN    = len(vI)
    dProb = 1/iN
    dRes  = 0
    
    for i in range(iN):
        vJ    = np.delete(vI, i)    # possible state transition
        dRes += dProb * vI[i]
        dProf = expProfit(vJ)       # expected profit of the new state
        dRes += dProb * dProf       # add this profit, times the probability of the transition, to the value function
    
    return dRes

def dataType(vI):
    """
    Purpose: to check whether the vector contains both positive and negative elements
    
    Input:
        vI, vector of integers
        
    Output:
        We return:
            1   if the vector contains only positive numbers (and zeros)
            0   if the vector contains both positive and negative numbers
            -1  if the vector contains no positive numbers
    """
    iPos = 0    # keep count of strictly positive elements of vI
    iNeg = 0    # keep count of strictly negative elements of vI
    
    for i in vI:
        if(i > 0):
            iPos += 1
        elif(i < 0):
            iNeg += 1
    
    if((iPos > 0) & (iNeg == 0)):
        return 1
    elif((iPos == 0) & (iNeg >= 0)):
        return -1
    else:
        return 0

def expProfit(vI):
    """
    Purpose: to compute the optimal expected profit given the state vector vI
    
    Input:
        vI, state vector containing integers, the numbers left to play with
    
    Output:
        dRes, optimal expected profit    
    """
    dRes = 0
    
    iType = dataType(vI) 
    
    # compute expected profit considering distribution of elements
    if(iType == 1):
        dRes += sum(vI)                             # vI contains positive numbers, so the expected profit is the sum of the elements
    elif(iType == 0):
        dRes += max(0, computeCombinations(vI))     # vI contains both positive and negative numbers, so we need to some more computations
    
    # note that if vI contains no positive numbers, the expected profit is 0
    
    return dRes

def valleyDistribution():
    vA = np.arange(iLow, iHigh+1)
    vP = np.array([15,13,11,9,6,2,6,9,11,13,15])
    vP = vP/110
    
    return(vA, vP)

def peakDistribution():
    vA = np.arange(iLow, iHigh+1)
    vP = np.array([1,3,5,7,9,60,9,7,5,3,1])
    vP = vP/110
    
    return(vA, vP)
    
def runSimulation(iN=1000, bPrint=0, sDistr='uniform'):
    """
    Purpose: to run a number of simulations and compute the optimal expected profit for the game 
    
    Input:
        iN, number of simulations to run
        bPrint, whether or not to print the drawn numbers and their outcome for each simulation
        sDistr, the distribution to be used to draw the 5 random numbers to play with (default is uniform)
        
    Output:
        We print, if wanted, the drawn numbers and their outcome for each simulation
        We return a vector of optimal expected profit for each simulation   
    """
    vSols = np.zeros(iN)    # vector to store solutions of simulations
    
    for i in range(iN):
        vX = np.zeros(iTurns)
        
        for j in range(iTurns):
            #np.random.seed(i*i+j)
            if(sDistr == 'binom'):
                dProb = np.random.rand()
                iX    = st.binom.ppf(dProb, n=iHigh, p=1/2)
            elif(sDistr == 'valley'):
                vA, vP = valleyDistribution()
                iX     = np.random.choice(vA,1,p=vP)
            elif(sDistr == 'peak'):
                vA, vP = peakDistribution()
                iX     = np.random.choice(vA,1,p=vP)
            else:
                iX    = np.random.randint(iLow, iHigh+1)
            vX[j] = iX
        
        vSols[i] = expProfit(vX-iCost)  # compute and store maximal expected profit
        
        if(bPrint == 1):
            print('We drew the following numbers:')
            print(vX)
            print('The maximal expected profit is: %.2f.\n' %vSols[i])
    
    return vSols

def variance(vX):
    """
    Purpose: to compute the variation for a vector of data
        
    Input:
        vX, vector of data
        
    Output:
        dVar, variance   
    """
    iN    = len(vX)
    dMean = np.mean(vX) 
    dSum  = 0
    
    for i in range(iN):
        dDiff = vX[i] - dMean
        dSum += dDiff**2
    
    dVar = dSum / (iN - 1)
    
    return dVar

def reportFindings(vData):
    """
    Purpose: to report some findings on the simulations we've run
    
    Input:
        vData, outcome of the simulation
       
    Output:
        We print the average maximal expected profit
        We print a 95% confidence interval for the maximal expected profit    
    """
    
    # process simulations
    iN   = len(vData)
    dAvg = np.mean(vData)
    dVar = variance(vData)
    dSE  = np.sqrt(dVar/iN)
    dLB  = dAvg - 1.96 * dSE    # lower bound of the 95% confidence interval
    dUB  = dAvg + 1.96 * dSE    # upper bound of the 95% confidence interval

    # report findings    
    print('Over %i simulations, the average optimal expected profit was: %.2f.' %(iN, np.mean(vData)))
    print('95%% confidence interval:(%.2f, %.2f)' %(dLB,dUB) ,'\n')
    
      
def partB():
    """
    Purpose: to solve the game for a specific set of numbers and to run ten more simulations
    
    Input:
    
    Output:
        We print:
            -the specific numbers (from the case assignment) and their maximal expected profit
            -for each simulation, the numbers drawn and the maximal expected outcome
            -the average maximal expected profit over the ten simulations
    """
    # first part: solve for the specific numbers
    vX = np.array([1,3,5,7,9])
    vY = vX - iCost
    
    #vZ= np.array([-1,4])
    #vY = np.array([-3,0,1])
    
    dProf = expProfit(vY)
    print('We drew the following numbers:')
    print(vX)
    print('The maximal expected profit is: %.2f.\n' %dProf)
    
    # second part: draw ten random samples and solve for those
    iN     = 10     # amount of simulations to run
    bPrint = 1
    vSims  = runSimulation(iN, bPrint)
    
    # report findings    
    print('Over %i simulations, the average maximal expected profit was: %.2f.\n' %(iN, np.mean(vSims)))
    
def partC():
    """
    Purpose: to run a number of simulations and compute the maximal expected profit for the game 
    
    Input:
        
    Output:
        We print the average maximal expected profit
        We print a 95% confidence interval for the maximal expected profit
    """
    iN     = 1000
    bPrint = 0
    vSims  = runSimulation(iN, bPrint)
    
    reportFindings(vSims)
    
def partD():
    """
    Purpose: to run some simulations where the numbers we play with come from a non-uniform distribution
    
    Input:
        
    Output:
        For each distribution,
            We print the average maximal expected profit
            We print a 95% confidence interval for the maximal expected profit
        
    """
    # first distribution: binomial with n=10 and p=0.2
    print('First, we try the binomial distribution with n=%i and p=%.2f:' %(iHigh, 1/iCost))
    vBinom = runSimulation(sDistr='binom')
    reportFindings(vBinom)
    
    vNext = runSimulation(sDistr='valley')
    reportFindings(vNext)
    
    vNext = runSimulation(sDistr='peak')
    reportFindings(vNext)
    
    
def main(): 
    partB()
    partC()
    partD()
    

if __name__ == "__main__":
    main()
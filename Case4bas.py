# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 14:15:11 2021

@author: basbr
"""
import numpy as np

# numbers that determine the game
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

def expProfit(vI):
    """
    Purpose: to compute the maximal expected profit given the state vector vI
    
    Input:
        vI, state vector containing integers, the numbers left to play with
    
    Output:
        dRes, maximal expected profit    
    """
    dRes = 0
    iPos = 0    # keep count of strictly positive elements of vI
    iNeg = 0    # keep count of strictly negative elements of vI
    
    # go through elements to count strictly positive and strictly negative elements
    for i in vI:
        if(i > 0):
            iPos += 1
        elif(i < 0):
            iNeg += 1
    
    # compute expected profit considering distribution of elements
    if((iPos > 0) & (iNeg > 0)):
        dRes += max(0, computeCombinations(vI))
    elif(iPos > 0):
        dRes += sum(vI)
    
    return dRes

def runSimulation(iN=1000, bPrint=0):
    """
    Purpose: to run a number of simulations and compute the maximal expected profit for the game 
    
    Input:
        iN, number of simulations to run
        bPrint, whether or not to print the drawn numbers and their outcome for each simulation
        
    Output:
        We print, if wanted, the drawn numbers and their outcome for each simulation
        We return a vector of maximal expected profit for each simulation
        
    """
    vSols = np.zeros(iN)    # vector to store solutions of simulations
    
    for i in range(iN):
        vX = np.zeros(iTurns)
        
        for j in range(iTurns):
            #np.random.seed(i*i+j)
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
    
    # process simulations
    dAvg = np.mean(vSims)
    dVar = variance(vSims)
    dSE  = np.sqrt(dVar/iN)
    dLB  = dAvg - 1.96 * dSE    # lower bound of the 95% confidence interval
    dUB  = dAvg + 1.96 * dSE    # upper bound of the 95% confidence interval

    # report findings    
    print('Over %i simulations, the average maximal expected profit was: %.2f.' %(iN, np.mean(vSims)))
    print('95%% confidence interval:(%.2f, %.2f)' %(dLB,dUB) ,'\n')
    
    
def main(): 
    partB()
    partC()
    

if __name__ == "__main__":
    main()

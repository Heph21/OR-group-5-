# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 14:15:11 2021

@author: basbr
"""
import numpy as np


def computeCombinations(vI):
    """
    Purpose: to compute the maximal expected profit given the state vector vI 
    
    Input: 
        -vI, state vector containing integers (both positive and negative), the numbers left to play with
        
    Output:
        -dRes, maximal expected profit
    """
    iN    = len(vI)
    dProb = 1/iN
    dRes  = 0
    
    for i in range(iN):
        vJ    = np.delete(vI, i)    # possible state transition
        dProf = expProfit(vJ)       # expected profit of the new state
        dRes += dProb * dProf       # add this profit, times the probability of the transition, to the value function
    
    return dRes

def expProfit(vI):
    """
    Purpose: to compute the maximal expected profit given the state vector vI
    
    Input:
        -vI, state vector containing integers, the numbers left to play with
    
    Output:
        -dRes, maximal expected profit    
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
        dRes += computeCombinations(vI)
    elif(iPos > 0):
        dRes += sum(vI)
    
    return dRes
      
def partB():
    """
    Purpose:
    
    Input:
        
    
    Output:
        
    """
    vX = np.array([1,3,5,7,9])
    vY = vX - 5
    
    vZ= np.array([-5,4])
    
    dProf = expProfit(vZ)
    print(dProf)

def main():
    
    partB()
    

if __name__ == "__main__":
    main()
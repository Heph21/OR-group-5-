# -*- coding: utf-8 -*-
"""
Created on Tue May 11 19:17:25 2021

@author: basbr
"""
import numpy as np
np.set_printoptions(precision=2)

transitions = {
        1: 
            [[.6,.4,0,0,0],
             [0,.5,.3,.2,0],
             [0,0,.4,.3,.3],
             [0,0,0,.5,.5],
             [0,0,0,0,1]],
        2:
            [[.8,.2,0,0,0],
             [0,.8,.2,0,0],
             [0,.2,.6,.2,0],
             [0,0,.3,.6,.1],
             [0,0,0,.5,.5]],
        3:
            [[1,0,0,0,0],
             [1,0,0,0,0],
             [1,0,0,0,0],
             [1,0,0,0,0],
             [1,0,0,0,0]]
        }

rewards = {
    1: [900,820,100],
    2: [800,720,0],
    3: [600,520,-200],
    4: [400,320,-400],
    5: [100,20,-700]
    }
"""
def areEqual(vA, vB):
    iM = len(vA)
    iN = len(vB)
    
    if(iM != iN):
        return False
    else:
        for i in range(iM):
            if(vA[i] != vB[i]):
                return False
    
    return True
"""

def absoluteError(vA, vB):
    iM   = len(vA)
    iN   = len(vB)
    
    if(iM != iN):
        print('Something went terribly wrong!')
        return 0
    
    vAbsErr = vA - vB
    
    return np.linalg.norm(vAbsErr)

def relativeError(vA, vB):
    iM   = len(vA)
    iN   = len(vB)
    
    if(iM != iN):
        print('Something went terribly wrong!')
        return 0
    
    vRelErr = (vA - vB) / vA
    
    return np.linalg.norm(vRelErr)

def newValueFunc(vV, dAlpha, bPrint=True):
    iM      = len(vV)
    iN      = len(rewards[1])
    mValues = np.zeros((iM,iN))
    
    for i in range(iM):
        iState = i+1
        for j in range(iN):
            iA      = j+1
            dReward = rewards[iState][j]
            
            mValues[i][j] += dReward
            mValues[i][j] += vV @ transitions[iA][i] 
             
    vNewV = np.zeros(iM)
    vR    = np.zeros(iM)
    
    for i in range(iM):
        vValues = mValues[i]
        dMaxV   = np.max(vValues)
        iIndex  = np.argmax(vValues)
        
        vNewV[i] = dMaxV
        vR[i]    = iIndex + 1   # action corresponding to index
    
    return vNewV, vR
    
def totalDiscountedCosts(vInitV, dAlpha, bPrint=True, dEps=.05):
    bConverged = False
    vNewV      = vInitV 
    dStopCrit  = (1-dAlpha) / dAlpha * dEps
    vPolicies  = []
    vValues    = []
    
    if(bPrint):
        print('\nWe use value iteration to estimate the total discounted rewards.')
        print('Using discount factor %.1f and initial value vector' %dAlpha, vInitV, ': \n')
    
    while(bConverged == False):
        vCurrentV = vNewV
        
        # new value computation and determine correpsonding policy
        vNewV, vR = newValueFunc(vCurrentV, dAlpha, bPrint)
    
        # add value function and policy to the record
        vPolicies.append(vR)
        vValues.append(vNewV)
        
        # convergence test
        dRelErr    = relativeError(vNewV, vCurrentV)
        dAbsErr    = absoluteError(vNewV, vCurrentV)
        bConverged = (dRelErr < dStopCrit)
        
        # do some reporting
        print('The updated value function is', vNewV)
        print('The corresponding policy is', vR, '\n')
    
    iIter = len(vPolicies)
    print('It took %i iterations to converge.' %iIter)
    
    return vPolicies, vValues

def main():
    
    # part A
    iStates = len(rewards)
    vAlpha  = [.3,.6,.9]
    vInitV = np.zeros(iStates)
    
    #vPolicies, vValues = totalDiscountedCosts(vInitV, vAlpha[1])
    
    for dAlpha in vAlpha:
        vPolicies, vValues = totalDiscountedCosts(vInitV, dAlpha)
    
   
    
if __name__ == "__main__":
    main()
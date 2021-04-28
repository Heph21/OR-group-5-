# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 13:09:41 2021

@author: basbr
"""
import numpy as np

transitions = {
        1: 
            [[0,1,0,0,0],
             [0,0,.3,.7,0],
             [0,0,0,.5,.5],
             [.6,0,.4,0,0],
             [0,.8,0,0,.2]],
        2:
            [[0,.5,.5,0,0],
             [0,0,0,1,0],
             [.6,0,0,0,.4],
             [0,0,.3,0,.7],
             [.8,.2,0,0,0]],
        3:
            [[0,0,0,0,1],
             [0,0,.5,.5,0],
             [.6,0,.4,0,0],
             [.7,0,0,0,.3],
             [0,1,0,0,0]],
        4:
            [[.5,0,0,.5,0],
             [.4,0,0,0,.6],
             [0,.7,0,.3,0],
             [0,0,1,0,0],
             [0,.8,0,0,.2]]  
    }
    
costs = {
    1: [6,5,2,6,9],
    2: [4,3,3,1,5],
    3: [8,1,5,5,6],
    4: [7,2,3,2,7]
    }




def stationaryDistr(vR, iPow=15):
    iN = len(vR)
    mP = np.zeros((iN,iN))
    
    for i in range(iN):
        iA    = vR[i]                   # action iA corresponding to state i
        mP[i] = transitions[iA][i]  	# get the corresponding row from the dictionary of transition matrices
    
    mPower = np.linalg.matrix_power(mP, iPow)
    
    return mPower[0]

def costsOfPolicy(vR):
    iN     = len(vR)
    vCosts = np.zeros(iN)
    
    for i in range(iN):
        iA        = vR[i]           # action iA corresonding to state i
        vCosts[i] = costs[iA][i]    # get the corresponding number from the cost dictionary
    
    return vCosts

def averageCosts(vR):
    # compute expected costs V given the policy R
    vPi    = stationaryDistr(vR)
    vCosts = costsOfPolicy(vR)
    dV     = np.dot(vPi, vCosts)
    
    print(vPi, vCosts, dV)
    print(sum(vPi))
      
def valueComputation(vR, dAlpha):  
    iN     = len(vR)
    mTrans = np.zeros((iN,iN))
    vCosts = np.zeros(iN)
    
    for i in range(iN):
        iA       = vR[i] # action iA corresponding to state i
        
        # set up the transition matrix corresponding to the policy
        vTrans    = transitions[iA][i]
        mTrans[i] = vTrans 
        
        # compute the costs for each action in the policy
        vCosts[i] = costs[iA][i]
        
    # set up the system of linear equations with all x's on one side and the costs on the other
    mSystem = np.zeros((iN,iN))
    
    for i in range(iN):
        for j in range(iN):
            mSystem[i][j] = -1 *dAlpha * mTrans[i][j] # the coefficients of the system
            
            if(i == j):
                mSystem[i][j] += 1  # for the diagonal coefficients, we add 1 
    
    # solve the system
    vX = np.linalg.solve(mSystem, vCosts)
    
    return vX

def improvePolicy(vR, vX, dAlpha):
    iM      = len(vX)
    iN      = len(costs)
    mValues = np.zeros((iM,iN))
    vNewR   = np.zeros(iM)
    vDiscV  = np.zeros(iM) 
    
    for i in range(iM):
        for j in range(iN):
            iA = j+1 # action iA
            
            dCosts        = costs[iA][i]
            dDiscountTerm = dAlpha * np.dot(vX, transitions[iA][i])
            #print(iA, dCosts, dDiscountTerm)
            
            mValues[i][j] = dCosts + dDiscountTerm
    #print(mValues)
    
    # find the smallest discounted costs and corresponding action for each state i
    for i in range(iM):
        vOutcomes = mValues[i]
        dDisc     = np.min(vOutcomes)
        iA        = np.argmin(vOutcomes) + 1 # add one since the actions start at 1 rather than 0
        
        vDiscV[i] = dDisc
        vNewR[i]  = iA
    
    print('Policy:', vNewR)
    print('Corresponding discounted costs:', vDiscV, '\n')
        
    return vNewR, vDiscV

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
     
def totalCosts(vR, dAlpha=.5):
    bConverged = False
    vCurrentR = vR
    
    print('The initial policy is:', vR, '\n')
    
    while(bConverged == False):
        vPreviousR = vCurrentR
        
        # value computation
        vX = valueComputation(vR, dAlpha)
    
        # policy improvement
        vCurrentR, vDiscV = improvePolicy(vCurrentR, vX, dAlpha) 
    
        # convergence test
        bConverged = areEqual(vCurrentR, vPreviousR)
    
    print('The optimal policy is:', vCurrentR)
    print('The corresponding discounted costs are:', vDiscV, '\n')


def main():
    vRa = [1,1,2,1,1]
    vRb = [2,2,2,2,2]
    totalCosts(vRa)
    totalCosts(vRb)


if __name__ == "__main__":
    main()
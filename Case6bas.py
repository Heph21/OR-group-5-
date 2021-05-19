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

def newValueFunc(vV, dAlpha):
    iM      = len(vV)
    iN      = len(transitions)
    mValues = np.zeros((iM,iN))
    
    for i in range(iM):
        iState = i+1
        for j in range(iN):
            iA      = j+1
            dReward = rewards[iState][j]
            
            mValues[i][j] += dReward
            mValues[i][j] += dAlpha * (transitions[iA][i] @ vV)
             
    vNewV = np.zeros(iM)
    vR    = np.zeros(iM)
    
    for i in range(iM):
        vValues = mValues[i]
        dMaxV   = np.max(vValues)
        iIndex  = np.argmax(vValues)
        
        vNewV[i] = dMaxV
        vR[i]    = iIndex + 1   # action corresponding to index
    
    return vNewV, vR

def reportResults(vValues, vPolicies):
    iN = len(vValues)
    
    print('At iteration 1, we got a value vector of:', vValues[0])
    print('At iteration 2, we got a value vector of:', vValues[1])
    print('At iteration %i, we got a value vector of:' %(iN-1), vValues[iN-2])
    print('At iteration %i, the final iteration, we got a value vector of:' %iN, vValues[iN-1])
    
    print('The corresponding policies were, respectively:')
    print(vPolicies[0])
    print(vPolicies[1])
    print(vPolicies[iN-2])
    print(vPolicies[iN-1], '\n')
    
def totalDiscountedCosts(vInitV, dAlpha, bPrint=True, dEps=.05):
    bConverged = False
    vNewV      = vInitV 
    dStopCrit  = (1-dAlpha) / dAlpha * dEps
    vValues    = []
    vPolicies  = []
    
    print('\nWe use value iteration to estimate the total discounted rewards.')
    print('Using discount factor %.1f and initial value vector' %dAlpha, vInitV, ': \n')
    
    while(bConverged == False):
        vCurrentV = vNewV
        
        # compute new value and determine corresponding policy
        vNewV, vR = newValueFunc(vCurrentV, dAlpha)
    
        # add value function and policy to the record
        vValues.append(vNewV)
        vPolicies.append(vR)
        
        # convergence test
        dRelErr    = relativeError(vNewV, vCurrentV)
        dAbsErr    = absoluteError(vNewV, vCurrentV)
        bConverged = (dAbsErr < dStopCrit)
        
        # do some reporting
        if(bPrint):
            print('The updated value function is', vNewV)
            print('The corresponding policy is', vR, '\n')
    
    reportResults(vValues, vPolicies)
    
    return vValues, vPolicies

def valueComputation(vR, dAlpha):  
    iN       = len(vR)
    mTrans   = np.zeros((iN,iN))
    vRewards = np.zeros(iN)
    
    for i in range(iN):
        iState = i+1
        iA     = int(vR[i]) # action iA corresponding to state i
        
        # set up the transition matrix corresponding to the policy
        vTrans    = transitions[iA][i]
        mTrans[i] = vTrans 
        
        # compute the rewards for each action in the policy
        vRewards[i] = rewards[iState][iA-1]
        
    # set up the system of linear equations with all x's on one side and the costs on the other
    mSystem = np.zeros((iN,iN))
    
    for i in range(iN):
        for j in range(iN):
            mSystem[i][j] = -1 * dAlpha * mTrans[i][j] # the coefficients of the system
            
            if(i == j):
                mSystem[i][j] += 1  # for the diagonal coefficients, we add 1 
    
    # solve the system
    vX = np.linalg.solve(mSystem, vRewards)
    
    return vX

def qTable(vR, dAlpha):
    iM = len(vR)
    iN = len(transitions)
    mQ = np.zeros((iM,iN))
    
    vV = valueComputation(vR, dAlpha)
    print('The value function vector:')
    print(vV)
    
    for i in range(iM):
        iState = i+1
        for j in range(iN):
            iA      = j+1
            dReward = rewards[iState][j]
            vP      = transitions[iA][i]
            
            mQ[i][j] += dReward
            mQ[i][j] += dAlpha * (vP @ vV)
            
    print('The Q table is:')
    print(mQ)
    
    # maximize the Qs for each state and find optimal policy R
    vQ    = np.zeros(iM)
    vNewR = np.zeros(iM)
    
    for i in range(iM):
        vTempQ = mQ[i]
        dMaxQ  = np.max(vTempQ)
        iIndex = np.argmax(vTempQ)
        
        vQ[i]    = dMaxQ
        vNewR[i] = iIndex+1 
    
    return vV, vQ, vNewR 

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

def checkOptimality(vR, dAlpha):
    print('\nFor an alpha of %.1f:' %dAlpha)
    
    vV, vQ, vQR = qTable(vR, dAlpha)
    
    # check optimal policy R
    if(areEqual(vR, vQR)):
        print('We can confirm the optimality of our policy.')
    else:
        print('The policy we found does not seem optimal.')
      
    # compare values V(part a) to Q
    if(areEqual(np.around(vV,decimals=12), np.around(vQ,decimals=12))):
        print('The value function equals the Q-function.')
    else:
        print('The value function does not equal the Q-function.')

def boundaries(vCurrentV, vNewV):
    vDiff = vNewV - vCurrentV
    
    dMin = min(vDiff)
    dMax = max(vDiff)
    
    return dMin, dMax

def relativeValues(vR):
    iN       = len(vR)
    mTrans   = np.zeros((iN,iN))
    vRewards = np.zeros(iN+1)
    
    for i in range(iN):
        iState   = i+1
        iA       = int(vR[i]) # action iA corresponding to state i
        
        # set up the transition matrix corresponding to the policy
        vTrans    = transitions[iA][i]
        mTrans[i] = vTrans 
        
        # compute the costs for each action in the policy
        vRewards[i] = rewards[iState][iA-1]
    
    # set up the system of linear equations
    mSystem         = np.zeros((iN+1,iN+1))
    mSystem[:,0]    = 1     # set the g coefficients
    mSystem[iN][iN] = 1     # set the final v-parameter coeffecient
    
    for i in range(iN):
        for j in range(iN):
            mSystem[i][j+1] = -1 * mTrans[i][j] # the coefficients of the system
            
            if(i == j):
                mSystem[i][j+1] += 1  # for the diagonal coefficients, we add 1 
    
    # solve the system
    vGandVs = np.linalg.solve(mSystem, vRewards)
    
    return vGandVs
    
def avgRewards(vInitV, iN=100):
    vNewV     = vInitV
    vValues   = []
    vPolicies = []
    
    print('\nWe use value iteration to estimate the average rewards.')
    print('Using initial value vector', vInitV, ': \n')
    
    for i in range(iN):
        vCurrentV = vNewV
        
        # compute new value and determine corresponding policy
        vNewV, vR = newValueFunc(vCurrentV, dAlpha=1)   # use same function but set discount to 1
    
        # add value function and policy to the record
        vValues.append(vNewV)
        vPolicies.append(vR)
        
        # find lower and upper boundaries
        dMin, dMax = boundaries(vCurrentV, vNewV)
    
    vGandVs = relativeValues(vPolicies[-1])
    dG      = vGandVs[0]
    
    reportResults(vValues, vPolicies)
    
    print('For the average reward g, we found boundaries of: (%f, %f)' %(dMin, dMax))
    print('Solving a system of equations to find g, we find:', dG)
    
    if((dG>=dMin) & (dG<=dMax)):
        print('This is within the computed boundaries.')
    else:
        print('Unfortunately, this is outside the computed boundaries.')
    
def main():
    iStates = len(rewards)
    vAlpha  = [.3,.6,.9]
    vInitV = np.zeros(iStates)
    
    for dAlpha in vAlpha:
        vValues, vPolicies = totalDiscountedCosts(vInitV, dAlpha, bPrint=False)
        
        vR = vPolicies[-1]
        checkOptimality(vR, dAlpha)
        
        avgRewards(vInitV)
   
    
if __name__ == "__main__":
    main()
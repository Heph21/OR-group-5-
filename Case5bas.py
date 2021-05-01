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



"""
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
"""     
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
            mSystem[i][j] = -1 * dAlpha * mTrans[i][j] # the coefficients of the system
            
            if(i == j):
                mSystem[i][j] += 1  # for the diagonal coefficients, we add 1 
    
    # solve the system
    vX = np.linalg.solve(mSystem, vCosts)
    
    return vX

def improveDiscountPolicy(vR, vX, dAlpha, bPrint=True):
    iM      = len(vX)
    iN      = len(costs)
    mValues = np.zeros((iM,iN))
    
    for i in range(iM):
        for j in range(iN):
            iA = j+1 # action iA
            
            dCosts        = costs[iA][i]
            dDiscountTerm = dAlpha * np.dot(vX, transitions[iA][i])
            #print(iA, dCosts, dDiscountTerm)
            
            mValues[i][j] = dCosts + dDiscountTerm
    #print(vX)
    #print(mValues)
    
    # find the smallest discounted costs and corresponding action for each state i
    vNewR   = np.zeros(iM)
    vDiscV  = np.zeros(iM) 
    
    for i in range(iM):
        vOutcomes = mValues[i]
        dDisc     = np.min(vOutcomes)
        iA        = np.argmin(vOutcomes) + 1 # add one since the actions start at 1 rather than 0
        
        vDiscV[i] = dDisc
        vNewR[i]  = iA
    
    if(bPrint):
        print('Policy:', vNewR)
        print('Corresponding discounted costs:', vDiscV)
        
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
     
def totalDiscountedCosts(vR, dAlpha, bPrint=True):
    bConverged = False
    vCurrentR = vR
    
    if(bPrint):
        print('\nUsing discount factor %.1f and initial policy' %dAlpha, vR, ': \n')
    
    while(bConverged == False):
        vPreviousR = vCurrentR
        
        # value computation
        vX = valueComputation(vPreviousR, dAlpha)
    
        # policy improvement
        vCurrentR, vDiscV = improveDiscountPolicy(vPreviousR, vX, dAlpha, bPrint) 
    
        # convergence test
        bConverged = areEqual(vCurrentR, vPreviousR)
    
    #print('The optimal policy is:', vCurrentR)
    #print('The corresponding discounted costs are:', vDiscV, '\n')
    
    return vCurrentR

def relativeValues(vR):
    iN = len(vR)
    mTrans = np.zeros((iN,iN))
    vCosts = np.zeros(iN+1)
    
    for i in range(iN):
        iA       = vR[i] # action iA corresponding to state i
        
        # set up the transition matrix corresponding to the policy
        vTrans    = transitions[iA][i]
        mTrans[i] = vTrans 
        
        # compute the costs for each action in the policy
        vCosts[i] = costs[iA][i]
    
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
    vGandVs = np.linalg.solve(mSystem, vCosts)
    
    return vGandVs

def improveAvg(vR, vGandVs, bPrint=True):
    dG      = vGandVs[0] 
    vV      = np.delete(vGandVs, 0)
    
    iM      = len(vV)
    iN      = len(costs)
    mValues = np.zeros((iM,iN))
    
    for i in range(iM):
        for j in range(iN):
            iA = j+1 # action iA
            
            dCosts = costs[iA][i]
            dVTerm = np.dot(vV, transitions[iA][i])
            
            mValues[i][j] = dCosts + dVTerm
    
    # find the smallest discounted costs and corresponding action for each state i
    vNewR   = np.zeros(iM)
    vAvgV   = np.zeros(iM) 
    
    for i in range(iM):
        vOutcomes = mValues[i]
        dAvgV     = np.min(vOutcomes)
        iA        = np.argmin(vOutcomes) + 1 # add one since the actions start at 1 rather than 0
        
        vAvgV[i] = dAvgV
        vNewR[i]  = iA
    
    if(bPrint):
        print('Corresponding average costs are for each state: %.2f' %dG)
        print('The (possibly) improved policy is:', vNewR)
    
    return vNewR, dG

def averageCosts(vR, bPrint=True):
    bConverged = False
    vCurrentR = vR
    
    if(bPrint):
        print('Using initial policy', vR, ':')
    
    while(bConverged == False):
        vPreviousR = vCurrentR
        
        # determine relative values
        vGandVs = relativeValues(vPreviousR)
        
        # policy improvement
        vCurrentR, dG = improveAvg(vPreviousR, vGandVs, bPrint)
    
        #convergence test
        bConverged = areEqual(vCurrentR, vPreviousR)
    
    #print('The optimal policy is:', vCurrentR)
    #print('The corresponding average costs are:', dG, '\n')
    
    return vCurrentR

def medianMethod(dMinAlpha, dMaxAlpha, vInitR, vObjectiveR, iRun):
    if(iRun == 0):
        return dMaxAlpha
    else:
        dMedianAlpha = (dMinAlpha + dMaxAlpha) / 2
    
        vR = totalDiscountedCosts(vInitR, dMedianAlpha, False)
    
        if(areEqual(vR, vObjectiveR)):
            dRes = medianMethod(dMinAlpha, dMedianAlpha, vInitR, vObjectiveR, iRun-1)
        else:
            dRes = medianMethod(dMedianAlpha, dMaxAlpha, vInitR, vObjectiveR, iRun-1)
    
    return dRes
        
def partE(vAlpha, vInitR, vRdisc, vRavg, iRun=5):
    iN          = len(vAlpha)
    dFirstAlpha = 1
    
    for i in range(iN):
        dAlpha = vAlpha[i]
        vR     = vRdisc[i]
        
        if(areEqual(vR, vRavg)):
            dFirstAlpha = medianMethod(0, dAlpha, vInitR, vRavg, iRun)
     
    print('We have used two methods of finding an optimal policy: total discounted costs and long-run average costs.')
    print('In order for both methods to give the same optimal policy, the discount factor alpha needs to be at least %.3f.' %dFirstAlpha)
    print('\n')

def main():
    # parts A and B
    vAlpha  = [.2,.5,.8]
    vInitRa = [1,1,1,1,1]
    vInitRb = [2,2,2,2,2]
    
    vOptRa = []     # list to store optimal policies
    vOptRb = []     # list to store optimal policies
    
    print('\nPART A\n')
    for dAlpha in vAlpha:
        vR = totalDiscountedCosts(vInitRa, dAlpha)  # compute optimal policy given alpha and initial policy
        vOptRa.append(vR)                           # store optimal policy
    
    print('\nPART B\n')
    for dAlpha in vAlpha:
        vR = totalDiscountedCosts(vInitRb, dAlpha)  # compute optimal policy given alpha and initial policy
        vOptRb.append(vR)                           # store optimal policy
    
    # parts C and D
    print('\nPART C\n')
    vOptRc = averageCosts(vInitRa)
    
    print('\nPART D\n')
    vOptRd = averageCosts(vInitRb)
    
    print('\nPART E\n')
    partE(vAlpha, vInitRa, vOptRa, vOptRc)
    partE(vAlpha, vInitRa, vOptRb, vOptRd)
    
if __name__ == "__main__":
    main()
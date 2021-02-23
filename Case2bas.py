# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 14:54:22 2021

@author: basbr
"""
import pandas as pd
import numpy as np

### auxiliary functions ###

def getClubList(vDFs):
    """
    Purpose: to make a list of all clubs featured in the different seasons
    
    Input: 
        -vDFs, a vector of DataFrames containing the match results for a season
        
    Output:
        -vClubs, a vector with the names of clubs featured over all the seasons
    """
    iN         = len(vDFs)
    vClubs     = []
    vClubsAbbr = []
    
    for i in range(iN):
        df     = vDFs[i]
        iM     = len(df.index)
        
        for j in range(iM):
            sClub       = df.index[j]
            sClubAbbr   = df.columns[j]
            iL          = len(vClubs)
            bSeenBefore = 0
            
            for k in range(iL):
                if(sClub == vClubs[k]):
                    bSeenBefore+= 1
            if(bSeenBefore == 0):
                vClubs.append(sClub)
                vClubsAbbr.append(sClubAbbr)
    
    return vClubs, vClubsAbbr

def inspectClubList(vClubs, vDFs):
    """
    Purpose: to inspect the amount of seasons played by each club
    
    Input:
        -vClubs, a vector with the names of clubs featured over all the seasons
        -vDFs, a vector of DataFrames containing the match results for a season
        
    Output:
        We print:
            -the list of clubs and their appearances (max 10, min 1)
            -the amount of clubs
    """
    iSeasons = len(vDFs) 
    iN       = len(vClubs)
    vX       = np.zeros(iN)
    serClubs = pd.Series(vX, index=vClubs)
    
    for i in range(iSeasons):
        df= vDFs[i]
        iM= len(df.index)
        
        for j in range(iM):
            sClub                = df.index[j]
            serClubs.loc[sClub] += 1
    
    print(serClubs)
    print('\nThe amount of clubs appearing in the Premier League from 2009 to 2019 is:', iN, '\n')

def readCell(sScore, sDelimiter='-'):
    """
    Purpose: to process the information in a single cell from one of our seasonly dataframes
    
    Input:
        -sScore, the object found in the cell, treated as a string
    
    Output:
        -iHome, goals scored by the home team
        -iAway, goals scored by the away team
    """
    sPartition = sScore.partition(sDelimiter)
    print(sPartition)
    #iHome = int(sPartition[0])
    #iAway = int(sPartition[2])
    
    iHome = int(sScore[0])
    iAway = int(sScore[2])
    
    return iHome, iAway
     
def getGoalData(vDFs): 
    """
    Purpose: to create matrices of goals scored and conceded by each team, at home and away
    
    Input:
        -vDFs, a vector of DataFrames containing the match results for a season
    
    Output:
        -vScored, vector of DataFrames of goals scored per season
        -vConceded, vector of DataFrames of goals conceded per season
    """
    iSeasons  = len(vDFs)
    iN        = len(vDFs[0])
    vScored   = []
    vConceded = []
    mEmpty    = np.zeros((iN,iN),dtype=int)
    
    for k in range(iSeasons):
        df          = vDFs[k]
        vClubs      = df.index
        vClubsAbbr  = df.columns
        dfScored    = pd.DataFrame(mEmpty, index= vClubs, columns=vClubsAbbr).astype(int)
        dfConceded  = dfScored.copy()
        
        for i in range(iN):
            for j in range(iN):  
                if(i != j):
                    iHome, iAway         = readCell(df.iloc[i,j])
                    dfScored.iloc[i,j]   = iHome
                    dfConceded.iloc[i,j] = iAway
        
        vScored.append(dfScored)
        vConceded.append(dfConceded)
    
    return vScored, vConceded

def getMatchData(vScored, vConceded, vClubs, vClubsAbbr):
    """
    Purpose: to create the matrices of matches played, won, lost and drawn
    
    Input:
        -vScored, vector of DataFrames of goals scored per season
        -vConceded, vector of DataFrames of goals conceded per season
        -vClubs, a vector with the names of clubs featured over all the seasons
    
    Output:
        -dfMatches, matrix of matches played over all the seasons
        -dfLost, matrix of matches lost over all the seasons
        -dfDrawn, matrix of matches drawn over all the seasons
        -dfScored,matrix of goals scored over all the seasons
        -dfConceded, matrix of goals conceded over all the seasons
    """
    iSeason    = len(vScored)           # amount of seasons
    iN         = len(vClubs)            # amount of clubs over all the seasons
    
    mEmpty     = np.zeros((iN,iN), dtype=int)
    dfMatches  = pd.DataFrame(mEmpty, index= vClubs, columns=vClubsAbbr)
    dfLost     = dfMatches.copy()
    dfDrawn    = dfMatches.copy()
    dfScored   = dfMatches.copy()
    dfConceded = dfMatches.copy()
    
    for k in range(iSeason):
        dfScoredThisYear   = vScored[k]
        dfConcededThisYear = vConceded[k]
        vClubsThisYear     = dfScoredThisYear.index
        iM                 = len(dfScoredThisYear.iloc[0])
        
        for i in range(iM):
            sHomeTeam = vClubsThisYear[i]
            iRowIndex = vClubs.index(sHomeTeam)
            
            for j in range(iM):
                sAwayTeam = vClubsThisYear[j]
                iColIndex = vClubs.index(sAwayTeam)
                
                if(i != j):
                    iHomeGoals = dfScoredThisYear.iloc[i,j]
                    iAwayGoals = dfConcededThisYear.iloc[i,j]
                    
                    dfMatches.iloc[iRowIndex,iColIndex]  += 1
                    dfScored.iloc[iRowIndex,iColIndex]   += iHomeGoals
                    dfConceded.iloc[iRowIndex,iColIndex] += iAwayGoals
                    
                    if(iHomeGoals > iAwayGoals):    # away team lost
                       dfLost.iloc[iColIndex, iRowIndex]  += 1
                    elif(iHomeGoals < iAwayGoals):  # home team lost
                        dfLost.iloc[iRowIndex, iColIndex]  += 1
                    else:                           # draw
                        dfDrawn.iloc[iRowIndex, iColIndex] += 1
                    
    # for the goal matrices, add up goals for home and away games
    for i in range(iN):
        for j in range(iN):
            if(j > i): 
                dfScored.iloc[i,j]   += dfConceded.iloc[j,i] 
                dfConceded.iloc[i,j] += dfScored.iloc[j,i]
                dfScored.iloc[j,i]    = 0
                dfConceded.iloc[j,i]  = 0
    
    return dfMatches, dfLost, dfDrawn, dfScored, dfConceded

def getMatchesPlayed(dfMatches, iClub):
    """
    Purpose: to compute the amount of matches played by a given team
    
    Input:
        -dfMatches, matrix of matches played over all the seasons
        -iClub, integer representing the team we want to compute the amount of matches for
    
    Output:
        -iMatches, amount of matches played by the team in question
    """
    
    vMatchesAtHome = dfMatches.iloc[iClub].values
    vMatchesAway = dfMatches.iloc[:,iClub].values
    iMatchesAtHome = sum(vMatchesAtHome)
    iMatchesAway = sum(vMatchesAway)
    iMatches = iMatchesAtHome + iMatchesAway
    
    return iMatches

def inspectData(dfMatches, dfLost, dfDrawn, dfScored, dfConceded): 
    """
    Purpose: to inspect the data we processed to get a better feel of it
    
    Input: 
        -dfMatches, matrix of matches played over all the seasons
        -dfLost, matrix of matches lost over all the seasons
        -dfDrawn, matrix of matches drawn over all the seasons
        -dfScored,matrix of goals scored over all the seasons
        -dfConceded, matrix of goals conceded over all the seasons
        
    Output:
        We print:
            -the first few columns of:
                -the matrix of matches played
                -the matrix of losses
                -the matrix of ties
                -the matrix of goals scored
                -the matrix of goals conceded
            -per club:
                -amount of matches played
                -amount of matches won
                -amount of matches tied
                -amount of matches lost
    """
    iN     = len(dfMatches)
    vClubs = dfMatches.index
    
    # print the first few rows of the df's to get a quick look at the data
    print('Matches played:\n', dfMatches.head(),'\n')
    print('Matches lost:\n', dfLost.head(),'\n')
    print('Matches tied:\n', dfDrawn.head(),'\n')
    print('Goals scored:\n', dfScored.head(),'\n')
    print('Goals conceded:\n', dfConceded.head(),'\n')
    
    # matches played
    vMatches = np.zeros(iN)
    
    for i in range(iN):
        iMatches    = getMatchesPlayed(dfMatches, i)
        vMatches[i] = iMatches
    
    serMatches = pd.Series(vMatches, index=vClubs)
    print('Matches played:\n', serMatches, '\n')
    
    # matches won, drawn and lost
    vWon   = np.zeros(iN)
    vDrawn = np.zeros(iN)
    vLost  = np.zeros(iN)
    
    for i in range(iN):
        vWonClub   = dfLost.iloc[:,i].values
        vDrawnClub = dfDrawn.iloc[i].values + dfDrawn.iloc[:,i].values
        vLostClub  = dfLost.iloc[i].values
        iWon   = sum(vWonClub)
        iDrawn = sum(vDrawnClub)
        iLost  = sum(vLostClub) 
        
        vWon[i]   += iWon
        vDrawn[i] += iDrawn
        vLost[i]  += iLost
        
        
    serWon = pd.Series(vWon, index=vClubs)
    print('Won:\n', serWon, '\n')
    serDrawn = pd.Series(vDrawn, index=vClubs)
    print('Drawn:\n', serDrawn, '\n')
    serLost = pd.Series(vLost, index=vClubs)
    print('Lost:\n', serLost, '\n')
    
    serPoints = serWon*3 + serDrawn
    serPPG    = serPoints / serMatches
    serPPG    = serPPG.sort_values(ascending=False)
    print('Points:\n', serPoints, '\n')
    print('Points per match:\n', serPPG, '\n')
 
    
def constructWebGraph(df):
    """
    Purpose: to compute the 'internet graph' or transition matrix of a Markov chain
    
    Input: 
        -df, DataFrame containing data to be used (most likely integers), e.g. amount of losses, or points earned
    
    Output:
        -dfWebGraph, DataFrame (of doubles), transition matrix (containing for each state, the probability of moving to each state on the next step)
        -note: 
            we do not prevent zero-rows, but given the data, we do not expect any zero-rows to occur
            if a zero row does come up, we notify the user with a print statement
            the user would then be required to alter dfGraph in the supervising method, before working with it as a full-grown transition matrix
    """
    iN = len(df)
    
    mData  = df.values.astype(float) 
    mGraph = np.copy(mData)
    
    for i in range(iN):
        vData = mData[i]
        iSum = sum(vData)
        
        if(iSum != 0):
            mGraph[i] = vData / iSum
        else:
            print('Found a zero row!')
            print(df.index[i])
            mGraph[i,i] = 1
    
    dfWebGraph = pd.DataFrame(mGraph, index=df.index, columns= df.columns)
    
    return dfWebGraph

def adaptGraph(dfSgraph, dAlpha=.85):
    """
    Purpose: to turn the given stochastic matrix into an irreducible ('Google') matrix to be used for PageRank
    
    Input:
        -dfSgraph, DataFrame (of doubles) containing the transition matrix
        -dAlpha, damping factor, used to create a linear combination of dfSgraph with another graph to get rid of any zeros
    
    Output:
        -dfGgraph, DataFrame (of doubles), the irreducible matrix we can use in executing PageRank
    """
    # create the matrix E that ensures irreducibilty
    iN       = len(dfSgraph)
    dEvalue  = 1/iN
    mE       = np.zeros((iN,iN)) + dEvalue
    dfEgraph = pd.DataFrame(mE, index=dfSgraph.index, columns=dfSgraph.columns)
    
    # create the matrix G, a stochastic linear combination of S and E
    dfGgraph  = dfSgraph.copy()
    dfGgraph *= dAlpha
    dfGgraph += (1 - dAlpha) * dfEgraph
    
    return dfGgraph

def power(dfGgraph, iMax=10, dEps=1e-5):
    """
    Purpose: to repeatedly square a given transition matrix so as to find the limiting probability distribution vector vPi
    
    Input:
        -dfGgraph, DataFrame (of doubles), the irreducible matrix we can use in executing PageRank
        -iMax, the maximum of times we square the matrix
        -dEps, the stop condition 
    
    Output:
        -mP, matrix of doubles, containing some power of the original matrix, to be used in determining vPi 
    """
    mG = dfGgraph.values
    
    for i in range(iMax):
        mX = mG @ mG
        dDiff = np.linalg.norm(mX - mG)
        
        if(dDiff < dEps):
            iPower = 2**i
            print('The transition matrix converged in the %i-th power ' %iPower)
            return mX
        else:
            mG = mX
    
    iPower = 2**(iMax - 1)
    print('We needed the %i-th power for the transition matrix to converge' %iPower)
    
    return mG

### the three parts

def processData(sFile, vSheets):
    """
    Purpose: to process the data of ten seasons
    
    Input: 
        -sFile, name of the file that contains the data
        -vSheets, vector of names of sheets containing the data for each season
        
    Output:
        -dfMatches, matrix of matches played over all the seasons
        -dfLost, matrix of matches lost over all the seasons
        -dfDrawn, matrix of matches drawn over all the seasons
        -dfScored,matrix of goals scored over all the seasons
        -dfConceded, matrix of goals conceded over all the seasons
    """
    iN   = len(vSheets)
    vDFs = []
    
    for i in range(iN):
        df= pd.read_excel(sFile, sheet_name= vSheets[i], index_col=0)
        vDFs.append(df)
        #print(df)
        
    vClubs, vClubsAbbr = getClubList(vDFs)
    
    vScored, vConceded                               = getGoalData(vDFs)
    dfMatches, dfLost, dfDrawn, dfScored, dfConceded = getMatchData(vScored, vConceded, vClubs, vClubsAbbr)
    
    #inspectClubList(vClubs, vDFs)
    inspectData(dfMatches, dfLost, dfDrawn, dfScored, dfConceded)
    
    return dfMatches, dfLost, dfDrawn, dfScored, dfConceded

def constructGraph(dfMatches, dfLost, dfDrawn, dfScored, dfConceded, sMethod='Losses'):
    """
    Purpose: to create an irreducible transition matrix we can use for PageRank
    
    Input: 
        -dfMatches, matrix of matches played over all the seasons
        -dfLost, matrix of matches lost over all the seasons
        -dfDrawn, matrix of matches drawn over all the seasons
        -dfScored,matrix of goals scored over all the seasons
        -dfConceded, matrix of goals conceded over all the seasons
        
    Output:
        -dfGgraph, DataFrame (of doubles), the irreducible matrix we can use in executing PageRank (depending on which method we want to use)    
    """
    # first try: weights based on losses
    if(sMethod=='Losses'):
        dfWebGraph = constructWebGraph(dfLost)
    
    # second idea: points earned
    if(sMethod=='Points'):
        iN       = len(dfMatches)
        mEmpty   = np.zeros((iN,iN), dtype=int)
        dfPoints = pd.DataFrame(mEmpty, index=dfMatches.index, columns=dfMatches.columns)
        
        for i in range(iN):
            for j in range(iN):
                if(i != j):
                    dfPoints.iloc[i,j] += 3 * dfLost.iloc[i,j]  # add 3 points for games lost
                    dfPoints.iloc[i,j] += dfDrawn.iloc[i,j]     # add 1 point for games tied at home
                    dfPoints.iloc[i,j] += dfDrawn.iloc[j,i]     # add 1 point for games ties away
        
        dfWebGraph = constructWebGraph(dfPoints)
    
    # third idea: goals conceded
    if(sMethod=='Goals'):
        dfGoals  = dfPoints.copy()
        dfGoals -= dfGoals          # make it a zero-matrix
        
        for i in range(iN):
            for j in range(iN):
                if((i != j) & (j > i)):
                    iScored   = dfScored.iloc[i,j]
                    iConceded = dfConceded.iloc[i,j]
                    
                    dfGoals.iloc[i,j] = iConceded
                    dfGoals.iloc[j,i] = iScored
        
        dfWebGraph = constructWebGraph(dfGoals)
    
    # fourth idea: goal differentials
    if(sMethod=='Diff'):
        dfDiff  = dfPoints.copy()
        dfDiff -= dfDiff            # make it a zero-matrix
        
        for i in range(iN):
            for j in range(iN):
                if((i != j) & (j > i)):
                    iScored   = dfScored.iloc[i,j]
                    iConceded = dfConceded.iloc[i,j]
                    
                    if(iScored > iConceded):
                        dfDiff.iloc[j,i] += (iScored - iConceded)
                    elif(iScored < iConceded):
                        dfDiff.iloc[i,j] += (iConceded - iScored)
        
        dfWebGraph = constructWebGraph(dfDiff)
    
    return dfWebGraph
    
def PageRank(dfWeb):
    """
    Purpose: to compute the limiting probability distribution of a given transition matrix, and find the according page ranking
    
    Input: 
        -dfTransition, (irreducible) transition matrix
        
    Output:
        We print the ranking        
    """
    
    # adapt web graph to make it irreducible (H to G in the first article)
    dfMarkov = adaptGraph(dfWeb)
    
    # compute the stationary distribution
    mProbs  = power(dfMarkov)
    vPi     = mProbs[0]
    
    # order clubs accordingly
    vClubs  = dfMarkov.index
    serPi   = pd.Series(vPi, index=vClubs)
    serRank = serPi.sort_values(ascending=False)
    
    # print resulting ranking
    print(serRank)

       
def main():
    sFile   = 'Premier League.xlsx'
    sFile2  = 'Eredivisie.xlsx'
    vSheets = ['2010','2011','2012','2013','2014','2015','2016','2017','2018','2019']
    
    dfMatches, dfLost, dfDrawn, dfScored, dfConceded = processData(sFile, vSheets)
    dfWebGraph = constructGraph(dfMatches, dfLost, dfDrawn, dfScored, dfConceded, 'Losses')
    PageRank(dfWebGraph)
    
    
if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 14:54:22 2021

@author: basbr
"""
import pandas as pd
import numpy as np

def getClubList(vDFs):
    """
    Purpose: to make a list of all clubs featured in the different seasons
    
    Input: 
        -vDFs, a vector of DataFrames containing the match results for a season
        
    Output:
        -vClubs, a vector with the names of clubs featured over all the seasons
    """
    iN     = len(vDFs)
    vClubs = []
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

def readCell(sScore):
    """
    Purpose: to process the information in a single cell from one of our seasonly dataframes
    
    Input:
        -sScore, the object found in the cell, treated as a string
    
    Output:
        -iHome, goals scored by the home team
        -iAway, goals scored by the away team
    """
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
    mEmpty    = np.zeros((iN,iN))
    
    for k in range(iSeasons):
        df          = vDFs[k]
        vClubs      = df.index
        vClubsAbbr  = df.columns
        dfScored    = pd.DataFrame(mEmpty, index= vClubs, columns=vClubsAbbr).astype(int)
        dfConceded  = dfScored.copy().astype(int)
        #dfConceded  = pd.DataFrame(mEmpty, index= vClubs, columns=vClubsAbbr)
        
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
        -dfWon, matrix of matches won over all the seasons
        -dfLost, matrix of matches lost over all the seasons
        -dfDrawn, matrix of matches drawn over all the seasons
    """
    iSeason    = len(vScored)           # amount of seasons
    iN         = len(vClubs)            # amount of clubs over all the seasons
    
    mEmpty     = np.zeros((iN,iN))
    dfMatches  = pd.DataFrame(mEmpty, index= vClubs, columns=vClubsAbbr).astype(int)
    dfLost     = dfMatches.copy().astype(int)
    dfDrawn    = dfMatches.copy().astype(int)
    dfScored   = dfMatches.copy().astype(int)
    dfConceded = dfMatches.copy().astype(int)
    
    for k in range(iSeason):
        dfScoredThisYear   = vScored[k]
        dfConcededThisYear = vConceded[k]
        vClubsThisYear     = dfScoredThisYear.index
        iM                 = len(dfScoredThisYear.iloc[0])
        
        for i in range(iM):
            sHomeTeam = vClubsThisYear[i]
            #print(sHomeTeam)
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

def processData(sFile, vSheets):
    """
    Purpose: to process the data of ten seasons
    
    Input: 
        -sFile, name of the file that contains the data
        -vSheets, vector of names of sheets containing the data for each season
        
    Output:
        -mGames, matrix of matches played
        -mWon, matrix of matches won
        -mLost, matrix of matches lost
        -mDrawn, matrix of matches drawn
        -mScored, matrix of goals scored
        -mConceded, matrix of goals conceded
    """
    iN   = len(vSheets)
    vDFs = []
    
    for i in range(iN):
        df= pd.read_excel(sFile, sheet_name= vSheets[i], index_col=0)
        vDFs.append(df)
        print(df)
        
    vClubs, vClubsAbbr = getClubList(vDFs)
    
    inspectClubList(vClubs, vDFs)
    
    vScored, vConceded                               = getGoalData(vDFs)
    dfMatches, dfLost, dfDrawn, dfScored, dfConceded = getMatchData(vScored, vConceded, vClubs, vClubsAbbr)
    
    print('Matches played:\n', dfMatches.head(),'\n')
    print('Matches lost:\n', dfLost.head(),'\n')
    print('Matches tied:\n', dfDrawn.head(),'\n')
    print('Goals scored:\n', dfScored.head(),'\n')
    print('Goals conceded:\n', dfConceded.head(),'\n')

def main():
    sFile   = 'Premier League.xlsx'
    vSheets = ['2010','2011','2012','2013','2014','2015','2016','2017','2018','2019']
    
    processData(sFile, vSheets)
    

if __name__ == "__main__":
    main()
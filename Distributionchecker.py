# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 13:28:30 2021

@author: basbr
"""
import numpy as np
import scipy.stats as st


def main(): 
    n=10
    l=.1
    p=.5
    t=5
    
    dProb = 0
    dExp  = 0
    
    for i in range(10+1):
        dP = st.binom.pmf(i,n,p)
        
        dProb += dP
        dExp  += dP*i
    
    print(dProb, dExp)
    
    dP2 = 0
    dE2 = 0
    
    for i in range(10+1):
        dP = st.boltzmann.pmf(i,l,n)
        
        dP2 += dP
        dE2 += dP*i
    
    print(dP2, dE2)

if __name__ == "__main__":
    main()
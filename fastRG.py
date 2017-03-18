#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 15:18:34 2017
Author: Yun-Jhong Wu
E-mail: yjwuam@gmail.com
"""

import numpy as np
from scipy.sparse import csc_matrix

def howManyEdges(X, S, Y=None):
    if Y is None:
        Y = X
    Cx = np.sum(X, axis=0)
    Cy = np.sum(Y, axis=0)
    
    em = Cx @ np.sum(S * Cy, axis=1)
    avDeg = em / X.shape[0]
    
    return em, avDeg


def fastRG(X, S, Y=None, avgDeg=None, simple=None, PoissonEdges=True, 
           directed=False, selfLoops=False, returnEdgeList=False, 
           returnParameters=False):
    """
    Rohe, K., Tao, J., Han, X., & Binkiewicz, N. (2017). A note on quickly 
    sampling a sparse matrix with low rank expectation. arXiv preprint 
    arXiv:1703.02998.
    
    Implementation of fastRG in R
    https://github.com/karlrohe/fastRG
    """
    
    if Y is not None and Y.size > 0:
        directed = True
        selfLoops = True
        simple = False
        returnY = True
    
    else:
        Y = X
        returnY = False
    
    if np.any(X < 0) or np.any(S < 0) or np.any(Y < 0):
        return None
    
    if simple is not None and simple:
        selfLoops = False
        directed = False
        PoissonEdges = False
    
    n, K1 = X.shape
    d, K2 = Y.shape
  
  
    if avgDeg is not None:
        _, eDbar = howManyEdges(X, S, Y)
        S *= avgDeg / eDbar
     
  
    if not directed:
        S = (S + S.T) * 0.25
        
    Cx = np.sum(X, axis=0)
    Cy = np.sum(Y, axis=0)
    Xt = (X * (1 / Cx)).T
    Yt = (Y * (1 / Cy)).T  
  
    St = Cx[:, None] * S * Cy
    m = np.random.poisson(np.sum(St))  

    if m == 0:
        A = csc_matrix((n, d))
        return (A, X, S, Y if returnY else None) if returnParameters else A
        
    tabUV = np.random.multinomial(m, pvals=St.ravel() * (1 / np.sum(St))).reshape((K1, K2))
    
    elist = np.empty((2, m))
    eitmp = np.empty(m)
    
    blockDegreesU = np.sum(tabUV, axis=1)
    tickerU = np.insert(np.cumsum(blockDegreesU), 0, 0) 
    
    for u in range(K1):
        if blockDegreesU[u] > 0:
            elist[0, tickerU[u]:tickerU[u+1]] = np.random.choice(n, size=blockDegreesU[u], 
                                                                 replace=True, p=Xt[u])
      
    blockDegreesV = np.sum(tabUV, axis=0)
    tickerV = np.insert(np.cumsum(blockDegreesV), 0, 0) 
    
    for v in range(K2):
        if blockDegreesV[v] > 0:
            eitmp[tickerV[v]:tickerV[v+1]] = np.random.choice(n, size=blockDegreesV[v], 
                                                              replace=True, p=Yt[v])

    ticker = 0
    for u in range(K1):
        for v in range(K2):   
            if tabUV[u,v] > 0:            
                elist[1, ticker:ticker + tabUV[u,v]] = eitmp[tickerV[v]:tickerV[v] + tabUV[u,v]]
                ticker += tabUV[u, v]
                
    elist = elist.T

    if not selfLoops:
        elist = elist[np.where(elist[:, 0] != elist[:, 1])]
        
    if not directed:
        if n != d:
            raise Exception("{0} != {1}: Undirected network requests n == d".format(n, d))
        
        elist = np.concatenate((elist, elist[:, ::-1]))

    if not PoissonEdges:       
        e = np.ascontiguousarray(elist)
        e_unique = np.unique(e.view([('', np.int), ('', np.int)]))
        elist = e_unique.view(np.int).reshape((e_unique.shape[0], 2))
          
    if returnEdgeList:
        return elist
    
    else:
        A = csc_matrix((np.ones(elist.shape[0], dtype=np.int), 
                        (elist[:, 0], elist[:, 1])), 
                       shape=(n, d), dtype=np.int)      

        return (A, X, S, Y if returnY else None) if returnParameters else A

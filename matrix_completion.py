# -*- coding: utf-8 -*-
# Created on Sun Jul 6 12:17:39 2014
# Implemented in Python 3.4.0
# Author: Yun-Jhong Wu
# E-mail: yjwu@umich.edu

import numpy as np

def MatrixCompletion(X, Omega, nu=1, rank=5, 
                     tol=1e-4, maxiters=1000):
    """
    Matrix completion by augmented Lagrange multiplier 
    (Lin, Z., Chen, M., & Ma, Y. (2010). The augmented lagrange multiplier 
     method for exact recovery of corrupted low-rank matrices. 
     arXiv preprint arXiv:1009.5055.)
     
    X: data matrix
    Omega: a tuple, given by numpy.where, contains row/column indexes 
           of observed elements
    """
    
    n = min(X.shape)
    Y = np.zeros(X.shape)
    E = np.zeros(X.shape)
    X_cp = X.copy()
    X_Fnorm = np.linalg.norm(X, 'fro')
    tol *= X_Fnorm
    nu *= X_Fnorm * Omega[0].size / X.size
    rho = 1 + Omega[0].size / X.size
    
    niters = 0
    Res = np.array([[np.inf]])
    
    while np.linalg.norm(Res, 'fro') > tol and niters < maxiters:
        niters += 1
        X_plus_Y = X + nu * Y
        U, D, V = np.linalg.svd(X_plus_Y - E, full_matrices=False)
        D = D[:rank]
        D -= nu
        D = D[D>0]
        X_cp = np.dot(U[:, :D.size] * D, V[:D.size, :])
        E = X_plus_Y - X_cp
        E[Omega] = 0
        Res = X - X_cp - E
        Y += 1 / nu * Res
        nu /= rho
        rank = min(D.size + 1 + (D.size < rank) * 4, n)
        
    return X_cp, D.size
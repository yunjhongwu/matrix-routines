# -*- coding: utf-8 -*-
# Created on Wed May 6 10:01:33 2015
# Implemented in Python 3.4.0
# Author: Yun-Jhong Wu
# E-mail: yjwu@umich.edu

import numpy as np

def RobustPCA(X, lbd=.01, nu=1, rho=1.5, tol=1e-3, maxiters=100):
    """
    Robust PCA by augmented Lagrange multiplier 
    (Lin et al. (2010) The Augmented Lagrange Multiplier Method 
     for Exact Recovery of Corrupted Low-Rank Matrices, arXiv:1009.5055)
     
    lbd: weight of sparse error (relative to low-rank term)
    nu: initial step size (nu = 1 / $mu$ for $mu$ in the above paper)
    rho: step size adjustment
    """
    
    niters = 0
    rank_new = 100
    L = np.zeros(X.shape)
    n = np.min(X.shape)
    Res = np.array([[np.inf]])   
    X_F_norm = np.linalg.norm(X, 'fro')
    tol *= X_F_norm
    nu *= X_F_norm
    Y = 1 / max(X_F_norm, np.max(np.abs(X)) / lbd) * X

    while np.linalg.norm(Res, 'fro') > tol and niters < maxiters:
        niters += 1
        X_plus_Y = X + nu * Y
        S = X_plus_Y - L
        S = np.maximum(S - lbd * nu, 0) + np.minimum(S + lbd * nu, 0)
        U, D, V = np.linalg.svd(X_plus_Y - S, full_matrices=False)
        D = D - nu
        rank = np.sum(D > 0)
        rank_new = min(rank + (1 if rank < rank_new else int(0.05 * n)), n)
        L = np.dot(U[:, :rank] * D[:rank], V[:rank, :])
        Res = X - L - S
        Y += (1 / nu) * Res 
        nu /= rho
        
    return L, S
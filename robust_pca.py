# -*- coding: utf-8 -*-
# Created on Wed Jul 9 10:01:33 2014
# Implemented in Python 3.4.0
# Author: Yun-Jhong Wu
# E-mail: yjwu@umich.edu

import numpy as np

def RobustPCA(X, lbd=.1, nu=1, rho=1.5, tol=1e-4, maxiters=1000):
    """
    Robust PCA by augmented Lagrange multiplier 
    (Lin, Z., Chen, M., & Ma, Y. (2010). The augmented lagrange multiplier 
     method for exact recovery of corrupted low-rank matrices. 
     arXiv preprint arXiv:1009.5055.)
     
    lbd: weight of sparse error (relative to low-rank term)
    nu: initial step size (nu = 1 / $mu$ for $mu$ in the above paper)
    rho: step size adjustment
    """
    
    niters = 0
    rank_new = 100
    L = np.zeros(X.shape)
    n = np.min(X.shape)
    Res = np.array([[np.inf]])   
    X_Fnorm = np.linalg.norm(X, 'fro')
    tol *= X_Fnorm
    nu *= X_Fnorm
    Y = 1 / max(X_Fnorm, np.max(np.abs(X)) / lbd) * X

    while np.linalg.norm(Res, 'fro') > tol and niters < maxiters:
        niters += 1
        X_plus_Y = X + nu * Y
        S = X_plus_Y - L
        S = np.maximum(S - lbd * nu, 0) + np.minimum(S + lbd * nu, 0)
        U, D, V = np.linalg.svd(X_plus_Y - S, full_matrices=False)
        D -= nu
        D = D[D > 0]
        rank_new = min(D.size + 1 + (D.size < rank_new) * int(0.05 * n), n)
        L = np.dot(U[:, :D.size] * D, V[:D.size, :])
        Res = X - L - S
        Y += (1 / nu) * Res 
        nu /= rho
        
    return L, S
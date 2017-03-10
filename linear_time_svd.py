#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 21:37:24 2017
Author: Yun-Jhong Wu
E-mail: yjwuam@gmail.com
"""

import numpy as np
import scipy as sp

def select(p):
    D = 0
    idx = -1
    q = 0
    for i, p_i in enumerate(p):
        D += p_i
        if np.random.uniform(0, D) < p_i:
           idx = i
           q = p_i
           
    return idx, q

def LinearTimeSVD(A, r, n_oversampling=10):
    """
    Drineas, P., Kannan, R., & Mahoney, M. W. (2006). Fast Monte Carlo 
    algorithms for matrices II: Computing a low-rank approximation to 
    a matrix. SIAM Journal on computing, 36(1), 158-183.
    
    A: data matrix

    return (the r leading left singular values,
            the r leading left singular vectors)
    """
    
    rowsums = np.sum(A * A, 0)
    p = rowsums / np.sum(rowsums)
    c = r + n_oversampling
    idx, q = map(np.array, zip(*[select(p) for _ in range(c)]))
    C = A[:, idx] * (1 / np.sqrt(c * q))
    w, H = sp.linalg.eigh(C.T @ C, eigvals=[n_oversampling, c - 1])
    d = np.sqrt(w)[::-1]

    return d, H[:, ::-1] * (1 / d)


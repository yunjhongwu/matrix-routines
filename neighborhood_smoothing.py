# -*- coding: utf-8 -*-
# Created on Wed Apr 13 19:52:56 2016
# Implemented in Python 3.5.1
# Author: Yun-Jhong Wu
# E-mail: yjwu@umich.edu

import numpy as np
from scipy.spatial.distance import cdist

def NeighborhoodSmoothing(A):
    """
    Python implementation of the algorithm proposed by
    Zhang, Y., Levina, E. and Zhu, J. (2016) Estimating neighborhood edge 
    probabilities by neighborhood smoothing. arXiv: 1509.08588.
    
    Input: Symmetric adjacency matrix
    Output: Estimated probaility matrix
    """
    n = A.shape[0]
    A2 = A @ A.T * (1 / n)
    D = cdist(A2, A2, 'chebyshev') 
    K = D < np.percentile(D, np.sqrt(np.log(n) / n) * 100, 0)
    P = A @ (K * (1 / (np.sum(K, 0) + 1e-10)))

    return (P + P.T) * 0.5
    
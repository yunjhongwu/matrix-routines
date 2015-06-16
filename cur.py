# -*- coding: utf-8 -*-
# Created on Sun Nov 9 16:35:39 2014
# Implemented in Python 3.4.0
# Author: Yun-Jhong Wu
# E-mail: yjwu@umich.edu

import numpy as np
from numpy.linalg import svd, qr, pinv, norm
from numpy.random import uniform
from scipy.stats import itemfreq

class CUR:
    """
    Based on rCUR R package
    Bodor, A., Csabai, I., Mahoney, M. W., & Solymosi, N. (2012). 
      rCUR: an R package for CUR matrix decomposition. 
      BMC bioinformatics, 13(1), 103.
    """
    def __init__(self, beta=None, alpha=1, method='random'):
        """
        beta: leverage scores are computed based on weighting 
              of the singular values with the power of beta
        alpha: the coefficient of orthogonality
        method: 'random', 'exact', 'top', 'orthotop', 'highest'
        """

        self.alpha = alpha
        self.beta = beta
        self.X = None
        
        self.sv = None
        self.X = None
        if method not in ['random', 'exactnum', 'top', 'orthotop', 'highest']:
            print("Unknown method: {0}; use 'random' instead.".format(method))
            self.method = 'random'
        else:
            self.method = method
    
    def fit(self, X, r=None, c=None, k=None, sv=None):
        """
        Input data X        
        
        X: data matrix
        r: number of selected rows in X
        c: number of selected columns in X
        k: maximum of rank
        sv: SVD given by numpy.linalg.svd
        """

        self.X = X
        self.c = min(self.X.shape[1], c) if c else self.X.shape[1]
        self.r = min(self.X.shape[0], r) if r else self.X.shape[0]

        self.scoresC = None
        self.scoresR = None
        self.idxC = np.zeros(0)
        self.idxR = np.zeros(0)

        self.err = None
        
        if not sv:
            sv = svd(X, full_matrices=False)
        if not k:
            sv_cumsum = np.cumsum(sv[1])
            self.k = np.sum(sv_cumsum < sv_cumsum[-1] * 0.8)
        else:
            self.k = min(k, min(X.shape))
            
        self.sv = (sv[0][:, :self.k], 
                   sv[1][:self.k], 
                   sv[2][:self.k, :])     
                   
        if c < self.X.shape[1]:
            self.scoresC = self.get_levscores(self.sv[2])
            if self.method == 'random':
                self.idxC = np.where(uniform(0, 1, self.X.shape[1]) < self.scoresC * self.c)[0]
            
            elif self.method == 'exactnum':
                self.idxC = np.argpartition(self.c * self.scoresC - 
                                               uniform(0, 1, self.X.shape[1]), -self.c)[-self.c:]

            elif self.method == 'top':
                self.idxC = np.argpartition(self.scoresC, -self.c)[-self.c:]

            elif self.method == 'orthotop':
                pi = self.scoresC
                self.idxC = np.zeros(self.c, dtype=int)
                vort = self.X.copy()
                vnormsqr = np.sum(self.X ** 2, axis=0)
                vortnormsqr = vnormsqr.copy()
                for i in range(self.c):
                    self.idxC[i] = np.argmax(pi + self.alpha * np.sqrt(vortnormsqr / vnormsqr))
                    sn = vortnormsqr[self.idxC[i]]
                    if sn:
                        delta = vort[:, self.idxC[i]].dot(vort)
                        vort -= np.outer(vort[:, self.idxC[i]], delta) / sn
                        vortnormsqr -= delta ** 2 / sn
                        vortnormsqr[vortnormsqr < 0] = 0
                    pi[self.idxC[i]] = 0
                    
            elif self.method == 'rank':
                v_cumsum = np.cumsum(self.sv[2] ** 2, axis=0)
                self.scoresC = 1 / (self.X.shape[1] + 1) * np.max(v_cumsum, axis=0)- np.argmax(v_cumsum, axis=0)
                self.idxC = np.argpartition(self.scoresC, c)[:c]

        else:
            self.idxC = np.arange(self.X.shape[1])            
            
        if r < self.X.shape[0]:
            self.scoresR = self.get_levscores(self.sv[0].T)
            if self.method == 'random':
                self.idxR = np.where(uniform(0, 1, self.X.shape[0]) < self.scoresR * self.r)[0]
            
            elif self.method == 'exactnum':
                self.idxR = np.argpartition(self.r * self.scoresR - 
                                               uniform(0, 1, self.X.shape[0]), -self.r)[-self.r:]

            elif self.method == 'top':
                self.idxR = np.argpartition(self.scoresR, -self.r)[-self.r:]

            elif self.method == 'orthotop':
                pi = self.scoresR
                self.idxR = np.zeros(self.r, dtype=int)
                vort = self.X.copy()
                vnormsqr = np.sum(self.X ** 2, axis=1)
                vortnormsqr = vnormsqr.copy()
                for i in range(self.r):
                    self.idxR[i] = np.argmax(pi + self.alpha * np.sqrt(vortnormsqr / vnormsqr))
                    sn = vortnormsqr[self.idxR[i]]
                    if sn:
                        delta = vort[self.idxR[i], :].dot(vort.T).T
                        vort -= np.outer(delta, vort[self.idxR[i], :]) / sn
                        vortnormsqr -= delta ** 2 / sn
                        vortnormsqr[vortnormsqr < 0] = 0
                    pi[self.idxR[i]] = 0
                    
                
            elif self.method == 'rank':
                v_cumsum = np.cumsum(self.sv[1].T ** 2, axis=0)
                self.scoresC = 1 / (self.X.shape[0] + 1) * np.max(v_cumsum, axis=0)- np.argmax(v_cumsum, axis=0)
                self.idxC = np.argpartition(self.scoresC, c)[:c]

        else:
            self.idxR = np.arange(self.X.shape[0]) 
            
        if not self.idxC.size:
            self.idxC = np.array([np.argmax(self.scoresC)])

        if not self.idxR.size:
            self.idxR = np.array([np.argmax(self.scoresR)])
            
    def get_levscores(self, v):
        """
        Compute leverage scores.
        """
        
        if self.beta is None:
            return np.mean(v[:self.k, :] ** 2, axis=0)
        else:
            scores = np.sum((v[:self.k, :].T ** 2 * \
                             self.sv[1][:self.k] ** self.beta).T, 
                            axis=0)
            scores /= np.sum(scores)
            return scores
        
    def transform(self, stable=True, rank_k=False):
        """
        Compute estimated X = CUR.
        stable: do StableCUR
        """
        
        if self.X is None:
            print("Run 'fit(X)' first.")
        else:
            if stable:
                Qc, _ = qr(self.getC(), 'reduced')
                Qr, _ = qr(self.getR().T, 'reduced')
                B = Qc.T.dot(self.X).dot(Qr)
                if rank_k:
                    U, _, _ = svd(B)
                    B = U[:, :self.k].dot(U[:, :self.k].T.dot(B))
                return Qc.dot(B).dot(Qr.T)
            else:
                return self.getC().dot(self.getU()).dot(self.getR())
            
            
    def getC(self):
        """
        Return C.
        """
        idx = itemfreq(self.idxC)
        C = self.X[:, idx[:, 0]]
        return C * np.sqrt(idx[:, 1])
    
    def getU(self):
        """
        Compute U.
        """
        return pinv(self.getC()).dot(self.X).dot(pinv(self.getR()))


    def getR(self):
        """
        Return R.
        """
        idx = itemfreq(self.idxR)
        R = self.X[idx[:, 0], :]

        return (R.T * np.sqrt(idx[:, 1])).T 

    def getError(self, stable=True, rank_k=False):
        """
        Return relative error of estimated X
        """
        return norm(self.X - self.transform(stable, rank_k)) / norm(self.X)

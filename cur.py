# -*- coding: utf-8 -*-
# Created on Fri Jun 12 16:35:39 2015
# Implemented in Python 3.4.0
# Author: Yun-Jhong Wu
# E-mail: yjwu@umich.edu

import numpy as np
from numpy.linalg import svd, pinv, norm
from numpy.random import uniform
class CUR:
    """
    Based on rCUR R package
    Mahoney and Drineas
    Method: 'random', 'exact', 'top', 'orthotop', 'highest'
    """
    def __init__(self, k=None, 
                 beta=4, alpha=1, weighted=True,
                 method='random'):
        self.k = k
        self.weighted = weighted
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
    
    def fit(self, X, r=None, c=None, sv=None):
        self.k = min(self.k, min(X.shape)) if self.k else None
        self.X = X
        self.c = min(self.X.shape[1], c) if c else self.X.shape[1]
        self.r = min(self.X.shape[0], r) if r else self.X.shape[0]

        self.scoresC = None
        self.scoresR = None
        self.idxC = None
        self.idxR = None

        self.err = None
        
        if not sv:
            sv = svd(X, full_matrices=False)
        if not self.k:
            sv_cumsum = np.cumsum(sv[1])
            self.k = np.sum(sv_cumsum < sv_cumsum[-1] * 0.8)

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
                self.idxC = np.zeros(self.c)
                vort = self.X.copy()
                vnormsqr = np.sum(self.X ** 2, axis=0)
                vortnormsqr = vnormsqr
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
                self.idxR = np.zeros(self.r)
                vort = self.X.copy()
                vnormsqr = np.sum(self.X ** 2, axis=1)
                vortnormsqr = vnormsqr
                for i in range(self.r):
                    self.idxR[i] = np.argmax(pi + self.alpha * np.sqrt(vortnormsqr / vnormsqr))
                    sn = vortnormsqr[self.idxR[i]]
                    if sn:
                        delta = vort[:, self.idxR[i]].dot(vort)
                        vort -= np.outer(vort[:, self.idxR[i]], delta) / sn
                        vortnormsqr -= delta ** 2 / sn
                        vortnormsqr[vortnormsqr < 0] = 0
                    pi[self.idxR[i]] = 0
                
            elif self.method == 'rank':
                v_cumsum = np.cumsum(self.sv[1].T ** 2, axis=0)
                self.scoresC = 1 / (self.X.shape[0] + 1) * np.max(v_cumsum, axis=0)- np.argmax(v_cumsum, axis=0)
                self.idxC = np.argpartition(self.scoresC, c)[:c]

        else:
            self.idxC = np.arange(self.X.shape[1]) 
            
        if not self.idxC.size:
            self.idxC = np.array([np.argmax(self.scoresC)])

        if not self.idxR.size:
            self.idxR = np.array([np.argmax(self.scoresR)])
            
    def get_levscores(self, v):
        if self.weighted:
            scores = np.sum((v[:self.k, :].T ** 2 * \
                             self.sv[1][:self.k] ** self.beta).T, 
                            axis=0)
            scores /= np.sum(scores)
            return scores
        else:
            return np.mean(v[:self.k, :] ** 2, axis=0)
        
    def transform(self):
        if self.X is not None:
            return self.getC().dot(self.getU().dot(self.getR()))
        else:
            print("Run 'fit(X)' first.")
            return 0
    
    def getC(self):
        return self.X[:, self.idxC]
    
    def getU(self):
        return pinv(self.X[:, self.idxC][self.idxR, :])

    def getR(self):
        return self.X[self.idxR, :]

    def getError(self):
        return norm(self.X - self.transform()) / norm(self.X)
        
cur = CUR(method='exactnum')  
#np.random.seed(1)
#X = np.random.normal(0, 1, (50, 10)).dot(np.random.normal(0, 1, (10, 100)))
X = np.loadtxt("X")
C = X[:, :10]
R = X[:5, :]
cur.fit(X, r=5, c=20)
print(norm(X - C.dot(pinv(X[:5,:][:,:10]).dot(R))) / norm(X))
print(cur.getError())
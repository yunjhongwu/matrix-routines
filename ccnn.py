# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 07:53:55 2016
Implemented in Python 3.5.2
Author: Yun-Jhong Wu
"""

import numpy as np

from typing import Union, Iterable
from sklearn.decomposition import TruncatedSVD
from sklearn.kernel_approximation import RBFSampler

from skimage.util import view_as_windows, view_as_blocks

class CCNNLayer:

    def __init__(self, name: str, input_size: int, filter_size: int,
                 gamma: float, m: int, R: float, r: int, lr: float):

        self.name = name
        self.input_size = input_size    
        self.filter_size = filter_size
        self.patch_size = filter_size ** 2
        self.output_size = self.input_size - self.filter_size + 1
        self.n_patchs = self.output_size ** 2        
        self.m = m
        self.R = R
        self.lr = lr
        
        self.rbf_feature = RBFSampler(gamma=gamma, n_components=m, random_state=1)
        self.svd = TruncatedSVD(n_components=r)


    def initPars(self, n_classes: int, batch_size: int):

        self.n_classes = n_classes        
        self.batch_size = batch_size
        self.lr /= batch_size
        
        self.A = np.random.normal(0, 0.1, size=(n_classes, self.n_patchs, self.m))
        
        
    def getZMatrix(self, X):
        """
        Input: (n_instances, n_channels, input_size, input_size)
        
        Output: (n_instances, n_patchs, m)
        """
        
        Z = view_as_windows(X, (1, X.shape[1], self.filter_size, self.filter_size))
        Z = Z.reshape(np.prod(Z.shape[:4]), np.prod(Z.shape[4:]))
        Q = self.rbf_feature.transform(Z).astype(np.float16)
        
        return Q.reshape(X.shape[0], self.n_patchs, -1)


    def predict(self, X, transform: bool=False):
        """
        Input: (batch_size, n_channels, input_size, input_size)
        
        Transformed input: (batch_size, n_patchs, m)
        
        Output: (batch_size, n_classes)
        """

        Z = self.getZMatrix(X) if transform else X
        p = np.exp(np.tensordot(Z, self.A, axes=[(1, 2), (1, 2)]))

        return (p.T / np.sum(p, axis=1)).T


    def fit(self, X, ylabel, n_epoch: int):

        assert X.shape[2] == X.shape[3] == self.input_size
        
        n = X.shape[0]
        self.rbf_feature.fit(np.zeros((1, X.shape[1] * self.filter_size ** 2)))
        
        print("Preparing patches...")
        
        Z_batches = [self.getZMatrix(X[i: i + self.batch_size]) 
                     for i in range(0, n, self.batch_size)]
        y_batches = ylabel.reshape(-1, self.batch_size)
        
        print("Starting PSGD...")
        
        loss = np.inf
        rhat = self.m

        for epoch in range(n_epoch):
            print("{0}: Epoch {1}: loss = {2}, r_hat = {3}".format(self.name, epoch + 1, loss / n, rhat))
            loss = 0
            for i, (Z_batch, y_batch) in enumerate(zip(Z_batches, y_batches)):
                p_batch = self.predict(Z_batch)
                loss += np.sum(-np.log(p_batch[np.arange(self.batch_size), y_batch]))
                dL_batch = -p_batch
                dL_batch[np.arange(self.batch_size), y_batch] += 1

                self.A += self.lr * np.tensordot(dL_batch, Z_batch, axes=[0, 0])
  
            A_unfold = self.A.reshape(-1, self.A.shape[2]).T
            U = self.svd.fit_transform(A_unfold)
            self.U = U.copy()
            d = np.linalg.norm(U, axis=0)
            U *= 1 / d            
            d_cum = np.cumsum(d) 
            rhat = np.searchsorted(d_cum - self.R > np.append(d[1:] * np.arange(1, d.size), 0), True) + 1

            if rhat >= d.size:
                print("Warning: Hard-thresholding applied")
                                
            if rhat <= d.size:                
                scale = np.maximum(0, d - (d_cum[rhat - 1] - self.R) / rhat)
                U = U[:, :rhat]
                d = d[:rhat]                  
                self.U = U * scale[:rhat] 

            self.A = ((self.U * (1 / d)) @ (U.T @ A_unfold)).T.reshape(*self.A.shape)
        
        Z_batches = None
        y_batches = None
        
        
            
    def transform(self, X):
        """
        Input: (batch_size, n_channels, input_size, input_size)
        
        Output: (batch_size, n_output_channels, output_size, output_size)
        """
        
        Z = np.rollaxis(np.tensordot(self.U, self.getZMatrix(X), axes=[0, 2]), 0, 2)

        return Z.reshape(Z.shape[0], Z.shape[1], self.output_size, self.output_size)


class AveragePooling:

    def __init__(self, name: str, input_size: int, pool_size: int):
        
        self.name = name
        self.pool_size = (1, 1, pool_size, pool_size)
        self.input_size = input_size
        self.output_size = input_size // pool_size
    
    def transform(self, X): 

        return np.mean(view_as_blocks(X, self.pool_size), axis=(4, 5, 6, 7))
     
     
    def initPars(self, *args, **kargs):

        pass
        
        

class DeepCCNN:
    
    def __init__(self, n_classes: int, batch_size: int):
                     
        self.layers = []
        self.n_classes = n_classes        
        self.batch_size = batch_size
        self.compiled = False


    def addLayer(self, layer):
        self.layers.append(layer)


    def compileCCNN(self):
        for prevLayer, nextLayer in zip(self.layers[:-1], self.layers[1:]):
            assert prevLayer.output_size == nextLayer.input_size
        
        for layer in self.layers:
            layer.initPars(n_classes=self.n_classes, 
                           batch_size=self.batch_size)
        
        self.compiled = True
        

    def fit(self, X_train, y_train, n_epoch: Union[int, Iterable]):
        assert self.compiled
        assert isinstance(self.layers[-1], CCNNLayer)
        
        n_conv = sum(isinstance(layer, CCNNLayer) for layer in self.layers)
        n_part = X_train.shape[0] // self.batch_size // n_conv * self.batch_size
        n_trained = 0

        for k, layer in enumerate(self.layers):
            if isinstance(layer, CCNNLayer):
                X = X_train.copy()
                y = y_train.copy()
                
                for prevLayer in self.layers[:k]:
                    X = prevLayer.transform(X)
                    
                if isinstance(n_epoch, int):
                    layer.fit(X, y, n_epoch)                    
                else:
                    layer.fit(X, y, n_epoch[n_trained])
                n_trained += 1


    def predict(self, X):
        
        return np.argmax(self.predict_prob(X), axis=1)


    def predict_prob(self, X):
        
        for layer in self.layers[:-1]:
            X = layer.transform(X)
            
        return self.layers[-1].predict(X, transform=True)


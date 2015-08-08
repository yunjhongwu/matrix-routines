import numpy as np
from functools import reduce

def parafac2(X, r=2, tol=1e-5, verbose=True):
    m = len(X)
    F = np.identity(r)
    D = np.ones((m, r))
    A = np.linalg.eigh(reduce(lambda A, B: A + B, map(lambda Xi: Xi.T.dot(Xi), X)))
    A = A[1][:, np.argsort(A[0])][:, -r:]

    H = [np.linalg.qr(Xi, mode='r') if Xi.shape[0] > Xi.shape[1] else Xi for Xi in X]
    G = [np.identity(r), np.identity(r), np.ones((r, r)) * m]
    
    err = 1
    conv = False
    niters = 0
    while not conv and niters < 100:
        P = [np.linalg.svd((F * D[i, :]).dot(H[i].dot(A).T), full_matrices=0) for i in range(m)]
        P = [(S[0].dot(S[2])).T for S in P]
        T = np.array([P[i].T.dot(H[i]) for i in range(m)])
        
        F = np.reshape(np.transpose(T, (0, 2, 1)), (-1, T.shape[1])).T.dot( _KhatriRao(D, A)).dot(np.linalg.pinv(G[2] * G[1]))
        G[0] = F.T.dot(F)
        A = np.reshape(np.transpose(T, (0, 1, 2)), (-1, T.shape[2])).T.dot( _KhatriRao(D, F)).dot(np.linalg.pinv(G[2] * G[0]))
        G[1] = A.T.dot(A)
        D = np.reshape(np.transpose(T, (2, 1, 0)), (-1, T.shape[0])).T.dot( _KhatriRao(A, F)).dot(np.linalg.pinv(G[1] * G[0]))
        G[2] = D.T.dot(D)        
        err_old = err
        err = np.sum(np.sum((H[i] - (P[i].dot(F) * D[i, :]).dot(A.T)) ** 2) for i in range(m))
        niters += 1        
        conv = abs(err_old - err) < tol * err_old
        if verbose: print("Iteration {0}; error = {1:.6f}".format(niters, err))

    P = [np.linalg.svd((F * D[i, :]).dot(X[i].dot(A).T), full_matrices=0) for i in range(m)]
    F = [(S[0].dot(S[2])).T.dot(F) for S in P]
    return F, D, A
    
def _KhatriRao(A, B):
    return np.repeat(A, B.shape[0], axis=0) * np.tile(B, (A.shape[0], 1))


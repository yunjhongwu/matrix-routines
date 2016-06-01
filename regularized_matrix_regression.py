import numpy as np
from sklearn.decomposition import TruncatedSVD

class RegularizedMatrixRegression:
    def __init__(self, r: int=1, C: float=1.0, alpha: float=1, max_iters: int=100):
        self.r = r
        self.C = C
        self.alpha = alpha
        self.max_iters = max_iters

    def _link(self, x, b) -> float:
        return 1 / (1 + np.exp(-max(-20, min(20, np.sum(x * b)))))
        
    def _PBCD(self, X, y):
        svd = TruncatedSVD(n_components=self.r)
        b = np.random.normal(0, 1, size=X[0].shape)
        niters = 0
        x_idx = [(i, k) for k in range(3) for i in range(len(y))]
        while niters < self.max_iters:
            b_old = b.copy()
            np.random.shuffle(x_idx)
            for i, k in x_idx:
                b[:, :, k] += self.alpha * (y[i] - self._link(X[i], b)) * X[i][:, :, k] 
                  
            for k in range(3):
                u = svd.fit_transform(b[:, :, k])
                u *= 1 / np.linalg.norm(u, axis=0)
                b[:, :, k] = u @ (u.T @ b[:, :, k])
            niters += 1
            err = np.sqrt(np.sum((b - b_old) ** 2))
            b_norm = np.sqrt(np.sum(b ** 2))
            b *= min(1, self.C * np.sqrt(np.size(b)) / b_norm)
            print("Iteration {0}: error = {1:.2f}, ||B||_F = {2:.2f}".format(niters, err, b_norm))
            if err < 1e-1:
                break
        else:
            print("Maximum number of iterations exceeded ({0})".format(self.max_iters))
        return b
        
    def fit(self, X, y):
        self.coef_ = self._PBCD(X, y)

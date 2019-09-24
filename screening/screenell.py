import numpy as np
from lightning.classification import LinearSVC
from screening.screentools import (
    fit_estimator,
    rank_dataset, 
    rank_dataset_accelerated, 
    compute_subgradient, 
    compute_A_g
)
import time 

class EllipsoidScreener:

    def __init__(self, lmbda, mu, loss, penalty, intercept, classification, n_ellipsoid_steps, 
                    better_init, better_radius, cut, clip_ell):

        self.lmbda = lmbda
        self.mu = mu
        self.loss = loss
        self.penalty = penalty
        self.intercept = intercept
        self.classification = classification
        self.n_steps = n_ellipsoid_steps
        self.better_init = better_init
        self.better_radius = better_radius
        self.cut = cut
        self.clip_ell = clip_ell

    def iterate_ellipsoids_accelerated(self, D, y, z_init, r_init):

        if self.intercept:
            X = np.concatenate((D, np.ones(D.shape[0]).reshape(1,-1).T), axis=1)
        else:
            X = D
        start = time.time()
        k = 0
        z = z_init
        p = z_init.size
        s = p ** 2 / (p ** 2 - 1)
        scaling = r_init * (s ** (self.n_steps))

        while k < self.n_steps:
            g = compute_subgradient(z, X, y, self.lmbda, self.mu, self.loss, self.penalty, self.intercept)
            if k == 0:
                A_g = r_init * g
                den = np.sqrt(g.dot(A_g))
                A_g = (1 / den) * A_g 
                z = z - (1 / (p + 1)) * A_g
                A_g = A_g.reshape(1,-1)
                L = A_g.T
                I_k_vec = s * (2 / (p + 1)) * np.ones(1)
            else:
                A_g = compute_A_g(r_init * (s ** k), L, I_k_vec, g)
                den = np.sqrt(g.dot(A_g))
                A_g = (1 / den) * A_g
                z = z - (1 / (p + 1)) * A_g
                A_g = A_g.reshape(1,-1)
                L = np.concatenate((L, A_g.T), axis=1)
                I_k_vec = np.insert(I_k_vec, 0, (s ** (k + 1)) * (2 / (p + 1)))
            k += 1
    
        g = compute_subgradient(z, X, y, self.lmbda, self.mu, self.loss, self.penalty, self.intercept)

        end = time.time()
        print('Time to compute z, A and g:', end - start)
        return z, scaling, L, I_k_vec, g
    
    def get_ellipsoid(self, X, y):

        if self.intercept:
            z_init = np.zeros(X.shape[1] + 1)
            r_init = X.shape[1] + 1
        else:
            z_init = np.zeros(X.shape[1])
            r_init = X.shape[1]

        if self.better_init != 0:
            est = fit_estimator(X, y, self.loss, self.penalty, self.mu, self.lmbda, self.intercept, max_iter=self.better_init)
            if self.classification:
                z_init = est.coef_[0]
                if self.intercept:
                    z_init = np.append(z_init, est.intercept_)
            else:
                z_init = est.coef_
                if self.intercept:
                    z_init = np.append(z_init, est.intercept_)
        if self.better_radius != 0:
            r_init = float(self.better_radius)                            
        z, scaling, L, I_k_vec, g = self.iterate_ellipsoids_accelerated(X, y, z_init, r_init)
        return z, scaling, L, I_k_vec, g, r_init

    def screen(self, X, y):

        z, scaling, L, I_k_vec, g, r_init = self.get_ellipsoid(X, y)
        if self.clip_ell:
            I_k_vec = I_k_vec.reshape(-1,1)
            A = scaling * np.identity(X.shape[1]) - L.dot(np.multiply(I_k_vec, np.transpose(L)))
            eigenvals, eigenvect = np.linalg.eigh(A)
            eigenvals = np.clip(eigenvals, 0, r_init)
            eigenvals = eigenvals.reshape(-1,1)
            A = eigenvect.dot(np.multiply(eigenvals, np.transpose(eigenvect)))
            scores = rank_dataset(X, y, z, A, g,
                                    self.mu, self.classification, self.intercept, self.cut)
        else:
            scores = rank_dataset_accelerated(X, y, z, scaling, L, I_k_vec, g,
                                    self.mu, self.classification, self.intercept, self.cut)
        return scores

if __name__ == "__main__":
    #we test that it works with MNIST
    from screening.tools import scoring_classif
    from sklearn.model_selection import train_test_split
    from screening.loaders import load_experiment
    
    X, y = load_experiment(dataset='mnist', synth_params=None, size=60000, redundant=0, 
                            noise=None, classification=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    screener = EllipsoidScreener(lmbda=0.0001, mu=1, loss='squared_hinge', penalty='l2', 
                                    intercept=False, classification=True, n_ellipsoid_steps=100, 
                                    better_init=0, better_radius=100, cut=False, clip_ell=False)
    print(screener.screen(X_train, y_train)[:10])
  
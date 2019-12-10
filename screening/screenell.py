import numpy as np
from screening.screentools import (
    rank_dataset, 
    rank_dataset_accelerated, 
    compute_subgradient, 
    compute_A_g,
    compute_loss
)
from screening.fit import (
    fit_estimator
)
import time 
import random
from scipy.sparse import (
    csr_matrix,
    csc_matrix,
    diags,
    identity,
)
from scipy.sparse.linalg import (
    eigsh
)

class EllipsoidScreener:

    def __init__(self, lmbda, mu, loss, penalty, intercept, classification, n_ellipsoid_steps, 
                    better_init=0, better_radius=0, cut=False, clip_ell=False, sgd=False, acceleration=True, dc=False,
                    use_sphere=False, ars=False):
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
        self.sgd = sgd
        self.acceleration = acceleration
        self.dc = dc
        self.use_sphere = use_sphere
        self.g = None
        self.ars = ars

    #@profile   
    def update_ell(self, z, A, g):
        p = z.size
        A_g = A.dot(g)
        den = np.sqrt(g.dot(A_g))
        g = (1 / den) * g
        A_g = A.dot(g)
        z = z - (1 / (p + 1)) * A_g
        A = (p ** 2 / (p ** 2 - 1)) * (A - (2 / (p + 1)) * np.outer(A_g, A_g))
        return z, A

    #@profile
    def iter_ell(self, X, y, z_init, r_init):
        k = 0
        z = z_init
        A = r_init * np.identity(X.shape[1])
        if self.sgd:
            random_subset = np.arange(X.shape[0])
            while k < self.n_steps:
                random.shuffle(random_subset)
                for i in range(X.shape[0]):
                    g = compute_subgradient(z, X[random_subset[i]].reshape(1,-1), np.array(y[random_subset[i]]), 
                                            self.lmbda, self.mu, self.loss, self.penalty, self.intercept, self.ars)
                    z, A = self.update_ell(z, A, g)
                k += 1 
                print(k)
        else:
            while k < self.n_steps:
                g = compute_subgradient(z, X, y, self.lmbda, self.mu, self.loss, self.penalty, self.intercept, self.ars)
                z, A = self.update_ell(z, A, g)
                k += 1 
        
        self.z = z
        self.A = A

        return
    
    #@profile   
    def update_ell_dc(self, z, A, g, f_best, f_):
        p = z.size
        A_g = A.dot(g)
        den = np.sqrt(g.dot(A_g))
        alpha = (f_ - f_best) / den
        g = (1 / den) * g
        A_g = A.dot(g)
        z = z - ((1 + p * alpha) / (p + 1)) * A_g
        A = (p ** 2 / (p ** 2 - 1)) * (1 - alpha ** 2) * (A - ((2 * (1 + p * alpha)) / ((p + 1) * (1 + alpha))) * np.outer(A_g, A_g))
        return z, A

    #@profile
    def iter_ell_dc(self, X, y, z_init, r_init):
        k = 0
        z = z_init
        A = r_init * np.identity(X.shape[1])
        while k < self.n_steps:
            g = compute_subgradient(z, X, y, self.lmbda, self.mu, self.loss, self.penalty, self.intercept, self.ars)
            f_ = compute_loss(z, X, y, self.loss, self.penalty, self.lmbda, self.mu)
            if k == 0:
                f_best = f_
            else:
                if f_ < f_best:
                    f_best = f_
            z, A = self.update_ell_dc(z, A, g, f_best, f_)
            k += 1 
        
        self.z = z
        self.A = A

        return

    #@profile
    def iter_ell_accelerated(self, D, y, z_init, r_init):
        if self.intercept:
            X = np.concatenate((D, np.ones(D.shape[0]).reshape(1,-1).T), axis=1)
        else:
            X = D
        k = 0
        z = z_init
        p = z_init.size
        s = (p ** 2) / (p ** 2 - 1)
        scaling = r_init * (s ** (self.n_steps))

        L = np.zeros((self.n_steps, X.shape[1]))
        while k < self.n_steps:
            if k % 10 == 0:
                print('Ellipsoid method iteration ', k)
            g = compute_subgradient(z, X, y, self.lmbda, self.mu, self.loss, self.penalty, 
                                    self.intercept, self.ars)
            if k == 0:
                A_g = r_init * g
                I_k_vec = s * (2 / (p + 1)) * np.ones(1)
            else:
                A_g = compute_A_g(r_init * (s ** k), L[:k,:].T, I_k_vec, g)
                I_k_vec = np.insert(I_k_vec, 0, (s ** (k + 1)) * (2 / (p + 1)))
            den = np.sqrt(g.dot(A_g))
            A_g = (1 / den) * A_g 
            z = z - (1 / (p + 1)) * A_g
            L[k,:] = A_g
            k += 1

        self.z = z
        self.scaling = scaling
        if type(D).__name__ == 'csr_matrix':
            L = csc_matrix(L)
        self.L = L.T
        self.I_k_vec = I_k_vec
        self.r_init = r_init
    
        return 
    
    def get_ell(self, X, y, init, rad):
        if self.intercept:
            z_init = np.zeros(X.shape[1] + 1)
            r_init = X.shape[1] + 1
        else:
            z_init = np.zeros(X.shape[1])
            r_init = X.shape[1]

        if self.better_init is not 0:
            est = fit_estimator(X, y, self.loss, self.penalty, self.mu, self.lmbda, self.intercept, max_iter=self.better_init, ars=self.ars)
            if self.classification:
                if self.ars and self.loss != 'safe_logistic':
                    z_init = est.w
                else:
                    z_init = est.coef_[0]
                if self.intercept:
                    z_init = np.append(z_init, est.intercept_)
            else:
                z_init = est.coef_
                if self.intercept:
                    z_init = np.append(z_init, est.intercept_)

        if init is not None and self.better_init == 0:
            z_init = init

        if rad != 0 and self.better_radius == 0:
            r_init = rad

        if self.better_radius != 0:
            r_init = float(self.better_radius)
        
        if self.acceleration:                            
            self.iter_ell_accelerated(X, y, z_init, r_init)
        elif self.dc:
            self.iter_ell_dc(X, y, z_init, r_init)
        else:
            self.iter_ell(X, y, z_init, r_init)
        
        return
    
    #@profile
    def fit(self, X_train, y_train, init=None, rad=0):
        start = time.time()
        if self.n_steps == 0:
            self.z = init
            self.A = rad * np.identity(X_train.shape[1])
        else:
            self.get_ell(X_train, y_train , init, rad)

            if self.acceleration and self.clip_ell:
                self.I_k_vec = self.I_k_vec.reshape(-1,1)
                A = self.scaling * np.identity(X_train.shape[1]) - self.L.dot(np.multiply(self.I_k_vec, self.L.T))
                eigenvals, eigenvect = np.linalg.eigh(A)
                eigenvals = np.clip(eigenvals, 0, self.r_init)
                eigenvals = eigenvals.reshape(-1,1)
                self.A = eigenvect.dot(np.multiply(eigenvals, np.transpose(eigenvect)))
            
            if self.acceleration and self.use_sphere:
                self.I_k_vec = self.I_k_vec.reshape(1,-1)
                if type(X_train).__name__ != 'csr_matrix':
                    U = np.multiply(self.L, np.sqrt(self.I_k_vec))
                    UTU = U.T.dot(U)
                    eigenval_max = np.linalg.eigvalsh(UTU)[-1]
                else:
                    U = csc_matrix(self.L.multiply(self.I_k_vec))
                    UTU = U.T.dot(U)
                    eigenval_max = eigsh(UTU, k=1, which='LM', return_eigenvectors=False)[0]
                self.squared_radius = self.scaling - eigenval_max
                print('Using minimal sphere of ellipsoid')

        if self.cut:
            self.g = compute_subgradient(self.z, X_train, y_train, self.lmbda, self.mu, self.loss, 
                                            self.penalty, self.intercept, self.ars)

        end = time.time()
        print('Time to fit EllipsoidScreener :', end - start)

        return self

    def score(self, X, y):
        if self.classification:
            outputs = np.sign(X.dot(self.z)) * y
            outputs_ = [1 if output > 0 else 0 for output in outputs]
            return np.sum(outputs_) / X.shape[0]
        else:
            pass

    def screen(self, X, y):
        if self.clip_ell or not(self.acceleration) or self.n_steps == 0:
            self.scores = rank_dataset(X, y, self.z, self.A, self.g,
                                    self.mu, self.classification, self.intercept, self.cut)
        elif self.use_sphere:
            self.scores = rank_dataset_accelerated(X, y, self.z, self.squared_radius, 0, 0, self.g, 
                                    self.mu, self.classification, self.intercept, self.cut)
        else:
            self.scores = rank_dataset_accelerated(X, y, self.z, self.scaling, self.L, self.I_k_vec, self.g,
                                    self.mu, self.classification, self.intercept, self.cut)
        return self.scores

if __name__ == "__main__":
    #we check that it works with MNIST
    from sklearn.model_selection import train_test_split
    from screening.loaders import load_experiment
    
    X, y = load_experiment(dataset='cifar10_kernel', synth_params=None, size=10000, redundant=0, 
                            noise=None, classification=True)
    
    #random.seed(0)
    #np.random.seed(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    z_init = np.random.rand(X_train.shape[1])
    screener = EllipsoidScreener(lmbda=0, mu=0, loss='safe_logistic', penalty='l2', 
                                intercept=False, classification=True, n_ellipsoid_steps=2000, 
                                better_init=20, better_radius=1, cut=False, clip_ell=False, 
                                sgd=False, acceleration=True, dc=False, use_sphere=False,
                                ars=True).fit(X_train, y_train)
    prop = np.unique(y_test, return_counts=True)[1]
    print('BASELINE : ', 1 - prop[1] / prop[0])
    print('SCORE SCREENER : ', screener.score(X_test, y_test))
    #print(screener.z)
    scores = screener.screen(X_train, y_train)
    idx_safeell = np.where(scores > 0)[0]
    print('NB TO KEEP', len(idx_safeell))
    if len(idx_safeell) !=0:
        estimator_whole = fit_estimator(X_train, y_train, loss='safe_logistic', penalty='l2', 
                                        mu=0, lmbda=0, intercept=False)
        print(y_train[idx_safeell][:10])
        print(estimator_whole.score(X_test, y_test))
        estimator_screened = fit_estimator(X_train[idx_safeell], y_train[idx_safeell], 
                                            loss='safe_logistic', penalty='l2', mu=0, 
                                            lmbda=0, intercept=False)
        print(estimator_screened.score(X_test, y_test))
        temp = np.array([estimator_whole.score(X_train, y_train), 
                    estimator_screened.score(X_train, y_train)])
        print('SAFE GUARANTEE : ', temp)


    print('GUARANTEE : ', )
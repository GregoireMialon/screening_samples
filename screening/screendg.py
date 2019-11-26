from sklearn.svm import LinearSVC
from screening.screentools import (
    rank_dataset,
    rank_dataset_accelerated,
    compute_squared_hinge,
    compute_squared_hinge_gradient,
    compute_squared_hinge_conjugate
)
from arsenic import BinaryClassifier
import numpy as np
import time

class DualityGapScreener:
    '''
    For the LinearSVC, liblinear does as many iterations in an epoch as training examples. Thus,
    1 Epoch of LinearSVC = 1 Ellipsoid Iteration in terms of complexity.
       
    Works with Squared Hinge and L2 Penalty only. Requires a strongly convex primal loss. 
    '''


    def __init__(self, lmbda, n_epochs, ars=False):
        self.lmbda = lmbda
        self.n_epochs = n_epochs
        self.ars = ars 


    def get_duality_gap(self, svc):
        '''
        Uses the trick in Sparse Coding for ML, Mairal, 2010, Appendix D.2.3.
        Liblinear and Lightning optimize (1/2) * ||x|| ^2 + C * loss(Ax).
        The loss is not normalized by the number of samples.
        '''
        if self.ars:
            info = svc.eval(self.X_train, self.y_train, self.lmbda)
            loss = info[1,-1]
            dg = loss - info[2, -1]
        else:
            coef = svc.coef_.reshape(-1,)
            pred = self.X_train.dot(coef) * self.y_train
            primal_loss = compute_squared_hinge(u=pred, mu=1)
            primal_reg = (self.lmbda / 2) * (np.linalg.norm(coef) ** 2)
            dual_coef = compute_squared_hinge_gradient(u=pred, mu=1)
            dual_pred = self.X_train.T.dot(self.y_train * dual_coef)
            dual_loss = compute_squared_hinge_conjugate(dual_coef)
            dual_reg = (1 / (2 * self.lmbda)) * (np.linalg.norm(dual_pred) ** 2)
            loss = primal_loss + primal_reg
            dg = loss + dual_loss + dual_reg
        return loss, dg
    

    def fit(self, X_train, y_train, init=None):
        start = time.time()
        self.X_train = X_train
        self.y_train = y_train
    
        if self.ars:
            svc = BinaryClassifier(loss='sqhinge', penalty='l2', intercept=False)
            if init is not None:
                svc.w = init
                restart = True
            else:
                restart = False
            #self.first_obj, self.first_dg =  self.get_duality_gap(svc)
            info = svc.fit(X_train, y_train, lambd=self.lmbda, solver='acc-svrg', 
                        nepochs=self.n_epochs, it0=1, tol=1.0e-20, restart=restart, verbose=False)
            self.loss = info[1,-1]
            self.dg = self.loss - info[2, -1]
            self.z = svc.w.reshape(-1,)
        else:
            first_svc = LinearSVC(loss='squared_hinge', dual=False, C=1/self.lmbda, fit_intercept=False, 
                            max_iter=0, tol=1.0e-20).fit(self.X_train, self.y_train)
            self.first_obj, self.first_dg = self.get_duality_gap(first_svc)
            svc = LinearSVC(loss='squared_hinge', dual=False, C=1/self.lmbda, fit_intercept=False, 
                            max_iter=self.n_epochs, tol=1.0e-20).fit(self.X_train, self.y_train)
            self.z = svc.coef_.reshape(-1,)
            self.loss, self.dg = self.get_duality_gap(svc)
        self.squared_radius = 2 * self.dg / self.lmbda
        end = time.time()
        print('Time to fit DualityGapScreener :', end - start)
        return self
    
    def score(self, X, y):
        outputs = np.sign(X.dot(self.z)) * y
        outputs_ = [1 if output > 0 else 0 for output in outputs]
        return np.sum(outputs_) / X.shape[0]

    def screen(self, X, y):
        # tol must be small enough so that max_iter is attained
        return rank_dataset_accelerated(D=X, y=y, z=self.z, scaling=self.squared_radius, 
                                        L=0, I_k_vec=0, g=None, mu=1, classification=True, intercept=False, 
                                        cut=False)
    

if __name__ == "__main__":
    #we check that it works with MNIST
    from sklearn.model_selection import train_test_split
    from screening.loaders import load_experiment
    import random

    X, y = load_experiment(dataset='mnist', synth_params=None, size=1000, redundant=0, 
                            noise=None, classification=True)
    random.seed(0)
    np.random.seed(0)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    prop = np.unique(y_test, return_counts=True)[1]
    print('BASELINE : ', 1 - prop[1] / prop[0])

    epoch=10
    
    screener = DualityGapScreener(lmbda=0.001, n_epochs=epoch, ars=True).fit(X_train, y_train)
    print('DUALITY GAP LAMBDA 0.001 ; ', screener.dg)
    print('SCORE', screener.score(X_test, y_test))
    print('COEF', screener.z)

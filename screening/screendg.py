from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from screening.screentools import (
    rank_dataset,
    rank_dataset_accelerated,
    compute_squared_hinge,
    compute_squared_hinge_gradient,
    compute_squared_hinge_conjugate
)
import numpy as np
import time

class DualityGapScreener:
    '''
    For the LinearSVC, liblinear does as many iterations in an epoch as training examples. Thus,
    1 Epoch of LinearSVC = 1 Ellipsoid Iteration in terms of complexity.
       
    Works with Squared Hinge and L2 Penalty only. Requires a strongly convex primal loss. 
    '''


    def __init__(self, lmbda, n_epochs):
        self.lmbda = lmbda
        self.n_epochs = n_epochs


    def get_duality_gap(self, svc):
        '''
        Uses the trick in Sparse Coding for ML, Mairal, 2010, Appendix D.2.3.
        Liblinear and Lightning optimize (1/2) * ||x|| ^2 + C * loss(Ax).
        The loss is not normalized by the number of samples.
        '''
        coef = svc.coef_.reshape(-1,)
        pred = self.X_train.dot(coef) * self.y_train
        primal_loss = compute_squared_hinge(u=pred, mu=1)
        primal_reg = (self.lmbda / 2) * (np.linalg.norm(coef) ** 2)
        dual_coef = compute_squared_hinge_gradient(u=pred, mu=1)
        dual_pred = X_train.T.dot(self.y_train * dual_coef)
        dual_loss = compute_squared_hinge_conjugate(dual_coef)
        dual_reg = (1 / (2 * self.lmbda)) * (np.linalg.norm(dual_pred) ** 2) 
        loss = primal_loss + primal_reg
        dg = primal_loss + primal_reg + dual_loss + dual_reg
        return loss, dg
    

    def fit(self, X_train, y_train, init=None):
        start = time.time()
        self.X_train = X_train
        self.y_train = y_train
        first_svc = LinearSVC(loss='squared_hinge', dual=False, C=1/self.lmbda, fit_intercept=False, 
                            max_iter=0, tol=1.0e-20).fit(self.X_train, self.y_train)
        if init is not None:
            first_svc.coef_ = init
        self.first_obj, self.first_dg = self.get_duality_gap(first_svc)
        if init is not None:
            svc = SGDClassifier(loss='squared_hinge', alpha=self.lmbda, l1_ratio=0, 
                                fit_intercept=False, max_iter=self.n_epochs, warm_start=True)
            svc.coef_ = init
            svc.intercept_ = 0
            svc.fit(X_train, y_train)
        else:
            svc = LinearSVC(loss='squared_hinge', dual=False, C=1/self.lmbda, fit_intercept=False, 
                            max_iter=self.n_epochs, tol=1.0e-20).fit(self.X_train, self.y_train)
        self.z = svc.coef_
        self.loss, self.dg = self.get_duality_gap(svc)
        self.squared_radius = 2 * self.dg / self.lmbda
        #self.A = self.squared_radius * np.identity(self.X_train.shape[1])
        end = time.time()
        print('Time to fit DualityGapScreener :', end - start)
        return self


    def screen(self, X, y):
        # tol must be small enough so that max_iter is attained
        #return rank_dataset(D=X, y=y, z=self.z.reshape(-1,), A=self.A, g=None, mu=1, 
        #                    classification=True, intercept=False, cut=False)
        return rank_dataset_accelerated(D=X, y=y, z=self.z.reshape(-1,), scaling=self.squared_radius, 
                                        L=0, I_k_vec=0, g=None, mu=1, classification=True, intercept=False, 
                                        cut=False)
    

if __name__ == "__main__":
    #we check that it works with MNIST
    from screening.tools import scoring_classif
    from sklearn.model_selection import train_test_split
    from screening.loaders import load_experiment
    import random

    X, y = load_experiment(dataset='mnist', synth_params=None, size=1000, redundant=0, 
                            noise=None, classification=True)
    random.seed(0)
    np.random.seed(0)

    epoch=20

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    screener = DualityGapScreener(lmbda=1.0, n_epochs=epoch).fit(X_train, y_train)
    print('DUALITY GAP LAMBDA 1.0 : ', screener.dg)
    print(screener.screen(X_train, y_train))
    #screener = DualityGapScreener(lmbda=0.9, n_epochs=1000).fit(X_train, y_train, init=screener.z)
    #print('DUALITY GAP LAMBDA 0.9 : ', screener.dg)
    #screener = DualityGapScreener(lmbda=0.8, n_epochs=1000).fit(X_train, y_train, init=screener.z)
    #print('DUALITY GAP LAMBDA 0.8 : ', screener.dg)
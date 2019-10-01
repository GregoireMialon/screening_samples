from sklearn.svm import LinearSVC
from screening.screentools import (
    rank_dataset,
    compute_squared_hinge,
    compute_squared_hinge_gradient,
    compute_squared_hinge_conjugate
)
import numpy as np

class DualityGapScreener:
    '''
    For the LinearSVC, liblinear does as many iterations in an epoch as training examples. Thus,
    1 Epoch of LinearSVC = 1 Ellipsoid Iteration in terms of complexity.
       
    Works with Squared Hinge and L2 Penalty only. Requires a strongly convex primal loss. 
    '''


    def __init__(self, lmbda, n_epochs):
        self.lmbda = lmbda
        self.n_epochs = n_epochs
    

    def get_duality_gap_gradient_old(self, svc):
        '''
        Uses the trick in Sparse Coding for ML, Mairal, 2010, Appendix D.2.3.
        Liblinear and Lightning optimize (1/2) * ||x|| ^2 + C * loss(Ax).
        The loss is not normalized by the number of samples.
        '''
        coef = svc.coef_.reshape(-1,1)
        pred = np.dot(self.X_train, coef).flatten() * self.y_train
        grad_pred = compute_squared_hinge_gradient(u=pred, mu=1)
        primal_reg = (self.lmbda / 2) * (np.linalg.norm(svc.coef_) ** 2)
        dual_reg = (self.lmbda / 2) * (np.linalg.norm(-np.dot((self.X_train.T * self.y_train) , grad_pred) / self.lmbda)) ** 2
        grad_loss = np.dot(pred.T, grad_pred)
        self.duality_gap = primal_reg + dual_reg + grad_loss
        return self.duality_gap
    

    def get_duality_gap_gradient(self, svc):
        coef = svc.coef_.reshape(-1,1)
        pred = np.dot(self.X_train, coef).flatten() * self.y_train
        primal_loss = compute_squared_hinge(u=pred, mu=1)
        primal_reg = (self.lmbda / 2) * (np.linalg.norm(svc.coef_) ** 2)
        dual_coef = np.clip(compute_squared_hinge_gradient(u=pred, mu=1), a_min=None, a_max=0.0)
        dual_pred = np.dot(self.X_train.T * self.y_train, dual_coef)
        dual_loss = compute_squared_hinge_conjugate(dual_coef)
        dual_reg = (1 / (2 * self.lmbda)) * (np.linalg.norm(dual_pred) ** 2) 
        self.duality_gap = primal_loss + primal_reg + dual_loss + dual_reg
        #print('PL', primal_loss, 'PR', primal_reg, 'DL', dual_loss, 'DR', dual_reg)
        return self.duality_gap
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        svc = LinearSVC(loss='squared_hinge', dual=False, C=1/self.lmbda, fit_intercept=False, 
                            max_iter=self.n_epochs, tol=1.0e-10).fit(self.X_train, self.y_train)
        self.z = svc.coef_
        dg = self.get_duality_gap_gradient(svc)
        self.squared_radius = 2 * dg / self.lmbda
        self.A = self.squared_radius * np.identity(X.shape[1])
        return self

    def screen(self, X, y):
        # tol must be small enough so that max_iter is attained
        return rank_dataset(D=X, y=y, z=self.z.reshape(-1, 1), A=self.A, g=None, mu=1, 
                            classification=True, intercept=False, cut=False)
    

if __name__ == "__main__":
    #we check that it works with MNIST
    from screening.tools import scoring_classif
    from sklearn.model_selection import train_test_split
    from screening.loaders import load_experiment
    
    X, y = load_experiment(dataset='mnist', synth_params=None, size=1000, redundant=0, 
                            noise=None, classification=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    for epoch in [1, 3, 10, 30, 100, 300, 1000]:
        screener = DualityGapScreener(lmbda=0.001, n_epochs=epoch).fit(X_train, y_train)
        screener.screen(X_train, y_train)
        print('Duality Gap at Epoch {}'.format(epoch), screener.squared_radius * screener.lmbda / 2)
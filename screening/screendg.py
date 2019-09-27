from sklearn.svm import LinearSVC
from screening.screentools import (
    rank_dataset,
    compute_squared_hinge_gradient
)
import numpy as np

class DualityGapScreener:
    '''We consider lambda = 1 / ( 2 * C) with lambda the squared norm penalty term WITHOUT a factor 2.
        
       For the LinearSVC, lightning does as many iterations in an epoch as training examples. Thus,
       1 Epoch of LinearSVC = 1 Ellipsoid Iteration.
       
       Works with Squared Hinge and L2 Penalty only. Requires a smooth primal loss in general. '''

    def __init__(self, lmbda, n_epochs):
        self.lmbda = lmbda
        self.n_epochs = n_epochs
  
    def loss_gradient(self, pred):
        '''
        Gradient of the Squared Hinge Loss.
        '''
        return compute_squared_hinge_gradient(u=pred, mu=1)
    
    def get_duality_gap_gradient(self, svc):
        '''
        Uses the trick in Sparse Coding for ML, Mairal, 2010, Appendix D.2.3.
        Liblinear and Lightning optimize (1/2) * ||x|| ^2 + C * loss(Ax).
        The loss is not normalized by the number of samples
        '''
        coef = svc.coef_.reshape(-1,1)
        pred = np.dot(self.X, coef).flatten() * self.y
        grad_pred = self.loss_gradient(pred)
        primal_reg = (self.lmbda / 2) * (np.linalg.norm(svc.coef_) ** 2)
        dual_reg = (self.lmbda / 2) * (np.linalg.norm(-np.dot((self.X.T * self.y) , grad_pred) / self.lmbda)) ** 2
        grad_loss = np.dot(pred.T, grad_pred)
        self.duality_gap = primal_reg + dual_reg + grad_loss
        return self.duality_gap
      
    def screen(self, X, y):
        # tol must be small enough so that max_iter is attained
        self.X = X
        self.y = y
        svc = LinearSVC(loss='squared_hinge', dual=False, C=1/self.lmbda, fit_intercept=False, 
                            max_iter=self.n_epochs, tol=1.0e-10).fit(self.X, self.y)
        self.z = svc.coef_
        dg = self.get_duality_gap_gradient(svc)
        self.squared_radius = 2 * dg / self.lmbda
        A = self.squared_radius * np.identity(X.shape[1])
        return rank_dataset(D=self.X, y=self.y, z=svc.coef_.reshape(-1, 1), A=A, g=None, mu=1, 
                            classification=True, intercept=False, cut=False)

if __name__ == "__main__":
    #we test that it works with MNIST
    from screening.tools import scoring_classif
    from sklearn.model_selection import train_test_split
    from screening.loaders import load_experiment
    
    X, y = load_experiment(dataset='mnist', synth_params=None, size=60000, redundant=0, 
                            noise=None, classification=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    screener = DualityGapScreener(lmbda=0.001, n_epochs=10)
    print('SCORES', screener.screen(X_train, y_train))
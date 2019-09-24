from lightning.classification import LinearSVC as LinearSVC_l
from screening.screentools import rank_dataset
import numpy as np

class DualityGapScreener:
    '''We consider lambda = 1 / ( 2 * C) with lambda the squared norm penalty term WITHOUT a factor 2.
        
       For the LinearSVC, lightning does as many iterations in an epoch as training examples. Thus,
       1 Epoch of LinearSVC = 1 Ellipsoid Iteration.
       
       Works with Squared Hinge and L2 Penalty only. Requires a smooth primal loss in general. '''

    def __init__(self, lmbda, n_epochs):
        self.lmbda = lmbda
        self.n_epochs = n_epochs

    def compute_coef_(self, svc):
        return np.dot(self.X.T, svc.dual_coef_.reshape(-1, 1))

    def squared_hinge(self, prediction):
        loss = np.maximum(np.ones(len(prediction)) - prediction, 0) ** 2
        return np.sum(loss)

    def squared_hinge_conjugate(self, dual_variable):
        dual_loss = dual_variable + (1 / 4) * (dual_variable ** 2)
        return np.sum(dual_loss)

    def compute_primal_objective(self, svc):
        #coef = self.compute_coef_(svc) if one does not have the primal coef
        coef = svc.coef_.reshape(-1,1)
        prediction = np.dot(self.X, coef).flatten() * self.y
        loss = self.squared_hinge(prediction)
        return (1 / self.X.shape[0]) * loss + (1 / (2 * svc.C)) * np.linalg.norm(coef)

    def compute_dual_objective(self, svc):
        dual_prediction = np.dot(self.X.T, svc.dual_coef_.reshape(-1, 1))
        dual_loss = - self.squared_hinge_conjugate(svc.dual_coef_.reshape(-1, 1))
        return (1 / self.X.shape[0]) * dual_loss - ( 2 * svc.C / (self.X.shape[0] ** 2) ) * np.linalg.norm(dual_prediction)
    
    def get_duality_gap(self, svc):
        self.duality_gap = self.compute_primal_objective(svc) - self.compute_dual_objective(svc)
        return self.duality_gap
      
    def screen(self, X, y):
        # tol must be small enough so that max_iter is attained
        self.X = X
        self.y = y
        svc = LinearSVC_l(C=1/self.lmbda, loss='squared_hinge', max_iter=self.n_epochs, tol=0.00001).fit(self.X, self.y)
        squared_radius = 2 * self.get_duality_gap(svc) / self.lmbda
        A = squared_radius * np.identity(X.shape[1])
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
    #svc = LinearSVC_l(C=1/lmbda, loss='squared_hinge', max_iter=100, tol=0.0001).fit(X_train, y_train)
    screener = DualityGapScreener(lmbda=0.001, n_epochs=10)
    print('SCORES', screener.screen(X_train, y_train))
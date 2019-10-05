from scipy.optimize import fmin_l_bfgs_b
import numpy as np
from screening.tools import scoring_classif
from sklearn.model_selection import train_test_split
from screening.loaders import load_experiment
from screening.screentools import (
    compute_loss,
    compute_subgradient
)
    

class SafeLogistic:


    def __init__(self, mu, lmbda, max_iter, penalty='l2'):
        #beware the change of variable for mu w.r.t the formula in the paper
        self.mu = mu
        self.lmbda = lmbda
        self.max_iter = max_iter
        self.penalty = penalty


    def penalized_safe_logistic(self, x, D, y):
        return compute_loss(x, D, y, loss='safe_logistic', penalty=self.penalty, lmbda=self.lmbda, mu=self.mu)


    def penalized_safe_logistic_gradient(self, x, D, y):
        return compute_subgradient(x, D, y, lmbda=self.lmbda, mu=self.mu, loss='safe_logistic', penalty=self.penalty, intercept=False)
    
    
    def fit(self, D, y):
        x_0 = np.random.rand(D.shape[1])
        print('Fitting...')
        coef, _, _ = fmin_l_bfgs_b(func=self.penalized_safe_logistic, x0=x_0, 
                            fprime=self.penalized_safe_logistic_gradient, 
                            args=(D, y), maxiter=self.max_iter)
        self.coef_ = np.array(coef).reshape(1,-1)
        print('...fit ok!')
        return self
    
    
    def predict(self, D):
        return np.sign(np.dot(D, self.coef_[0]))
    
    
if __name__ == "__main__":
    #we check that this penalty does well on MNIST

    X, y = load_experiment(dataset='mnist', synth_params=None, size=60000, redundant=0, 
                            noise=None, classification=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    for mu in [1.0]:
        for lmbda in [0.0001]:
            safelog = SafeLogistic(mu=mu, lmbda=lmbda, penalty='l1', max_iter=10000)
            safelog.fit(X_train, y_train)
            print('LMBDA:', lmbda, 'MU :', mu, scoring_classif(safelog, X_test, y_test))
    


    
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
from screening.tools import scoring_classif
from sklearn.model_selection import train_test_split
from screening.loaders import load_experiment

class SafeLogistic:


    def __init__(self, mu, lmbda, max_iter):
        self.mu = 1 - mu
        self.lmbda = lmbda
        self.max_iter = max_iter


    def penalized_safe_logistic(self, x, D, y):
        u = y * D.dot(x)
        output = np.zeros(D.shape[0])
        for i in range(D.shape[0]):
            if u[i] < 1 - self.mu:
                output[i] = np.exp(u[i] + self.mu - 1) - u[i] - self.mu
        return np.dot(output, np.ones(D.shape[0])) / D.shape[0] + self.lmbda * np.linalg.norm(x)


    # the gradient function has to support arrays
    def penalized_safe_logistic_gradient(self, x, D, y):
        u = y * D.dot(x)
        output = np.zeros(D.shape[1])
        for i in range(D.shape[0]):
            if u[i] < 1 - self.mu:
                temp = np.exp(u[i] + self.mu - 1) - 1
                output += temp * D[i] * y[i]
        return output / D.shape[0] + self.lmbda * x
    
    
    def fit(self, D, y):
        x_0 = np.random.rand(D.shape[1])
        coef, _, _ = fmin_l_bfgs_b(func=self.penalized_safe_logistic, x0=x_0, 
                            fprime=self.penalized_safe_logistic_gradient, 
                            args=(D, y), maxiter=self.max_iter)
        self.coef_ = np.array(coef).reshape(1,-1)
        return self
    
    
    def predict(self, D):
        return np.sign(np.dot(D, self.coef_[0]))
    
    
if __name__ == "__main__":
    #we test that this penalty does well on MNIST

    X, y = load_experiment(dataset='mnist', synth_params=None, size=60000, redundant=0, 
                            noise=None, classification=True, path='./datasets/')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    safelog = SafeLogistic(mu=0, lmbda=0.0, max_iter=10000)
    safelog.fit(X_train, y_train)
    print(scoring_classif(safelog, X_test, y_test))
    


    
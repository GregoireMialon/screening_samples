import numpy as np
from sklearn.model_selection import train_test_split
from screening.loaders import load_experiment
from screening.screentools import (
    compute_loss,
    compute_subgradient
)
from arsenic import BinaryClassifier
import random
    

class SafeLogistic:


    def __init__(self, lmbda, max_iter=1000, penalty='l2'):
        #beware the change of variable for mu w.r.t the formula in the paper
        self.lmbda = lmbda
        self.max_iter = max_iter
        self.penalty = penalty


    def fit(self, D, y):
        safelog = BinaryClassifier(loss='safe-logistic', penalty='l1')
        safelog.fit(D, y, lambd=self.lmbda, solver='qning-svrg', nepochs=self.max_iter, 
                    verbose=True)
        self.coef = safelog.w
        return self
    
    
    def predict(self, D):
        return np.sign(D.dot(self.coef))


    def score(self, D, y):
        outputs = self.predict(D) * y
        outputs_ = [1 if output > 0 else 0 for output in outputs]
        return np.sum(outputs_) / D.shape[0]

    
    
if __name__ == "__main__":
    #unit test
    print('START')
    X, y = load_experiment(dataset='rcv1', synth_params=None, size=100000, redundant=0, 
                            noise=None, classification=True)

    random.seed(0)
    np.random.seed(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    prop = np.unique(y_test, return_counts=True)[1]
    print('BASELINE : ', 1 - prop[1] / prop[0])
    safelog = SafeLogistic(lmbda=0.00001, penalty='l1').fit(X_train, y_train)
    print(safelog.score(X_test, y_test))
    


    
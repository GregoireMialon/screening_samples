from screening.safelog import SafeLogistic
from sklearn.svm import LinearSVC
from sklearn.linear_model import Lasso, LogisticRegression
from cyanure import BinaryClassifier


def fit_estimator(X, y, loss, penalty, mu, lmbda, intercept, max_iter=10000, ars=False):
    if loss == 'truncated_squared' and penalty == 'l1':
        estimator = Lasso(alpha=lmbda, fit_intercept=intercept, 
                        max_iter=max_iter).fit(X, y)
    elif loss == 'squared' and penalty == 'l1':
        estimator = Lasso(alpha=lmbda, fit_intercept=intercept, 
                        max_iter=max_iter).fit(X, y)
    elif loss == 'hinge' and penalty == 'l2':
        estimator = LinearSVC(C= 1 / lmbda, loss=loss, penalty=penalty, fit_intercept=intercept, 
                        max_iter=max_iter).fit(X, y)
    elif loss == 'squared_hinge' and not(ars):
        estimator = LinearSVC(C= 1 / lmbda, loss=loss, dual=False, penalty=penalty, fit_intercept=intercept, 
                        max_iter=1000).fit(X, y) 
    elif loss == 'squared_hinge' and ars:
        estimator = BinaryClassifier(loss='sqhinge', penalty=penalty, fit_intercept=intercept)
        estimator.fit(X, y, lambd=lmbda, solver='catalyst-miso', nepochs=max_iter, verbose=False)            
    elif loss == 'safe_logistic':
        estimator = SafeLogistic(lmbda=lmbda, penalty=penalty, max_iter=max_iter).fit(X, y)
    elif loss == 'logistic':
        estimator = LogisticRegression(C=1/lmbda, penalty=penalty, fit_intercept=intercept).fit(X, y)            
    else:
    	print('ERROR, you picked a combination which is not implemented.')
    return estimator

if __name__ == "__main__":
    #Test the estimators
    import numpy as np
    from sklearn.model_selection import train_test_split
    from utils.loaders import load_experiment
    import random

    X, y = load_experiment(dataset='cifar10_kernel', synth_params=None, size=1000, redundant=0, 
                            noise=None, classification=True)
    #random.seed(0)
    #np.random.seed(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    prop = np.unique(y_test, return_counts=True)[1]
    print('BASELINE : ', 1 - prop[1] / prop[0])

    estimator = fit_estimator(X_train, y_train, loss='safe_logistic', penalty='l2', mu=0, lmbda=0, 
                                intercept=False)
    print(estimator.score(X_test, y_test))
    print(estimator.predict(X_test))

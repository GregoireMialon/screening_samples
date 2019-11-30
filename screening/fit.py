from screening.safelog import SafeLogistic
from sklearn.svm import LinearSVC
from sklearn.linear_model import Lasso, LogisticRegression
from arsenic import BinaryClassifier


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
from screening.safelog import SafeLogistic
from sklearn.svm import LinearSVC
from sklearn.linear_model import Lasso, LogisticRegression


def fit_estimator(X, y, loss, penalty, mu, lmbda, intercept, max_iter=10000):
    if loss == 'truncated_squared' and penalty == 'l1':
        estimator = Lasso(alpha=lmbda, fit_intercept=intercept, 
                        max_iter=max_iter).fit(X, y)
    elif loss == 'squared' and penalty == 'l1':
        estimator = Lasso(alpha=lmbda, fit_intercept=intercept, 
                        max_iter=max_iter).fit(X, y)
    elif loss == 'hinge' and penalty == 'l2':
        estimator = LinearSVC(C= 1 / lmbda, loss=loss, penalty=penalty, fit_intercept=intercept, 
                        max_iter=max_iter).fit(X, y)
    elif loss == 'squared_hinge':
        estimator = LinearSVC(C= 1 / lmbda, loss=loss, dual=False, penalty=penalty, fit_intercept=intercept, 
                        max_iter=1000).fit(X, y) 
    elif loss == 'safe_logistic':
        estimator = SafeLogistic(mu=mu, lmbda=lmbda, penalty=penalty, max_iter=max_iter).fit(X, y)
    elif loss == 'logistic':
        estimator = LogisticRegression(C=1/lmbda, penalty=penalty, fit_intercept=intercept).fit(X, y)            
    else:
    	print('ERROR, you picked a combination which is not implemented.')
    return estimator
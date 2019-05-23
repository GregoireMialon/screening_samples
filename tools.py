from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from random import randint
import random
import numpy as np
from scipy.sparse import rand
import matplotlib.pyplot as plt

def make_data(n, p, sparsity, noise=True):
    X = np.zeros((n,p))
    y = np.zeros(n)
    true_params = rand(p, 1, density = sparsity).A.ravel()
    #print(np.linalg.norm(true_params, ord=1))
    sparse_ones = np.zeros(p)
    for i in range(p):
        if true_params[i] != 0:
            sparse_ones[i] = 1
    true_params = 2 * true_params - sparse_ones

    noise_norm = 0

    for i in range(n):
        # sample x in [-1,1]^p box
        x = 2 * np.random.rand(p) - np.ones(p) 
        X[i,:] = x
        y[i] = np.dot(true_params,x)
        if noise:
            w = np.random.randn(1) / 10
            y[i] += w
            noise_norm += (np.linalg.norm(w)) ** 2

    return X, y, true_params, np.sqrt(noise_norm)

def make_redundant_data(X, y, nb_points_to_add, noise=0):
    X_redundant = X
    y_redundant = y
    compt = 0
    while compt< nb_points_to_add:
        compt+=1
        random.seed(compt)
        np.random.seed(compt)
        convex_comb = np.random.random(X.shape[0])
        convex_comb /= convex_comb.sum()
        X_redundant_new = (X.T).dot(convex_comb).reshape(-1, X.shape[1])
        y_redundant_new = y.dot(convex_comb)
        if noise != 0:
            y_redundant_new += np.random.randn(1) * noise
        X_redundant = np.concatenate((X_redundant, X_redundant_new), axis=0)
        y_redundant = np.append(y_redundant, y_redundant_new)
    return X_redundant, y_redundant

def make_redundant_data_classification(X, y, nb_points_to_add):
    X_redundant = X
    y_redundant = y
    compt = 0
    X_pos = X[np.where(y == 1)]
    X_neg = X[np.where(y == 1)]

    while compt < nb_points_to_add:
        compt+=1
        random.seed(compt)
        np.random.seed(compt)

        convex_comb = np.random.random(X_pos.shape[0])
        convex_comb /= convex_comb.sum()
        X_redundant_new = (X_pos.T).dot(convex_comb).reshape(-1, X.shape[1])
        X_redundant = np.concatenate((X_redundant, X_redundant_new), axis=0)
        y_redundant = np.append(y_redundant, 1)

        convex_comb = np.random.random(X_neg.shape[0])
        convex_comb /= convex_comb.sum()
        X_redundant_new = (X_neg.T).dot(convex_comb).reshape(-1, X.shape[1])
        X_redundant = np.concatenate((X_redundant, X_redundant_new), axis=0)
        y_redundant = np.append(y_redundant, -1)

    return X_redundant, y_redundant

def balanced_subsample(x,y,subsample_size=1):
    class_xs = []
    min_elems = None
    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]
    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)
    xs = []
    ys = []
    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)
        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)
        xs.append(x_)
        ys.append(y_)
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    return xs,ys


def find_best_lasso(X, y):
    param_grid = {'alpha':[0.001,0.01,0.1,1,10], 'fit_intercept':[True, False] }
    clf = GridSearchCV(estimator = Lasso(max_iter=10000), param_grid = param_grid)
    clf.fit(X,y)
    best_lasso = clf.best_estimator_
    return best_lasso, clf.best_score_

def find_best_svm(X, y):
    param_grid = {'C':[0.1, 1, 10, 100, 1000], 'fit_intercept':[True, False] }
    clf = GridSearchCV(estimator = LinearSVC(max_iter=10000), param_grid = param_grid)
    clf.fit(X,y)
    best_svm = clf.best_estimator_
    return best_svm, clf.best_score_

def random_screening(X, y, nb_points_to_keep):
    X_screened = X
    y_screened = y
    while X_screened.shape[0] > nb_points_to_keep:
        i = randint(0,X_screened.shape[0]-1)
        X_screened = np.delete(X_screened, i, 0)
        y_screened = np.delete(y_screened, i, 0)
    return X_screened, y_screened

def k_medioids(X, y, nb_points_to_keep):
    print('TODO')
    return

def screen(X, y, scores, nb_to_delete):
    X_screened = X
    y_screened = y
    idx_to_delete = np.argsort(scores)[0:nb_to_delete]
    X_screened = np.delete(X, idx_to_delete, 0)
    y_screened = np.delete(y, idx_to_delete, 0)
    return X_screened, y_screened

def get_idx_not_safe(scores, mu):
    scores = np.array(scores)
    idx = np.where(scores < mu)[0]
    idx_not_safe = len(idx)
    return idx_not_safe

def scoring_classif(estimator, X, y):
    score = 0
    for i in range(len(y)):
        if estimator.predict(X[i].reshape(1, -1))* y[i] > 0:
            score += 1
    return score / len(y)

def scoring_interval_regression(y, y_predict, y_predict_screened, mu):
    score = 0
    score_screened = 0
    for i in range(len(y)):
        if np.abs(y_predict[i] - y[i]) < mu:
            score += 1
            if np.abs(y_predict_screened[i] - y[i]) < mu:
                score_screened += 1
    return score_screened / score
    
def plot_experiment(data, title, dataset, dataset_size):

    nb_to_del_table = data[0] / dataset_size
    scores_regular_all = np.array(data[1])
    scores_screened_all = np.array(data[2])
    scores_r_all = np.array(data[3])
    idx_not_safe = data[4]

    scores_regular_mean = np.mean(scores_regular_all, 0)
    scores_screened_mean = np.mean(scores_screened_all, 0)
    scores_r_mean = np.mean(scores_r_all, 0)

    scores_regular_var = np.sqrt(np.var(scores_regular_all, 0))
    scores_screened_var = np.sqrt(np.var(scores_screened_all, 0))
    scores_r_var = np.sqrt(np.var(scores_r_all, 0))

    fig, ax1 = plt.subplots()
    ax1.errorbar(nb_to_del_table, scores_regular_mean, yerr=scores_regular_var, label='Whole dataset')
    ax1.errorbar(nb_to_del_table, scores_screened_mean, yerr=scores_screened_var, label='Screened dataset')
    ax1.errorbar(nb_to_del_table, scores_r_mean, yerr=scores_r_var, label='Random subset')
    ax1.set_xlabel('Fraction of points deleted')
    ax1.set_ylabel('Accuracy on test set')
    ax1.tick_params('y')
    plt.axvline(idx_not_safe / dataset_size, color='red')
    fig.tight_layout()
    ax1.legend()
    plt.title(dataset + ', ' + title)
    plt.show()

    return


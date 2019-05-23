import numpy as np
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from tools import *
from screening import *
from sklearn.datasets import load_diabetes, load_boston, fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import random
import scipy.io
import plasp #contains the solver for interval regression

def load_leukemia(cluster):
    if cluster:
        data = pd.read_csv('/sequoia/data1/gmialon/screening/datasets/leukemia_big.csv')
    else:
        data = pd.read_csv('/nas/home2/g/gmialon/research/datasets/leukemia_big.csv')
    X = np.transpose(data.values)
    y_ = data.columns.values
    y = np.ones(len(y_))
    for i in range(len(y_)):
        if 'AML' in y_[i]:
            y[i] = -1
    return X, y

def load_20newsgroups():
    cats = ['comp.graphics','talk.religion.misc']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=cats)
    X = newsgroups_train.data
    y = newsgroups_train.target
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2)
    X = vectorizer.fit_transform(X).toarray()
    y = 2 * y - np.ones(len(y))
    return X, y

def load_mnist(pb=1):
    mat = scipy.io.loadmat('/sequoia/data1/gmialon/screening/datasets/ckn_mnist.mat')
    X = mat['psiTr'].T
    print(X.shape)
    y = mat['Ytr']
    y = np.array(y, dtype=int).reshape(y.shape[0])
    for i in range(len(y)):
        if y[i] != 9:
            y[i] = - 1
    X, y = balanced_subsample(X, y)
    return X, y

#@profile

def fit_estimator(X, y, loss, penalty, lmbda, intercept, max_iter=5000):
    if loss == 'truncated_squared' and penalty == 'l1':
        estimator = Lasso(alpha=lmbda, fit_intercept=intercept, 
                        max_iter=max_iter).fit(X, y)
    elif loss == 'hinge' and penalty == 'l2':
        estimator = LinearSVC(C= 1 / lmbda, loss=loss, penalty=penalty, fit_intercept=intercept, 
                        max_iter=max_iter).fit(X, y)
    return estimator

def experiment(dataset, redundant, noise, nb_delete_steps, lmbda, mu, classification, loss, penalty, intercept, classif_score,
                    n_ellipsoid_steps, better_init, better_radius, reverse, cut, nb_exp, nb_test, cluster):


    exp_title = '/lmbda_{}_n_ellipsoid_{}_intercept_{}_mu_{}_redundant_{}_noise_{}_loss_{}_penalty_{}_better_init_{}_better_radius_{}'.format(lmbda, n_ellipsoid_steps, 
        intercept, mu, redundant, noise, loss, penalty, better_init, better_radius)

    if dataset == 'leukemia':
        X, y = load_leukemia(cluster)
    elif dataset == 'boston':
        boston = load_boston(return_X_y=True)
        X = boston[0]
        y = boston[1]
    elif dataset == 'diabetes':
        diabetes = load_diabetes(return_X_y=True)
        X = diabetes[0]
        y = diabetes[1]
    elif dataset == '20newsgroups':
        X, y = load_20newsgroups()
    elif dataset == 'mnist':
        X, y = load_mnist()
    elif dataset == 'synthetic':
        X, y, _, _ = make_data(100, 500, 10 / 500)
        
    if redundant != None and not(classification):
        dataset+= '_redundant'
        X, y = make_redundant_data(X, y, int(redundant), noise)
    elif redundant != None and classification:
        dataset+= '_redundant'
        X, y = make_redundant_data_classification(X, y, int(redundant))

    scores_regular_all = []
    scores_screened_all = []
    scores_r_all = []

    compt_exp = 0
    idx_not_safe_all = 0

    while compt_exp < nb_exp:
        random.seed(compt_exp)
        np.random.seed(compt_exp)
        compt_exp += 1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        if intercept:
            z_init = np.zeros(X_train.shape[1] + 1)
            r_init = np.sqrt(X_train.shape[1] + 1)
        else:
            z_init = np.zeros(X_train.shape[1])
            r_init = np.sqrt(X_train.shape[1])
        if better_init != None:
            est = fit_estimator(X_train, y_train, loss, penalty, lmbda, intercept, max_iter=int(better_init))
            z_init = est.coef_
            if intercept:
                z_init = np.append(z_init, est.intercept_)
        if better_radius != None:
            r_init = float(better_radius)
        z, scaling, L, I_k_vec = iterate_ellipsoids_accelerated_(X_train, y_train, z_init,
                                r_init, lmbda, mu, loss, penalty, n_ellipsoid_steps, intercept)
        scores = rank_dataset_accelerated(X_train, y_train, z, scaling, L, I_k_vec,
                                             lmbda, mu, classification, loss, penalty, intercept, reverse,cut)
        truc = np.argsort(scores)
        #print(scores[0:20], y_train[0:20])
        #print(y_train[truc[0:431]])
        #print(y_train[truc[450:]])
        idx_not_safe = get_idx_not_safe(scores, mu)
        idx_not_safe_all += idx_not_safe
        scores_regular = []
        scores_screened = []
        scores_r= []
        
        nb_to_del_table = np.linspace(1, X_train.shape[0] - 10, nb_delete_steps, dtype='int') #delete that if small dataset

        X_r = X_train
        y_r = y_train
         
        for nb_to_delete in nb_to_del_table:
            score_regular = 0
            score_screened = 0
            score_r = 0
            compt = 0
            X_screened, y_screened = screen(X_train, y_train, scores, nb_to_delete)
            X_r, y_r = random_screening(X_r, y_r, X_train.shape[0] - nb_to_delete)
            print(X_train.shape, X_screened.shape, X_r.shape)
            while compt < nb_test:
                compt += 1
                estimator_regular = fit_estimator(X_train, y_train, loss, penalty, lmbda, intercept)
                estimator_screened = fit_estimator(X_screened, y_screened, loss, penalty, lmbda, intercept)
                estimator_r = fit_estimator(X_r, y_r, loss, penalty, lmbda, intercept)

                if classif_score:
                    score_regular += scoring_classif(estimator_regular, X_test, y_test)
                    score_screened += scoring_classif(estimator_screened, X_test, y_test)
                    score_r += scoring_classif(estimator_r, X_test, y_test)
                else:
                    score_regular += estimator_regular.score(X_test, y_test)
                    score_screened += estimator_screened.score(X_test, y_test)
                    score_r += estimator_r.score(X_test, y_test)
            scores_regular.append(score_regular / nb_test)
            scores_screened.append(score_screened / nb_test)
            scores_r.append(score_r / nb_test)

        scores_regular_all.append(scores_regular)
        scores_screened_all.append(scores_screened)
        scores_r_all.append(scores_r)

    print('Number of datapoints we can screen:', idx_not_safe / nb_exp)

    data = (nb_to_del_table, scores_regular_all, scores_screened_all, scores_r_all, 
        np.floor(idx_not_safe / nb_exp))
    
    if cluster:
        np.save('/sequoia/data1/gmialon/screening/results/' + dataset + exp_title, data)
    else:
        np.save('/nas/home2/g/gmialon/research/safe_datapoints/results/' + dataset + exp_title, data)
    
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='leukemia', help='dataset used for the experiment')
    parser.add_argument('--redundant', default=None)
    parser.add_argument('--noise', default=0, type=float)
    parser.add_argument('--nb_delete_steps', default=20, type=int)
    parser.add_argument('--lmbda', default=0.01, type=float)
    parser.add_argument('--mu', default=1, type=float)
    parser.add_argument('--classification', action='store_true')
    parser.add_argument('--loss', default='truncated_squared')
    parser.add_argument('--penalty', default='l1', help='determines the penalty in the loss of the problem to screen')
    parser.add_argument('--intercept', action='store_true')
    parser.add_argument('--classif_score', action='store_true', help='determines the score that is used')
    parser.add_argument('--n_ellipsoid_steps', default=1000, type=int)
    parser.add_argument('--better_init', default=None)
    parser.add_argument('--better_radius', default=None)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--cut', action='store_true')
    parser.add_argument('--nb_exp', default=5, type=int)
    parser.add_argument('--nb_test', default=3, type=int)
    parser.add_argument('--cluster', action='store_true', help='save on data1')
    args = parser.parse_args()

    print('START')

    experiment(args.dataset, args.redundant, args.noise, args.nb_delete_steps, args.lmbda, args.mu, 
        args.classification, args.loss, args.penalty, args.intercept, args.classif_score, 
        args.n_ellipsoid_steps, args.better_init, args.better_radius, args.cut, args.reverse,
         args.nb_exp, args.nb_test, args.cluster)
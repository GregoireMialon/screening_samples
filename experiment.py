#TODO ARGPARSE

import numpy as np
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from tools import *
from screening import *
from sklearn.datasets import load_diabetes, load_boston
from sklearn.feature_extraction.text import TfidfVectorizer
import random

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

def experiment(dataset, nb_delete_steps, lmbda, mu, penalty, intercept, classification,
                    n_ellipsoid_steps, nb_exp, nb_test, cluster):

    exp_title = '/lmbda_{}_n_ellipsoid_{}_intercept_{}_mu_{}_classification_{}_penalty_{}'.format(lmbda, n_ellipsoid_steps, 
        intercept, mu, classification, penalty)

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

    #if not(intercept):
        #X = np.concatenate((X, np.ones(X.shape[0]).reshape(1,-1).T), axis=1)

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
        z, scaling, L, I_k_vec = iterate_ellipsoids_accelerated_(X_train, y_train, z_init,
                                r_init, lmbda, mu, penalty, n_ellipsoid_steps, intercept)
        scores = rank_dataset_accelerated(X_train, y_train, z, scaling, L, I_k_vec,
                                             lmbda, mu, penalty, intercept)
        idx_not_safe = get_idx_not_safe(scores, mu)
        idx_not_safe_all += idx_not_safe
        scores_regular = []
        scores_screened = []
        scores_r= []
        
        nb_to_del_table = np.linspace(1, X_train.shape[0] - 10, nb_delete_steps, dtype='int')

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
                lasso_regular = Lasso(alpha=lmbda, fit_intercept=intercept, 
                    max_iter=10000).fit(X_train,y_train)
                lasso_screened = Lasso(alpha=lmbda, fit_intercept=intercept, 
                    max_iter=10000).fit(X_screened, y_screened)
                lasso_r = Lasso(alpha=lmbda,fit_intercept=intercept, 
                    max_iter=10000).fit(X_r, y_r)

                if classification:
                    score_regular += scoring_classif(lasso_regular, X_test, y_test)
                    score_screened += scoring_classif(lasso_screened, X_test, y_test)
                    score_r += scoring_classif(lasso_r, X_test, y_test)
                else:
                    score_regular += lasso_regular.score(X_test, y_test)
                    score_screened += lasso_screened.score(X_test, y_test)
                    score_r += lasso_r.score(X_test, y_test)
            
            scores_regular.append(score_regular / nb_test)
            scores_screened.append(score_screened / nb_test)
            scores_r.append(score_r / nb_test)

        scores_regular_all.append(scores_regular)
        scores_screened_all.append(scores_screened)
        scores_r_all.append(scores_r)

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
    parser.add_argument('--nb_delete_steps', default=20, type=int)
    parser.add_argument('--lmbda', default=0.01, type=float)
    parser.add_argument('--mu', default=1, type=float)
    parser.add_argument('--penalty', default='l1', help='determines the penalty in the loss of the problem to screen')
    parser.add_argument('--intercept', action='store_true', help=
                            'if not intercept, a dimension is added to the dataset')
    parser.add_argument('--classification', action='store_true', help='determines the score that is used')
    parser.add_argument('--n_ellipsoid_steps', default=100, type=int)
    parser.add_argument('--nb_exp', default=5, type=int)
    parser.add_argument('--nb_test', default=3, type=int)
    parser.add_argument('--cluster', action='store_true', help='save on data1')
    args = parser.parse_args()

    experiment(args.dataset, args.nb_delete_steps, args.lmbda, args.mu, args.penalty, args.intercept, 
        args.classification, args.n_ellipsoid_steps, args.nb_exp, args.nb_test, args.cluster)
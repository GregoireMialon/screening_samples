#TODO ARGPARSE

import numpy as np
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from tools import *
from screening import *

def load_leukemia():
    data = pd.read_csv('/nas/home2/g/gmialon/research/datasets/leukemia_big.csv')
    X = np.transpose(data.values)
    y_ = data.columns.values
    y = np.ones(len(y_))
    for i in range(len(y_)):
        if 'AML' in y_[i]:
            y[i] = -1
    return X, y

def experiment(dataset, nb_delete_steps, lmbda, mu, penalty, intercept, classification,
                    n_ellipsoid_steps, nb_test, cluster):

    if dataset == 'leukemia':
        X, y = load_leukemia()
    
    if not(intercept):
        X = np.concatenate((X, np.ones(X.shape[0]).reshape(1,-1).T), axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
    z_init = np.zeros(X_train.shape[1])
    A_init = np.sqrt(X_train.shape[1]) * np.identity(X_train.shape[1])
    z, scaling, L, I_k_vec = iterate_ellipsoids_accelerated_(X_train, y_train, z_init,
                                              np.sqrt(X_train.shape[1]), 
                                              lmbda, mu, penalty, n_ellipsoid_steps)
    scores = rank_dataset_accelerated(X_train, y_train, z, scaling, L, I_k_vec,
                                         lmbda, mu, penalty)
    
    scores_regular = []
    scores_screened = []
    scores_r= []
    
    nb_to_del_table = np.linspace(1, X_train.shape[0] - 10, nb_delete_steps, dtype='int')
     
    for nb_to_delete in nb_to_del_table:
        score_regular = 0
        score_screened = 0
        score_r = 0
        compt = 0
        X_screened, y_screened = screen(X_train, y_train, scores, nb_to_delete)
        print(X_train.shape, X_screened.shape)
        while compt < nb_test:
            compt += 1
            X_r, y_r = random_screening(X_train, y_train, X_train.shape[0] - nb_to_delete)
            if compt == 1:
                print(X_r.shape)
            lasso_regular, _ = find_best_lasso(X_train,y_train, intercept=intercept)
            lasso_screened, _ = find_best_lasso(X_screened, y_screened, intercept=intercept)
            lasso_r, _ = find_best_lasso(X_r, y_r,  intercept=intercept)

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

    data = (nb_to_del_table, scores_regular, scores_screened, scores_r)
    exp_title = '/lmbda_' + str(lmbda) + '_n_ellipsoid_' + str(n_ellipsoid_steps) + '_intercept_' + str(intercept)
    if cluster:
        np.save()
    else:
        np.save('/nas/home2/g/gmialon/research/safe_datapoints/results/' + dataset + exp_title, data)
    
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='dataset used for the experiment')
    parser.add_argument('--nb_delete_steps', default=20, type=int)
    parser.add_argument('--lmbda', default=0.01, type=float)
    parser.add_argument('--mu', default=1, type=float)
    parser.add_argument('--penalty', help='determines the penalty in the loss of the problem to screen')
    parser.add_argument('--intercept', action='store_true', help=
                            'if not intercept, a dimension is added to the dataset')
    parser.add_argument('--classification', action='store_true', help='determines the score that is used')
    parser.add_argument('--n_ellipsoid_steps', default=100, type=int)
    parser.add_argument('--nb_test', default=5, type=int)
    parser.add_argument('--cluster', action='store_true', help='save on data1')
    args = parser.parse_args()

    experiment(args.dataset, args.nb_delete_steps, args.lmbda, args.mu, args.penalty, args.intercept, 
        args.classification, args.n_ellipsoid_steps, args.nb_test, args.cluster)
import numpy as np
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from tools import (
    make_data, make_redundant_data, make_redundant_data_classification, 
    balanced_subsample, random_screening, check_dataset, screen, get_idx_not_safe, 
    scoring_classif, plot_experiment
)
from screening import (
    iterate_ellipsoids_accelerated_,
    rank_dataset_accelerated
)
from sklearn.datasets import load_diabetes, load_boston, fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import Lasso
from sklearn.utils import shuffle
import random
import scipy.io
import pickle
import os

def load_leukemia(path):
    data = pd.read_csv(path + 'leukemia_big.csv')
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

def load_mnist(path, pb=1):
    mat = scipy.io.loadmat(path + 'ckn_mnist.mat')
    X = mat['psiTr'].T
    print(X.shape)
    y = mat['Ytr']
    y = np.array(y, dtype=int).reshape(y.shape[0])
    for i in range(len(y)):
        if y[i] != 9:
            y[i] = - 1
    X, y = balanced_subsample(X, y)
    return X, y

def load_higgs(path):
    dir_higgs = path + 'higgs.small.p'
    with open(dir_higgs, 'rb') as handle:
        data_higgs = pickle.load(handle)
    X = data_higgs[0]
    y =data_higgs[1]
    return X, y

def load_experiment(dataset, size, redundant, noise, classification, path):
    if dataset == 'leukemia':
        X, y = load_leukemia(path)
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
        X, y = load_mnist(path)
    elif dataset == 'higgs':
        X, y = load_higgs(path)
    elif dataset == 'synthetic':
        X, y, true_params, _ = make_data(100, 2, 0.5)
        print('TRUE SYNTHETIC PARAMETERS', true_params)
    if redundant != 0 and not(classification):
        dataset+= '_redundant'
        X, y = make_redundant_data(X, y, int(redundant), noise)
    elif redundant != 0 and classification:
        dataset+= '_redundant'
        X, y = make_redundant_data_classification(X, y, int(redundant))

    if size != None:
        X = X[:int(size)]
        y = y[:int(size)]
    return X, y

def fit_estimator(X, y, loss, penalty, lmbda, intercept, max_iter=10000):
    if loss == 'truncated_squared' and penalty == 'l1':
        estimator = Lasso(alpha=lmbda, fit_intercept=intercept, 
                        max_iter=max_iter).fit(X, y)
    elif loss == 'hinge' and penalty == 'l2':
        estimator = LinearSVC(C= 1 / lmbda, loss=loss, penalty=penalty, fit_intercept=intercept, 
                        max_iter=max_iter).fit(X, y)
    else:
    	print('ERROR, you can only combine squared loss with l1 and hinge with l2 for now.')
    return estimator

def experiment_get_ellipsoid(X, y, intercept, better_init, better_radius, loss, penalty, 
                                lmbda, classification, mu, n_ellipsoid_steps):
    if intercept:
        z_init = np.zeros(X.shape[1] + 1)
        r_init = X.shape[1] + 1
    else:
        z_init = np.zeros(X.shape[1])
        r_init = X.shape[1]
    if better_init != 0:
        est = fit_estimator(X, y, loss, penalty, lmbda, intercept, max_iter=better_init)
        if classification:
            z_init = est.coef_[0]
            if intercept:
                z_init = np.append(z_init, est.intercept_)
        else:
            z_init = est.coef_
            if intercept:
                z_init = np.append(z_init, est.intercept_)
    if better_radius != 0:
        r_init = float(better_radius)                            
    z, scaling, L, I_k_vec, g = iterate_ellipsoids_accelerated_(X, y, z_init,
                                r_init, lmbda, mu, loss, penalty, n_ellipsoid_steps, intercept)
    return z, scaling, L, I_k_vec, g

#@profile
def experiment(path, dataset, size, redundant, noise, nb_delete_steps, lmbda, mu, classification, 
                loss, penalty, intercept, classif_score, n_ellipsoid_steps, better_init, 
                better_radius, cut, get_ell_from_subset, nb_exp, nb_test, plot, zoom):  
    exp_title = 'X_size_{}_sub_ell_{}_lmbda_{}_n_ellipsoid_{}_intercept_{}_mu_{}_redundant_{}_noise_{}_better_init_{}_better_radius_{}'.format(size, 
        get_ell_from_subset, lmbda, n_ellipsoid_steps, intercept, mu, redundant, noise, better_init, 
        better_radius)

    X, y = load_experiment(dataset, size, redundant, noise, classification, path + 'datasets/')

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
        if get_ell_from_subset != 0:
            X_train_, y_train_ = balanced_subsample(X_train, y_train)
            X_train_, y_train_ = shuffle(X_train_, y_train_)
            X_train_ = X_train_[:get_ell_from_subset]
            y_train_ = y_train_[:get_ell_from_subset]
            z, scaling, L, I_k_vec, g = experiment_get_ellipsoid(X_train_, y_train_, intercept, better_init, 
                                                            better_radius, loss, penalty, lmbda, 
                                                            classification, mu, n_ellipsoid_steps)
        else:
            z, scaling, L, I_k_vec, g = experiment_get_ellipsoid(X_train, y_train, intercept, better_init, 
                                                            better_radius, loss, penalty, lmbda, 
                                                            classification, mu, n_ellipsoid_steps)
        scores = rank_dataset_accelerated(X_train, y_train, z, scaling, L, I_k_vec, g,
                                             lmbda, mu, classification, loss, penalty, intercept, cut)
        idx_not_safe = get_idx_not_safe(scores, mu)
        idx_not_safe_all += idx_not_safe
        scores_regular = []
        scores_screened = []
        scores_r= []
        
        nb_to_del_table = np.linspace(1, X_train.shape[0], nb_delete_steps, dtype='int')

        X_r = X_train
        y_r = y_train
        
        for nb_to_delete in nb_to_del_table:
            score_regular = 0
            score_screened = 0
            score_r = 0
            compt = 0
            X_screened, y_screened = screen(X_train, y_train, scores, nb_to_delete)
            X_r, y_r = random_screening(X_r, y_r, X_train.shape[0] - nb_to_delete)
            if not(check_dataset(y_r) and check_dataset(y_screened)):
                break
            print(X_train.shape, X_screened.shape, X_r.shape)
            while compt < nb_test:
                compt += 1
                estimator_regular = fit_estimator(X_train, y_train, loss, penalty, lmbda, intercept)
                estimator_screened = fit_estimator(X_screened, y_screened, loss, penalty, lmbda, 
                intercept)
                estimator_r = fit_estimator(X_r, y_r, loss, penalty, lmbda, intercept)
                if compt == 1 and compt_exp == 1:
                    #print('ESTIMATOR SOLUTION', estimator_regular.coef_)
                    pass
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

    print('Number of datapoints we can screen (if safe rules apply to the experiment):', idx_not_safe / nb_exp)

    data = (nb_to_del_table, scores_regular_all, scores_screened_all, scores_r_all, 
        np.floor(idx_not_safe / nb_exp), X_train.shape[0])
    save_dataset_folder = os.path.join(path, 'results', dataset)
    os.makedirs(save_dataset_folder, exist_ok=True)
    np.save(os.path.join(save_dataset_folder, exp_title), data)

    if plot:
        plot_experiment(data, zoom=zoom, name=None, save=False)
    
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./', type=str)
    parser.add_argument('--dataset', default='diabetes', help='dataset used for the experiment')
    parser.add_argument('--size', default=442, type=int, help='number of samples of the dataset to use')
    parser.add_argument('--redundant', default=400, type=int, help='add redundant examples to the dataset. Do not use redundant with --size')
    parser.add_argument('--noise', default=0.1, type=float, help='standard deviation of the noise to add to the redundant examples')
    parser.add_argument('--nb_delete_steps', default=20, type=int, help='at each step of the experiment, we delete size / nb_delete_steps data points')
    parser.add_argument('--lmbda', default=0.01, type=float, help='regularization parameter of the estimator')
    parser.add_argument('--mu', default=10, type=float, help='regularization parameter of the dual')
    parser.add_argument('--classification', action='store_true')
    parser.add_argument('--loss', default='truncated_squared', choices=['hinge','truncated_squared'])
    parser.add_argument('--penalty', default='l1', choices=['l1','l2'])
    parser.add_argument('--intercept', action='store_true')
    parser.add_argument('--classif_score', action='store_true', help='determines the score that is used')
    parser.add_argument('--n_ellipsoid_steps', default=1000, type=int, help='number of iterations of the ellipsoid method')
    parser.add_argument('--better_init', default=10, type=int, help='number of optimizer gradient steps to initialize the center of the ellipsoid')
    parser.add_argument('--better_radius', default=10, type=float, help='radius of the initial l2 ball')
    parser.add_argument('--cut', action='store_true', help='cut the final ellipsoid in half using a subgradient of the loss')
    parser.add_argument('--get_ell_from_subset', default=0, type=int, help='train the ellipsoid on a random subset of the dataset')
    parser.add_argument('--nb_exp', default=10, type=int)
    parser.add_argument('--nb_test', default=3, type=int)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--zoom', default=[0.2, 0.6], nargs='+', type=float, help='zoom in the final plot')
    
    args = parser.parse_args()

    print('START')

    experiment(args.path, args.dataset, args.size, args.redundant, args.noise, args.nb_delete_steps, 
        args.lmbda, args.mu, 
        args.classification, args.loss, args.penalty, args.intercept, args.classif_score, 
        args.n_ellipsoid_steps, args.better_init, args.better_radius, args.cut, 
        args.get_ell_from_subset, args.nb_exp, args.nb_test, args.plot, args.zoom)
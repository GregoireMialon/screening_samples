import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from tools import (
    make_data, make_redundant_data, make_redundant_data_classification, 
    balanced_subsample, random_screening, dataset_has_both_labels, screen, get_idx_safe, 
    scoring_classif, plot_experiment, screen_baseline_margin
)
from screening import (
    iterate_ellipsoids_accelerated,
    rank_dataset_accelerated,
    rank_dataset
)
from loaders import load_experiment
from safelog import SafeLogistic
from sklearn.svm import LinearSVC
from sklearn.linear_model import Lasso
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import random
import os

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
    elif loss == 'squared_hinge' and penalty == 'l2':
        estimator = LinearSVC(C= 1 / lmbda, loss=loss, dual=False, penalty=penalty, fit_intercept=intercept, 
                        max_iter=max_iter).fit(X, y) 
    elif loss == 'safe_logistic' and penalty == 'l2':
        estimator = SafeLogistic(mu=mu, lmbda=lmbda, max_iter=max_iter).fit(X, y)            
    else:
    	print('ERROR, you picked a combination which is not implemented.')
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
        est = fit_estimator(X, y, loss, penalty, mu, lmbda, intercept, max_iter=better_init)
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
    z, scaling, L, I_k_vec, g = iterate_ellipsoids_accelerated(X, y, z_init,
                                r_init, lmbda, mu, loss, penalty, n_ellipsoid_steps, intercept)
    return z, scaling, L, I_k_vec, g, r_init

def run_experiment(X_train, y_train, model, experiment, intercept, loss, penalty, lmbda, classification, 
                    mu, n_ellipsoid_steps, better_init, better_radius, cut, get_ell_from_subset, clip_ell):
    if experiment == 'whole':
        idx_to_del = []
    elif experiment == 'random':
        idx_to_del = []
    elif experiment == 'screen':
        scores = ellipsoid_screening(X_train, y_train, intercept, loss, penalty, lmbda, classification, mu, n_ellipsoid_steps, 
                        better_init, better_radius, cut, get_ell_from_subset, clip_ell)
        idx_to_del = [scores]
    elif experiment == 'margin':
        idx_to_del = []
    else:
        print('ERROR, experiment not implemented')
    return idx_to_del

#@profile
def experiment(path, dataset, synth_params, size, scale_data, redundant, noise, nb_delete_steps, lmbda, mu, classification, 
                loss, penalty, intercept, classif_score, n_ellipsoid_steps, better_init, 
                better_radius, cut, get_ell_from_subset, clip_ell, nb_exp, nb_test, plot, zoom, 
                dontsave):
    exp_title = 'X_size_{}_sub_ell_{}_loss_{}_lmbda_{}_n_ellipsoid_{}_intercept_{}_mu_{}_redundant_{}_noise_{}_better_init_{}_better_radius_{}_cut_ell_{}_clip_ell_{}'.format(size, 
        get_ell_from_subset, loss, lmbda, n_ellipsoid_steps, intercept, mu, redundant, noise, better_init, 
        better_radius, cut, clip_ell)
    print(exp_title)

    X, y = load_experiment(dataset, synth_params, size, redundant, noise, classification, path + 'datasets/')

    scores_regular_all = []
    scores_screened_all = []
    scores_r_all = []
    #scores_margin_all = []

    compt_exp = 0
    idx_safe_all = 0
    
    while compt_exp < nb_exp:
        random.seed(compt_exp)
        np.random.seed(compt_exp)
        compt_exp += 1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        if scale_data:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        if get_ell_from_subset != 0:
            X_train_, y_train_ = balanced_subsample(X_train, y_train)
            X_train_, y_train_ = shuffle(X_train_, y_train_)
            if get_ell_from_subset < X_train_.shape[0]:
                X_train_ = X_train_[:get_ell_from_subset]
                y_train_ = y_train_[:get_ell_from_subset]
            z, scaling, L, I_k_vec, g, r_init = experiment_get_ellipsoid(X_train_, y_train_, intercept, better_init, 
                                                            better_radius, loss, penalty, lmbda, 
                                                            classification, mu, n_ellipsoid_steps)
        else:
            z, scaling, L, I_k_vec, g, r_init = experiment_get_ellipsoid(X_train, y_train, intercept, better_init, 
                                                            better_radius, loss, penalty, lmbda, 
                                                            classification, mu, n_ellipsoid_steps)
        if clip_ell:
            I_k_vec = I_k_vec.reshape(-1,1)
            A = scaling * np.identity(X.shape[1]) - L.dot(np.multiply(I_k_vec, np.transpose(L)))
            eigenvals, eigenvect = np.linalg.eigh(A)
            eigenvals = np.clip(eigenvals, 0, r_init)
            eigenvals = eigenvals.reshape(-1,1)
            A = eigenvect.dot(np.multiply(eigenvals, np.transpose(eigenvect)))
            scores = rank_dataset(X_train, y_train, z, A, g,
                                             lmbda, mu, classification, loss, penalty, intercept, cut)
        else:                            
            scores = rank_dataset_accelerated(X_train, y_train, z, scaling, L, I_k_vec, g,
                                             lmbda, mu, classification, loss, penalty, intercept, cut)
        print('SCORES', scores)
        idx_safe = get_idx_safe(scores, mu, classification)
        #print(idx_safe)
        idx_safe_all += idx_safe
        scores_regular = []
        scores_screened = []
        scores_r = []
        #scores_margin = []
        
        nb_to_del_table = np.linspace(1, X_train.shape[0], nb_delete_steps, dtype='int')

        X_r = X_train
        y_r = y_train
        
        for nb_to_delete in nb_to_del_table:
            score_regular = 0
            score_screened = 0
            score_r = 0
            #score_margin = 0
            compt = 0
            X_screened, y_screened = screen(X_train, y_train, scores, nb_to_delete)
            X_r, y_r = random_screening(X_r, y_r, X_train.shape[0] - nb_to_delete)
            #model = fit_estimator(X_train, y_train, loss, penalty, mu, lmbda, intercept, max_iter=10)
            #X_margin, y_margin = screen_baseline_margin(X_train, y_train, model, nb_to_delete)
            if not(dataset_has_both_labels(y_r)):
                print('Warning, only one label in randomly screened dataset')
            if not(dataset_has_both_labels(y_screened)):
                print('Warning, only one label in screened dataset')
            #if not(dataset_has_both_labels(y_margin)):
                #print('Warning, only one label in margin dataset')
            if not(dataset_has_both_labels(y_r) and dataset_has_both_labels(y_screened)): # and dataset_has_both_labels(y_margin)):
                break
            print(X_train.shape, X_screened.shape, X_r.shape) #, X_margin.shape)
            while compt < nb_test:
                compt += 1
                estimator_regular = fit_estimator(X_train, y_train, loss, penalty, mu, lmbda, intercept)
                estimator_screened = fit_estimator(X_screened, y_screened, loss, penalty, mu, lmbda, 
                intercept)
                estimator_r = fit_estimator(X_r, y_r, loss, penalty, mu, lmbda, intercept)
                #estimator_margin = fit_estimator(X_margin, y_margin, loss, penalty, mu, lmbda, intercept)
                if classif_score:
                    score_regular += scoring_classif(estimator_regular, X_test, y_test)
                    score_screened += scoring_classif(estimator_screened, X_test, y_test)
                    score_r += scoring_classif(estimator_r, X_test, y_test)
                    #score_margin += scoring_classif(estimator_margin, X_test, y_test)
                else:
                    score_regular += estimator_regular.score(X_test, y_test)
                    score_screened += estimator_screened.score(X_test, y_test)
                    score_r += estimator_r.score(X_test, y_test)
                    #score_margin += estimator_margin.score(X_test, y_test)
            scores_regular.append(score_regular / nb_test)
            scores_screened.append(score_screened / nb_test)
            scores_r.append(score_r / nb_test)
            #scores_margin.append(score_margin / nb_test)

        scores_regular_all.append(scores_regular)
        scores_screened_all.append(scores_screened)
        scores_r_all.append(scores_r)
        #scores_margin_all.append(scores_margin)

    print('Number of datapoints we can screen (if safe rules apply to the experiment):', idx_safe_all / nb_exp)

    data = (nb_to_del_table, scores_regular_all, scores_screened_all, scores_r_all, 
        idx_safe_all / nb_exp, X_train.shape[0]) #, scores_margin_all)
    save_dataset_folder = os.path.join(path, 'results', dataset)
    os.makedirs(save_dataset_folder, exist_ok=True)
    if not dontsave:
        np.save(os.path.join(save_dataset_folder, exp_title), data)
        print('RESULTS SAVED!')

    if plot:
        plot_experiment(data, zoom=zoom)
    
    return

def ellipsoid_screening(X_train, y_train, intercept, loss, penalty, lmbda, classification, mu, n_ellipsoid_steps, 
                        better_init, better_radius, cut, get_ell_from_subset, clip_ell):
    if get_ell_from_subset != 0:
        X_train_, y_train_ = balanced_subsample(X_train, y_train)
        X_train_, y_train_ = shuffle(X_train_, y_train_)
        if get_ell_from_subset < X_train_.shape[0]:
            X_train_ = X_train_[:get_ell_from_subset]
            y_train_ = y_train_[:get_ell_from_subset]
        z, scaling, L, I_k_vec, g, r_init = experiment_get_ellipsoid(X_train_, y_train_, intercept, better_init, 
                                                        better_radius, loss, penalty, lmbda, 
                                                        classification, mu, n_ellipsoid_steps)
    else:
        z, scaling, L, I_k_vec, g, r_init = experiment_get_ellipsoid(X_train, y_train, intercept, better_init, 
                                                            better_radius, loss, penalty, lmbda, 
                                                            classification, mu, n_ellipsoid_steps)
    if clip_ell:
        I_k_vec = I_k_vec.reshape(-1,1)
        A = scaling * np.identity(X_train.shape[1]) - L.dot(np.multiply(I_k_vec, np.transpose(L)))
        eigenvals, eigenvect = np.linalg.eigh(A)
        eigenvals = np.clip(eigenvals, 0, r_init)
        eigenvals = eigenvals.reshape(-1,1)
        A = eigenvect.dot(np.multiply(eigenvals, np.transpose(eigenvect)))
        scores = rank_dataset(X_train, y_train, z, A, g,
                                        lmbda, mu, classification, loss, penalty, intercept, cut)
    else:                            
        scores = rank_dataset_accelerated(X_train, y_train, z, scaling, L, I_k_vec, g,
                                        lmbda, mu, classification, loss, penalty, intercept, cut)
    return scores

def experiment_(path, dataset, synth_params, size, scale_data, redundant, noise, nb_delete_steps, lmbda, mu, classification, 
            loss, penalty, intercept, classif_score, baseline, n_ellipsoid_steps, better_init, 
            better_radius, cut, get_ell_from_subset, clip_ell, nb_exp, nb_test, 
            dontsave):
    exp_title = 'X_size_{}_baseline_{}_loss_{}_lmbda_{}_n_ellipsoid_{}_intercept_{}_mu_{}_redundant_{}_noise_{}_better_init_{}_better_radius_{}_cut_ell_{}_clip_ell_{}'.format(size, 
        baseline, loss, lmbda, n_ellipsoid_steps, intercept, mu, redundant, noise, better_init, 
        better_radius, cut, clip_ell)
    print(exp_title)

    X, y = load_experiment(dataset, synth_params, size, redundant, noise, classification, path + 'datasets/')
    scores_all = []
    compt_exp = 0
    idx_safe_all = 0

    while compt_exp < nb_exp:
        random.seed(compt_exp)
        np.random.seed(compt_exp)
        compt_exp += 1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        if scale_data:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        if experiment == 'screen':
            scores = []
            idx_safe = get_idx_safe(scores, mu, classification)
            idx_safe_all += idx_safe
            idx_to_del = []    
        else:
            model = [] #TODO
            idx_to_del = run_experiment(X_train, y_train, model, experiment, intercept, loss, penalty, lmbda, classification, 
                    mu, n_ellipsoid_steps, better_init, better_radius, cut, get_ell_from_subset, clip_ell)

        nb_to_del_table = np.linspace(1, X_train.shape[0], nb_delete_steps, dtype='int')
        for nb_to_delete in nb_to_del_table:
            score = 0
            compt = 0
            if not(dataset_has_both_labels(y_train[idx_to_del[nb_to_delete:]])):
                print('Warning, only one label in {} dataset'.format(baseline))
                break
            print(X_train[idx_to_del[nb_to_delete:]].shape)
            while compt < nb_test:
                compt += 1
                estimator = fit_estimator(X_train[idx_to_del[nb_to_delete:]], y_train[idx_to_del[nb_to_delete:]], loss, penalty, mu, lmbda, intercept)
                if classif_score:
                    score += scoring_classif(estimator, X_test, y_test)
                else:
                    score += estimator.score(X_test, y_test)
            scores.append(score / nb_test)
        scores_all.append(scores)

    print('Number of datapoints we can screen (if safe rules apply to the experiment):', idx_safe_all / nb_exp)

    data = (nb_to_del_table, scores_all, idx_safe_all / nb_exp, X_train.shape[0])
    save_dataset_folder = os.path.join(path, 'results', dataset)
    os.makedirs(save_dataset_folder, exist_ok=True)
    if not dontsave:
        np.save(os.path.join(save_dataset_folder, exp_title), data)
        print('RESULTS SAVED!')

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./', type=str)
    parser.add_argument('--dataset', default='diabetes', help='dataset used for the experiment')
    parser.add_argument('--synth_params', default=[100, 500, 10 / 500], nargs='+', type=float, help='in order: nb_samples, dimension, sparsity')
    parser.add_argument('--size', default=442, type=int, help='number of samples of the dataset to use')
    parser.add_argument('--scale_data', action='store_true')
    parser.add_argument('--redundant', default=400, type=int, help='add redundant examples to the dataset. Do not use redundant with --size')
    parser.add_argument('--noise', default=0.1, type=float, help='standard deviation of the noise to add to the redundant examples')
    parser.add_argument('--nb_delete_steps', default=20, type=int, help='at each step of the experiment, we delete size / nb_delete_steps data points')
    parser.add_argument('--lmbda', default=0.01, type=float, help='regularization parameter of the estimator')
    parser.add_argument('--mu', default=10, type=float, help='regularization parameter of the dual')
    parser.add_argument('--classification', action='store_true')
    parser.add_argument('--loss', default='truncated_squared', choices=['hinge', 'squared_hinge', 'squared','truncated_squared', 'safe_logistic'])
    parser.add_argument('--penalty', default='l1', choices=['l1','l2'])
    parser.add_argument('--intercept', action='store_true')
    parser.add_argument('--classif_score', action='store_true', help='determines the score that is used')
    parser.add_argument('--n_ellipsoid_steps', default=1000, type=int, help='number of iterations of the ellipsoid method')
    parser.add_argument('--better_init', default=0, type=int, help='number of optimizer gradient steps to initialize the center of the ellipsoid')
    parser.add_argument('--better_radius', default=0, type=float, help='radius of the initial l2 ball')
    parser.add_argument('--cut_ell', action='store_true', help='cut the final ellipsoid in half using a subgradient of the loss')
    parser.add_argument('--get_ell_from_subset', default=0, type=int, help='train the ellipsoid on a random subset of the dataset')
    parser.add_argument('--clip_ell', action='store_true', help='clip the eigenvalues of the ellipsoid')
    parser.add_argument('--nb_exp', default=10, type=int)
    parser.add_argument('--nb_test', default=3, type=int)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--zoom', default=[0.2, 0.6], nargs='+', type=float, help='zoom in the final plot')
    parser.add_argument('--dontsave', action='store_true', help='do not save your experiment, but no plot possible')
    args = parser.parse_args()

    print('START')

    experiment(args.path, args.dataset, args.synth_params, args.size, args.scale_data, args.redundant, args.noise, args.nb_delete_steps, 
        args.lmbda, args.mu, 
        args.classification, args.loss, args.penalty, args.intercept, args.classif_score, 
        args.n_ellipsoid_steps, args.better_init, args.better_radius, args.cut_ell, 
        args.get_ell_from_subset, args.clip_ell, args.nb_exp, args.nb_test, args.plot, args.zoom,
        args.dontsave)
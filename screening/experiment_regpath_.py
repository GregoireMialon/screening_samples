import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from screening.tools import (
    make_data, make_redundant_data, make_redundant_data_classification, 
    balanced_subsample, dataset_has_both_labels, get_nb_safe, 
    plot_experiment, screen_baseline_margin
)
from screening.screentools import (
    rank_dataset_accelerated,
    rank_dataset
)
from screening.fit import (
    fit_estimator
)
from screening.loaders import load_experiment
from screening.screenell import EllipsoidScreener
from screening.screendg import DualityGapScreener
from screening.safelog import SafeLogistic
from sklearn.svm import LinearSVC
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from screening.settings import RESULTS_PATH
from arsenic import BinaryClassifier
import random
import os
import time

#@profile
def experiment_regpath_(dataset, synth_params, size, scale_data, redundant, noise, lmbda_grid, mu, loss, penalty, 
                        intercept, n_ellipsoid_steps, n_epochs, n_epochs_ell_path, cut, get_ell_from_subset, clip_ell, 
                        use_sphere, nb_exp, dontsave):
    
    print('START')

    exp_title = 'X_size_{}_ell_subset_{}_loss_{}_intercept_{}_n_ell_{}_mu_{}_redundant_{}_noise_{}_cut_ell_{}_clip_ell_{}_n_epochs_{}_n_ell_path_{}_use_sphere_{}_regpath_'.format(size, 
        get_ell_from_subset, loss, intercept, n_ellipsoid_steps, mu, redundant, noise, cut, clip_ell, n_epochs, n_epochs_ell_path, use_sphere)
    print(exp_title)

    X, y = load_experiment(dataset, synth_params, size, redundant, noise, classification=True)

    data = {}
    for lmbda in lmbda_grid:
        data['budget_ell_lmbda_{}'.format(lmbda)] = 0
        data['budget_noscreen_lmbda_{}'.format(lmbda)] = 0
        #data['time_ell_lmbda_{}'.format(lmbda)] = 0
        #data['time_noscreen_lmbda_{}'.format(lmbda)] = 0 
        data['score_ell_lmbda_{}'.format(lmbda)] = 0
        data['score_noscreen_lmbda_{}'.format(lmbda)] = 0
    compt_exp = 0

    while compt_exp < nb_exp:
        random.seed(compt_exp + 1)
        np.random.seed(compt_exp + 1)
        compt_exp += 1
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)
        
        for lmbda in lmbda_grid:
            print(' LMBDA', lmbda)
            budget_ell = 0
            budget_noscreen = 0
            if lmbda == lmbda_grid[0]:
                start = time.time()
                screener_ell = EllipsoidScreener(lmbda=lmbda, mu=mu, loss=loss, penalty=penalty, 
                                                intercept=intercept, classification=True, 
                                                n_ellipsoid_steps=n_ellipsoid_steps, 
                                                cut=cut, clip_ell=clip_ell, use_sphere=use_sphere, 
                                                ars=True)
                screener_dg = DualityGapScreener(lmbda=lmbda, n_epochs=n_epochs, ars=True)
                screener_dg.fit(X_train, y_train)
                print('INIT RADIUS : ', screener_dg.squared_radius)
                random_subset = random.sample(range(0, X_train.shape[0]), get_ell_from_subset)
                screener_ell.fit(X_train[random_subset], y_train[random_subset], 
                                init=screener_dg.z, rad=screener_dg.squared_radius)
                tokeep_ell = np.arange(X_train.shape[0])
                stop = time.time()
                time_ell = stop - start

                start = time.time()
                svc = BinaryClassifier(loss='sqhinge', penalty=penalty, intercept=intercept)
                svc.fit(X_train, y_train, solver='acc-svrg', lambd=lmbda, verbose=False)
                stop = time.time()
                time_noscreen = stop - start
                svc_ell = svc
                z_svc_ell = svc_ell.w

                print('SCORE DG', screener_dg.score(X_train, y_train), 'SCORE ELL', screener_ell.score(X_train, y_train))

            else:
                start = time.time()
                screener_ell = EllipsoidScreener(lmbda=lmbda, mu=mu, loss=loss, penalty=penalty, 
                                                intercept=intercept, classification=True, 
                                                n_ellipsoid_steps=n_ellipsoid_steps, 
                                                cut=cut, clip_ell=clip_ell, use_sphere=use_sphere, 
                                                ars=True)
                screener_dg = DualityGapScreener(lmbda=lmbda, n_epochs=n_epochs_ell_path, ars=True)
                
                random_subset = random.sample(range(0, X_train.shape[0]), get_ell_from_subset)
                screener_dg.fit(X_train, y_train, init=z_svc_ell)
                print('INIT RAD : ', 2 * screener_dg.dg / lmbda)
                budget_init_ell = n_epochs_ell_path * X_train.shape[0]
                screener_ell.fit(X_train[random_subset], y_train[random_subset], 
                                init=screener_dg.z, rad=2 * screener_dg.dg / lmbda)
                print('FINAL RAD : ', screener_ell.squared_radius)
                budget_fit_ell = n_ellipsoid_steps * get_ell_from_subset
                print('SCORE DG', screener_dg.score(X_train, y_train), 'SCORE ELL', screener_ell.score(X_train, y_train))
                scores_ell = screener_ell.screen(X_train, y_train)
                budget_screen = X_train.shape[0]
                tokeep_ell = np.where(scores_ell > - mu)[0]
                svc_ell = BinaryClassifier(loss='sqhinge', penalty=penalty, intercept=intercept)
                print('CURRENT SHAPE : ', len(tokeep_ell))
                print(X_train[tokeep_ell].shape)
                budget_fit_solver = svc_ell.fit(X_train, y_train, solver='acc-svrg', it0=1, lambd=lmbda, verbose=False)[0,-1] #* X_train.shape[0] #len(tokeep_ell)
                z_svc_ell = svc_ell.w
                print(budget_init_ell, budget_fit_ell, budget_screen, budget_fit_solver)
                budget_ell += budget_init_ell + budget_fit_ell + budget_screen + budget_fit_solver
                stop = time.time()
                time_ell = stop - start

                start = time.time()
                svc = BinaryClassifier(loss='sqhinge', penalty=penalty, intercept=intercept)
                budget_fit_solver_noscreen = svc.fit(X_train, y_train, solver='acc-svrg', it0=1, lambd=lmbda, verbose=False)[0,-1] #* X_train.shape[0]
                print(budget_fit_solver_noscreen)
                budget_noscreen += budget_fit_solver_noscreen
                stop = time.time()
                time_noscreen = stop - start

            score_ell = svc_ell.score(X_train, y_train)
            score_noscreen = svc.score(X_train, y_train)

            data['budget_ell_lmbda_{}'.format(lmbda)] += budget_ell
            data['budget_noscreen_lmbda_{}'.format(lmbda)] += budget_noscreen      
            #data['time_ell_lmbda_{}'.format(lmbda)] += time_ell
            #data['time_noscreen_lmbda_{}'.format(lmbda)] += time_noscreen
            data['score_ell_lmbda_{}'.format(lmbda)] += score_ell
            data['score_noscreen_lmbda_{}'.format(lmbda)] += score_noscreen

    data = {k: float(data[k]/nb_exp) for k in data}
    save_dataset_folder = os.path.join(RESULTS_PATH, dataset)
    os.makedirs(save_dataset_folder, exist_ok=True)
    if not dontsave:
        np.save(os.path.join(save_dataset_folder, exp_title), data)
        print('RESULTS SAVED!')

    print('END')

    print(data)
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist', help='dataset used for the experiment')
    parser.add_argument('--synth_params', default=[100, 500, 10 / 500], nargs='+', type=float, help='in order: nb_samples, dimension, sparsity')
    parser.add_argument('--size', default=60000, type=int, help='number of samples of the dataset to use')
    parser.add_argument('--scale_data', action='store_true')
    parser.add_argument('--redundant', default=0, type=int, help='add redundant examples to the dataset. Do not use redundant with --size')
    parser.add_argument('--noise', default=0.1, type=float, help='standard deviation of the noise to add to the redundant examples')
    parser.add_argument('--lmbda_grid', default=[0.0001, 0.00009, 0.00008, 0.00007], nargs='+', type=float, help='regularization parameter grid of the estimator')
    parser.add_argument('--mu', default=1.0, type=float, help='regularization parameter of the dual')
    parser.add_argument('--loss', default='squared_hinge', choices=['hinge', 'squared_hinge', 'squared','truncated_squared', 'safe_logistic', 'logistic'])
    parser.add_argument('--penalty', default='l2', choices=['l1','l2'])
    parser.add_argument('--intercept', action='store_true')
    parser.add_argument('--n_ellipsoid_steps', default=100, type=int, help='number of ellipsoid step in screen ell')
    parser.add_argument('--n_epochs', default=10, type=int, help='number of epochs of the solver in ellipsoid method for screening')
    parser.add_argument('--n_epochs_ell_path', default=3, type=int, help='number of epochs of the solver in ellipsoid method for screening in path')
    parser.add_argument('--cut_ell', action='store_true', help='cut the final ellipsoid in half using a subgradient of the loss')
    parser.add_argument('--get_ell_from_subset', default=48000, type=int, help='train the ellipsoid on a random subset of the dataset')
    parser.add_argument('--clip_ell', action='store_true', help='clip the eigenvalues of the ellipsoid')
    parser.add_argument('--use_sphere', action='store_true', help='the region is a sphere whose radius is the smallest semi-axe of the ellipsoid')
    parser.add_argument('--nb_exp', default=2, type=int)
    parser.add_argument('--dontsave', action='store_true', help='do not save your experiment, but no plot possible')
    args = parser.parse_args()


    experiment_regpath_(args.dataset, args.synth_params, args.size, args.scale_data, args.redundant, args.noise, 
                args.lmbda_grid, args.mu, args.loss, args.penalty, args.intercept, args.n_ellipsoid_steps, 
                args.n_epochs, args.n_epochs_ell_path, args.cut_ell, args.get_ell_from_subset, args.clip_ell, 
                args.use_sphere, args.nb_exp, args.dontsave)
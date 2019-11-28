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
import random
import os

#@profile
def experiment_tradeoff(dataset, synth_params, size, scale_data, redundant, noise, lmbda, mu, 
                loss, penalty, intercept, acc, n_ellipsoid_steps, better_init, cut, get_ell_from_subset, clip_ell, 
                use_sphere, guarantee, nb_exp, plot, zoom, dontsave):
    
    print('START')

    X, y = load_experiment(dataset, synth_params, size, redundant, noise, classification=True)

    if acc:
        exp_title = 'X_size_{}_ell_subset_{}_loss_{}_lmbda_{}_n_ellipsoid_{}_mu_{}_better_init_{}_cut_ell_{}_clip_ell_{}_use_sphere_{}_acc'.format(size, 
            get_ell_from_subset, loss, lmbda, n_ellipsoid_steps, mu, better_init, 
            cut, clip_ell, use_sphere)
    else:
        exp_title = 'X_size_{}_ell_subset_{}_loss_{}_lmbda_{}_n_ellipsoid_{}_mu_{}_better_init_{}_cut_ell_{}_clip_ell_{}_use_sphere_{}_tradeoff'.format(size, 
            get_ell_from_subset, loss, lmbda, n_ellipsoid_steps, mu, better_init, 
            cut, clip_ell, use_sphere)
    print(exp_title)

    nb_epochs = int(better_init + n_ellipsoid_steps * get_ell_from_subset / (0.8 * X.shape[0]))
    scores_screening_all = np.zeros(nb_epochs)
    safe_guarantee = np.array([0., 0.])
    
    compt_exp = 0
    
    while compt_exp < nb_exp:
        #random.seed(compt_exp + 1)
        #np.random.seed(compt_exp + 1)
        compt_exp += 1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        if acc:
            for i in range(nb_epochs):
                estimator = LinearSVC(loss='squared_hinge', dual=False, C=1/lmbda, fit_intercept=False, 
                            max_iter=i+1, tol=1.0e-20).fit(X_train, y_train)
                scores_screening_all[i] += estimator.score(X_test, y_test)
                print(scores_screening_all[i])
            print('SCORES', scores_screening_all)
        else:
            for i in range(nb_epochs):
                i = i + 1
                if i <= better_init:
                    screener_dg = DualityGapScreener(lmbda=lmbda, n_epochs=i).fit(X_train, y_train)
                    z_init = screener_dg.z
                    rad_init = screener_dg.squared_radius
                    scores = screener_dg.screen(X_train, y_train)
                    scores_screening_all[i - 1] += get_nb_safe(scores, mu, classification=True)
                    print('SCREEN DG RADIUS', screener_dg.squared_radius)
                elif better_init < i <= nb_epochs:
                    random_subset = random.sample(range(0, X_train.shape[0]), get_ell_from_subset)
                    screener_ell = EllipsoidScreener(lmbda=lmbda, mu=mu, loss=loss, penalty=penalty, 
                                            intercept=intercept, classification=True, 
                                            n_ellipsoid_steps= int((i - better_init) * X_train.shape[0] / get_ell_from_subset), 
                                            better_init=0, better_radius=0, 
                                            cut=cut, clip_ell=clip_ell, 
                                            use_sphere=use_sphere).fit(X_train[random_subset], y_train[random_subset], 
                                                                                init=z_init, rad=rad_init)
                    scores = screener_ell.screen(X_train, y_train)
                    scores_screening_all[i - 1] += get_nb_safe(scores, mu, classification=True)
                    if use_sphere:
                        print('SCREEN ELL RADIUS', screener_ell.squared_radius)

            if guarantee:
                idx_safeell = np.where(scores > - mu)[0]
                print('SCORES ', scores)
                print('NB TO KEEP', len(idx_safeell))
                if len(idx_safeell) !=0:
                    estimator_whole = fit_estimator(X_train, y_train, loss, penalty, mu, lmbda, intercept)
                    estimator_screened = fit_estimator(X_train[idx_safeell], y_train[idx_safeell], loss, 
                                    penalty, mu, lmbda, intercept)
                    temp = np.array([estimator_whole.score(X_train, y_train), 
                                estimator_screened.score(X_train, y_train)])
                    print('SAFE GUARANTEE : ', temp)
                    safe_guarantee += temp
    
    if acc:
        scores_screening_all = scores_screening_all * X_train.shape[0]
    data = {
        'step_table': better_init + n_ellipsoid_steps * (get_ell_from_subset / X_train.shape[0]),
        'scores_screening': scores_screening_all / (X_train.shape[0] * nb_exp),
        'safe_guarantee': safe_guarantee / nb_exp
    }
    print(data)
    save_dataset_folder = os.path.join(RESULTS_PATH, dataset)
    os.makedirs(save_dataset_folder, exist_ok=True)
    if not dontsave:
        np.save(os.path.join(save_dataset_folder, exp_title), data)
        print('RESULTS SAVED!')

    if plot:
        plot_experiment(data, zoom=zoom)
    
    print('END')
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist', help='dataset used for the experiment')
    parser.add_argument('--synth_params', default=[100, 500, 10 / 500], nargs='+', type=float, help='in order: nb_samples, dimension, sparsity')
    parser.add_argument('--size', default=60000, type=int, help='number of samples of the dataset to use')
    parser.add_argument('--scale_data', action='store_true')
    parser.add_argument('--redundant', default=0, type=int, help='add redundant examples to the dataset. Do not use redundant with --size')
    parser.add_argument('--noise', default=0.1, type=float, help='standard deviation of the noise to add to the redundant examples')
    parser.add_argument('--lmbda', default=1.0, type=float, help='regularization parameter of the estimator')
    parser.add_argument('--mu', default=1.0, type=float, help='regularization parameter of the dual')
    parser.add_argument('--loss', default='squared_hinge', choices=['hinge', 'squared_hinge', 'squared','truncated_squared', 'safe_logistic', 'logistic'])
    parser.add_argument('--penalty', default='l2', choices=['l1','l2'])
    parser.add_argument('--intercept', action='store_true')
    parser.add_argument('--acc', action='store_true', help='for plot accuracy of estimator vs epoch')
    parser.add_argument('--n_ellipsoid_steps', default=10, type=int, help='number of ellipsoid steps to be done')
    parser.add_argument('--better_init', default=1, type=int, help='number of optimizer gradient steps to initialize the center of the ellipsoid')
    parser.add_argument('--cut_ell', action='store_true', help='cut the final ellipsoid in half using a subgradient of the loss')
    parser.add_argument('--get_ell_from_subset', default=int(0.8 * 60000), type=int, help='train the ellipsoid on a random subset of the dataset')
    parser.add_argument('--clip_ell', action='store_true', help='clip the eigenvalues of the ellipsoid')
    parser.add_argument('--use_sphere', action='store_true', help='the region is a sphere whose radius is the smallest semi-axe of the ellipsoid')
    parser.add_argument('--guarantee', action='store_true', help='check whether the deleted points were safe')
    parser.add_argument('--nb_exp', default=2, type=int)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--zoom', default=[0, 1], nargs='+', type=float, help='zoom in the final plot')
    parser.add_argument('--dontsave', action='store_true', help='do not save your experiment, but no plot possible')
    args = parser.parse_args()


    experiment_tradeoff(args.dataset, args.synth_params, args.size, args.scale_data, args.redundant, args.noise, 
                args.lmbda, args.mu, args.loss, args.penalty, args.intercept, args.acc,
                args.n_ellipsoid_steps, args.better_init, args.cut_ell, 
                args.get_ell_from_subset, args.clip_ell, args.use_sphere, args.guarantee, 
                args.nb_exp, args.plot, args.zoom, args.dontsave)
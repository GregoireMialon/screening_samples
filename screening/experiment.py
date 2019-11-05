import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from screening.tools import (
    make_data, make_redundant_data, make_redundant_data_classification, 
    balanced_subsample, random_screening, dataset_has_both_labels, get_nb_safe, 
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
def experiment(dataset, synth_params, size, scale_data, redundant, noise, nb_delete_steps, lmbda, mu, classification, 
                loss, penalty, intercept, n_ellipsoid_steps, better_init, better_radius, cut, get_ell_from_subset, clip_ell, 
                use_sphere, guarantee, n_epochs_dg, nb_exp, nb_test, plot, zoom, dontsave):
    
    print('START')

    exp_title = 'X_size_{}_ell_subset_{}_loss_{}_lmbda_{}_n_ellipsoid_{}_intercept_{}_mu_{}_redundant_{}_noise_{}_better_init_{}_better_radius_{}_cut_ell_{}_clip_ell_{}_n_dg_{}_use_sphere_{}_nds_{}'.format(size, 
        get_ell_from_subset, loss, lmbda, n_ellipsoid_steps, intercept, mu, redundant, noise, better_init, 
        better_radius, cut, clip_ell, n_epochs_dg, use_sphere, nb_delete_steps)
    print(exp_title)

    X, y = load_experiment(dataset, synth_params, size, redundant, noise, classification)

    scores_regular_all = []
    scores_ell_all = []
    scores_dg_all = []
    scores_r_all = []
    safe_guarantee = np.array([0, 0])
    
    compt_exp = 0
    nb_safe_ell_all = 0
    nb_safe_dg_all = 0
    
    while compt_exp < nb_exp:
        random.seed(compt_exp + 1)
        np.random.seed(compt_exp + 1)
        compt_exp += 1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        screener_ell = EllipsoidScreener(lmbda=lmbda, mu=mu, loss=loss, penalty=penalty, 
                                            intercept=intercept, classification=classification, 
                                            n_ellipsoid_steps=n_ellipsoid_steps, 
                                            better_init=0, better_radius=0, 
                                            cut=cut, clip_ell=clip_ell, use_sphere=use_sphere)
        screener_dg = DualityGapScreener(lmbda=lmbda, n_epochs=better_init)

        if scale_data:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        if get_ell_from_subset != 0:
            random_subset = random.sample(range(0, X_train.shape[0]), get_ell_from_subset)
            screener_dg.fit(X_train, y_train)
            print('INIT RADIUS :', screener_dg.squared_radius, X_train.shape[1], screener_dg.first_obj)
            rad = np.min([screener_dg.squared_radius, X_train.shape[1], screener_dg.first_obj])
            if better_init == 0:
                init = None
            else:
                init = screener_dg.z.reshape(-1,)
            screener_ell.fit(X_train[random_subset], y_train[random_subset], 
                                                    init=init, rad=rad)
            screener_dg = DualityGapScreener(lmbda=lmbda, n_epochs=n_epochs_dg).fit(X_train, y_train)
        else:
            screener_dg.fit(X_train, y_train)
            print('INIT RADIUS :', screener_dg.squared_radius, X_train.shape[1], screener_dg.first_obj)
            rad = np.min([screener_dg.squared_radius, X_train.shape[1], screener_dg.first_obj])
            if better_init == 0:
                init = None
            else:
                init = screener_dg.z.reshape(-1,)
            screener_ell.fit(X_train, y_train, init=init, rad=rad)
            screener_dg = DualityGapScreener(lmbda=lmbda, n_epochs=n_epochs_dg).fit(X_train, y_train)
                                            
        print('SQUARED RADIUS OF THE DUALITY GAP BALL :', screener_dg.squared_radius)
            
        scores_screendg = screener_dg.screen(X_train, y_train)
        scores_screenell = screener_ell.screen(X_train, y_train)
        
        idx_screendg = np.argsort(scores_screendg)
        idx_screenell = np.argsort(scores_screenell)

        print('SCORES_DG', scores_screendg[:10])
        print('SCORES_ELL', scores_screenell[:10])

        nb_safe_ell = get_nb_safe(scores_screenell, mu, classification)
        nb_safe_ell_all += nb_safe_ell
        nb_safe_dg = get_nb_safe(scores_screendg, mu, classification)
        nb_safe_dg_all += nb_safe_dg

        scores_regular = []
        scores_ell = []
        scores_dg = []
        scores_r = []

        nb_to_del_table=None

        if guarantee:
            idx_safeell = np.where(scores_screenell < -mu)[0]
            if len(idx_safeell) !=0:
                estimator_whole = fit_estimator(X_train, y_train, loss, penalty, mu, lmbda, intercept)
                estimator_screened = fit_estimator(X_train[idx_safeell], y_train[idx_safeell], loss, 
                                penalty, mu, lmbda, intercept)
                safe_guarantee = np.array([estimator_whole.score(X_train, y_train),
                                            estimator_screened.score(X_train, y_train)])

        if nb_delete_steps != 0:
            nb_to_del_table = np.sqrt(np.linspace(1, X_train.shape[0], nb_delete_steps, dtype='int'))
            nb_to_del_table = np.ceil(nb_to_del_table * (X_train.shape[0] / nb_to_del_table[-1])).astype(int)

            X_r = X_train
            y_r = y_train
            
            for i, nb_to_delete in enumerate(nb_to_del_table):
                if i == 0:
                    score_regular = 0
                score_ell = 0
                score_dg = 0
                score_r = 0
                compt = 0
                
                X_screenell, y_screenell = X_train[idx_screenell[nb_to_delete:]], y_train[idx_screenell[nb_to_delete:]]
                X_screendg, y_screendg = X_train[idx_screendg[nb_to_delete:]], y_train[idx_screendg[nb_to_delete:]]
                X_r, y_r = random_screening(X_r, y_r, X_train.shape[0] - nb_to_delete)
                if not(dataset_has_both_labels(y_r)):
                    print('Warning, only one label in randomly screened dataset')
                if not(dataset_has_both_labels(y_screenell)):
                    print('Warning, only one label in screenell dataset')
                if not(dataset_has_both_labels(y_screendg)):
                    print('Warning, only one label in screendg dataset')
                if not(dataset_has_both_labels(y_r) and dataset_has_both_labels(y_screenell) and dataset_has_both_labels(y_screendg)): 
                    break
                print('X_train :', X_train.shape,'X_screenell :', X_screenell.shape, 'X_screendg : ', X_screendg.shape,
                    'X_random : ', X_r.shape) 
                while compt < nb_test:
                    compt += 1
                    if i == 0:
                        estimator_regular = fit_estimator(X_train, y_train, loss, penalty, mu, lmbda, intercept)
                    estimator_screenell = fit_estimator(X_screenell, y_screenell, loss, penalty, mu, lmbda, 
                    intercept)
                    estimator_screendg = fit_estimator(X_screendg, y_screendg, loss, penalty, mu, lmbda, 
                    intercept)
                    estimator_r = fit_estimator(X_r, y_r, loss, penalty, mu, lmbda, intercept)
                    if i == 0:
                        score_regular += estimator_regular.score(X_test, y_test)
                    score_ell += estimator_screenell.score(X_test, y_test)
                    score_dg += estimator_screendg.score(X_test, y_test)
                    score_r += estimator_r.score(X_test, y_test)

                scores_regular.append(score_regular / nb_test)
                scores_ell.append(score_ell / nb_test)
                scores_dg.append(score_dg / nb_test)
                scores_r.append(score_r / nb_test)

            scores_regular_all.append(scores_regular)
            scores_ell_all.append(scores_ell)
            scores_dg_all.append(scores_dg)
            scores_r_all.append(scores_r)

    print('Number of datapoints we can safely screen with ellipsoid method:', nb_safe_ell_all / nb_exp)
    print('Number of datapoints we can safely screen with duality gap method:', nb_safe_dg_all / nb_exp)

    data = {
        'nb_to_del_table': nb_to_del_table,
        'scores_regular': scores_regular_all,
        'scores_ell': scores_ell_all,
        'scores_dg': scores_dg_all,
        'scores_r': scores_r_all,
        'nb_safe_ell': nb_safe_ell_all / nb_exp,
        'nb_safe_dg': nb_safe_dg_all / nb_exp,
        'train_set_size': X_train.shape[0],
        'safe_guarantee': safe_guarantee / nb_exp
    }
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
    parser.add_argument('--nb_delete_steps', default=15, type=int, help='at each step of the experiment, we delete size / nb_delete_steps data points')
    parser.add_argument('--lmbda', default=0.1, type=float, help='regularization parameter of the estimator')
    parser.add_argument('--mu', default=1.0, type=float, help='regularization parameter of the dual')
    parser.add_argument('--classification', action='store_true')
    parser.add_argument('--loss', default='squared_hinge', choices=['hinge', 'squared_hinge', 'squared','truncated_squared', 'safe_logistic', 'logistic'])
    parser.add_argument('--penalty', default='l2', choices=['l1','l2'])
    parser.add_argument('--intercept', action='store_true')
    parser.add_argument('--n_ellipsoid_steps', default=10, type=int, help='number of iterations of the ellipsoid method')
    parser.add_argument('--better_init', default=0, type=int, help='number of optimizer gradient steps to initialize the center of the ellipsoid')
    parser.add_argument('--better_radius', default=0, type=float, help='DEPRECATED, radius of the initial l2 ball')
    parser.add_argument('--cut_ell', action='store_true', help='cut the final ellipsoid in half using a subgradient of the loss')
    parser.add_argument('--get_ell_from_subset', default=0, type=int, help='train the ellipsoid on a random subset of the dataset')
    parser.add_argument('--clip_ell', action='store_true', help='clip the eigenvalues of the ellipsoid')
    parser.add_argument('--use_sphere', action='store_true', help='the region is a sphere whose radius is the smallest semi-axe of the ellipsoid')
    parser.add_argument('--guarantee', action='store_true', help='check whether the deleted points were safe')
    parser.add_argument('--n_epochs_dg', default=5, type=int, help='number of epochs of the solver in the duality gap baseline for screening')
    parser.add_argument('--nb_exp', default=3, type=int)
    parser.add_argument('--nb_test', default=3, type=int)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--zoom', default=[0, 1], nargs='+', type=float, help='zoom in the final plot')
    parser.add_argument('--dontsave', action='store_true', help='do not save your experiment, but no plot possible')
    args = parser.parse_args()


    experiment(args.dataset, args.synth_params, args.size, args.scale_data, args.redundant, args.noise, 
                args.nb_delete_steps, args.lmbda, args.mu, args.classification, args.loss, args.penalty, 
                args.intercept, args.n_ellipsoid_steps, args.better_init, args.better_radius, args.cut_ell, 
                args.get_ell_from_subset, args.clip_ell, args.use_sphere, args.guarantee, args.n_epochs_dg, 
                args.nb_exp, args.nb_test, args.plot, args.zoom, args.dontsave)
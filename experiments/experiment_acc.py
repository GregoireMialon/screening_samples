from screening.fit import fit_estimator
from utils.settings import RESULTS_PATH
from utils.loaders import load_experiment
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
import argparse

def experiment_acc(dataset, loss, penalty, lmbda):

    acc = {}
    random.seed(0)
    np.random.seed(0)
    if dataset == 'mnist':
        size=60000
    elif dataset == 'svhn':
        size=604388
    elif dataset == 'rcv1':
        size=781265
    X, y = load_experiment(dataset=dataset, synth_params=None, size=size, redundant=0, noise=0, classification=True)
    score = 0
    k = 0
    while k < 3:
        k+=1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        estimator = fit_estimator(X_train, y_train, loss=loss, penalty=penalty, mu=1, lmbda=lmbda, 
                                    intercept=False, max_iter=10000)
        score += estimator.score(X_test, y_test)
    acc['{}_{}_{}_{}'.format(dataset, loss, penalty, lmbda)] = score / 3
    print('{}_{}_{}_{}'.format(dataset, loss, penalty, lmbda), ' : Done !')

    save_dataset_folder = os.path.join(RESULTS_PATH, 'accuracies')
    os.makedirs(save_dataset_folder, exist_ok=True)
    np.save(os.path.join(save_dataset_folder, '{}_{}_{}_{}'.format(dataset, loss, penalty, lmbda)), acc)
    print('RESULTS SAVED!')

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist', choices=['mnist', 'svhn', 'rcv1'])
    parser.add_argument('--loss', default='squared_hinge',  choices=['hinge', 'squared_hinge', 'squared','truncated_squared', 'safe_logistic', 'logistic'])
    parser.add_argument('--penalty', default='l2', choices=['l1', 'l2'])
    parser.add_argument('--lmbda', default=0.1, type=float)
    args = parser.parse_args()

    experiment_acc(args.dataset, args.loss, args.penalty, args.lmbda)

    #for lmbda in [0.0001, 0.001, 0.01, 0.1, 1.0]:
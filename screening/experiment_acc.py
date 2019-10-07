from screening.fit import fit_estimator
from screening.settings import RESULTS_PATH
from screening.loaders import load_experiment
from sklearn.model_selection import train_test_split
from screening.tools import scoring_classif
import os
import random
import numpy as np

def experiment_acc():

    acc = {}

    for dataset in ['mnist', 'svhn']:
        random.seed(0)
        np.random.seed(0)
        if dataset == 'mnist':
            size=60000
        elif dataset == 'svhn':
            size=604388
        X, y = load_experiment(dataset=dataset, synth_params=None, size=size, redundant=0, noise=0, classification=True)
        for loss in ['squared_hinge', 'logistic', 'safe_logistic']:
            for penalty in ['l1', 'l2']:
                for lmbda in [0.0001, 0.001, 0.01, 0.1, 1.0]:
                    score = 0
                    k = 0
                    while k < 3:
                        k+=1
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                        estimator = fit_estimator(X_train, y_train, loss=loss, penalty=penalty, mu=1, lmbda=lmbda, 
                                                    intercept=False, max_iter=10000)
                        score += scoring_classif(estimator, X_test, y_test)
                    acc['{}_{}_{}_{}'.format(dataset, loss, penalty, lmbda)] = score / 3
                    print('{}_{}_{}_{}'.format(dataset, loss, penalty, lmbda), ' : Done !')

    save_dataset_folder = os.path.join(RESULTS_PATH, 'accuracies')
    os.makedirs(save_dataset_folder, exist_ok=True)
    np.save(os.path.join(save_dataset_folder, 'accuracies_dict'), acc)
    print('RESULTS SAVED!')

    return

if __name__ == "__main__":
    experiment_acc()


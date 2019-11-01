import numpy as np
import os
import re
from screening.settings import RESULTS_PATH

def get_lmbda(filename):
    lmbda = re.findall('(?<=l2_).*?(?=.npy)', filename)
    if lmbda is None:
        lmbda = re.findall('(?<=l1_).*?(?=.npy)', filename)
    if lmbda is None:
        'LMBDA not found'
    return lmbda[0]

def get_best_score(model, dataset):
    lmbda = 0
    score = 0
    acc_path = os.path.join(RESULTS_PATH, 'accuracies')
    for file in os.listdir(acc_path):
        filename = os.fsdecode(file)
        if (dataset + '_' + model) in filename:
            to_load = os.path.join(acc_path, filename)
            dic = np.load(to_load).item()
            index = filename.replace('.npy','')
            score_ = dic.get(index)
            if score_ > score:
                score = score_
                lmbda = get_lmbda(filename)
    return (str(lmbda), str(round(score, 3)))

list_dataset = ['mnist', 'svhn', 'rcv1']

line_positions = set()
list_model = ['logistic_l1', 
                'logistic_l2', 
                'safe_logistic_l1', 
                'safe_logistic_l2',
                'squared_hinge_l1',
                'squared_hinge_l2']
line_positions.update([10])


dic_model = {'logistic_l1':'Logistic + $\ell_1$',
             'logistic_l2':'Logistic + $\ell_2$',
             'safe_logistic_l1':'Safelog + $\ell_1$',
             'safe_logistic_l2':'Safelog + $\ell_2$',
             'squared_hinge_l1':'Squared Hinge + $\ell_1$',
             'squared_hinge_l2':'Squared Hinge + $\ell_2$',
        }

dic_dataset = {'mnist': 'MNIST', 
        'svhn': 'SVHN',
        'rcv1': 'RCV-1'
        }

def print_table(list_model, list_dataset):
    n = len(list_model)
    m = len(list_dataset)
    scores = np.zeros((n, m, 2))
    table = [['-' for _ in range(m)] for _ in range(n)]
  
    for i in range(n):
        for j in range(m):
            couple = get_best_score(list_model[i], list_dataset[j])
            table[i][j] = couple[1] + '(' + couple[0] + ')'

    print(r'\begin{tabular}{ | l |', 'c | ' * m, '}')
    print(r'\hline')
    print(r'Dataset &', ' & '.join(dic_dataset[truc] for truc in list_dataset), r'\\ \hline')
    print(r'\hline')
    for i in range(n):
        if i in line_positions:
            print(r'\hline')
        print(dic_model[list_model[i]],'&', ' & '.join(table[i]), r'\\') #  \hline')
    print(r'\hline')
    print(r'\end{tabular}')


if __name__ == '__main__':
    #print(get_best_score('logistic_l2', 'mnist'))
    print_table(list_model, list_dataset)

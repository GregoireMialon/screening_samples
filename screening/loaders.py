import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_boston, fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.io import loadmat
import pickle
from screening.tools import (
    make_data, make_redundant_data, 
    make_redundant_data_classification
)
from screening.settings import DATASETS_PATH


def load_leukemia():
    data = pd.read_csv(DATASETS_PATH + 'leukemia_big.csv')
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

def load_mnist(pb=1):
    mat = loadmat(DATASETS_PATH + 'ckn_mnist.mat')
    X = mat['psiTr'].T
    print('Shape of original MNIST', X.shape)
    y = mat['Ytr']
    y = np.array(y, dtype=int).reshape(y.shape[0])
    for i in range(len(y)):
        if y[i] != 9:
            y[i] = - 1
        else:
            y[i] = 1
    #X, y = balanced_subsample(X, y)
    return X, y

def load_higgs():
    dir_higgs = DATASETS_PATH + 'higgs'
    with open(dir_higgs, 'rb') as handle:
        data_higgs = pickle.load(handle)
    X = data_higgs[0]
    y =data_higgs[1]
    return X, y

def load_synthetic(extension):
    X = np.load(DATASETS_PATH + 'synthetic_X' + extension + '.npy')
    y = np.load(DATASETS_PATH + 'synthetic_y' + extension + '.npy') 
    return X, y

def load_experiment(dataset, synth_params, size, redundant, noise, classification):
    if dataset == 'leukemia':
        X, y = load_leukemia()
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
        X, y = load_mnist()
    elif dataset == 'higgs':
        X, y = load_higgs()
    elif dataset == 'synthetic':
        X, y, _, _ = make_data(synth_params[0], synth_params[1], synth_params[2]) #old params: 100, 2, 0.5
        #print('TRUE SYNTHETIC PARAMETERS', true_params)
    elif 'fixed_synthetic' in dataset:
        extension = dataset.replace('fixed_synthetic','')
        X, y = load_synthetic(extension)
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
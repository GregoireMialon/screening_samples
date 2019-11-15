import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import (
    rand, csr_matrix, csc_matrix
)
import random
import time
from sklearn.model_selection import train_test_split


#TODO : eliminate the loops
def compute_truncated_squared_loss_gradient(u, mu):
    g = np.zeros(u.size)
    for i in range(u.size):
        if np.abs(u[i]) > mu:
            g[i] = 2 * (u[i] - np.sign(u[i]) * mu)
    return g


def compute_hinge_subgradient(u, mu):
    g = np.zeros(u.size)
    for i in range(u.size):
        if u[i] < mu:
            g[i] = -1
        elif u[i] == mu:
            g[i] = np.random.rand(1)[0] - 1
    return g


def compute_squared_hinge(u, mu):
    return np.sum(np.maximum((mu * np.ones(len(u)) - u), 0) ** 2)


def compute_squared_hinge_gradient(u, mu):
    return np.minimum(-2 * (mu * np.ones(len(u)) - u), 0)


def compute_squared_hinge_conjugate(u):
    return np.sum((1 / 4) * (u ** 2) + u)


def compute_l1_subgradient(u):
    return np.sign(u)


def compute_safe_logistic(u, mu):
    #we make a change of variable mu = 1 - mu w.r.t the actual formula
    output = np.minimum(u - mu, 0)
    return np.sum(np.exp(output) - output - np.ones(len(u)))


def compute_safe_logistic_gradient(u, mu):
    #we make a change of variable mu = 1 - mu w.r.t the actual formula
    return np.exp(np.minimum(u - mu, 0)) - np.ones(len(u))


def compute_logistic(u):
    return np.log(np.exp(-u) + 1)
    

def compute_logistic_gradient(u):
    return - np.exp(-u) / (np.exp(-u) + 1)


def compute_loss(z, X, y, loss, penalty, lmbda, mu):
    if loss == 'squared_hinge':
        pred = X.dot(z)
        if pred.shape == (X.shape[0],) and y.shape == (X.shape[0],):
            loss = compute_squared_hinge(u=y * pred, mu=mu)
        else:
            raise ValueError('pred does not have the right shape')
    elif loss == 'safe_logistic':
        pred = X.dot(z)
        if pred.shape == (X.shape[0],) and y.shape == (X.shape[0],):
            loss = compute_safe_logistic(u=y * pred, mu=mu) / X.shape[0]
        else:
            raise ValueError('pred does not have the right shape')
    if penalty == 'l2':
        reg = (np.linalg.norm(z) ** 2) / 2
    elif penalty == 'l1':
        reg = np.linalg.norm(z, ord=1)
    return loss + lmbda * reg


def compute_subgradient(x, D, y, lmbda, mu, loss, penalty, intercept, ars):
    if loss == 'truncated_squared':
        output = D.dot(x) - y
        g_1 = compute_truncated_squared_loss_gradient(output, mu)
        g_1 = (1 / (2 * D.shape[0])) * np.transpose(D).dot(g_1)
    elif loss == 'squared':
        output = D.dot(x) - y
        g_1 = (1 / D.shape[0]) * np.transpose(D).dot(output)
    elif loss == 'hinge':
        output = y * (D.dot(x))
        g_1 = compute_hinge_subgradient(output, mu)
        g_1 = (np.transpose(D).dot(y * g_1))
    elif loss == 'squared_hinge':
        output = y * (D.dot(x))
        g_1 = compute_squared_hinge_gradient(output, mu)
        g_1 = (np.transpose(D).dot(y * g_1))
        if ars:
            g_1 /= (2 * D.shape[0]) 
    elif loss == 'safe_logistic':
        output = y * (D.dot(x))
        g_1 = compute_safe_logistic_gradient(output, mu)
        g_1 = (1 / D.shape[0]) * (np.transpose(D).dot(y * g_1))
    elif loss == 'logistic':
        output = y * (D.dot(x))
        g_1 = compute_logistic_gradient(output)
        g_1 = (1 / D.shape[0]) * (np.transpose(D).dot(y * g_1))
    if penalty == 'l2':
        g_2 = np.copy(x)
        if intercept:
            g_2[D.shape[1]-1] = 0
    elif penalty == 'l1':   
        g_2 = compute_l1_subgradient(x)
        if intercept:
            g_2[D.shape[1]-1] = 0       
    g = g_1 + lmbda * g_2
    return g


def compute_A_g(scaling, L, I_k_vec, g):
    if L is not 0 and I_k_vec is not 0:
        if type(g).__name__ == 'csr_matrix':
            g = g.toarray().reshape(-1,)
        L_g = L.T.dot(g)
        I_k_L = np.multiply(I_k_vec, L_g)
        A_g = scaling * g - L.dot(I_k_L)
    else:
        A_g = scaling * g
    return A_g


def compute_A_X(scaling, L, I_k_vec, X):
    if L is not 0 and I_k_vec is not 0:
        L_X = L.T.dot(X)
        I_k_L = csr_matrix(L_X.multiply(I_k_vec.reshape(-1,1)))
        A_X = scaling * csr_matrix(X) - L.dot(I_k_L)
    else:
        A_X = scaling * X
    return A_X


def compute_test_accelerated(D_i, y_i, z, scaling, L, I_k_vec, g, classification, cut):
    A_D_i = compute_A_g(scaling, L, I_k_vec, D_i).T
    if classification:
        if cut:
            A_g = compute_A_g(scaling, L, I_k_vec, g)
            nu = - g.dot(A_D_i) / g.dot(A_g)
            if nu < 0:
                test = D_i.dot(z) - np.sqrt(D_i.dot(A_D_i))
            else:
                new_D_i = D_i + nu * g
                A_new_D_i = compute_A_g(scaling, L, I_k_vec, new_D_i)
                nu_ = np.sqrt(new_D_i.dot(A_new_D_i))
                body = D_i.dot(A_new_D_i) / nu_
                test = D_i.dot(z) - body
        else:
            test = D_i.dot(z) - np.sqrt(D_i.dot(A_D_i))

    else:
        if cut:
            A_g = compute_A_g(scaling, L, I_k_vec, g)
            nu = g.dot(A_D_i) / g.dot(A_g)
            if nu < 0:
                test = D_i.dot(z) + np.sqrt(D_i.dot(A_D_i)) - y_i
            else:
                new_D_i = D_i - nu * g
                A_new_D_i = compute_A_g(scaling, L, I_k_vec, new_D_i)
                nu_ = np.sqrt(new_D_i.dot(A_new_D_i))
                body = D_i.dot(A_new_D_i) / nu_
                test = D_i.dot(z) + body - y_i
        else:
            test = D_i.dot(z) + np.sqrt(D_i.dot(A_D_i)) - y_i
    return test


def compute_test_accelerated_(Xy, z, scaling, L, I_k_vec, g, cut):
    A_X = compute_A_X(scaling, L, I_k_vec, Xy.T)
    if cut:
        pass
    else:
        test = Xy.dot(z) - np.sqrt(np.array((A_X.T.multiply(Xy)).sum(1)).reshape(-1,))
    return test


def rank_dataset_accelerated(D, y, z, scaling, L, I_k_vec, g, mu, classification, intercept, cut):
    '''
    Gives score to each sample, does not re-order the dataset
    '''
    if intercept:
        X = np.concatenate((D, np.ones(D.shape[0]).reshape(1,-1).T), axis=1)
    else:
        X = D
    scores = np.zeros(X.shape[0])
    
    if classification:
        if type(D).__name__ == 'csr_matrix':
            Xy = csr_matrix(X.multiply(y.reshape(-1,1)))
            scores = - compute_test_accelerated_(Xy, z, scaling, L, I_k_vec, g, cut)
        else:
            Xy = y.reshape(-1,1) * X
            scores = - np.array([compute_test_accelerated(sample, None, z, scaling, L, I_k_vec, 
                                                    g, classification, cut) for sample in Xy])
    else:
        for i in range(X.shape[0]):
            test_1 = compute_test_accelerated(X[i], y[i], z, scaling, L, I_k_vec, g,
             classification, cut)
            test_2 = compute_test_accelerated(-X[i], -y[i], z, scaling, L, I_k_vec, g,
                classification, cut)
            scores[i] = np.maximum(test_1, test_2)
    return scores


def compute_test(D_i, y_i, z, A, g, classification, cut):
    A_D_i = A.dot(D_i) 
    if classification:
        if cut:
            A_g = A.dot(g)
            nu = - g.dot(A_D_i) / g.dot(A_g)
            if nu < 0:
                test = D_i.dot(z) - np.sqrt(D_i.dot(A_D_i))
            else:
                new_D_i = D_i + nu * g
                A_new_D_i = A.dot(new_D_i)
                nu_ = np.sqrt(new_D_i.dot(A_new_D_i)) 
                body = D_i.dot(A_new_D_i) / nu_
                test = D_i.dot(z) - body
        else:
            test = D_i.dot(z) - np.sqrt(D_i.dot(A_D_i))
        
    else:
        if cut:
            A_g = A.dot(g)
            nu = g.dot(A_D_i) / g.dot(A_g)
            if nu < 0:
                test = D_i.dot(z) + np.sqrt(D_i.dot(A_D_i)) - y_i
            else:
                new_D_i = D_i - nu * g
                A_new_D_i = A.dot(new_D_i)
                nu_ = np.sqrt(new_D_i.dot(A_new_D_i)) 
                body = D_i.dot(A_new_D_i) / nu_
                test = D_i.dot(z) + body - y_i
        else:
            test = D_i.dot(z) + np.sqrt(D_i.dot(A_D_i)) - y_i
    return test


def rank_dataset(D, y, z, A, g, mu, classification, intercept, cut):
    if intercept:
        X = np.concatenate((D, np.ones(D.shape[0]).reshape(1,-1).T), axis=1)
    else:
        X = D
    scores = np.zeros(X.shape[0])
    
    if classification:
        for i in range(X.shape[0]):
            x_i = y[i] * X[i]
            scores[i] = - compute_test(x_i, None, z, A, g, classification, cut)
    else:
        for i in range(X.shape[0]):
            test_1 = compute_test(X[i], y[i], z, A, g,
             classification, cut)
            test_2 = compute_test(-X[i], -y[i], z, A, g,
                classification, cut)
            scores[i] = np.maximum(test_1, test_2)
    return scores

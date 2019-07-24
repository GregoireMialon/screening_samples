import numpy as np
import matplotlib.pyplot as plt
import sklearn
from scipy.sparse import rand
import random
import time
from sklearn.model_selection import train_test_split


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


def compute_l1_subgradient(u):
    g = np.zeros(u.size)
    for i in range(u.size):
        if u[i] != 0:
            g[i] = np.sign(u[i])
        else:
            g[i] = 2 * np.random.rand(1)[0] - 1
    return g


def compute_subgradient(x, D, y, lmbda, mu, loss, penalty, intercept):
    if loss == 'truncated_squared':
        output = D.dot(x) - y
        g_1 = compute_truncated_squared_loss_gradient(output, mu)
        g_1 = (1 / (2 * D.shape[0])) * np.transpose(D).dot(g_1)
    elif loss == 'hinge':
        output = y * D.dot(x)
        g_1 = compute_hinge_subgradient(output, mu)
        g_1 = (np.transpose(D).dot(y * g_1))
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
    L_g = np.dot(np.transpose(L), g)
    I_k_L = np.multiply(I_k_vec, L_g)
    A_g = scaling * g - L.dot(I_k_L)
    return A_g


#@profile
def iterate_ellipsoids_accelerated(D, y, z_init, r_init, lmbda, mu, loss, penalty, n_steps, intercept):
    if intercept:
        X = np.concatenate((D, np.ones(D.shape[0]).reshape(1,-1).T), axis=1)
    else:
        X = D
    start = time.time()
    k = 0
    z = z_init
    p = z_init.size
    s = p ** 2 / (p ** 2 - 1)
    scaling = r_init * (s ** (n_steps))

    while k < n_steps:
        g = compute_subgradient(z, X, y, lmbda, mu, loss, penalty, intercept)
        if k == 0:
            A_g = r_init * g
            den = np.sqrt(g.dot(A_g))
            A_g = (1 / den) * A_g 
            z = z - (1 / (p + 1)) * A_g
            A_g = A_g.reshape(1,-1)
            L = A_g.T
            I_k_vec = s * (2 / (p + 1)) * np.ones(1)
        else:
            A_g = compute_A_g(r_init * (s ** k), L, I_k_vec, g)
            den = np.sqrt(g.dot(A_g))
            A_g = (1 / den) * A_g
            z = z - (1 / (p + 1)) * A_g
            A_g = A_g.reshape(1,-1)
            L = np.concatenate((L, A_g.T), axis=1)
            I_k_vec = np.insert(I_k_vec, 0, (s ** (k + 1)) * (2 / (p + 1)))
        k += 1
 
    g = compute_subgradient(z, X, y, lmbda, mu, loss, penalty, intercept)

    end = time.time()
    print('Time to compute z, A and g:', end - start)
    return z, scaling, L, I_k_vec, g


def compute_test_accelerated(D_i, y_i, z, scaling, L, I_k_vec, g, classification, cut):
    A_g = compute_A_g(scaling, L, I_k_vec, g)
    A_D_i = compute_A_g(scaling, L, I_k_vec, D_i)
    if classification:
        if cut:
            nu = - g.dot(A_D_i) / g.dot(A_g)
            if nu < 0:
                test = D_i.dot(z) - np.sqrt(D_i.dot(A_D_i))
            else:
                new_D_i = D_i + nu * g
                A_new_D_i = compute_A_g(scaling, L, I_k_vec, new_D_i)
                mu = np.sqrt(new_D_i.dot(A_new_D_i))
                body = D_i.dot(A_new_D_i) / mu
                test = D_i.dot(z) - body
        else:
            test = D_i.dot(z) - np.sqrt(D_i.dot(A_D_i))

    else:
        if cut:
            nu = g.dot(A_D_i) / g.dot(A_g)
            if nu < 0:
                test = D_i.dot(z) + np.sqrt(D_i.dot(A_D_i)) - y_i
            else:
                new_D_i = D_i - nu * g
                A_new_D_i = compute_A_g(scaling, L, I_k_vec, new_D_i)
                mu = np.sqrt(new_D_i.dot(A_new_D_i))
                body = D_i.dot(A_new_D_i) / mu
                test = D_i.dot(z) + body - y_i
        else:
            test = D_i.dot(z) + np.sqrt(D_i.dot(A_D_i)) - y_i
    return test


def test_dataset_accelerated(D, y, lmbda, mu, classification, loss, penalty, n_steps, intercept, cut):
    if intercept:
        X = np.concatenate((D, np.ones(D.shape[0]).reshape(1,-1).T), axis=1)
        z_init = np.zeros(X.shape[1] + 1)
        r_init = np.sqrt(X.shape[1] + 1)
    else:
        X = D
        z_init = np.zeros(X.shape[1])
        r_init = np.sqrt(X.shape[1])
    z, scaling, L, I_k_vec, _ = iterate_ellipsoids_accelerated(X, y, z_init, r_init, lmbda, mu, 
        loss, penalty, n_steps, intercept)
    results = np.zeros(X.shape[0])
    g = compute_subgradient(z, X, y, lmbda, mu, loss, penalty, intercept)
    start = time.time()
    if classification:
        for i in range(X.shape[0]):
            x_i = y[i] * X[i]
            test = compute_test_accelerated(x_i, None, z, scaling, L, I_k_vec, g, 
                classification, cut)
            if test > mu:
                results[i] = 1
    else:
        for i in range(X.shape[0]):
            test_1 = compute_test_accelerated(X[i], y[i], z, scaling, L, I_k_vec, g, 
                classification, cut)
            test_2 = compute_test_accelerated(-X[i],-y[i],z,scaling, L, I_k_vec, g, 
                classification, cut)
            if test_1 < mu and test_2 < mu:
                results[i] = 1
    end = time.time()
    print('Time to test the entire dataset:', end - start)
    return results


def rank_dataset_accelerated(D, y, z, scaling, L, I_k_vec, g, lmbda, mu, classification, 
    loss, penalty, intercept, cut):
    if intercept:
        X = np.concatenate((D, np.ones(D.shape[0]).reshape(1,-1).T), axis=1)
    else:
        X = D
    scores = np.zeros(X.shape[0])
    
    start = time.time()

    if classification:
        for i in range(X.shape[0]):
            x_i = y[i] * X[i]
            scores[i] = - compute_test_accelerated(x_i, None, z, scaling, L,
             I_k_vec, g, classification, cut)
    else:
        for i in range(X.shape[0]):
            test_1 = compute_test_accelerated(X[i], y[i], z, scaling, L, I_k_vec, g,
             classification, cut)
            test_2 = compute_test_accelerated(-X[i], -y[i], z, scaling, L, I_k_vec, g,
                classification, cut)
            scores[i] = np.maximum(test_1, test_2)
    end = time.time()
    print('Time to rank the entire dataset:', end - start)
    return scores


def compute_test(D_i, y_i, z, A, g, classification, cut):
    A_g = A.dot(g)
    A_D_i = A.dot(D_i)
    if classification:
        if cut:
            nu = - g.dot(A_D_i) / g.dot(A_g)
            if nu < 0:
                test = D_i.dot(z) - np.sqrt(D_i.dot(A_D_i))
            else:
                new_D_i = D_i + nu * g
                A_new_D_i = A.dot(new_D_i)
                mu = np.sqrt(new_D_i.dot(A_new_D_i)) / 2
                body = D_i.dot(A_new_D_i) / (2 * mu)
                test = D_i.dot(z) - body
        else:
            test = D_i.dot(z) - np.sqrt(D_i.dot(A_D_i))
        
    else:
        if cut:
            nu = g.dot(A_D_i) / g.dot(A_g)
            if nu < 0:
                test = D_i.dot(z) + np.sqrt(D_i.dot(A_D_i)) - y_i
            else:
                new_D_i = D_i - nu * g
                A_new_D_i = A.dot(new_D_i)
                mu = np.sqrt(new_D_i.dot(A_new_D_i)) / 2
                body = D_i.dot(A_new_D_i) / (2 * mu)
                test = D_i.dot(z) + body - y_i
        else:
            test = D_i.dot(z) + np.sqrt(D_i.dot(A_D_i)) - y_i
    return test


def rank_dataset(D, y, z, A, g, lmbda, mu, classification, loss, penalty, intercept, cut):
    if intercept:
        X = np.concatenate((D, np.ones(D.shape[0]).reshape(1,-1).T), axis=1)
    else:
        X = D
    scores = np.zeros(X.shape[0])
    
    start = time.time()

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
    end = time.time()
    print('Time to rank the entire dataset:', end - start)
    return scores

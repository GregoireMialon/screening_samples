import numpy as np
import matplotlib.pyplot as plt
import sklearn
from scipy.sparse import rand
import random
import time
from sklearn.model_selection import train_test_split

def compute_loss_gradient(u, mu):
    g = np.zeros(u.size)
    for i in range(u.size):
        if np.abs(u[i]) > mu:
            g[i] = 2 * (u[i] - np.sign(u[i]) * mu)
    return g

def compute_l1_subgradient(u):
    g = np.zeros(u.size)
    for i in range(u.size):
        if u[i] != 0:
            g[i] = np.sign(u[i])
        else:
            g[i] = 2 * np.random.rand(1)[0] - 1
    return g

def compute_subgradient(x, D, y, lmbda, mu, penalty):
    output = D.dot(x) - y
    g_1 = compute_loss_gradient(output, mu)
    #g_1 = np.transpose(D).dot(g_1)
    g_1 = (1 / (2 * D.shape[0])) * np.transpose(D).dot(g_1)
    if penalty == 'l2':
        g_2 = 2 * x
    elif penalty == 'l1':   
        g_2 = compute_l1_subgradient(x)
    g = g_1 + lmbda * g_2
    return g

def compute_A_g(scaling, L, I_k_vec, g):
    
    L_g = np.dot(np.transpose(L), g)
    I_k_L = np.multiply(I_k_vec, L_g)
    A_g = scaling * g - L.dot(I_k_L)
    return A_g

def iterate_ellipsoids_accelerated_(D, y, z_init, r_init, lmbda, mu, penalty, n_steps):
    start = time.time()
    k = 0
    z = z_init
    p = z_init.size
    s = p ** 2 / (p ** 2 - 1)
    scaling = r_init * (s ** (n_steps))
    I = np.diag(scaling * np.ones(p))
     
    while k < n_steps:
        g = compute_subgradient(z, D, y, lmbda, mu, penalty)
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
 
    end = time.time()
    print('Time to compute z and A:', end - start)
    return z, scaling, L, I_k_vec

def compute_test_with_linear_ineq_accelerated(D_i, y_i, z, scaling, L, I_k_vec, g):
    A_g = compute_A_g(scaling, L, I_k_vec, g)
    A_D_i = compute_A_g(scaling, L, I_k_vec, D_i)
    nu = g.dot(A_D_i) / g.dot(A_g)
    if nu < 0:
        test = D_i.dot(z) + np.sqrt(D_i.dot(A_D_i)) - y_i
    else:
        new_D_i = D_i - nu * g
        A_new_D_i = compute_A_g(scaling, L, I_k_vec, new_D_i)
        mu = np.sqrt(new_D_i.dot(A_new_D_i)) / 2
        body = D_i.dot(A_new_D_i) / (2 * mu)
        test = D_i.dot(z) + body - y_i
    return test

def test_dataset_accelerated(D, y, lmbda, mu, penalty, n_steps):
    z_init = np.zeros(D.shape[1])
    r_init = np.sqrt(D.shape[1])
    z, scaling, L, I_k_vec = iterate_ellipsoids_accelerated_(D, y, z_init, r_init, lmbda, mu, 
        penalty, n_steps)
    results = np.zeros(D.shape[0])
    g = compute_subgradient(z, D, y, lmbda, mu, penalty)
    start = time.time()
    for i in range(D.shape[0]):
        test_1 = compute_test_with_linear_ineq_accelerated(D[i], y[i], z, scaling, L, I_k_vec, g)
        test_2 = compute_test_with_linear_ineq_accelerated(-D[i],-y[i],z,scaling, L, I_k_vec, g)
        if test_1 < mu and test_2 < mu:
            results[i] = 1
    end = time.time()
    print('Time to test the entire dataset:', end - start)
    return results

def rank_dataset_accelerated(D, y, z, scaling, L, I_k_vec, lmbda, mu, penalty):
    scores = np.zeros(D.shape[0])
    g = compute_subgradient(z, D, y, lmbda, mu, penalty)
    start = time.time()
    for i in range(D.shape[0]):
        test_1 = compute_test_with_linear_ineq_accelerated(D[i], y[i], z, scaling, L, I_k_vec, g)
        test_2 = compute_test_with_linear_ineq_accelerated(-D[i],-y[i],z,scaling, L, I_k_vec, g)
        scores[i] = np.maximum(test_1, test_2)
    end = time.time()
    print('Time to rank the entire dataset:', end - start)
    return scores
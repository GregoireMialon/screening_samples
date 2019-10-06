import datetime
import itertools
import os
import numpy as np
from screening.pyapt import apt_run
from screening.settings import LOGS_PATH


mus = [1.0]
lmbdas = [0.0001, 0.001, 0.01, 0.1, 1.0]
n_ellipsoid_stepss = [1920]
sizes = [60000]
sub_ells = [100]
init_dgs = [(1, 5), (5, 9), (10, 14), (15, 19), (20, 24), (25, 29), (30, 34), (35, 39)]

parallel_args = []
for (mu, lmbda, n_ellipsoid_steps, size, sub_ell, init_dg) in itertools.product(
    mus, lmbdas, n_ellipsoid_stepss, sizes, sub_ells, init_dgs):
	args = {
		'mu': mu,
		'lmbda': lmbda,
		'n_ellipsoid_steps': n_ellipsoid_steps,
		'dataset': 'mnist',
		'size': size,
		'redundant': 0,
		'penalty': 'l2',
		'nb_delete_steps': 8,
		'nb_exp': 3,
		'nb_test': 2,
		'classif_score': True,
		'loss': 'squared_hinge',
		'classification': True,
		'better_init': init_dg[0],
		'get_ell_from_subset': sub_ell,
		'use_sphere': True,
		'cut': True,
		'n_epochs_dg': init_dg[1],
		'zoom': 0
		}
	parallel_args.append(args)




cd_project_folder = os.path.join('cd ', LOGS_PATH)
conda_activate = 'conda activate yana'
conda_var = ('CONDA_PREFIX',
	 '/sequoia/data1/gmialon/miniconda/envs/yana')
paths = ('PATH', '/sequoia/data1/gmialon/miniconda/etc/profile.d/conda.sh')
shell_vars = [conda_var, paths]
prepend_cmd = ['source ~/.bashrc', cd_project_folder, conda_activate, 'which python']

queues = ['all.q', 'bigmem.q', 'goodboy.q']
python_cmd = '-m screening.experiment'
apt_run(
        python_cmd,
        parallel_args=parallel_args,
        queues=queues,
        shell_var=shell_vars,
        prepend_cmd=prepend_cmd,
        group_by=1,
        memory=20000,
        memory_hard=20000,
        max_parrallel_jobs=40,
        multi_threading=1)

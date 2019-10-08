import datetime
import itertools
import os
import numpy as np
from screening.pyapt import apt_run
from screening.settings import LOGS_PATH


mus = [1.0]
lmbdas = [0.0001, 0.001, 0.01, 0.1, 1.0]
n_ellipsoid_stepss = [10, 100, 1000, 3000]
sizes = [604388]
sub_ells = [100]
init_rads = [(10, 1000), (10, 100), (20, 100), (20, 10), (30, 10), (30, 1)]

parallel_args = []
for (mu, lmbda, n_ellipsoid_steps, size, sub_ell, init_rad) in itertools.product(
    mus, lmbdas, n_ellipsoid_stepss, sizes, sub_ells, init_rads):
	args = {
		'mu': mu,
		'lmbda': lmbda,
		'n_ellipsoid_steps': n_ellipsoid_steps,
		'dataset': 'svhn',
		'size': size,
		'redundant': 0,
		'penalty': 'l1',
		'nb_delete_steps': 0,
		'nb_exp': 3,
		'nb_test': 2,
		'classif_score': True,
		'loss': 'safe_logistic',
		'classification': True,
		'better_init': init_rad[0],
        'better_radius': init_rad[1],
		'get_ell_from_subset': sub_ell,
		#'use_sphere': True,
		'cut': True,
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

queues = ['all.q', 'goodboy.q']
python_cmd = '-m screening.experiment_reg'
apt_run(
        python_cmd,
        parallel_args=parallel_args,
        queues=queues,
        shell_var=shell_vars,
        prepend_cmd=prepend_cmd,
        group_by=1,
        memory=50000,
        memory_hard=50000,
        max_parrallel_jobs=40,
        multi_threading=1)

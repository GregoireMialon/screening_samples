import datetime
import itertools
import os
import numpy as np
from screening.pyapt import apt_run
from screening.settings import LOGS_PATH


mus = [1.0]
lmbdas = [0.001, 0.01, 0.1]
n_ellipsoid_stepss = [10, 100, 1000]
better_inits = [0, 1, 15]
sizes = [604388]
sub_ells = [0, 100, 1000]
n_dgs = [2, 16]

parallel_args = []
for (mu, lmbda, n_ellipsoid_steps, better_init, size, sub_ell, n_dg) in itertools.product(
    mus, lmbdas, n_ellipsoid_stepss, better_inits, sizes, sub_ells, n_dgs):
	args = {
		'mu': mu,
		'lmbda': lmbda,
		'n_ellipsoid_steps': n_ellipsoid_steps,
		'dataset': 'svhn',
		'size': size,
		'redundant': 0,
		'penalty': 'l2',
		'nb_delete_steps': 12,
		'nb_exp': 3,
		'nb_test': 2,
		'classif_score': True,
		'loss': 'squared_hinge',
		'classification': True,
		'better_init': better_init,
		'get_ell_from_subset': sub_ell,
		'use_sphere': True,
		#'cut': True,
		'n_epochs_dg': n_dg,
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
        memory=30000,
        memory_hard=50000,
        max_parrallel_jobs=5,
        multi_threading=1)

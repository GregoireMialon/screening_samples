import datetime
import itertools
import os

import pyapt
import numpy as np


mus = [0, 0.001, 0.01, 0.1, 1.0, 10.0]
lmbdas = [0.0001, 0.001, 0.01, 0.1, 1.0]
n_ellipsoid_stepss = [1, 10, 50]
better_inits = [0, 1, 10, 100]
better_radiuss = [0, 1, 10, 100]
sizes = [100]

parallel_args = []
for (mu, lmbda, n_ellipsoid_steps, better_init, better_radius, size) in itertools.product(
    mus, lmbdas, n_ellipsoid_stepss, better_inits, better_radiuss, sizes):
	args = {
		'mu': mu,
		'lmbda': lmbda,
		'n_ellipsoid_steps': n_ellipsoid_steps,
		'dataset': 'fixed_synthetic_n_100_p_2_s_0.5',
		'size': sizes,
		'redundant': 0,
		'penalty': 'l1',
		'nb_delete_steps':10,
		'nb_exp': 10,
		'nb_test': 3,
		'loss': 'truncated_squared',
		'better_init': better_init,
		'better_radius': better_radius,
		'zoom': 0,
		#'cut': True,
		}
	parallel_args.append(args)




cd_project_folder = 'cd /sequoia/data1/gmialon/screening'
conda_activate = 'conda activate yana'
conda_var = ('CONDA_PREFIX',
	 '/sequoia/data1/gmialon/miniconda/envs/yana')
paths = ('PATH', '/sequoia/data1/gmialon/miniconda/etc/profile.d/conda.sh')
shell_vars = [conda_var, paths]
prepend_cmd = ['source ~/.bashrc', cd_project_folder, conda_activate, 'which python']

queues = ['all.q', 'bigmem.q', 'goodboy.q']
python_cmd = 'experiment.py'
pyapt.apt_run(
        python_cmd,
        parallel_args=parallel_args,
        queues=queues,
        shell_var=shell_vars,
        prepend_cmd=prepend_cmd,
        group_by=1,
        memory=6000,
        memory_hard=20000,
        max_parrallel_jobs=10,
        multi_threading=1)

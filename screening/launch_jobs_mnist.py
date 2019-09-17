import datetime
import itertools
import os
import numpy as np
from screening.pyapt import apt_run
from screening.settings import LOGS_PATH


mus = [0.1, 0.5, 1]
lmbdas = [0, 0.0001, 0.001] #0.01, 0.1, 1, 10] 
n_ellipsoid_stepss = [10, 100, 1000, 10000]
better_inits = [0, 1, 10, 100]
better_radiuss = [0, 10, 100, 1000]
sizes = [60000]
sub_ells = [0]

parallel_args = []
for (mu, lmbda, n_ellipsoid_steps, better_init, better_radius, size, sub_ell) in itertools.product(
    mus, lmbdas, n_ellipsoid_stepss, better_inits, better_radiuss, sizes, sub_ells):
	args = {
		'mu': mu,
		'lmbda': lmbda,
		'n_ellipsoid_steps': n_ellipsoid_steps,
		'dataset': 'mnist',
		'size': size,
		'redundant': 0,
		'penalty': 'l2',
		'nb_delete_steps': 10,
		'nb_exp': 3,
		'nb_test': 3,
		'classif_score': True,
		'loss': 'safe_logistic',
		'classification': True,
		'better_init': better_init,
		'better_radius': better_radius,
		'zoom': 0,
		#'cut': False,
		'get_ell_from_subset': sub_ell
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
        max_parrallel_jobs=8,
        multi_threading=1)

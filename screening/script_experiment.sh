#!/bin/bash
export SGE_ROOT=/cm/shared/apps/sge/2011.11p1;
COUNTERJOBS=`/cm/shared/apps/sge/2011.11p1/bin/linux-x64/qstat -u gmialon | wc -l`
memory=3000M
vmemory=3000M
nb_node=6


for lmbda in 0.0001 0.001 0.01 0.1 1.0; do
  for n_ellipsoid_steps in 100 1000 10000 100000; do
    for mu in 0.01 0.1 1 10; do
        COUNTERJOBS=`qstat -u gmialon | wc -l`
        echo "  job count : ${COUNTERJOBS}"
        while [ $COUNTERJOBS -ge $nb_node ]; do
            sleep 10
            COUNTERJOBS=`qstat -u gmialon | wc -l`
        done

        NAME="screening_leukemia_$lmbda_$n_ellipsoid_steps_$mu"
        echo "#$ -l mem_req=${memory},h_vmem=${vmemory}
              #$ -pe serial 1
              #$ -q all.q
              #$ -e /sequoia/data1/gmialon/screening/logs/$NAME.err
              #$ -o /sequoia/data1/gmialon/screening/logs/$NAME.out
              #$ -N $NAME

              echo 00
              export MKL_NUM_THREADS=1
              export NUMEXPR_NUM_THREADS=1
	      export PATH='/sequoia/data1/gmialon/miniconda/etc/profile.d/conda.sh'
	      source ~/.bashrc
	      conda activate yana
              python /sequoia/data1/gmialon/screening/experiment.py --dataset 'leukemia' --nb_delete_steps 10 --lmbda $lmbda --mu $mu --penalty 'l1' --n_ellipsoid_steps $n_ellipsoid_steps --nb_test 1 --cluster
              " | sed "s/^ *//g" > /sequoia/data1/gmialon/screening/cluster_script/logs/$NAME.pbs

        qsub /sequoia/data1/gmialon/screening/cluster_script/logs/$NAME.pbs
    done
  done
done

>#!/bin/bash

export SGE_ROOT=/cm/shared/apps/sge/2011.11p1;

COUNTERJOBS=`/cm/shared/apps/sge/2011.11p1/bin/linux-x64/qstat -u mabarre | wc -l`

memory=3500M

vmemory=3800M

nb_node=6



cd /sequoia/data1/mabarre/mstar/



#for mask in 0 1; do

for mask in 0 1; do

  #for k in 2 ; do

  for Image in 'boat' 'barbara' 'cameraman' 'fish' 'house' 'jetplane' 'lake' 'lena_gray_512' 'livingroom' 'mandril_gray' 'people' 'peppers_gray' 'pirate' 'walkbridge'; do

   #for Image in 'jetplane'; do

    for k in 2 4 6 8; do

        COUNTERJOBS=`qstat -u mabarre | wc -l`

        echo "  job count : ${COUNTERJOBS}"

        while [ $COUNTERJOBS -ge $nb_node ]; do

            sleep 10

            COUNTERJOBS=`qstat -u mabarre | wc -l`

        done



        NAME="inpainting_$Image"_"$k"_"$mask"_"1e8"

        echo "#$ -l mem_req=${memory},h_vmem=${vmemory}

              #$ -pe serial 1

              #$ -q all.q

              #$ -e /sequoia/data1/mabarre/mstar/logs/$NAME.err

              #$ -o /sequoia/data1/mabarre/mstar/logs/$NAME.out

              #$ -N $NAME



              echo 00

              export MKL_NUM_THREADS=1

              export NUMEXPR_NUM_THREADS=1

              export OMP_NUM_THREADS=1

              /sequoia/data2/matlab/matlab-2017a/linux64/bin/matlab -singleCompThread -nodisplay -nodesktop -nosplash -nojvm -r \"cd '/sequoia/data1/mabarre/mstar/';ImageName = '${Image}';k = ${k};mk = ${mask};Inpainting\"

              " | sed "s/^ *//g" > /sequoia/data1/mabarre/mstar/cluster_script/logs/$NAME.pbs



        qsub /sequoia/data1/mabarre/mstar/cluster_script/logs/$NAME.pbs

    done

  done

done

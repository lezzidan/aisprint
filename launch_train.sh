#!/bin/sh
export COMPSS_PYTHON_VERSION=None
#module load dislib/0.6.4-COMPSs_2.9-qr
module load dislib/0.7.0
module load COMPSs/2.10
module load python/3.7.4
export PYTHONPATH=$PWD:$PYTHONPATH

export ComputingUnits=8

worker_working_dir=/gpfs/scratch/bsc19/bsc19275/workdir
base_log_dir=/gpfs/scratch/bsc19/bsc19275/

#queue=bsc_cs
queue=debug
time_limit=2*60
num_nodes=2

# log level off for better performance
enqueue_compss --qos=$queue \
 --log_level=off \
 --job_name=csvm_small \
 --worker_in_master_cpus=0 \
 --jvm_master_opts="-Xms16000m,-Xmx50000m,-Xmn1600m" \
 --max_tasks_per_node=48 \
 --exec_time=$time_limit \
 --num_nodes=$num_nodes \
 --base_log_dir=${base_log_dir} \
 --worker_working_dir=${worker_working_dir} \
 train_csvm_dislib.py /home/bsc19/bsc19275/csvm/models/saved.sav



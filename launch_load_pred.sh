#!/bin/sh
export COMPSS_PYTHON_VERSION=None
#module load dislib/0.6.4-COMPSs_2.9-qr
module load dislib/master
module load COMPSs/2.10
module load python/3.7.4
export PYTHONPATH=$PWD:$PYTHONPATH

worker_working_dir=/gpfs/scratch/bsc19/bsc19756/aisprint_other_params/KNN/1_worker
base_log_dir=/gpfs/scratch/bsc19/bsc19756/aisprint_other_params/KNN/1_worker
export ComputingUnits=4

#queue=bsc_cs
queue=debug

time_limit=120

num_nodes=2

# log level off for better performance
enqueue_compss --qos=$queue \
 --log_level=off \
 --cpu_affinity="dlb" \
 --job_name=csvm_small \
 --worker_in_master_cpus=0 \
 --jvm_master_opts="-Xms16000m,-Xmx50000m,-Xmn1600m" \
 --max_tasks_per_node=48 \
 --exec_time=$time_limit \
 --num_nodes=$num_nodes \
 --base_log_dir=${base_log_dir} \
 --worker_working_dir=${worker_working_dir} \
 --pythonpath=/gpfs/scratch/bsc19/bsc19756/aisprint_other_params/KNN/1_worker \
 load_knn_predict.py knn_1 pickle /gpfs/scratch/bsc19/bsc19756/aisprint_other_params/PCA/balanced_training2017/ 250 250 250



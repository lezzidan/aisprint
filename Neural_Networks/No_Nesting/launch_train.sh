#!/bin/sh
source  /home/bsc19/bsc19756/MLESmap/eddl-1.0.4b/pyeddl-1.2.0/pyenv/bin/activate

export CPATH="/apps/EIGEN/3.3.9/GCC/include/eigen3:${CPATH}"
export EDDL_WITH_CUDA="true"
export EDDL_DIR=$HOME/MLESmap/eddl-1.0.4b
export ComputingUnits=40
export ComputingGPUs=1
# module load gcc/7.3.0 python/3.8.2
# export COMPSS_PYTHON_VERSION=3.8
export COMPSS_PYTHON_VERSION=3.7-ML
module load COMPSs/TrunkFVN
module load openmpi/3.0.0 protobuf/3.14.0 eigen/3.3.9
export PYTHONPATH=$HOME/dislib:/home/bsc19/bsc19756/MLESmap/eddl-1.0.4b/pyeddl-1.2.0/pyenv/lib/python3.7/site-packages:$PYTHONPATH
# module load gcc/7.3.0 openmpi/3.0.0 cuda/10.1


worker_working_dir=/gpfs/scratch/bsc19/bsc19756/aisprint_other_params/KNN/New_Test_Set/Neural_Networks/Definitive_Scripts
base_log_dir=/gpfs/scratch/bsc19/bsc19756/aisprint_other_params/KNN/New_Test_Set/Neural_Networks/Definitive_Scripts
export ComputingUnits=4

#queue=bsc_cs
queue=debug

time_limit=40

num_nodes=2

# log level off for better performance
enqueue_compss --qos=$queue \
 --log_level=off \
 --cpu_affinity="dlb" \
 --job_name=csvm_small \
 --worker_in_master_cpus=0 \
 --jvm_master_opts="-Xms16000m,-Xmx50000m,-Xmn1600m" \
 --max_tasks_per_node=160 \
 --exec_time=$time_limit \
 --num_nodes=$num_nodes \
 --base_log_dir=${base_log_dir} \
 --worker_working_dir=${worker_working_dir} \
 --pythonpath=/gpfs/scratch/bsc19/bsc19756/aisprint_other_params/KNN/New_Test_Set/Neural_Networks/Definitive_Scripts \
 train_cnn_2_classes_kfold_4_work_1_gpu.py



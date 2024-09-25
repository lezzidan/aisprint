# aisprint
AI-SPRINT

## Reexecution instructions

To reexecute some of the experiments it is required to download the PhysioNet dataset. Select the desired python script and the corresponding launching script.
In the launching script it is required to specify the location of the dataset. The location of the dataset should be specified as the third entry argument to the python script.
Other aspect to modify of the launching script is the number of nodes, to evaluate the scalability of the algorithms the number of nodes used should range from 2 to 17 (there is always 1 node that works as master node). This data is contained inside the different bash scripts. For example, the number of nodes is defined as a variable called num_nodes that should be modified previously to the execution of the bash script.

In the case of local execution, this scripts are not going to work, the last section explains how to execute the experiments in local.

## Execution Machines
The machine learning algorithms were executed in the supercomputer MareNostrum4. If the scripts are going to be executed on another supercomputer, it may be required to load other software modules.
The Neural Networks scripts are specific for the Power-9 Supercomputer. In another infrastructure it may be required to load other software modules.

## Launching scripts
In order to launch an experiment it is required only to place the python and the .sh script in the correct directory and launch the .sh script. An example would be the following command:

$ ./launch_train_rf.sh

## Local Execution
For the local execution the command enqueue_compss will not work. Instead using enqueue_compss it is required to use the command runcompss. The different script can be executed using a command similar to the next one:

$ runcompss --python_interpreter=python3 train_knn_kfold.py knn_model pickle $PATH_TO_DATASET/balanced_training2017/ 250 250 250

Then, for the configuration of the COMPSs runtime, refer to the COMPSs [COMPSs documentation]https://www.genome.gov/(https://compss-doc.readthedocs.io/en/stable/Sections/03_Execution_Environments/03_Deployments/01_Master_worker/01_Local.html)

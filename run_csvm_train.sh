#!/bin/bash

/etc/init.d/ssh start
runcompss \
    --pythonpath=$(pwd) \
    --python_interpreter=python3 \
    /home/user/train_csvm_dislib.py $1 $2 $3 $4 $5 $6 &> >(tee output.log)

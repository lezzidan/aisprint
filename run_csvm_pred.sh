#!/bin/bash

/etc/init.d/ssh start
runcompss \
    --pythonpath=$(pwd) \
    --python_interpreter=python3 \
    /home/user/pred_csvm_dislib.py $1 $2 &> >(tee output.log)

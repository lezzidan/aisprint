#!/bin/bash

/etc/init.d/ssh start
runcompss \
    --pythonpath=$(pwd) \
    --python_interpreter=python3 \
    /home/user/valid_csvm_dislib.py $1 &> >(tee output.log)

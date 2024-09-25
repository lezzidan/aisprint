FROM compss/compss:latest

EXPOSE 43000-44000

COPY train_csvm_dislib.py /home/user/
COPY pred_csvm_dislib.py /home/user/
COPY valid_csvm_dislib.py /home/user/

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir graphviz && \
# Remove enum34 since causes issues with dislib required libraries
    python3 -m pip uninstall -y enum34 && \
# Dislib requirements:
    python3 -m pip install scikit-learn==0.22.1 && \
    python3 -m pip install scipy>=1.3.0 && \
    python3 -m pip install numpy==1.19.5 && \
    python3 -m pip install numpydoc>=0.8.0 && \
    python3 -m pip install cvxpy>=1.1.5

# Install dislib
RUN python3 -m pip install dislib
#RUN cd /home/user/dislib && \
#    python3 setup.py install

RUN sed -i '/<ComputingUnits>4<\/ComputingUnits>/c<ComputingUnits>2<\/ComputingUnits>' /opt/COMPSs/Runtime/configuration/xml/resources/default_resources.xml

COPY run_csvm_train.sh /home/user
COPY run_csvm_pred.sh /home/user
COPY run_csvm_valid.sh /home/user

# Expose SSH port and run SSHD
EXPOSE 22

#CMD ["/home/user/run_csvm_train.sh /home/user/models/saved.sav"]

import scipy.io
import numpy as np
import glob
import time
import dislib as ds
from dislib.classification import CascadeSVM
from dislib.data.array import Array

from pycompss.api.api import compss_barrier
from pycompss.api.api import compss_wait_on
from scipy import signal
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree._tree import Tree as SklearnTree
from sklearn.svm import SVC as SklearnSVC
from sklearn.tree import DecisionTreeClassifier as SklearnDTClassifier
from sklearn.tree import DecisionTreeRegressor as SklearnDTRegressor
import pickle
from scipy.sparse import csr_matrix
from scipy import sparse as sp
from collections import Counter
import random
import json
import os


def zero_pad(data, length):
    extended = np.zeros(length)
    signal_length = np.min([length, data.shape[0]])
    extended[:signal_length] = data[:signal_length]
    return extended

def spectrogram(data, fs=300, nperseg=64, noverlap=32):
    f, t, Sxx = signal.spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    Sxx = np.transpose(Sxx, [0, 2, 1])
    Sxx = np.abs(Sxx)
    mask = Sxx > 0
    Sxx[mask] = np.log(Sxx[mask])
    return f, t, Sxx

def load_n_preprocess(dataDir):
    
    max_length = 61
    freq = 300

    ## Loading labels and time serie signals (A and N)
    import csv
    csvfile = list(csv.reader(open(dataDir+'REFERENCE.csv')))

    files = [dataDir+i[0]+".mat" for i in csvfile]
    dataset = np.zeros((len(files),18810))
    count = 0
    for f in files:
        mat_val = zero_pad(scipy.io.loadmat(f)['val'][0], length=max_length * freq)
        sx = spectrogram(np.expand_dims(mat_val, axis=0))[2] # generate spectrogram
        sx_norm = (sx - np.mean(sx)) / np.std(sx) # normalize the spectrogram
        dataset[count,] = sx_norm.flatten()
        count += 1
   
    labels = np.zeros((dataset.shape[0],1))
    classes = ['A','N', 'O', '~']
    for row in range(len(csvfile)):
        labels[row, 0] = 0 if classes.index(csvfile[row][1]) == 0 else 1 if classes.index(csvfile[row][1]) == 1 else 2 if classes.index(csvfile[row][1]) == 2 else 3

    return(dataset,labels)

if __name__ == "__main__":
    args = sys.argv[1:]
    start_time = time.time()
    model_saved = args[0]
    format_model = args[1]
    dataset_to_use = args[2]
    block_size_x = (int(args[3]), int(args[4]))
    block_size_y = int(args[5])
    seed = 1234
    csvm = CascadeSVM(kernel='rbf', c=1, gamma='auto', tol=1e-2, random_state=seed)
    
    
    X_test, y_test = load_n_preprocess(dataset_to_use)
    print([X_test.shape, y_test.shape])
    print(Counter(y_test.flatten()))
    load_time = time.time()
    csvm.load_model(model_saved, load_format=format_model)
    #model = load_ds_csvm_model(model_saved)
    

    x_t = ds.array(X_test, block_size=block_size_x)
    y_t = ds.array(y_test, block_size=(block_size_y, 1))

    x_test_shuffle, y_test_shuffle = ds.utils.base.shuffle(x_t,y_t)

    labels_pred = csvm.predict(x_test_shuffle)
    compss_barrier()


    merged_labels = labels_pred.collect()
    merged_y_test = y_test_shuffle.collect()
    print("Labels predict: ", labels_pred)
    print("Blocks predict: ", merged_labels)
    cm = confusion_matrix(merged_y_test, merged_labels)
    print(cm)
    acc= accuracy_score(merged_y_test,merged_labels)
    print(acc)
    print (classification_report(merged_y_test, merged_labels))

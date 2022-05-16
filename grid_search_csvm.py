import sys

import scipy.io
import numpy as np
import glob
import time
import dislib as ds
from dislib.classification import CascadeSVM
from dislib.data.array import Array
from dislib.model_selection import GridSearchCV
from sklearn.datasets import make_classification
import pandas as pd
from pycompss.api.api import compss_wait_on, compss_barrier
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
    csvfile = list(csv.reader(open(dataDir + 'REFERENCE.csv')))

    files = [dataDir + i[0] + ".mat" for i in csvfile]
    dataset = np.zeros((len(files), 18810))
    count = 0
    for f in files:
        mat_val = zero_pad(scipy.io.loadmat(f)['val'][0], length=max_length * freq)
        sx = spectrogram(np.expand_dims(mat_val, axis=0))[2]  # generate spectrogram
        sx_norm = (sx - np.mean(sx)) / np.std(sx)  # normalize the spectrogram
        dataset[count,] = sx_norm.flatten()
        count += 1

    labels = np.zeros((dataset.shape[0], 1))
    classes = ['A', 'N', 'O', '~']
    for row in range(len(csvfile)):
        labels[row, 0] = 0 if classes.index(csvfile[row][1]) == 0 else 1 if classes.index(
            csvfile[row][1]) == 1 else 2 if classes.index(csvfile[row][1]) == 2 else 3

    return (dataset, labels)


if __name__ == "__main__":
    args = sys.argv[1:]
    start_time = time.time()
    dataset_to_use = args[0]
    block_size_x = (int(args[1]), int(args[2]))
    block_size_y = int(args[3])
    seed = 1234
    parameters = {'kernel': ('rbf', 'linear'),
                  'c': (0.75, 1.0, 2.0),
                  'tol': (1e-3, 1e-4, 1e-5)}
    csvm = CascadeSVM()
    # csvm = CascadeSVC(fold_size=500)

    X_train, y_train = load_n_preprocess(dataset_to_use)
    print([X_train.shape, y_train.shape])
    print(Counter(y_train.flatten()))
    idx = random.sample(list(np.where(y_train == 1.0)[0]),
                        Counter(y_train.flatten())[1.0] - Counter(y_train.flatten())[0.0])
    y_train = np.delete(y_train, idx, axis=0)
    X_train = np.delete(X_train, idx, axis=0)
    print(Counter(y_train.flatten()))
    load_time = time.time()

    x = ds.array(X_train, block_size=block_size_x)
    y = ds.array(y_train, block_size=(block_size_y, 1))

    searcher = GridSearchCV(csvm, parameters, cv=5)
    np.random.seed(0)
    searcher.fit(x, y)
    print(searcher.cv_results_['params'])
    print(searcher.cv_results_['mean_test_score'])
    pd_df = pd.DataFrame.from_dict(searcher.cv_results_)
    print(pd_df[['params', 'mean_test_score']])
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None):
        print(pd_df)

    print('best_estimator')
    print(searcher.best_estimator_)
    print(searcher.best_score_)
    print(searcher.best_params_)
    print(searcher.best_index_)
    print(searcher.scorer_)
    print(searcher.n_splits_)



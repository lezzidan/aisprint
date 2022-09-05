import sys

import scipy.io
import numpy as np
import time

import dislib as ds
from dislib.classification import CascadeSVM
from dislib.decomposition import PCA
from dislib.preprocessing.standard_scaler import StandardScaler

from pycompss.api.api import compss_wait_on, compss_barrier
from scipy import signal

from collections import Counter
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from knn.base import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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

def main():
    args = sys.argv[1:]
    start_time = time.time()
    model_saved = args[0]
    format_model = args[1]
    dataset_to_use = args[2]
    block_size_x = (int(args[3]), int(args[4]))
    block_size_y = int(args[5])
    seed = 1234
    knn = KNeighborsClassifier(n_neighbors=5)

    # X = ds.random_array((2852, 716), block_size_x)
    X_train, y_train = load_n_preprocess(dataset_to_use)
    X = ds.array(X_train, block_size_x)
    Y = ds.array(y_train, (block_size_y, 1))

    scal = StandardScaler()
    print('SCALER TRANSFORM')
    X = scal.fit_transform(X)

    print('START FIT')
    knn.fit(X, Y)

    X_test, y_test = load_n_preprocess('/gpfs/scratch/bsc19/bsc19756/aisprint_other_params/PCA/balanced_validation2017/')
    x_t = ds.array(X_test, block_size=(250, 250))
    y_t = ds.array(y_test, block_size=(250, 1))
    X_test = scal.transform(x_t)

    print('SCORE')
    print(accuracy_score(y_t.collect(), knn.predict(X_test).collect()))
    print("FULL TIME", time.time() - start_time)


if __name__ == "__main__":
    main()
    # df = pd.read_csv('dataset.csv')
    # df_test = pd.read_csv('dataset_val.csv')
    
    # X, Y = df[df.columns[:-1]], df[df.columns[-1]]
    # X_test, Y_test = df_test[df_test.columns[:-1]], df_test[df_test.columns[-1]]

    # clf = RandomForestClassifier(n_estimators=300, n_jobs=20, class_weight='balanced')

    # clf.fit(X, Y)

    # print(clf.score(X_test, Y_test))

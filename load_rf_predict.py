import sys

import scipy.io
import numpy as np
import time

import dislib as ds
from dislib.classification import CascadeSVM, RandomForestClassifier
from dislib.decomposition import PCA
from dislib.preprocessing.standard_scaler import StandardScaler

from pycompss.api.api import compss_wait_on, compss_barrier
from scipy import signal

from collections import Counter
import pandas as pd
from sklearn.utils import shuffle
from dislib.classification import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from os import listdir
from os.path import isfile, join


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
    #csvfile = list(csv.reader(open(dataDir + 'REFERENCE.csv')))
    mat_files = [f for f in listdir(dataDir) if isfile(join(dataDir, f))]
    files = [dataDir + "/" + "201m (0).mat"]#i for i in mat_files]
    dataset = np.zeros((len(files), 18810))
    count = 0
    for f in files:
        mat_val = zero_pad(scipy.io.loadmat(f)['val'][0], length=max_length * freq)
        sx = spectrogram(np.expand_dims(mat_val, axis=0))[2]  # generate spectrogram
        sx_norm = (sx - np.mean(sx)) / np.std(sx)  # normalize the spectrogram
        dataset[count,] = sx_norm.flatten()
        count += 1

    return dataset

def main():
    args = sys.argv[1:]
    start_time = time.time()
    model_saved = args[0]
    format_model = args[1]
    dataset_to_use = args[2]
    block_size_x = (int(args[3]), int(args[4]))
    block_size_y = int(args[5])
    seed = 1234
    rf = RandomForestClassifier(n_estimators=40)

    rf.load_model(model_saved, load_format=format_model)

    X_test = load_n_preprocess(dataset_to_use)
    x_t = ds.array(X_test, block_size=block_size_x)

    print('Prediction')
    print(rf.predict(x_t).collect())
    print("Full Time", time.time() - start_time)

if __name__ == "__main__":
    main()
    # df = pd.read_csv('dataset.csv')
    # df_test = pd.read_csv('dataset_val.csv')

    # X, Y = df[df.columns[:-1]], df[df.columns[-1]]
    # X_test, Y_test = df_test[df_test.columns[:-1]], df_test[df_test.columns[-1]]

    # clf = RandomForestClassifier(n_estimators=300, n_jobs=20, class_weight='balanced')

    # clf.fit(X, Y)

    # print(clf.score(X_test, Y_test))



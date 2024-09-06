import sys

import scipy.io
import numpy as np
import time

import dislib as ds
from dislib.model_selection import KFold
from dislib.classification import CascadeSVM, RandomForestClassifier
from dislib.decomposition import PCA
from dislib.preprocessing.standard_scaler import StandardScaler

from pycompss.api.api import compss_wait_on, compss_barrier
from scipy import signal

from collections import Counter
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix

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
    rows_to_delete = []
    for row in range(len(csvfile)):
        labels[row, 0] = 0 if classes.index(csvfile[row][1]) == 0 else 1 if classes.index(
            csvfile[row][1]) == 1 else 2 if classes.index(csvfile[row][1]) == 2 else 3
        if labels[row, 0] == 2 or labels[row, 0] == 3:
            rows_to_delete.append(row)
    dataset = np.delete(dataset, rows_to_delete, 0)
    labels = np.delete(labels, rows_to_delete, 0)
    
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
    rf = RandomForestClassifier(n_estimators=40, random_state=0)

    X_train, y_train = load_n_preprocess(dataset_to_use)
    X = ds.array(X_train, block_size_x)
    Y = ds.array(y_train, (block_size_y, 1))
    cv = KFold(n_splits = 5, shuffle = True)
    scaler_time = time.time()
    total_score = 0
    predictions = []
    confusion_matrices = []
    truth_values = []
    pca = PCA()
    print([X_train.shape, y_train.shape])
    transformed_data = pca.fit_transform(X)
    variance = pca.explained_variance_.collect()
    total_variance = np.sum(variance)
    variance_until_component = 0
    for i in range(len(variance)):
        variance_until_component = variance_until_component + variance[i] / total_variance
        if variance_until_component >= 0.95:
            print("Number components kept: " +str (i))
            break
    load_time = time.time()
    compss_barrier()
    print("Scale time:" + str(time.time() - scaler_time))
    fit_time = time.time()
    for train_ds, test_ds in cv.split(transformed_data[:, 0:i], Y):
        rf.fit(train_ds[0], train_ds[1])
        truth_values.append(test_ds[1])
        predictions.append(rf.predict(test_ds[0]))
    
    for i in range(len(predictions)):
        true_values = truth_values[i].collect()
        prediction = predictions[i].collect()
        total_score += accuracy_score(true_values, prediction)
        confusion_matrices.append(confusion_matrix(true_values, prediction))
    print("Fit time", time.time() - fit_time)
    total_score = total_score/5
    print("Average score of 5 models: " + str(total_score))
    print("Confusion Matrices")
    print(confusion_matrices)


if __name__ == "__main__":
    main()

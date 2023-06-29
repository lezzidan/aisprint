import sys

import scipy.io
import numpy as np
import time
import dislib as ds
from dislib.model_selection import KFold
import math
from pycompss.api.constraint import constraint

from pycompss.api.task import task
from pycompss.api.parameter import Type, COLLECTION_IN, COLLECTION_OUT, Depth

from pycompss.api.api import compss_wait_on, compss_barrier
from scipy import signal
from ds_tensor import Tensor as ds_tensor, load_dataset, change_shape
from pyeddl.tensor import Tensor
from EncapsulateFunctionTensor_nested import EncapsulatedFunctionsTensor
from collections import Counter
import random
import pyeddl.eddl as eddl
from net_utils import net_parametersToTensor
from sklearn.metrics import accuracy_score, confusion_matrix
import sys

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


@constraint(computing_units=160)
@task(x_train=COLLECTION_IN,
      y_train=COLLECTION_IN,
      x_test=COLLECTION_IN,
      y_test=COLLECTION_IN,
      returns=2)
def training_nested(x_train, y_train, x_test, y_test):
    in_ = eddl.Input([18810])
    layer = in_
    layer = eddl.Reshape(layer, [1, 18810])
    layer = eddl.LeakyReLu(eddl.Conv1D(layer, filters=32, kernel_size=[209]))
    layer = eddl.LeakyReLu(eddl.Conv1D(layer, filters=32, kernel_size=[209]))
    layer = eddl.Reshape(layer, [-1])
    layer = eddl.LeakyReLu(eddl.Dense(layer, 32))
    layer = eddl.Flatten(layer)
    out = eddl.Softmax(eddl.Dense(layer, 2))
    net = eddl.Model([in_], [out])
    fit_time = time.time()
    x_train = ds.array(np.block(x_train), (2600, 18810))
    y_train = ds.array(np.block(y_train), (2600, 2))
    x_test = ds.array(np.block(x_test), (300, 18810))
    y_test = ds.array(np.block(y_test), (300, 2))
    encaps_function = EncapsulatedFunctionsTensor(num_workers=1)
    optimizer = {'optimizer': 'adam', 'lr': 0.00175}
    encaps_function.build(net, optimizer, "soft_cross_entropy", "categorical_accuracy", num_nodes=1, num_gpu=4)
    parameters = encaps_function.fit_synchronous_shuffle_with_GPU(x_train, y_train, 150, 8)
    total_fit_time = time.time() - fit_time
    x_test = x_test.collect()
    y_test = y_test.collect()
    local_arrays = []
    for i in range(math.ceil(x_test.shape[0]/17)):
        if i == math.ceil(x_test.shape[0]/17) - 1:
            x_test_t = Tensor.fromarray(x_test[i*17:])
            y_test_t = Tensor.fromarray(y_test[i*17:])
        else:
            x_test_t = Tensor.fromarray(x_test[i*17:(i+1)*17])
            y_test_t = Tensor.fromarray(y_test[i*17:(i+1)*17])
        local_arrays.append([x_test_t, y_test_t])
    net = eddl.Model([in_], [out])
    eddl.build(
        net,
        eddl.adam(lr=0.001),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_GPU([1])
        )
    eddl.set_parameters(net, net_parametersToTensor(parameters))
    eddl.set_mode(net, 0)
    x_test_to_evaluate = Tensor.fromarray(x_test)
    x_test_to_evaluate.div_(255.0)
    y_test_to_evaluate = Tensor.fromarray(y_test)
    eddl.evaluate(net, [x_test_to_evaluate], [y_test_to_evaluate], bs=1003)
    predictions = []
    true_outputs = []
    for local_array in local_arrays:
        x_test_t = Tensor.fromarray(local_array[0])
        y_test_t = Tensor.fromarray(local_array[1])
        x_test_t.div_(255.0)
        predictions.extend(eddl.predict(net, [x_test_t])[0].getdata())
        true_outputs.extend(y_test_t.getdata())
    sys.stdout.write("Fit time fold:\n")
    sys.stdout.write(str(total_fit_time))
    sys.stdout.flush()
    return predictions, true_outputs

@task()
def main():
    seed = 1234

    cv = KFold(n_splits = 5, shuffle = True)

    in_ = eddl.Input([18810])
    layer = in_
    layer = eddl.Reshape(layer, [1, 18810])
    layer = eddl.LeakyReLu(eddl.Conv1D(layer, filters=32, kernel_size=[209]))
    layer = eddl.LeakyReLu(eddl.Conv1D(layer, filters=32, kernel_size=[209]))
    layer = eddl.Reshape(layer, [-1])
    layer = eddl.LeakyReLu(eddl.Dense(layer, 32))
    layer = eddl.Flatten(layer)
    out = eddl.Softmax(eddl.Dense(layer, 2))
    net = eddl.Model([in_], [out])

    x_train_tensor = ds.data.load_npy_file('/gpfs/scratch/bsc19/bsc19756/aisprint_other_params/KNN/New_Test_Set/Neural_Networks/X_train/x_train_2_classes.npy', (2600, 18810))
    y_train_tensor = ds.data.load_npy_file('/gpfs/scratch/bsc19/bsc19756/aisprint_other_params/KNN/New_Test_Set/Neural_Networks/Y_train/y_train_2_classes.npy', (2600, 1))
    y_train = y_train_tensor.collect()
    y_one_hot = np.zeros((y_train.size, 2))
    y_one_hot[np.arange(y_train.size,dtype=int), y_train.flatten().astype(np.int64)] = 1
    print(y_one_hot[0:1000])
    y_train_tensor = ds.array(y_one_hot, (2600, 2))
    x_train_tensor._blocks = compss_wait_on(x_train_tensor._blocks)
    y_train_tensor._blocks = compss_wait_on(y_train_tensor._blocks)
    start_time = time.time()
    predictions = []
    true_outputs = []
    tests_ds = []
    parameters_list = []
    i = 0
    total_fit_time = time.time()
    j = 0
    for train_ds, test_ds in cv.split(x_train_tensor, y_train_tensor):
        prediction, true_output = training_nested(train_ds[0]._blocks, train_ds[1]._blocks, test_ds[0]._blocks, test_ds[1]._blocks)
        predictions.append(prediction)
        true_outputs.append(true_output)
    predictions = compss_wait_on(predictions)
    true_outputs = compss_wait_on(true_outputs)
    print("Confusion Matrices:", flush=True)
    total_fit_time = time.time() - total_fit_time
    total_score = 0
    confusion_matrices = []
    for i in range(len(predictions)):
        true_values = np.array(true_outputs[i])
        prediction = np.array(predictions[i])
        total_score += accuracy_score(np.argmax(true_values, axis=1), np.argmax(prediction, axis=1))
        confusion_matrices.append(confusion_matrix(np.argmax(true_values, axis=1), np.argmax(prediction, axis=1)))
    print("Fit time" + str(total_fit_time), flush=True)
    print("Full time" + str(time.time() - start_time), flush=True)
    total_score = total_score/5
    print("Average Score: " + str(total_score), flush=True)
    print("Confusion Matrices " + str(confusion_matrices), flush=True)

if __name__ == "__main__":
    main()
    


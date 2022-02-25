import scipy.io
import numpy as np
import glob
import time
import dislib as ds
from base import CascadeSVM
from dislib.data.array import Array

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
from CascadeSync import CascadeSVC
from dislib.trees.decision_tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    _Node,
    _ClassificationNode,
    _RegressionNode,
    _InnerNodeInfo,
    _LeafInfo,
    _SkTreeWrapper,
)


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

def _sync_obj(obj):
    """Recursively synchronizes the Future objects of a list or dictionary
    by using `compss_wait_on(obj)`.
    """
    if isinstance(obj, dict):
        iterator = iter(obj.items())
    elif isinstance(obj, list):
        iterator = iter(enumerate(obj))
    else:
        raise TypeError("Expected dict or list and received %s." % type(obj))
    for key, val in iterator:
        if isinstance(val, (dict, list)):
            _sync_obj(obj[key])
        else:
            obj[key] = compss_wait_on(val)
            if isinstance(getattr(obj[key], "__dict__", None), dict):
                _sync_obj(obj[key].__dict__)

def _encode_helper(obj):
    """Special encoder for dislib that serializes the different objectes
    and stores their state for future loading.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, csr_matrix):
        return {
            "class_name": "csr_matrix",
            **obj.__dict__,
        }
    elif isinstance(obj, np.ndarray):
        return {
            "class_name": "ndarray",
            "dtype_list": len(obj.dtype.descr) > 1,
            "dtype": str(obj.dtype),
            "items": obj.tolist(),
        }
    elif isinstance(obj, Array):
        return {"class_name": "dsarray", **obj.__dict__}
    elif isinstance(obj, np.random.RandomState):
        return {"class_name": "RandomState", "items": obj.get_state()}
    elif callable(obj):
        return {
            "class_name": "callable",
            "module": obj.__module__,
            "name": obj.__name__,
        }
    elif isinstance(obj, SklearnTree):
        return {
            "class_name": obj.__class__.__name__,
            "n_features": obj.n_features,
            "n_classes": obj.n_classes,
            "n_outputs": obj.n_outputs,
            "items": obj.__getstate__(),
        }
    elif isinstance(
        obj, tuple(DISLIB_CLASSES.values()) + tuple(SKLEARN_CLASSES.values())
    ):
        return {
            "class_name": obj.__class__.__name__,
            "module_name": obj.__module__,
            "items": obj.__dict__,
        }
    raise TypeError("Not JSON Serializable:", obj)

DISLIB_CLASSES = {
    "_Node": _Node,
    "_ClassificationNode": _ClassificationNode,
    "_RegressionNode": _RegressionNode,
    "_InnerNodeInfo": _InnerNodeInfo,
    "_LeafInfo": _LeafInfo,
    "_SkTreeWrapper": _SkTreeWrapper,
}
SKLEARN_CLASSES = {
    "SVC": SklearnSVC,
    "DecisionTreeClassifier": SklearnDTClassifier,
    "DecisionTreeRegressor": SklearnDTRegressor,
}

def load_ds_csvm_model(model_path):

    loaded_model = pickle.load(open(model_path, 'rb'))
    model_name = loaded_model["model_name"]
    print("MODEL LOADED: ", model_name)
    model_clf = loaded_model["_clf"]
    model_module = getattr(ds, "classification")
    model_class = getattr(model_module, model_name)
    model = model_class()
    for key, val in loaded_model.items():
        setattr(model, key, val)
    return model

if __name__ == "__main__":
    #args = sys.argv[1:]
    start_time = time.time()
    model_saved = "multiclassCSVM"
    seed = 1234
    csvm = CascadeSVM(cascade_arity=3, kernel='rbf', c=1, gamma=0.05, tol=1e-6, random_state=seed)
    #csvm = CascadeSVC(fold_size=500)
    
    X_train, y_train = load_n_preprocess('./training2017/')
    print([X_train.shape, y_train.shape])
    print(Counter(y_train.flatten()))
    
    # downsample the majority class to balance the training
    idx = random.sample(list(np.where(y_train == 1.0)[0]), Counter(y_train.flatten())[1.0]-Counter(y_train.flatten())[0.0])
    y_train = np.delete(y_train, idx, axis=0)
    X_train = np.delete(X_train, idx, axis=0)
    print(Counter(y_train.flatten()))
    load_time = time.time()
    
    x = ds.array(X_train, block_size=(200, 200))
    y = ds.array(y_train, block_size=(200, 1))

    x_train_shuffle, y_train_shuffle = ds.utils.base.shuffle(x,y)
    csvm.fit(x_train_shuffle, y_train_shuffle)
    compss_barrier()
    fit_time = time.time()
    csvm.save_model("./"+model_saved, save_format="pickle")
    X_train, y_train = load_n_preprocess('./Validation/')
    print([X_train.shape, y_train.shape])
    print(Counter(y_train.flatten()))
    X_train = ds.array(X_train, block_size=(200, 200))
    y_train = ds.array(y_train, block_size=(200, 1))
    #prediction = csvm.predict(X_train)
    #equal = np.equal(prediction, y_train)
    #print(equal)
    #print(np.sum(equal))
    #print(len(prediction))
    #print("SCORE")
    #print(np.sum(equal) / len(prediction))
    print("SCORE:")
    print(compss_wait_on(csvm.score(X_train, y_train)))
    print("Score time", time.time() - fit_time)

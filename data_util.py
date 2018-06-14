import pickle
import platform
import numpy as np
import tensorflow as tf

def load_dr_dataset(ROOT):
    with open('diabetic_dataset/data_batch_1', 'rb') as f:
#         filecontent = f.read()
#         dataset1 = tf.data.Dataset.from_tensor_slices('diabetic_dataset/data_batch_1')
        datadict = pickle.load(f,encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
#         X = np.array(X)
        
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
    return X,Y
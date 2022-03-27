import numpy as np
import tensorflow as tf
from metrics import *
from data import Data

if __name__ == '__main__':
    
    data=Data(n_feature=2)
    xor_data=data.xor_dataset(split=0.6,size=200)
    
    X_train,Y_train= tf.convert_to_tensor(xor_data['train_x']),tf.convert_to_tensor(xor_data['train_y'])
    X_test, Y_test   = tf.convert_to_tensor(xor_data['test_x']),tf.convert_to_tensor(xor_data['test_y'])

    tf_model = tf.keras.models.Sequential([tf.keras.layers.Dense(units=1, activation='sigmoid',use_bias=False)])
    
    tf_model.compile(optimizer=tf.keras.optimizers.SGD(lr=1e-1),loss=tf.keras.losses.BinaryCrossentropy(from_logits=False))
    tf_model.fit(X_train, Y_train, epochs=5000, verbose=1)
    
    Y_hat=tf.squeeze(tf_model(X_test, training=False)).numpy()
    Y_hat=np.where(Y_hat < 0.5, 0, 1)
    Y_test=Y_test.numpy()

    print('Accuracy:',accuracy(Y_hat, Y_test))
    for cls in np.unique(Y_test):
        print('Class: ', cls)
        print('Precision:', precision(Y_hat, Y_test, cls))
        print('Recall:',recall(Y_hat, Y_test, cls))
import numpy as np
import tensorflow as tf
from metrics import *
from data import Data

tf.random.set_seed(45)

if __name__ == '__main__':
    
    data=Data(n_feature=2)
    X_train, Y_train, X_test, Y_test =data.iris(path="iris.csv",split=0.6,type='sepal')
    
    X_train, Y_train = tf.convert_to_tensor(X_train) , tf.convert_to_tensor(Y_train)
    X_test, Y_test   = tf.convert_to_tensor(X_test) , tf.convert_to_tensor(Y_test)

    tf_model = tf.keras.models.Sequential([tf.keras.layers.Dense(units=3, activation='softmax',use_bias=False)])
    
    tf_model.compile(optimizer=tf.keras.optimizers.SGD(lr=1e-1),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False))
    tf_model.fit(X_train, Y_train, epochs=5000, verbose=1)
    
    Y_hat=tf.squeeze(tf_model(X_test, training=False)).numpy()
    Y_hat  = tf.argmax(Y_hat, axis=-1)
    Y_hat  = Y_hat.numpy()
    Y_test = Y_test.numpy()

    print('Accuracy:',accuracy(Y_hat, Y_test))
    for cls in np.unique(Y_test):
        print('Class: ', cls)
        print('Precision:', precision(Y_hat, Y_test, cls))
        print('Recall:',recall(Y_hat, Y_test, cls))
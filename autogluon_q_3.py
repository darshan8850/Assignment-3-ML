import numpy as np
import pandas as pd 
from data import Data
from autogluon.tabular import  TabularPredictor

np.random.seed(45)

if __name__ == '__main__':
    data=Data(n_feature=2)
    XOR_data=data.xor_dataset(N=200,split=0.6)
    X_train,Y_train=XOR_data['train_x'],XOR_data['train_y']
    X_test,Y_test=XOR_data['test_x'],XOR_data['test_y']    

    X_train=pd.DataFrame(X_train)
    X_train['class']=Y_train
    X_test=pd.DataFrame(X_test)
    X_test['class']=Y_test

    path='autogluon_models'  # specifies folder to store trained models
    predictor=TabularPredictor(label='class', path=path).fit(X_train)

    predictor = TabularPredictor.load(path)
    predictor.leaderboard(X_test)
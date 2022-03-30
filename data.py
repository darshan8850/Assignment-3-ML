import pandas as pd
import autograd.numpy as np
from sklearn.preprocessing import LabelEncoder


class Data():
    def __init__(self,n_feature):
        self.n_feature=n_feature

    def iris(self,loc,split=0.8,type='sepal'):
        data=pd.read_csv(loc,header=0)
        l_e=LabelEncoder()
        data['label']=l_e.fit_transform(data['variety'])
        data=data.drop(columns=['variety'])
        train_size=int(data.shape[0]*split)
        train_data=data.iloc[:train_size, :]
        test_data=data.iloc[train_size:, :]

        if type == 'sepal':
            X_train,Y_train=train_data[['sepal.length','sepal.width']].to_numpy(),train_data['label'].to_numpy()
            X_test,Y_test=test_data[['sepal.length','sepal.width']].to_numpy(),test_data['label'].to_numpy()
        elif type == 'petal':
            X_train,Y_train=train_data[['petal.length','petal.width']].to_numpy(),train_data['label'].to_numpy()
            X_test,Y_test=test_data[['petal.length','petal.width']].to_numpy(),test_data['label'].to_numpy()
        else:
            X_train,Y_train=train_data[['petal.length','petal.width','sepal.length','sepal.width']].to_numpy(),train_data['label'].to_numpy()
            X_test,Y_test=test_data[['petal.length','petal.width','sepal.length','sepal.width']].to_numpy(),test_data['label'].to_numpy()

        return X_train,Y_train,X_test,Y_test


    def xor(self,size,split=0.6):
        data={}
        train_size=int(size*split)
        X=np.random.randn(size,self.n_feature)
        Y=np.int64(np.logical_xor(X[:,0]>0,X[:,1]>0))
          
        data['train_x']=X[:train_size]
        data['train_y']=Y[:train_size]
        data['test_x']=X[train_size:]
        data['test_y']=Y[train_size:]
        return data
import autograd.numpy as np
from data import Data
from model import Multiclass_Logistic_Reg
import matplotlib.pyplot as plt
from matplotlib import style
from tqdm import tqdm
from metrics import *


if __name__ == '__main__':

    data=Data(n_feature=2)
    X_train,Y_train,X_test,Y_test=data.iris(loc='iris.csv',split=0.8,type='sepal')
    
    cls = np.unique(Y_train)
    n_feature=X_train.shape[1]

    model=Multiclass_Logistic_Reg(n_feature=n_feature,bias=False)

    model.model_train(X_train,Y_train,cls=cls,lamda=0.0,lr=1e-1,epch=1000)
    Y_hat = np.squeeze(model.test(X_test))
    
    total_acc = 0.0
   
    for cls in np.unique(Y_test):
        Y_temp  = np.where(Y_test == cls, 1, 0)
        Y_hat_t = np.where(Y_hat < 0.5, 0, 1)
        print('Class: ', cls)
        print('Accuracy: ', accuracy(Y_hat_t, Y_temp))
        total_acc += accuracy(Y_hat_t, Y_temp)

    print('\nModel Acc: {}'.format(total_acc/3.0))
    
    model.draw_decision_surface(X_train, Y_train)

    lamda=[0.0, 0.5]

    cls = np.unique(Y_train)

    for lamb in tqdm(lamda):
        new_bin_model = Multiclass_Logistic_Reg(n_feat=n_feature, use_bias=False)
        i,loss=model.train(X_train, Y_train,cls=cls,lamda=lamda, lr=1e-1,epch=1000)
        i=np.array(i)
        loss=np.array(loss)
        loss=(loss - np.min(loss)) / (np.max(loss) - np.min(loss) + 0.001)
        plt.plot(i,loss,label='Lamda: {}'.format(lamb), alpha=0.6)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
    plt.legend()
    plt.show()
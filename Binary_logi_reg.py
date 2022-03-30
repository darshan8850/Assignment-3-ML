import autograd.numpy as np
from data import Data
from model import Binary_Logistic_Reg
import matplotlib.pyplot as plt
from matplotlib import style
from tqdm import tqdm
from metrics import *


if __name__ == '__main__':

    data=Data(n_feature=2)
    xor_data=data.xor(split=0.6,size=200)

    model=Binary_Logistic_Reg(n_feature=2,bias=False)

    model.model_train(xor_data['train_x'], xor_data['train_y'],lamda=0.0,lr=1e-1,epch=1000)
    Y_hat=np.squeeze(model.test(xor_data['test_x']))

    print('Accuracy: ', accuracy(Y_hat, xor_data['test_y']))
    for cls in np.unique(xor_data['test_y']):
        print('Class: ', cls)
        print('Precision: ', precision(Y_hat, xor_data['test_y'], cls))
        print('Recall: ', recall(Y_hat, xor_data['test_y'], cls))

        
    model.draw_decision_surface(xor_data['train_x'], xor_data['train_y'])
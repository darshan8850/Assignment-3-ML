import numpy as np 
import pandas as pd
from autograd import grad
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import style

style.use('seaborn')



class Binary_Logistic_Reg():
    def __init__(self,bias=False,n_feature):
        self.bias=bias
        if self.bias:
            self.theta=np.random.normal(size=(n_feature+1,1))
        else:
            self.theta=np.random.normal(size=(n_feature,1))
    def model_train(self,X,Y,epch=1,lr=0.01,lamda=0.0):
        def sigmoid(X):
          return 1/(1+np.exp(-X))
        def loss(y,y_hat):
          l=np.mean(-1*((y*np.log(y_hat))+((1-y)*np.log(1-y_hat))))
          return l
        def fit(theta):
            t=sigmoid(np.dot(X_new,theta))
            l=loss(Y,t)+(lamda*np.sqrt(np.sum(np.square(theta))))
            return l
            
            
        if self.bias:
            X_new=np.insert(X, 0, np.ones(shape=(X.shape[0],)), axis=1)
        else:
            X_new=X

        grad_function = grad(fit)
        tq = tqdm(range(1,epch+1))

        l= []
        i= []

        for epoch in tq:
            loss = fit(self.theta)
            self.theta -= lr * grad_function(self.theta)
            tq.set_description('Loss_train {}'.format(loss))
            i.append(epoch)
            l.append(loss)


    def model_test(self,X):

        def sigmoid(X):
          return 1/(1+np.exp(-X))  
        
        if self.bias:
          X_new=np.insert(X, 0, np.ones(shape=(X.shape[0],)), axis=1)
        else:
          X_new=X

        t=sigmoid(np.dot(X_new,self.theta))

        if(t>=0.5):
          pred=1
        else:
          pred=0
        
        return pred
    
    def decision_surface(self,X,Y,split="train"):
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
        h=70

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, h), np.linspace(y_min, y_max, h))
        z  = np.reshape(self.test(np.c_[xx.ravel(), yy.ravel()]), xx.shape)
        plt.contourf(xx, yy, z, cmap='viridis', alpha=0.5)
        plt.scatter(X[:, 0], X[:, 1], c=Y, alpha=1.0, edgecolors='k')
        plt.title('Logistic Regression {}'.format(split))
        plt.tight_layout()
        plt.show()




class Multiclass_Logistic_Reg():
    def __init__(self,bias=False,n_feature):
        self.bias=bias
        if self.bias:
            self.theta=np.random.normal(size=(n_feature+1,1))
        else:
            self.theta=np.random.normal(size=(n_feature,1))

    def model_train(self,X,Y,cls,epch=1,lr=0.01,lamda=0.0):
        def sigmoid(X):
          return 1/(1+np.exp(-X))
        def loss(y,y_hat):
          l=np.mean(-1*((y*np.log(y_hat))+((1-y)*np.log(1-y_hat))))
          return l
        def fit(theta):
            t=sigmoid(np.dot(X_new,theta))
            loss_per_class=0
            for i in cls:
                if(Y==i):
                  y=1
                else:
                  y=0
                
                loss_per_class+=loss(y,t)

            total_loss=loss_per_class+(lamda*np.sqrt(np.sum(np.square(theta))))
            return total_loss
            
            
        if self.bias:
            X_new=np.insert(X,0,np.ones(shape=(X.shape[0],)),axis=1)
        else:
            X_new=X

        grad_function = grad(fit)
        tq = tqdm(range(1,epch+1))

        l= []
        i= []

        for epoch in tq:
            loss = fit(self.theta)
            self.theta -= lr * grad_function(self.theta)
            tq.set_description('Loss_train {}'.format(loss))
            i.append(epoch)
            l.append(loss)


    def model_test(self,X):

        def sigmoid(X):
          return 1/(1+np.exp(-X))  
        
        if self.bias:
          X_new=np.insert(X, 0, np.ones(shape=(X.shape[0],)), axis=1)
        else:
          X_new=X

        t=sigmoid(np.dot(X_new,self.theta))

        return t
    
    def decision_surface(self,X,Y,split='train'):
        x_min,x_max=X[:,0].min(),X[:,0].max()
        y_min,y_max=X[:,1].min(),X[:, 1].max()
        M=len(np.unique(Y))
        plt.figure(figsize=(9,9))
        for c,cls in enumerate(np.unique(Y)):
            h= 70
            xx,yy=np.meshgrid(np.linspace(x_min,x_max,h),np.linspace(y_min,y_max,h))
            ax=plt.subplot2grid((1,3), (0, c), rowspan=1, colspan=1)
            z=self.test(np.c_[xx.ravel(),yy.ravel()])
            z=np.where(z<0.5,0,1)
            z=np.reshape(z,xx.shape)
            ax.contourf(xx,yy,z,cmap='viridis',alpha=0.5)
            ax.scatter(X[:,0],X[:,1],c=np.where(Y==cls,1,0),alpha=1.0,edgecolors='k')
            ax.set_title('Multiclass Logisitic Regression [class: {}]'.format(cls))

        plt.tight_layout()
        plt.show()


        




        







# здесь будут собираться все методы МО, которые пишем мы
# интерфейс - тот же, что и у scipy: контруктор, fit и predict
# from scipy.interpolate import Rbf,NearestNDInterpolator,LinearNDInterpolator
import numpy as np
import pandas as pd
import math, copy, geometry
# from neighbourConvexHull import interpByNeighbourConvexHull
# from sklearn.linear_model import LogisticRegression
# from keras.models import Sequential
# from keras.layers.core import Dense, Activation
# from keras.layers.recurrent import LSTM
# from keras import optimizers
# from keras.layers import Dropout, Conv1D, GlobalAveragePooling1D, MaxPooling1D, Embedding, Flatten

def scoreFast(x,y,predictY):
    if len(y.shape) == 1:
        u = np.mean((y - predictY)**2)
        v = np.mean((y - np.mean(y))**2)
    else:
        u = np.mean(np.linalg.norm(y - predictY, axis=1, ord=2)**2)
        v = np.mean(np.linalg.norm(y - np.mean(y, axis=0).reshape([1,y.shape[1]]), axis=1, ord=2)**2)
    return 1-u/v

def score(x,y,predictor):
    predictY = predictor(x)
    return scoreFast(x,y,predictY)

def transformFeatures2Quadric(x):
    n = x.shape[1]
    newX = np.zeros([x.shape[0], n + n*(n+1)//2 + 1])
    newX[:,0:n] = x
    k = n
    for i1 in range(n):
        for i2 in range(i1,n):
            newX[:,k] = x[:,i1]*x[:,i2]
            k += 1
    newX[:,k] = 1
    k += 1
    assert k == n + n*(n+1)//2 + 1
    return newX

def transformFeaturesSeqRelations(x):
    return np.hstack((x, x[:,1:]/x[:,:-1]))

class makeQuadric:
    def __init__(self, learner):
        self.learner = learner
    def fit(self, x, y):
        self.learner.fit(transformFeatures2Quadric(x), y)
    def predict(self, x):
        return self.learner.predict(transformFeatures2Quadric(x))
    def score(self, x, y): return score(x,y,self.predict)

class makeMulti:
    def __init__(self, learner):
        self.learner0 = learner
    def fit(self, x, y0):
        assert len(y0.shape)<=2
        y = y0 if len(y0.shape)==2 else y0.reshape(-1,1)
        n = y.shape[1]
        self.learners = [None]*n
        for i in range(n):
            learner = copy.deepcopy(self.learner0)
            learner.fit(x, y[:,i])
            self.learners[i] = learner
    def predict(self, x):
        n = len(self.learners)
        res = np.zeros([x.shape[0],n])
        for i in range(n): res[:,i] = self.learners[i].predict(x)
        return res
    def score(self, x, y): return score(x,y,self.predict)

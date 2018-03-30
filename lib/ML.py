# здесь будут собираться все методы МО, которые пишем мы
# интерфейс - тот же, что и у scipy: контруктор, fit и predict
from scipy.interpolate import Rbf,NearestNDInterpolator,LinearNDInterpolator
import numpy as np
import pandas as pd
import math, copy, geometry
from neighbourConvexHull import interpByNeighbourConvexHull
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras import optimizers
from keras.layers import Dropout, Conv1D, GlobalAveragePooling1D, MaxPooling1D, Embedding, Flatten

# исправить все статические поля классов-типов!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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

class neighborRegressor:
    def __init__(self, typeName, **neighbourConvexHullParams):
        self.typeName = typeName # поддерживаются следующие типы: 'neighbourConvexHull', 'nearest', 'linear', 'rbf function' (function: multiquadric, inverse, gaussian, linear, cubic, quintic, thin_plate)
        self.neighbourConvexHullParams = neighbourConvexHullParams
        self.rbf = None # модель
        self.train_x = None # (Npoints, NdimsX)
        self.train_y = None # (Npoints, NdimsY)

    def fit(self, x, y):
        self.train_x = np.copy(x.values if type(x) is pd.DataFrame else x)
        self.train_y = np.copy(y.values if type(y) is pd.DataFrame else y)
        if self.typeName == 'neighbourConvexHull':
            self.meanX = np.mean(self.train_x, axis=0)
            self.sigmaX = np.sqrt(np.sum((self.train_x-self.meanX)**2))
            self.train_x -= self.meanX
            if self.sigmaX != 0: self.train_x /= self.sigmaX

    def predict(self, x):
        if type(x) is pd.DataFrame: x = x.values
        x = np.copy(x)
        y = np.copy(self.train_y)
        NdimsX = self.train_x.shape[1]
        NdimsY = y.shape[1] if len(y.shape) > 1 else 1
        res = np.zeros([x.shape[0], NdimsY])
        if self.typeName == 'neighbourConvexHull':
            x -= self.meanX
            if self.sigmaX != 0: x /= self.sigmaX
            for k in range(x.shape[0]):
                res[k,:] = interpByNeighbourConvexHull(self.train_x, y, x[k,:], **self.neighbourConvexHullParams)
            return res
        train_points = [self.train_x[:,j] for j in range(NdimsX)]
        for i in range(NdimsY):
            if self.typeName == 'nearest':
                interp = NearestNDInterpolator(self.train_x, y[:,i])
                res[:,i] = interp(x)
            elif self.typeName == 'linear':
                interp = LinearNDInterpolator(self.train_x, y[:,i])
                res[:,i] = interp(x)
            else:
                assert self.typeName.split(' ')[0] == 'rbf', "Unknown type name"
                func = self.typeName.split(' ')[1]
                interp = Rbf(*train_points, y[:,i], function=func)
                points = [x[:,j] for j in range(NdimsX)]
                res[:,i] = interp(*points)
        return res

    def score(self, x, y): return score(x,y,self.predict)

class myLinearRegression:
    def fit(self, x0, y):
        m = x0.shape[0]
        n = x0.shape[1]
        x = np.hstack((x0,np.ones([m,1])))
        G = np.dot(np.transpose(x), x)
        G1 = np.linalg.inv(G)
        rightPart = np.dot(np.transpose(x), y)
        self.coeffs = np.dot(G1,rightPart) # один столбец - один столбец y

    def predict(self, x0):
        m = x0.shape[0]
        n = x0.shape[1]
        x = np.hstack((x0,np.ones([m,1])))
        y = np.dot(x, self.coeffs)
        return y

    def score(self, x, y): return score(x,y,self.predict)

class myLogisticRegression:
    def fit(self, x, y):
        for j in range(y.shape[1]):
            lr = LogisticRegression()
            lr.fit(x,y[:,j])
            self.learners.append(lr)

    def predict(self, x):
        n = len(self.learners)
        y = np.zeros((x.shape[0], n))
        for j in range(n):
            y[:,j] = slef.learners[j].predict(x)
        return y

    def score(self, x, y): return score(x,y,self.predict)

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

class NeuralNet:
    def fit(self, x, y):
        hidden_neurons = 50
        input_dim = x.shape[1]
        output_dim = y.shape[1]
        model = Sequential()
        model.add(Dense(units=output_dim*10, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.01), activation='linear', input_dim=input_dim))
        model.add(Dense(units=output_dim*2, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.01), activation='linear'))
        model.add(Dense(units=output_dim, kernel_regularizer=regularizers.l2(0.01), kernel_initializer='normal'))
        sgd = optimizers.SGD(lr=0.3, decay=0, momentum=0.9, nesterov=True)
        model.compile(loss='mean_squared_error', optimizer=sgd)
        model.fit(x, y, epochs=100, batch_size=10)
        self.model = model

    def predict(self, x):
        return self.model.predict(x)

    def score(self, x, y): return score(x,y,self.predict)

class NeuralNetDirect:
    def fit(self, x0, y):
        if self.d3: x = np.expand_dims(np.copy(x0), axis=2)
        else: x = np.copy(x0)
        for j in range(y.shape[1]):
            input_dim = x.shape[1]
            output_dim = y.shape[1]
            print('input_dim =', input_dim, 'output_dim =', output_dim)
            model = Sequential()
            # model.add(Conv1D(output_dim, 10, activation='relu', input_shape=(input_dim,1)))
            # model.add(LSTM(32, return_sequences=False, input_shape=(input_dim,1)))
            model.add(Dense(units=1, kernel_initializer='normal', activation='relu', input_dim=input_dim))
            # model.add(Dense(units=output_dim, kernel_initializer='normal', activation='relu'))
            if self.d3: model.add(Flatten())
            # model.add(Dense(units=output_dim, kernel_initializer='normal'))
            sgd = optimizers.SGD(lr=0.3, decay=0, momentum=0.9, nesterov=True)
            model.compile(loss='mean_squared_error', optimizer=sgd)
            model.fit(x, y[:,j], epochs=100, batch_size=1)
            self.models.append(model)

    def predict(self, x0):
        if self.d3: x = np.expand_dims(np.copy(x0), axis=2)
        else: x = x0
        y = np.zeros((x.shape[0], len(self.models)))
        for j in range(len(self.models)):
            y[:,j] = self.models[j].predict(x)
        return y

    def score(self, x, y): return score(x,y,self.predict)

def enlargeDataset(moleculas, values, newCount):
    # поворачиваем на случайные углы
    np.random.seed(0)
    if type(moleculas) is pd.DataFrame: moleculas = moleculas.values
    if type(values) is pd.DataFrame: values = values.values
    Nmol = moleculas.shape[0]
    Natom = moleculas.shape[1]//3
    res = np.zeros([newCount, Natom*3])
    resY = np.zeros([newCount, values.shape[1]])
    for i in range(newCount):
        imol = np.random.randint(0,Nmol)
        #imol = i
        mol = moleculas[imol]
        res[i,:] = mol[:]
        resY[i,:] = values[imol,:]
        center = mol[0:3]
        #print(moleculas[imol])
        for icoord in range(3):
            # вращаем вокруг оси № icoord
            phi = np.random.rand()*2*math.pi
            #phi=0
            cphi = math.cos(phi)
            sphi = math.sin(phi)
            inds = np.arange(3)
            inds = np.delete(inds,icoord)
            c = center[inds]
            for j in range(Natom):
                res[i,3*j+inds] = geometry.turnCoords(res[i,3*j+inds]-c, cphi, sphi) + c
        #if i == 0:
        #    # for testing
        #    df = assignPosToPandasMol(moleculas[imol])
        #    save_to_file(df, 'rotation_before')
        #    df2 = assignPosToPandasMol(res[i,:])
        #    save_to_file(df2, 'rotation_after')
        #print(res[i,:])
    return res, resY

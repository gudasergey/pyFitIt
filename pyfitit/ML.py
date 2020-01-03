# здесь будут собираться все методы МО, которые пишем мы
# интерфейс - тот же, что и у scipy: контруктор, fit и predict
from scipy.interpolate import Rbf, NearestNDInterpolator, LinearNDInterpolator
import numpy as np
import pandas as pd
import math, copy, os, time, gc, warnings
from . import geometry, utils
from sklearn.linear_model import LogisticRegression, RidgeCV
import sklearn
if utils.isLibExists("keras"):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=FutureWarning)
        from keras.models import Sequential
        from keras.layers.core import Dense, Activation
        from keras.layers.recurrent import LSTM
        from keras import optimizers, Model
        from keras.layers import Dropout, Conv1D, GlobalAveragePooling1D, MaxPooling1D, Embedding, Flatten, Input
        from keras.layers.normalization import BatchNormalization
        from keras.utils import np_utils


class Sample:
    def __init__(self, params, spectra):
        assert isinstance(params, pd.DataFrame), 'params should be pandas DataFrame object'
        assert isinstance(spectra, pd.DataFrame), 'spectra should be pandas DataFrame object'
        self.params = params
        self._spectra = spectra
        self.paramNames = params.columns.values
        self.folder = None
        e_names = spectra.columns
        self.energy = np.array([float(e_names[i][2:]) for i in range(e_names.size)])

    def setSpectra(self, spectra):
        self._spectra = copy.deepcopy(spectra)
        e_names = spectra.columns
        self.energy = np.array([float(e_names[i][2:]) for i in range(e_names.size)])
        self.folder = None

    def getSpectra(self): return self._spectra

    spectra = property(getSpectra, setSpectra)

    @classmethod
    def readFolder(cls, folder):
        paramFile = utils.fixPath(folder+'/params.txt')
        spectraFile = utils.fixPath(folder+'/spectra.txt')
        res = cls(pd.read_csv(paramFile, sep=' '), pd.read_csv(spectraFile, sep=' '))
        res.folder = folder
        return res

    def saveToFolder(self, folder):
        if not os.path.exists(folder): os.makedirs(folder)
        self.spectra.to_csv(folder+'/spectra.txt', sep=' ', index=False)
        self.params.to_csv(folder+'/params.txt', sep=' ', index=False)

    def addParam(self, paramGenerator, paramName, project):
        n = self.params.shape[0]
        newParam = np.zeros(n)
        for i in range(n):
            params = {self.paramNames[j]:self.params.loc[i,self.paramNames[j]] for j in range(self.paramNames.size)}
            m = project.moleculeConstructor(params)
            newParam[i] = paramGenerator(params, m)
        self.params[paramName] = newParam
        self.paramNames = self.params.columns.values

def scoreFast(x,y,predictY):
    if len(y.shape) == 1:
        u = np.mean((y - predictY)**2)
        v = np.mean((y - np.mean(y))**2)
    else:
        u = np.mean(np.linalg.norm(y - predictY, axis=1, ord=2)**2)
        v = np.mean(np.linalg.norm(y - np.mean(y, axis=0).reshape([1,y.shape[1]]), axis=1, ord=2)**2)
    if v == 0: return 0
    return 1-u/v


def score(x,y,predictor):
    predictY = predictor(x)
    return scoreFast(x,y,predictY)


class RBF:
    def __init__(self, function, baseRegression='quadric'):
        # function: multiquadric, inverse, gaussian, linear, cubic, quintic, thin_plate
        # baseRegression: linear quadric
        self.function = function
        self.baseRegression = baseRegression
        self.trained = False

    def fit(self, x, y):
        self.train_x = x.values if (type(x) is pd.DataFrame) or (type(x) is pd.Series) else x
        self.train_y = y.values if (type(y) is pd.DataFrame) or (type(y) is pd.Series) else y
        if len(self.train_y.shape) == 1: self.train_y = self.train_y.reshape(-1, 1)
        if self.baseRegression == 'quadric': self.base = makeQuadric(RidgeCV())
        else: self.base = RidgeCV()
        self.base.fit(self.train_x, self.train_y)
        self.train_y = self.train_y - self.base.predict(self.train_x)
        NdimsX = self.train_x.shape[1]
        NdimsY = self.train_y.shape[1]
        train_points = [self.train_x[:, j] for j in range(NdimsX)]
        self.interp = [None]*NdimsY
        for i in range(NdimsY):
            self.interp[i] = Rbf(*train_points, self.train_y[:,i], function=self.function)
        self.trained = True

    def predict(self, x):
        assert self.trained
        if type(x) is pd.DataFrame: x = x.values
        y = self.train_y
        if len(y.shape) == 1: y = y.reshape(-1, 1)
        NdimsX = self.train_x.shape[1]
        NdimsY = y.shape[1] if len(y.shape) > 1 else 1
        res = np.zeros([x.shape[0], NdimsY])
        for i in range(NdimsY):
            points = [x[:,j] for j in range(NdimsX)]
            res[:,i] = self.interp[i](*points)
        res = res + self.base.predict(x)
        return res

    def score(self, x, y): return score(x,y,self.predict)


def transformFeatures2Quadric(x):
    if type(x) is pd.DataFrame: x = x.values
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

def transformFeaturesAddDiff(x):
    return np.hstack((x, x[:,1:]-x[:,:-1]))
def transformFeaturesAddDiff2(x):
    dx = x[:,1:]/x[:,:-1]
    return np.hstack((x, dx, dx[:,1:]-dx[:,:-1]))

class makeQuadric:
    def __init__(self, learner):
        self.learner = learner
    def fit(self, x, y):
        self.learner.fit(transformFeatures2Quadric(x), y)
    def predict(self, x):
        return self.learner.predict(transformFeatures2Quadric(x))
    def score(self, x, y): return score(x,y,self.predict)

class addDiffs:
    def __init__(self, learner, diffNumber):
        self.learner = learner
        self.diffNumber = diffNumber
        if not hasattr(learner, 'name'): self.name = str(type(learner))
        else: self.name = 'diff '+str(diffNumber)+' '+learner.name
    def fit(self, x, y):
        if self.diffNumber == 1:
            self.learner.fit(transformFeaturesAddDiff(x), y)
        elif self.diffNumber == 2:
            self.learner.fit(transformFeaturesAddDiff2(x), y)
        else: assert False
    def predict(self, x):
        if self.diffNumber == 1:
            return self.learner.predict(transformFeaturesAddDiff(x))
        else:
            return self.learner.predict(transformFeaturesAddDiff2(x))
    def score(self, x, y): return score(x,y,self.predict)

def norm(x, minX, maxX):
    res = 2*(x-minX)/(maxX-minX) - 1
    if type(x) is pd.DataFrame:
        res.loc[:, minX == maxX] = 0
    else:
        if minX.size == 1:
            if minX==maxX: res[:] = 0
        else:
            res[:,minX==maxX] = 0
    return res
def invNorm(x, minX, maxX): return (x+1)/2*(maxX-minX) + minX

class Normalize:
    def __init__(self, learner, xOnly):
        self.learner = learner
        self.xOnly = xOnly
        if not hasattr(learner, 'name'): self.name = str(type(learner))
        else: self.name = 'normalized '+learner.name

    # args['xyRanges'] = {'minX':..., 'maxX':..., ...}
    def fit(self, x, y, **args):
        y_is_df = type(y) is pd.DataFrame
        if y_is_df: columns = y.columns
        if 'xyRanges' in args: self.xyRanges = args['xyRanges']; del args['xyRanges']
        else: self.xyRanges = {}
        if len(self.xyRanges)>=2:
            self.minX = self.xyRanges['minX']; self.maxX = self.xyRanges['maxX']
            if len(self.xyRanges)==4: self.minY = self.xyRanges['minY']; self.maxY = self.xyRanges['maxY']
        else:
            self.minX = np.min(x, axis=0); self.maxX = np.max(x, axis=0)
            if self.xOnly:
                self.minY = -np.ones(y.shape[1])
                self.maxY = np.ones(y.shape[1])
            else:
                self.minY = np.min(y.values, axis=0) if y_is_df else np.min(y, axis=0)
                self.maxY = np.max(y.values, axis=0) if y_is_df else np.max(y, axis=0)
        if type(self.minX) is pd.Series: self.minX = self.minX.values; self.maxX = self.maxX.values
        if type(self.minY) is pd.Series: self.minY = self.minY.values; self.maxY = self.maxY.values
        # print(self.minX, self.maxX, self.minY, self.maxY)
        if 'validation_data' in args:
            (xv, yv) = args['validation_data']
            validation_data = (norm(xv, self.minX, self.maxX), norm(yv, self.minY, self.minY))
            args['validation_data'] = validation_data
        if 'yRange' in args: args['yRange'] = [norm(args['yRange'][0], self.minY, self.minY), norm(args['yRange'][1], self.minY, self.minY)]
        self.learner.fit(norm(x, self.minX, self.maxX), norm(y, self.minY, self.maxY), **args)

    def predict(self, x):
        if type(x) is pd.DataFrame: x = x.values
        yn = self.learner.predict(norm(x, self.minX, self.maxX))
        return invNorm(yn,self.minY, self.maxY)

    def predict_proba(self, x):
        pyn = self.learner.predict_proba(norm(x, self.minX, self.maxX))
        return pyn

    def score(self, x, y): return score(x,y,self.predict)

class makeMulti:
    def __init__(self, learner):
        self.learner = learner

    def fit(self, x, y0, **args):
        if (type(y0) is pd.DataFrame) or (type(y0) is pd.Series): y0 = y0.values
        assert len(y0.shape)<=2
        y = y0 if len(y0.shape)==2 else y0.reshape(-1,1)
        n = y.shape[1]
        self.learners = [None]*n
        if 'validation_data' in args: validation_data_all = args['validation_data']
        for i in range(n):
            learner = copy.deepcopy(self.learner)
            if 'validation_data' in args:
                validation_data = (validation_data_all[0], validation_data_all[1][:,i])
                args['validation_data'] = validation_data
                learner.fit(x, y[:,i], **args)
            else: learner.fit(x, y[:,i], **args)
            self.learners[i] = learner

    def predict(self, x):
        n = len(self.learners)
        res = np.zeros([x.shape[0],n])
        for i in range(n): res[:,i] = self.learners[i].predict(x).reshape(x.shape[0])
        return res

    def predict_proba(self, x):
        n = len(self.learners)
        res = []
        # для каждого классификатора вообще говоря свое число классов
        for i in range(n): res.append(self.learners[i].predict_proba(x))
        return res

    def score(self, x, y): return score(x,y,self.predict)

class NeuralNetDirect:
    d3 = True

    def __init__(self, epochs, batch_size, showProgress=True):
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = 1 if showProgress else 0

    def fit(self, x0, y, validation_data=None):
        if self.d3:
            x = np.expand_dims(x0, axis=2)
            if validation_data is not None:
                validation_data = (np.expand_dims(validation_data[0], axis=2), validation_data[1])
        else: x = x0
        input_dim = x.shape[1]
        model = Sequential()
        model.add(Conv1D(100, 5, activation='relu', input_shape=(input_dim,1)))
        model.add(Flatten())
        # model.add(Dropout(0.3, seed=0))
        model.add(Dense(units=10, kernel_initializer='normal', activation='relu'))
        model.add(Dense(units=1, kernel_initializer='normal'))
        sgd = optimizers.SGD() # параметры lr=0.3, decay=0, momentum=0.9, nesterov=True уводят в nan
        model.compile(loss='mean_squared_error', optimizer=sgd)
        t1 = time.time()
        if validation_data is None: model.fit(x, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        else: model.fit(x, y, epochs=self.epochs, batch_size=self.batch_size, validation_data=validation_data, verbose=self.verbose)
        t2 = time.time()
        # print("Train time=", t2 - t1)
        self.model = model

    def predict(self, x0):
        if self.d3: x = np.expand_dims(x0, axis=2)
        else: x = x0
        return self.model.predict(x)

    def score(self, x, y): return score(x,y,self.predict)

class NeuralNetDirectClass:
    def __init__(self, epochs):
        self.epochs = epochs
    d3 = True
    def fit(self, x0, y, classNum, yRange, validation_data=None):
        if self.d3:
            x = np.expand_dims(x0, axis=2)
            if validation_data is not None:
                validation_data = (np.expand_dims(validation_data[0], axis=2), validation_data[1])
        else: x = x0
        sgd = optimizers.SGD() # параметры lr=0.3, decay=0, momentum=0.9, nesterov=True уводят в nan
        inp = Input(shape=(x.shape[1],1))
        out = Conv1D(10, 3, activation='relu')(inp)
        out = Flatten()(out)
        # out = Dropout(0.3, seed=0)(out)
        # out = BatchNormalization()(out)
        out = Dense(units=20, kernel_initializer='normal')(out)
        # out = BatchNormalization()(out) #!!!!!!!!!!!! - только этот!!!
        out = Activation('relu')(out)
        # out = Dropout(0.1, seed=0)(out)

        newLayerCount = 0
        out_class = Dense(units=classNum, kernel_initializer='normal')(out); newLayerCount+=1
        # out_class = BatchNormalization()(out_class); newLayerCount+=1
        out_class = Activation('softmax')(out_class); newLayerCount+=1
        model_class = Model(input=inp, output=out_class)

        out_regr = Dense(units=1, kernel_initializer='normal')(out)
        model_regr = Model(input=inp, output=out_regr)
        model_regr.compile(loss='mean_squared_error', optimizer=sgd)
        # model_regr.summary()

        # print('Training regression network...')
        # if validation_data is None: model_regr.fit(x, y, epochs=2, batch_size=1)
        # else: model_regr.fit(x, y, epochs=10, batch_size=1, validation_data=validation_data)

        y_class = makeClasses(y, yRange, classNum)
        dummy_y = np_utils.to_categorical(y_class)
        (vx,vy) = validation_data
        vy_class = makeClasses(vy, yRange, classNum)
        dummy_vy = np_utils.to_categorical(vy_class)

        # for i in range(len(model_class.layers)-newLayerCount): model_class.layers[i].trainable = False
        model_class.compile(loss='categorical_crossentropy', optimizer=sgd)
        # model_class.summary()
        print('Training last layer of classification network...')
        if validation_data is None: model_class.fit(x, dummy_y, epochs=10, batch_size=1, verbose=1)
        else: model_class.fit(x, dummy_y, epochs=100, batch_size=16, validation_data=(vx,dummy_vy))

        # print('Training all classification network...')
        # for l in model_class.layers: l.trainable = True
        # model_class.compile(loss='categorical_crossentropy', optimizer=sgd)
        # # model_class.summary()
        # if validation_data is None: model_class.fit(x, dummy_y, epochs=30, batch_size=1, verbose=1)
        # else: model_class.fit(x, dummy_y, epochs=30, batch_size=1, validation_data=(vx,dummy_vy))

        self.model = model_class

    def predict_proba(self, x0):
        if self.d3: x = np.expand_dims(x0, axis=2)
        else: x = x0
        return self.model.predict(x)

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

def cross_val_predict(method, X, y, cv=10):
    kf = sklearn.model_selection.KFold(n_splits=cv, shuffle=True, random_state=0)
    predictions = np.zeros(y.shape)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index,:], X[test_index,:]
        y_train, y_test = y[train_index,:], y[test_index,:]
        if y_train.shape[1] == 1:
            y_train = y_train.reshape(-1)
        method.fit(X_train, y_train)
        predictions[test_index,:] = method.predict(X_test).reshape(test_index.size, -1)
    return predictions

def getOneDimPrediction(estimator, x0, y0, verticesNum = 10, intermediatePointsNum = 10):
    x = x0.values; y = y0.values
    stdx = np.std(x, axis=0)
    ind_x = np.argmin( np.abs( stdx - np.mean(stdx[stdx>0]) ) )
    stdy = np.std(y, axis=0)
    ind_y = np.argmin( np.abs( stdy - np.mean(stdy[stdy>0]) ) )
    grid = np.linspace(np.min(x[:,ind_x]), np.max(x[:,ind_x]), verticesNum)
    ind0 = [np.argmin( np.abs( x[:,ind_x] - grid[i] ) ) for i in range(verticesNum)]
    ind = []
    for i in ind0:
        if i not in ind: ind.append(i) # to preserve order
    assert len(ind)>1
    vertices = x[ind,:]

    Ntraj = vertices.shape[0]
    NtrajFull = (Ntraj-1)*(intermediatePointsNum+1)+1
    trajectoryFull = np.zeros([NtrajFull, vertices.shape[1]])
    k = 0
    for i in range(Ntraj-1):
        trajectoryFull[k] = vertices[i]; k+=1
        for j in range(intermediatePointsNum):
            lam = (j+1)/(intermediatePointsNum+1)
            trajectoryFull[k] = vertices[i]*(1-lam) + vertices[i+1]*lam; k+=1
    trajectoryFull[k] = vertices[-1]; k+=1
    assert k == NtrajFull

    estimator.fit(x,y[:,ind_y])
    prediction = estimator.predict(trajectoryFull)

    return [trajectoryFull[:,ind_x], prediction], [x[ind,ind_x], y[ind,ind_y]], [x0.columns[ind_x], y0.columns[ind_y]], trajectoryFull

import sklearn.neighbors
import scipy.spatial
import scipy.stats
import statsmodels.api as sm

def kde(responses, grid, bandwidth):
    """Calculates the kernel density estimate.

    Arguments
    ---------
    responses : numpy matrix
       The training responses; each row corresponds to an observation,
       each column corresponds to a variable.
    grid : numpy matrix
        The grid points at which the KDE is evaluated.
    bandwidth : numpy array or string
        The bandwidth for the kernel density estimate; array specifies
        the diagonal of the bandwidth matrix. Strings include
        "scott", "silverman", and "normal_reference" for univariate densities and
        "normal_reference", "cv_ml", and "cv_ls" for multivariate densities.

    Returns
    -------
    numpy array
       The density evaluated at the grid points.

    """

    if len(grid.shape) == 1:
        grid = grid.reshape(-1, 1)
    if len(responses.shape) == 1:
        responses = responses.reshape(-1, 1)


    n_grid, n_dim = grid.shape
    n_obs, _ = responses.shape
    density = np.zeros(n_grid)

    if n_dim == 1:
        kde = sm.nonparametric.KDEUnivariate(responses[:, 0])
        kde.fit(bw = bandwidth, fft = False)
        return kde.evaluate(grid[:, 0])
    else:
        if isinstance(bandwidth, (float, int)):
            bandwidth = [bandwidth] * n_dim
        kde = sm.nonparametric.KDEMultivariate(responses, var_type = "c" * n_dim,
                                               bw = bandwidth)
        return kde.pdf(grid)


class NNKCDE(object):
    def __init__(self, x_train, z_train):

        if len(z_train.shape) == 1:
            z_train = z_train.reshape(-1, 1)
        if len(x_train.shape) == 1:
            x_train = x_train.reshape(-1, 1)

        self.z_train = z_train
        self.tree = sklearn.neighbors.BallTree(x_train)

    def predict(self, x_test, z_grid, k, bandwidth):
        n_test = x_test.shape[0]

        if k is None:
            k = self.k

        if len(x_test.shape) == 1:
            x_test = x_test.reshape(-1, 1)
        if len(z_grid.shape) == 1:
            z_grid = z_grid.reshape(-1, 1)
        n_grid = z_grid.shape[0]

        ids = self.tree.query(x_test, k=k, return_distance=False)

        cdes = np.empty((n_test, n_grid))
        for idx in range(n_test):
            cdes[idx, :] = kde(self.z_train[ids[idx], :], z_grid, bandwidth)

        return cdes

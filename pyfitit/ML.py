import shutil

from scipy.interpolate import Rbf, RBFInterpolator
import numpy as np
import pandas as pd
import math, copy, os, time, warnings, glob, sklearn, inspect
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pyfitit.enhancedGpr import EnhancedGaussianProcessRegressor
from . import geometry, utils, plotting
from sklearn.linear_model import RidgeCV
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels


if utils.isLibExists("tensorflow"):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # with warnings.catch_warnings():
    #     warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf


class Sample:
    def __init__(self, params, spectra, energy=None, spType='default'):
        """
        Main data container for ML tasks
        :param params: DataFrame of geometry parameters
        :param spectra: list or Dataframe of spectra (column names: 'e_7443 e_7444.5 ...). For the case of several types of spectra - can be dict 'spectraType':list/spectraDataFrame. Spectra in list can have different energies
        :param energy: to initialize Sample by numpy matrix (to build correct column names of dataframe). Dict, if spectra is dict
        """
        assert isinstance(params, pd.DataFrame), 'params should be pandas DataFrame object'
        params = copy.deepcopy(params)
        spectra = copy.deepcopy(spectra)
        if energy is not None: energy = copy.deepcopy(energy)
        if not isinstance(spectra, dict):
            spectras = {spType:spectra}
            energies = {spType:energy}
        else:
            spectras = spectra
            energies = energy
            if energy is None: energies = {spType:None for spType in spectras}
            else: assert isinstance(energy, dict)
        first = True
        for spType in spectras:
            spectra = spectras[spType]
            energy = energies[spType]
            if first: sp0 = spectra
            if isinstance(spectra, np.ndarray):
                assert energy is not None
                assert len(energy) == spectra.shape[1], f'{len(energy)} != {spectra.shape[1]} energy vector must contain values for all columns of spectra matrix'
                spectra = pd.DataFrame(data=spectra, columns=['e_' + str(e) for e in energy])
            elif isinstance(spectra, list):
                spectra = utils.makeDataFrameFromSpectraList(spectra, energy)
            else:
                assert isinstance(spectra, pd.DataFrame), 'spectra should be pandas DataFrame object'
            assert len(params)==0 or params.shape[0] == spectra.shape[0], str(params.shape[0]) + ' != ' + str(spectra.shape[0])
            assert len(sp0) == len(spectra), 'All spectra collections in dict must have the same count'
            spectras[spType] = spectra
            energies[spType] = utils.getEnergy(spectra)
        self._spectra = spectras
        self._energy = energies
        self.paramNames = params.columns.to_numpy()
        self._params = params
        self.folder = None
        self.nameColumn = None
        self.defaultSpType = self.spTypes()[0]

    def getLength(self):
        for spType in self._spectra:
            n = self.getSpectra(spType).shape[0]
            break
        return n

    def spTypes(self):
        return sorted(list(self._spectra.keys()))

    def getDefaultSpType(self):
        return self.defaultSpType

    def setDefaultSpType(self, spType):
        assert spType in self.spTypes()
        self.defaultSpType = spType

    def renameSpType(self, oldName, newName):
        assert oldName in self.spTypes()
        assert newName not in self.spTypes()
        self._spectra[newName] = self._spectra[oldName]
        del self._spectra[oldName]

    def delSpType(self, spType):
        assert spType in self._spectra
        del self._spectra[spType]
        del self._energy[spType]

    def setSpectra(self, spectra, energy=None, spType=None):
        """
        Setter for spectra
        :param spectra: DataFrame or np.ndarray (in last case energy should be given)
        :param energy:
        :param spType: spectrum type name for the case of multiple spectra matrixes inside one sample
        """
        if spType is None: spType = self.getDefaultSpType()
        if isinstance(spectra, pd.DataFrame):
            self._spectra[spType] = copy.deepcopy(spectra)
            self._energy[spType] = utils.getEnergy(spectra)
            self.folder = None
        else:
            assert isinstance(spectra, np.ndarray)
            assert energy is not None
            self._spectra[spType] = utils.makeDataFrame(energy, copy.deepcopy(spectra))
            self.folder = None
            self._energy[spType] = energy

    def getSpectra(self, spType=None):
        if spType is None: spType = self.getDefaultSpType()
        return self._spectra[spType]

    spectra = property(getSpectra, setSpectra)

    def getSpectrum(self, i, spType=None, returnIntensityOnly=False):
        """
        By default returns spectrum of default spType
        """
        res = {}
        for spT in self._spectra:
            intensity = self._spectra[spT].loc[i].to_numpy().reshape(-1)
            if returnIntensityOnly:
                res[spT] = intensity
            else:
                res[spT] = utils.Spectrum(self.getEnergy(spType=spT), intensity)
        if spType is None: spType = self.getDefaultSpType()
        return res[spType]

    def getIndByName(self, name):
        assert self.nameColumn is not None
        i = np.where(self._params[self.nameColumn].to_numpy() == name)[0]
        assert len(i) == 1, str(i)
        return i[0]

    def setSpectrum(self, i, spectrum, spType=None):
        """
        Set spectrum
        :param i: index
        :param spectrum: spectrum
        :param spType: if spType==None and there are several spTypes, spectrum should be dict of spectra for all spTypes
        """
        if spType is None:
            if len(self._spectra) > 1:
                assert isinstance(spectrum, dict) and set(spectrum.keys()) == set(self._spectra.keys())
                for spType in spectrum:
                    self._spectra[spType].loc[i] = spectrum[spType]
                return
            else:
                spType = self.getDefaultSpType()
        if isinstance(spectrum, dict):
            assert len(spectrum) == 1
            spectrum = spectrum[list(spectrum.keys())[0]]
        self._spectra[spType].loc[i] = spectrum

    def setParams(self, params):
        assert isinstance(params, pd.DataFrame)
        self._params = params
        self.paramNames = params.columns.to_numpy()

    def getParams(self): return self._params

    params = property(getParams, setParams)

    def getEnergy(self, spType=None):
        if spType is None: spType = self.getDefaultSpType()
        return self._energy[spType]

    def setEnergy(self, energy, spType=None):
        assert False, 'Do not set energy explicitly. It is done after setting spectra'

    energy = property(getEnergy, setEnergy)

    def shiftEnergy(self, shift, spType=None, inplace=False):
        if spType is None: spType = self.getDefaultSpType()
        newEnergy = self._energy[spType] + shift
        sam = self if inplace else self.copy()
        sam._energy[spType] = newEnergy
        sam._spectra[spType].columns = ['e_' + str(e) for e in newEnergy]
        if not inplace: return sam

    def changeEnergy(self, newEnergy, spType=None, inplace=False):
        if spType is None: spType = self.getDefaultSpType()
        oldEnergy = self._energy[spType]
        sam = self if inplace else self.copy()
        spectra = np.zeros((self.getLength(), len(newEnergy)))
        oldSpectra = sam._spectra[spType].to_numpy()
        for i in range(self.getLength()):
            spectra[i] = np.interp(newEnergy, oldEnergy, oldSpectra[i])
        sam.setSpectra(spectra, energy=newEnergy, spType=spType)
        if not inplace: return sam

    @classmethod
    def readFolder(cls, folder):
        paramFile = utils.fixPath(folder+os.sep+'params.txt')
        files = glob.glob(folder + os.sep + '*' + '_spectra.txt')
        if len(files) == 1:
            spectra = pd.read_csv(files[0], sep=' ')
        elif len(files) > 1:
            spectra = {}
            for f in files:
                base = os.path.split(f)[1]
                spType = base[:-len('_spectra.txt')]
                spectra[spType] = pd.read_csv(f, sep=' ')
        else:
            spectraFile = utils.fixPath(folder+os.sep+'spectra.txt')
            spectra = pd.read_csv(spectraFile, sep=' ')
        res = cls(pd.read_csv(paramFile, sep=' '), spectra)
        res.folder = folder
        return res

    def saveToFolder(self, folder, plot=False, colorParam=None):
        if os.path.exists(folder): shutil.rmtree(folder)
        if not os.path.exists(folder): os.makedirs(folder)
        if len(self._spectra) == 1:
            self.spectra.to_csv(folder+os.sep+'spectra.txt', sep=' ', index=False)
        else:
            for spType in self._spectra:
                self._spectra[spType].to_csv(folder + os.sep + f'{spType}_spectra.txt', sep=' ', index=False)
        self.params.to_csv(folder + os.sep + 'params.txt', sep=' ', index=False)
        self.folder = folder
        if plot:
            for spType in self._spectra:
                plotting.plotSample(self._energy[spType], self._spectra[spType].to_numpy(), fileName=folder + os.sep + f'plot_{spType}.png', colorParam=colorParam)

    def copy(self):
        return Sample(self._params, self._spectra)

    def addParam(self, paramGenerator=None, paramName='', project=None, paramData=None):
        """
        Add new parameters to sample.params
        :param paramGenerator: function(paramDict, molecula) to calculate new params (single or multiple)
        :param paramName: name or list of new param names
        :param project: to call moleculeConstructor(sample.params) and pass molecula to paramGenerator
        :param paramData: already calculated params - alternative to paramGenerator
        """
        assert (paramData is None) or (paramGenerator is None and project is None)
        assert paramName != ''
        assert paramName not in self.paramNames, f'Parameter {paramName} already exists'
        if isinstance(paramData, list): paramData = np.array(paramData)
        n = self.params.shape[0]
        if paramData is None:
            if isinstance(paramName, str):  # need to construct one parameter
                paramName = [paramName]
            newParam = np.zeros((n, len(paramName)))
            for i in range(n):
                params = {self.paramNames[j]:self.params.loc[i,self.paramNames[j]] for j in range(self.paramNames.size)}
                m = project.moleculeConstructor(params)
                t = paramGenerator(params, m)
                assert len(t) == len(paramName)
                newParam[i] = t
            for p,j in zip(paramName, range(len(paramName))): self.params[p] = newParam[:,j]
        else:
            assert paramData.size == n
            self.params[paramName] = paramData
        self.paramNames = self.params.columns.values
        self.folder = None

    def delParam(self, paramName):
        assert self.params.shape[1]>1, 'Can\'t delete last parameter'
        if isinstance(paramName, str): paramName = [paramName]
        for p in paramName: del self.params[p]
        self.paramNames = self.params.columns.to_numpy()
        self.folder = None

    def delRow(self, i, inplace=True):
        if inplace:
            sample = self
        else:
            sample = self.copy()
        sample.params.drop(i, inplace=True)
        sample.params.reset_index(inplace=True, drop=True)
        for spType in sample._spectra:
            sample._spectra[spType].drop(i, inplace=True)
            sample._spectra[spType].reset_index(inplace=True, drop=True)
        if not inplace: return sample

    def takeRows(self, ind):
        if isinstance(ind, list): ind = np.array(ind)
        if ind.dtype == bool: ind = np.where(ind)[0]
        toDel = np.setdiff1d(np.arange(len(self.params)), ind)
        sample = self.copy()
        if len(toDel) > 0:
            sample.delRow(toDel)
        return sample

    def unionWith(self, other):
        if other is None: return
        assert isinstance(other, self.__class__)
        assert np.all(self.params.shape[1] == other.params.shape[1]), 'Params differ: self = '+str(self.paramNames)+' other = '+str(other.paramNames)
        assert self._spectra.keys() == other._spectra.keys()
        for spType in self._spectra:
            assert np.all(self._spectra[spType].shape[1] == other._spectra[spType].shape[1]), f'spType={spType} {self._spectra[spType].shape[1]} != {other._spectra[spType].shape[1]}'
            assert np.all(self._energy[spType] == other._energy[spType])
        assert set(self.paramNames) == set(other.paramNames), 'Params differ: self = ' + str(self.paramNames) + ' other = ' + str(other.paramNames)
        self.params = pd.concat((self.params, other.params), ignore_index=True)
        for spType in self._spectra:
            self._spectra[spType] = pd.concat((self._spectra[spType], other._spectra[spType]), ignore_index=True)
        self.folder = None

    def addSpectrumType(self, spectra, spType, energy=None):
        assert spType not in self.spTypes(), f'Spectrum type {spType} already exists'
        assert self.getDefaultSpType() != 'default', 'Rename default spType by renameSpType before adding new one'
        if isinstance(spectra, list):
            spectra = utils.makeDataFrameFromSpectraList(spectra, energy)
        self.setSpectra(spectra, energy, spType=spType)

    def addRow(self, spectrum=None, params=None):
        i = self.params.shape[0]
        if spectrum is not None:
            if len(self.spTypes()) > 1:
                assert isinstance(spectrum, dict)
            if not isinstance(spectrum, dict): spectrum = {self.getDefaultSpType(): spectrum}
            for spType in spectrum:
                sp = spectrum[spType]
                if isinstance(sp, utils.Spectrum):
                    if len(self._energy[spType]) != len(sp.energy) or ~np.all(self._energy[spType] == sp.energy):
                        sp = np.interp(self._energy[spType], sp.energy, sp.intensity)
                    else: sp = sp.intensity
                else:
                    assert isinstance(sp, np.ndarray)
                    sp = sp.reshape(-1)
                    assert len(sp) == len(self._energy[spType])
                spectrum[spType] = sp
        else:
            spectrum = {}
            for spType in self.spTypes():
                spectrum[spType] = np.zeros(len(self._energy[spType]))
                spectrum[spType][:] = np.nan
        # print(spectrum.shape, self.spectra.shape)
        for spType in self.spTypes():
            self._spectra[spType].loc[i] = spectrum[spType]
        if params is not None:
            if isinstance(params, pd.Series): params = params.to_dict()
            assert isinstance(params, dict)
            assert set(params.keys()) <= set(self.paramNames), 'Unknown param names: ' + str(set(self.paramNames) - set(params.keys()))
        else: params = {}
        self.params.loc[i, :] = np.nan
        for p in params: self.params.loc[i, p] = params[p]
        self.folder = None

    def limit(self, energyRange, spType=None, inplace=True):
        if spType is None: spType = self.getDefaultSpType()
        ind = (energyRange[0] <= self._energy[spType]) & (self._energy[spType] <= energyRange[1])
        energy = self._energy[spType][ind]
        spectra = self._spectra[spType].to_numpy()[:, ind]
        spectra = utils.makeDataFrame(energy, spectra)
        self.folder = None
        if inplace:
            self._spectra[spType] = spectra
            self._energy[spType] = energy
        else:
            newSpectra = copy.deepcopy(self._spectra)
            newSpectra[spType] = spectra
            newEnergy = copy.deepcopy(self._energy)
            newEnergy[spType] = energy
            return Sample(self.params, newSpectra, newEnergy)

    def changeEnergy(self, energy, spType=None, inplace=True):
        if spType is None: spType = self.getDefaultSpType()
        if len(energy) == len(self._energy[spType]) and np.all(energy == self._energy[spType]): return
        spectra = np.zeros((len(self._spectra[spType]), len(energy)))
        selfspectra = self._spectra[spType].to_numpy()
        for i in range(len(self._spectra[spType])):
            spectra[i] = np.interp(energy, self._energy[spType], selfspectra[i])
        if inplace:
            self.setSpectra(spectra, energy, spType=spType)
        else:
            newSpectra = copy.deepcopy(self._spectra)
            newSpectra[spType] = spectra
            newEnergy = copy.deepcopy(self._energy)
            newEnergy[spType] = energy
            return Sample(self.params, newSpectra, newEnergy)

    def encode(self, columnName, labelEncoder=None):
        """Run LabelEncoder and returns dict: oldValue -> code as it is used in labelMaps. If labelEncoder is None - create one and return with label Maps"""
        assert columnName in self.paramNames
        labelEncoder0 = labelEncoder
        notNan = pd.notnull(self.params[columnName])
        if labelEncoder is None:
            labelEncoder = sklearn.preprocessing.LabelEncoder()
            labelEncoder.fit(self.params.loc[notNan,columnName])
        self.params.loc[notNan,columnName] = labelEncoder.transform(self.params.loc[notNan,columnName])
        labelMap = {c:i for i,c in enumerate(labelEncoder.classes_)}
        if labelEncoder0 is None: return labelMap, labelEncoder
        else: return labelMap

    def splitUnknown(self, columnNames=None, returnInd=False):
        """
        Divide sample into two parts: known (all columns are not NaN) and unknown (in each row there is NaN). Analyse only float64 columns
        :param columnNames: column name or list of names to analyse (default - all)
        :return: known, unknown
        """
        p = self.params
        if columnNames is None:
            columnsToAnalyse = p.select_dtypes(include=['float64'])
        else:
            if isinstance(columnNames, str): columnNames = [columnNames]
            columnsToAnalyse = p.loc[:, columnNames]
        nan = np.any(np.isnan(columnsToAnalyse), axis=1)
        if np.all(nan):
            known, unknown = None, self
        elif not np.any(nan):
            known, unknown = self, None
        else:
            s = self._spectra
            known = Sample(p.loc[~nan].reset_index(drop=True), {spType: s[spType].loc[~nan].reset_index(drop=True) for spType in s})
            unknown = Sample(p.loc[nan].reset_index(drop=True), {spType: s[spType].loc[nan].reset_index(drop=True) for spType in s})
        known.nameColumn = self.nameColumn
        if unknown is not None: unknown.nameColumn = self.nameColumn
        if returnInd:
            return known, unknown, np.where(~nan)[0], np.where(nan)[0]
        else:
            return known, unknown

    def plot(self, **kw):
        plotting.plotSample(self.energy, self.spectra, **kw)

    def convertEnergyToWeights(self):
        e = self.energy
        de2 = e[2:]-e[:-2]
        w = np.insert(de2, 0, e[1]-e[0])
        w = np.append(w, e[-1]-e[-2])
        return w/2


readSample = Sample.readFolder


def scoreFast(y,predictY):
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
    return scoreFast(y,predictY)


def score_cv(model, X, y, cv_count, returnPrediction=True):
    if isinstance(y, pd.Series): y = y.to_numpy().reshape(-1,1)
    if isinstance(X, pd.Series): X = X.to_numpy().reshape(-1,1)
    if len(y.shape) == 1: y = y.reshape(-1,1)
    if len(X.shape) == 1: X = X.reshape(-1,1)
    if cv_count < len(X)/4:
        cv = sklearn.model_selection.KFold(cv_count, shuffle=True)
    else:
        cv = sklearn.model_selection.LeaveOneOut()
    try:
        if len(warnings.filters) > 0:
            action = warnings.filters[0][0]
            warnings.filterwarnings("default")
        with warnings.catch_warnings(record=True) as warn:
            pred = sklearn.model_selection.cross_val_predict(model, X, y, cv=cv)
        if len(warnings.filters) > 0:
            warnings.filterwarnings(action)
    except Warning:
        pass
    res = sklearn.metrics.accuracy_score(y, pred) if isClassification(y) else sklearn.metrics.r2_score(y, pred)
    if returnPrediction: return res, pred
    else: return res


def getWeightsForNonUniformSample(x):
    """
    Calculates weights for each object x[i] = NN_dist^dim. These weights make uniform the error of ML models fitted on non-uniform samples
    """
    assert len(x.shape) == 2
    if len(x) <= 1: return np.ones(len(x))
    NNdists, _ = geometry.getNNdistsStable(x, x)
    w = NNdists**x.shape[1]
    w /= np.sum(w)
    w[w<1e-6] = 1e-6
    w /= np.sum(w)
    return w


def crossValidation(estimator, X, Y, CVcount, YColumnWeights=None, nonUniformSample=False):
    if isinstance(X, pd.DataFrame): X = X.to_numpy()
    if isinstance(Y, pd.DataFrame): Y = Y.to_numpy()
    if isinstance(Y, list): Y = np.array(Y)
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    if len(Y.shape) == 1: Y = Y.reshape(-1, 1)
    N = Y.shape[0]
    assert len(X) == N
    if YColumnWeights is None:
        YColumnWeights = np.ones((1,Y.shape[1]))
    if N > 20:
        kf = sklearn.model_selection.KFold(n_splits=CVcount, shuffle=True, random_state=0)
    else:
        kf = sklearn.model_selection.LeaveOneOut()
    predictedY = np.zeros(Y.shape)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index,:], X[test_index,:]
        y_train, y_test = Y[train_index, :], Y[test_index, :]
        estimator.fit(X_train, y_train)
        predictedY[test_index] = estimator.predict(X_test)
    if nonUniformSample: rowWeights = getWeightsForNonUniformSample(X)
    else: rowWeights = np.ones(N)
    individualErrors = np.array([np.sqrt(np.sum(np.abs(Y[i] - predictedY[i])**2 * YColumnWeights)) for i in range(N)])
    u = np.sum(individualErrors*rowWeights)
    y_mean = np.mean(Y, axis=0)
    v = np.sum(np.array([np.sqrt(np.sum(np.abs(Y[i] - y_mean)**2 * YColumnWeights)) for i in range(N)]) * rowWeights)
    error = u / v
    return error, individualErrors, predictedY


class RBF:
    def __init__(self, function='linear', baseRegression='quadric', scaleX=True, removeDublicates=False):
        """
        RBF predictor
        :param function: string. Possible values: multiquadric, inverse, gaussian, linear, cubic, quintic, thin_plate
        :param baseRegression: string, base estimator. Possible values: quadric, linear, None
        :param scaleX: bool. Scale X by gradients of y
        """
        # function: multiquadric, inverse, gaussian, linear, cubic, quintic, thin_plate
        # baseRegression: linear quadric
        self.function = function
        self.baseRegression = baseRegression
        self.trained = False
        self.scaleX = scaleX
        self.train_x = None
        self.train_y = None
        self.base = None
        self.scaleGrad = None
        self.minX = None
        self.maxX = None
        self.interp = None
        self.removeDublicates = removeDublicates

    def get_params(self, deep=True):
        return {'function': self.function, 'baseRegression': self.baseRegression, 'scaleX': self.scaleX}

    def set_params(self, **params):
        self.function = copy.deepcopy(params['function'])
        self.baseRegression = copy.deepcopy(params['baseRegression'])
        self.scaleX = copy.deepcopy(params['scaleX'])
        return self

    def fit(self, x, y):
        x = copy.deepcopy(x)
        y = copy.deepcopy(y)
        self.train_x = x.values if (type(x) is pd.DataFrame) or (type(x) is pd.Series) else x
        self.train_y = y.values if (type(y) is pd.DataFrame) or (type(y) is pd.Series) else y
        if len(self.train_y.shape) == 1: self.train_y = self.train_y.reshape(-1, 1)
        if self.baseRegression == 'quadric': self.base = makeQuadric(RidgeCV())
        elif self.baseRegression is None: self.base = None
        else:
            assert self.baseRegression == 'linear'
            self.base = RidgeCV()
        if self.scaleX:
            n = self.train_x.shape[1]
            self.minX = np.min(self.train_x, axis=0)
            self.maxX = np.max(self.train_x, axis=0)
            self.train_x = norm(self.train_x, self.minX, self.maxX)
            quadric = makeQuadric(RidgeCV())
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                quadric.fit(self.train_x, self.train_y)
            center_x = np.zeros(n)
            center_y = quadric.predict(center_x.reshape(1,-1))
            grad = np.zeros(n)
            for i in range(n):
                h = 1
                x2 = np.copy(center_x)
                x2[i] = center_x[i] + h
                y2 = quadric.predict(x2.reshape(1,-1))
                x1 = np.copy(center_x)
                x1[i] = center_x[i] - h
                y1 = quadric.predict(x1.reshape(1,-1))
                grad[i] = np.max([np.linalg.norm(y2 - center_y, ord=np.inf) / h, np.linalg.norm(center_y - y1, ord=np.inf) / h])
            if np.max(grad) == 0:
                if self.train_x.shape[0] > 2:
                    warnings.warn(f'Constant function. Gradient = 0. x.shape={self.train_x.shape}')
                self.scaleGrad = np.ones((1,n))
            else:
                grad = grad / np.max(grad)
                eps = 0.01
                if len(grad[grad <= eps]) > 0:
                    grad[grad <= eps] = np.min(grad[grad > eps]) * 0.01
                self.scaleGrad = grad.reshape(1,-1)
                self.train_x = self.train_x * self.scaleGrad
        if self.removeDublicates:
            # RBF crashes when dataset includes close or equal points
            self.train_x, uniq_ind = geometry.unique_mulitdim(self.train_x)
            self.train_y = self.train_y[uniq_ind,:]
        w = getWeightsForNonUniformSample(self.train_x)
        if self.baseRegression is not None:
            with warnings.catch_warnings():
                # warnings.simplefilter("ignore")
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                self.base.fit(self.train_x, self.train_y, sample_weight=w)
            self.train_y = self.train_y - self.base.predict(self.train_x)
        NdimsY = self.train_y.shape[1]
        assert NdimsY > 0
        self.interp = RBFInterpolator(self.train_x, self.train_y, kernel=self.function, degree=0)
        self.trained = True

    def predict(self, x):
        assert self.trained
        if type(x) is pd.DataFrame: x = x.values
        assert len(x.shape) == 2, f'x = '+str(x)
        assert x.shape[1] == self.train_x.shape[1], f'{x.shape[1]} != {self.train_x.shape[1]}'
        if self.scaleX:
            x = norm(x, self.minX, self.maxX)
            x = x * self.scaleGrad
        res = self.interp(x)
        if self.baseRegression is not None:
            res = res + self.base.predict(x)
        return res

    def score(self, x, y): return score(x,y,self.predict)


class RBFWrapper(RBF):
    def predict(self, x):
        result = RBF.predict(self, x).flatten()
        return result


def transformFeatures2Quadric(x, addConst=True):
    isDataframe = type(x) is pd.DataFrame
    if isDataframe:
        col_names = np.array(x.columns)
        x = x.values
    n = x.shape[1]
    new_n = n + n*(n+1)//2
    if addConst: new_n += 1
    newX = np.zeros([x.shape[0], new_n])
    newX[:,:n] = x
    if isDataframe:
        new_col_names = np.array(['']*newX.shape[1], dtype=object)
        new_col_names[:n] = col_names
    k = n
    for i1 in range(n):
        for i2 in range(i1,n):
            newX[:,k] = x[:,i1]*x[:,i2]
            if isDataframe:
                if i1 != i2:
                    new_col_names[k] = col_names[i1]+'*'+col_names[i2]
                else:
                    new_col_names[k] = col_names[i1] + '^2'
            k += 1
    if addConst:
        newX[:,k] = 1
        if isDataframe:
            new_col_names[k] = 'const'
        k += 1
        assert k == n + n*(n+1)//2 + 1
    else: assert k == n + n*(n+1)//2
    if isDataframe:
        newX = pd.DataFrame(newX, columns=new_col_names)
    return newX


def transformFeaturesAddDiff(x):
    return np.hstack((x, x[:,1:]-x[:,:-1]))
def transformFeaturesAddDiff2(x):
    dx = x[:,1:]/x[:,:-1]
    return np.hstack((x, dx, dx[:,1:]-dx[:,:-1]))


class makeQuadric:
    def __init__(self, learner):
        self.learner = learner

    def get_params(self, deep=True):
        return {'learner': self.learner}

    def set_params(self, **params):
        self.learner = copy.deepcopy(params['learner'])
        return self

    def fit(self, x, y, **args):
        x2 = transformFeatures2Quadric(x)
        self.learner.fit(x2, y, **args)

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
    """
    Do not norm columns in x for which minX == maxX
    :param x:
    :param minX:
    :param maxX:
    :return:
    """
    dx = maxX-minX
    ind = dx != 0
    res = copy.deepcopy(x)
    if type(x) is pd.DataFrame:
        res.loc[:, ind] = 2 * (x.loc[:, ind] - minX[ind]) / dx[ind] - 1
        res.loc[:,~ind] = 0
    else:
        if minX.size == 1:
            if dx != 0: res = 2 * (x - minX) / dx - 1
            else: res[:] = 0
        else:
            res[:, ind] = 2 * (x[:, ind] - minX[ind]) / dx[ind] - 1
            res[:, ~ind] = 0
    return res


def invNorm(x, minX, maxX):
    """
    Do not norm columns in x for which minX == maxX
    :param x:
    :param minX:
    :param maxX:
    :return:
    """
    dx = maxX - minX
    ind = dx != 0
    res = copy.deepcopy(x)
    if type(x) is pd.DataFrame:
        res.loc[:, ind] = (x.loc[:, ind]+1)/2*dx[ind] + minX[ind]
        res.loc[:, ~ind] = minX[~ind]
    else:
        if minX.size == 1:
            if dx != 0: res = (x+1)/2*(maxX-minX) + minX
            else: res[:] = minX
        else:
            res[:, ind] = (x[:, ind]+1)/2*dx[ind] + minX[ind]
            res[:, ~ind] = minX[~ind]
    return res


class Normalize:
    def __init__(self, learner, xOnly):
        self.learner = learner
        self.xOnly = xOnly
        if not hasattr(learner, 'name'): self.name = str(type(learner))
        else: self.name = 'normalized '+learner.name

    def get_params(self, deep=True):
        return {'learner':self.learner, 'xOnly':self.xOnly}

    def set_params(self, **params):
        self.learner = copy.deepcopy(params['learner'])
        self.xOnly= params['xOnly']
        return self

    def isFitted(self):
        return isFitted(self.learner)

    # args['xyRanges'] = {'minX':..., 'maxX':..., ...}
    def fit(self, x, y, **args):
        if isinstance(y,np.ndarray) and (len(y.shape)==1): y = y.reshape(-1,1)
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
        return self

    def predict(self, x, **predictArgs):
        if type(x) is pd.DataFrame: x = x.values
        yn = self.learner.predict(norm(x, self.minX, self.maxX), **predictArgs)
        if isinstance(yn, tuple):
            return (invNorm(yn[0], self.minY, self.maxY),) + yn[1:]
        else:
            return invNorm(yn,self.minY, self.maxY)

    def predict_proba(self, x):
        pyn = self.learner.predict_proba(norm(x, self.minX, self.maxX))
        return pyn

    def score(self, x, y): return score(x,y,self.predict)


class SeparateNorm:
    def __init__(self, learner, normLearner=None, normMethod='max'):
        """
        For multi-dimensional y normilize each y row by normMethod and predict normed_y and norm separately
        :param learner: base learner
        :param normMethod: 'max', 'mean', 'first', 'last' or function(y_row) to calculate norm
        """
        self.learner = learner
        self.normLearner = copy.deepcopy(learner) if normLearner is None else normLearner
        self.normMethod = normMethod
        self.norm = None
        if not hasattr(learner, 'name'): self.name = 'SeparateNorm '+str(type(learner))
        else: self.name = 'SeparateNorm '+learner.name

    def get_params(self, deep=True):
        return {'learner': self.learner, 'normLearner': self.normLearner, 'normMethod': self.normMethod}

    def set_params(self, **params):
        self.learner = copy.deepcopy(params['learner'])
        self.normLearner = copy.deepcopy(params['normLearner'])
        self.normMethod = params['normMethod']
        return self

    def normalize(self, sample):
        assert isinstance(sample, np.ndarray)
        if self.normMethod == 'max': norm = np.max(sample, axis=1)
        elif self.normMethod == 'mean': norm = np.mean(sample, axis=1)
        elif self.normMethod == 'first': norm = sample[:,0]
        elif self.normMethod == 'last': norm = sample[:,-1]
        else:
            assert callable(self.normMethod)
            norm = np.array([self.normMethod(sample[i]) for i in range(len(sample))])
        norm = norm.reshape(-1, 1)
        return sample/norm, norm

    # args['xyRanges'] = {'minX':..., 'maxX':..., ...}
    def fit(self, x, y, **args):
        assert len(y.shape) == 2 and y.shape[1] > 1
        if isinstance(y, pd.DataFrame): y = y.values
        y, self.norm = self.normalize(y)
        self.learner.fit(x,y)
        self.normLearner.fit(x, self.norm)
        return self

    def predict(self, x):
        yn = self.learner.predict(x)
        no = self.normLearner.predict(x)
        return yn*no

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
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv1D(100, 5, activation='relu', input_shape=(input_dim,1)))
        model.add(tf.keras.layers.Flatten())
        # model.add(tf.keras.layers.Dropout(0.3, seed=0))
        model.add(tf.keras.layers.Dense(units=10, kernel_initializer='normal', activation='relu'))
        model.add(tf.keras.layers.Dense(units=1, kernel_initializer='normal'))
        sgd = tf.keras.optimizers.SGD() # параметры lr=0.3, decay=0, momentum=0.9, nesterov=True уводят в nan
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
    if isinstance(cv, int):       
        kf = sklearn.model_selection.KFold(n_splits=cv, shuffle=True, random_state=0)
    else:
        kf = cv
    if type(X) is pd.DataFrame:
        X = X.to_numpy()
    if type(y) is pd.DataFrame:
        y = y.to_numpy()
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


class KrigingGaussianProcess:
    def __init__(self, n_restarts_optimizer=9, kernel=None, alpha=1e-10):
        if kernel is None:
            # kernel = sklearn.gaussian_process.kernels.ExpSineSquared()
            # kernel = sklearn.gaussian_process.kernels.RBF()
            pass
        self.model = GaussianProcessRegressor(n_restarts_optimizer=n_restarts_optimizer, kernel=kernel, alpha=alpha)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x, return_std=True)


class KrigingEnhancedGaussianProcess:
    def __init__(self, n_restarts_optimizer=9, kernel=None, alpha=1e-10):
        if kernel is None:
            # kernel = sklearn.gaussian_process.kernels.ExpSineSquared()
            # kernel = sklearn.gaussian_process.kernels.RBF()
            pass
        self.model = EnhancedGaussianProcessRegressor(n_restarts_optimizer=n_restarts_optimizer, kernel=kernel, alpha=alpha)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x, return_std=True)


class KrigingNNKCDE:
    def __init__(self, bandwidth=0.2, k=10):
        self.regressor = RBF(function='linear')
        self.nnkcde = None
        self.bandwidth = bandwidth
        self.k = k
        self.yBounds = None

    def fit(self, x, y):
        self.regressor.fit(x, y)
        self.nnkcde = NNKCDE(x, y)
        self.yBounds = [np.min(y, axis=0), np.max(y, axis=0)]

    def predict(self, x):
        assert self.yBounds[0].size == 1, str(self.yBounds[0].shape)
        y_grid = np.linspace(self.yBounds[0][0], self.yBounds[1][0], 100)
        dy = y_grid[1]-y_grid[0]
        dens = self.nnkcde.predict(x, y_grid, k=self.k, bandwidth=self.bandwidth)
        # print(dens.shape, x.shape, y_grid.shape)
        def int_dens(f):
            return np.sum(f*dens, axis=1)*dy / (np.sum(dens, axis=1)*dy)
        m = int_dens(y_grid)
        sigma = np.sqrt(int_dens(np.abs(y_grid.reshape(1,-1)-m.reshape(-1,1))**2))
        y = self.regressor.predict(x)
        # print(y)
        # print(m)
        # print(sigma)
        return y, sigma


class KrigingJointXY:
    def __init__(self, n_restarts_optimizer=9, kernel=None, alpha=1e-10):
        self.y_mult = 1
        if kernel is None:
            kernel = sklearn.gaussian_process.kernels.RationalQuadratic()
            # kernel = sklearn.gaussian_process.kernels.RBF()
            pass
        self.model = Normalize(GaussianProcessRegressor(n_restarts_optimizer=n_restarts_optimizer, kernel=kernel, alpha=alpha), xOnly=False)
        # self.model = Normalize(EnhancedGaussianProcessRegressor(n_restarts_optimizer=n_restarts_optimizer, kernel=kernel, alpha=alpha), xOnly=False)
        self.regressor = RBF()

    def fit(self, x0, y):
        self.y_mult = x0.shape[1]  # increase importance of y coordinate in multidimensional case
        self.regressor.fit(x0,y)
        x = np.hstack((x0,y*self.y_mult))
        self.model.fit(x, y)

    def predict(self, x0):
        y_regr = self.regressor.predict(x0)
        x = np.hstack((x0,y_regr*self.y_mult))
        # print(x.shape)
        return self.model.predict(x, return_std=True)


def isClassification(data, column=None):
    if column is not None: data = data[column].to_numpy()
    else:
        if isinstance(data, list):
            data = np.array(data)
    if data.dtype != 'float64': return True
    ind = ~np.isnan(data)
    return np.all(np.round(data[ind]) == data[ind]) and len(np.unique(data[ind]))<100


def plotPredictionError(x, y, params, method, pathToSave):
    '''
    Parameters
    ----------
    x : DataFrame
        Parameters of samples.
    y : DataFrame
        Smoothed spectra.
    params : [param1, param2]
        Parameters for which plot map.
    method : ml model
    pathToSave : folder name
        Name of folder in which save the figure.
    Returns
    -------
    None.
    '''
    
    cv_result = cross_val_predict(method, x, y, cv=sklearn.model_selection.LeaveOneOut())
    L2_norm = np.linalg.norm(cv_result - y, axis=1)
    fig, ax = plotting.createfig()
    sc = ax.scatter(x[params[0]], x[params[1]], cmap='plasma', c=L2_norm, alpha=0.5, norm=LogNorm())
    plt.colorbar(sc)
    ax.set_xlabel(params[0])
    ax.set_ylabel(params[1])
    plotting.savefig(f"./{pathToSave}/scatter-{params[0]}-{params[1]}.png", fig)
    plotting.closefig(fig)
    

def auc(true, pred):
    """
    Calculate AUC. Can work even for real-valued true and pred values
    :param true: true values (0,1) or true order defined by some real values
    :param pred: predicted values (or probabilities of 1) or predicted order defined by some real values
    """
    assert len(true) == len(pred)
    un = np.unique(true)
    assert len(un) > 1
    if len(un) == 2 and un[0] == 0 and un[1] == 1:
        return sklearn.metrics.roc_auc_score(true, pred)
    ind = np.argsort(pred)
    if isinstance(true, list): true = np.array(true)
    strue = true[ind]
    n = len(true)
    invCount = 0
    maxInvCount = 0
    strueTrue = np.sort(true)
    for i in range(n-1):
        invCount += np.sum(strue[i] > strue[i+1:])
        maxInvCount += np.sum(strueTrue[i] < strueTrue[i+1:])
    return 1-invCount/maxInvCount


def isFitted(estimator):
    if isinstance(estimator, sklearn.base.BaseEstimator):
        try:
            sklearn.utils.validation.check_is_fitted(estimator)
        except sklearn.exceptions.NotFittedError:
            return False
        return True
    if hasattr(estimator, 'isFitted') and callable(getattr(estimator, 'isFitted')):
        return estimator.isFitted()
    if hasattr(estimator, "classes_"): return True
    if 0 < len( [k for k,v in inspect.getmembers(estimator) if k.endswith('_') and not k.startswith('__')] ): return True
    assert hasattr(estimator, 'trained'), 'Your estimator is very unusual. Use your custom isFitted method'
    return estimator.trained

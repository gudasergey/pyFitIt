from typing import Optional
from scipy.interpolate import RBFInterpolator
import numpy as np
import pandas as pd
import math, copy, os, time, warnings, glob, sklearn, sklearn.linear_model, inspect, json, re, shutil, scipy, scipy.stats, scipy.special
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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
    def __init__(self, params, spectra, energy=None, spType='default', meta=None, encodeLabels=False, makeCommonEnergy=True, interpArgs=None):
        """
        Main data container for ML tasks
        :param params: DataFrame of geometry parameters
        :param spectra: list or Dataframe of spectra (column names: 'e_7443 e_7444.5 ...). For the case of several types of spectra - can be dict 'spectraType':list/spectraDataFrame. Spectra in list can have different energies
        :param energy: to initialize Sample by numpy matrix (to build correct column names of dataframe). Dict, if spectra is dict
        :param spType: spType name
        :param meta: dict with keys 'nameColumn', 'labels', 'labelMaps', 'features'. labelMaps=dict{label: map(str->int)}
        :param encodeLabels: whether to encode labels
        :param makeCommonEnergy: whether to interp all the spectra to make common energy
        :params interpArgs: np.interp arguments when interp all the spectra to make common energy
        """
        assert isinstance(params, pd.DataFrame), 'params should be pandas DataFrame object'
        params = copy.deepcopy(params)
        params = fixDtypes(params)
        self.paramNames = params.columns.to_numpy()
        self._params = params
        spectra = copy.deepcopy(spectra)
        if energy is not None: energy = copy.deepcopy(energy)
        meta = copy.deepcopy(meta)
        if not isinstance(spectra, dict):
            spectras = {spType:spectra}
            energies = {spType:energy}
        else:
            spectras = spectra
            energies = energy
            if energy is None: energies = {spType:None for spType in spectras}
            else: assert isinstance(energy, dict)
        self._spectra, self._energy = {}, {}
        first = True
        for spType in spectras:
            if first: sp0 = spectras[spType]
            else:
                assert len(sp0) == len(spectras[spType]), 'All spectra collections in dict must have the same count'
            first = False
            self.setSpectra(spectra=spectras[spType], energy=energies[spType], spType=spType, makeCommonEnergy=makeCommonEnergy, interpArgs=interpArgs)
        self.setDefaultSpType()
        if meta is None: meta = {'defaultSpType': self.defaultSpType}
        assert set(meta.keys()) <= {'nameColumn', 'labels', 'labelMaps', 'features', 'userDefined', 'defaultSpType'}
        if 'defaultSpType' in meta: self.defaultSpType = meta['defaultSpType']
        self._nameColumn: Optional[str] = None
        if 'nameColumn' in meta: self.setNameColumn(meta['nameColumn'])
        self.labels = []
        if 'labels' in meta:
            assert isinstance(meta['labels'], list)
            self.labels = meta['labels']
            for label in self.labels:
                assert label in self.paramNames, f'Label {label} not in params: {self.paramNames.tolist()}'
        self.labelMaps = {}
        if 'labelMaps' in meta:
            assert isinstance(meta['labelMaps'], dict)
            self.labelMaps = copy.deepcopy(meta['labelMaps'])
            for label in self.labelMaps:
                assert label in self.paramNames, f'Label {label} not in params: {self.paramNames.tolist()}'
                assert label in self.labels, f'Label {label} not in labels: {self.labels}'
                if encodeLabels:
                    # Check the data is not encoded yet
                    d = self.params[label]
                    d = d[pd.notnull(d)]
                    assert set(d) <= set(self.labelMaps[label].keys()), f'For label {label} unknown label values detected: {set(d)}. Known: {set(self.labelMaps[label].keys())}. If you have encoded data, set encodeLabels=False'
        self.features = []
        if 'features' in meta:
            assert isinstance(meta['features'], list)
            self.features = meta['features']
            for f in self.features:
                assert f in self.paramNames, f'Feature {f} not in params: {self.paramNames.tolist()}'
        if encodeLabels:
            if 'labelMaps' in meta:
                for label in meta['labelMaps']:
                    del self.labelMaps[label]
                    init_label = self.params[label]
                    self.encode(label, meta['labelMaps'][label])
                    decoded = self.decode(label, values=self.params[label])
                    ind = pd.notnull(init_label)
                    assert np.all(init_label[ind] == decoded[ind]), f'\n{init_label[ind].tolist()} !=\n{decoded[ind].tolist()}'
            else:
                self.calcLabelMaps()
        self.userDefined = None
        if 'userDefined' in meta:
            assert isinstance(meta['userDefined'], dict)
            self.userDefined = meta['userDefined']
        self.check()

    def check(self):
        nparam = self.params.shape[0]
        for spType,spectra in self._spectra.items():
            assert len(spectra) == nparam, f'{len(spectra)} != {nparam} for spType={spType}.\nAll spectra Sample must have the same count'
            if self.isCommonEnergy(spType):
                assert len(self.getEnergy(spType)) == spectra.shape[1], f'{len(self.getEnergy(spType))} != {spectra.shape[1]}'
        assert np.all(self.paramNames == self.params.columns)
        for l in self.labels: assert l in self.paramNames, f'{l} not in params: {self.paramNames.tolist()}'
        self.checkNameColumn(self.nameColumn)
        for label in self.labelMaps:
            assert label in self.paramNames, f'{l} not in params: {self.paramNames.tolist()}'
            self.checkEncodedLabel(label)
            assert self.params.dtypes[label] == 'float64', 'dtype of encoded label must be float, because int doesn\'t support NaN'
            # we can prepare dataframe by hand and then have not all the label values
            # lab_vals = sorted(list(self.labelMaps[label].keys()))
            # assert np.any(lab_vals != np.arange(len(lab_vals))), f'For label {label} trivial encoding detected: {self.labelMaps[label]}'

    def __len__(self):
        spType = self.getDefaultSpType()
        return len(self.getSpectra(spType))

    def __hash__(self):
        res = utils.hash((utils.hash(self._spectra), utils.hash(self.params), utils.hash(self.meta)))
        return res

    def getLength(self): return len(self)

    def spTypes(self):
        return sorted(list(self._spectra.keys()))

    def getDefaultSpType(self):
        return self.defaultSpType

    def setDefaultSpType(self, spType=None):
        if spType is None:
            if 'default' in self.spTypes(): spType = 'default'
            else: spType = self.spTypes()[0]
        assert spType in self.spTypes(), f'{spType} not in {self.spTypes()}'
        self.defaultSpType = spType

    def renameSpType(self, oldName, newName):
        assert oldName in self.spTypes(), f'{oldName} not in {self.spTypes()}'
        assert newName not in self.spTypes()
        self._spectra[newName] = self._spectra[oldName]
        self._energy[newName] = self._energy[oldName]
        del self._spectra[oldName]
        del self._energy[oldName]
        if self.getDefaultSpType() == oldName:
            self.setDefaultSpType(newName)

    def delSpType(self, spType):
        assert spType in self._spectra, f'spType {spType} not in sample spTypes: {self.spTypes()}'
        assert len(self.spTypes())>1, f'Can\'t delete the only existing spType'
        del self._spectra[spType]
        del self._energy[spType]
        if self.getDefaultSpType() == spType: self.setDefaultSpType()

    def setSpectra(self, spectra, energy=None, spType=None, makeCommonEnergy=True, interpArgs=None):
        """
        Setter for spectra
        :param spectra: DataFrame or np.ndarray (in last case energy should be given)
        :param energy:
        :param spType: spectrum type name for the case of multiple spectra matrixes inside one sample
        :param makeCommonEnergy: whether to interp all the spectra to make common energy
        :params interpArgs: np.interp arguments when interp all the spectra to make common energy: left=None, right=None, period=None
        """
        if spType is None: spType = self.getDefaultSpType()
        assert len(spectra) > 0
        if isinstance(spectra, np.ndarray):
            assert energy is not None
            assert len(energy) == spectra.shape[
                1], f'{len(energy)} != {spectra.shape[1]} energy vector must contain values for all columns of spectra matrix'
            spectra = pd.DataFrame(data=spectra, columns=['e_' + str(e) for e in energy])
        elif isinstance(spectra, list):
            for s in spectra:
                if energy is None:
                    assert isinstance(s, utils.Spectrum), type(s)
            if makeCommonEnergy:
                spectra = utils.makeDataFrameFromSpectraList(spectra, energy, interpArgs=interpArgs)
        else:
            assert isinstance(spectra, pd.DataFrame), 'Spectra should be pandas DataFrame object'
        assert len(self._params) == 0 or self._params.shape[0] == len(spectra), str(self._params.shape[0]) + ' != ' + str(len(spectra))
        self._spectra[spType] = spectra
        if isinstance(spectra, pd.DataFrame):
            self._energy[spType] = utils.getEnergy(spectra)
        else:
            self._energy[spType] = None

    def getSpectra(self, spType=None):
        if spType is None: spType = self.getDefaultSpType()
        return self._spectra[spType]

    spectra = property(getSpectra, setSpectra)

    def checkNameColumn(self, colName):
        if colName is not None:
            uniqNames, counts = np.unique(self.params[colName], return_counts=True)
            assert np.all(counts == 1), f'Duplicate names in column {colName} were found: ' + str(list(uniqNames[counts != 1]))

    def setNameColumn(self, colName):
        if colName is not None:
            assert colName in self.paramNames, f'{colName} not in params: {self.paramNames.tolist()}'
            assert self.params[colName].dtype == 'object', 'Wrong type of name column: ' + str(self.params[colName].dtype)
            for i in range(len(self.params)):
                if not isinstance(self.params.loc[i, colName], str):
                    self.params.loc[i, colName] = str(self.params.loc[i, colName])
            self.checkNameColumn(colName)
        self._nameColumn = colName

    nameColumn = property(lambda self: self._nameColumn, setNameColumn)

    def isCommonEnergy(self, spType=None):
        if spType is None: spType = self.getDefaultSpType()
        assert spType in self.spTypes(), f'Unknown spType: {spType}. All types: {self.spTypes()}'
        return self._energy[spType] is not None

    def getSpectrum(self, ind=None, name=None, spType=None, returnIntensityOnly=False):
        """
        By default returns spectrum of default spType. If spType=='all types' returns dict spType->spectrum
        """
        if name is None:
            assert ind is not None
        else:
            assert ind is None
            ind = self.getIndByName(name)
        res = {}
        for spT in self._spectra:
            if self.isCommonEnergy(spT):
                intensity = self._spectra[spT].loc[ind].to_numpy().reshape(-1)
                if returnIntensityOnly:
                    res[spT] = intensity
                else:
                    res[spT] = utils.Spectrum(self.getEnergy(spType=spT), intensity)
            else:
                res[spT] = self._spectra[spT][ind]
        if spType is None: spType = self.getDefaultSpType()
        if spType == 'all types': return res
        assert spType in res, f'Unknown spType: {spType}. All types: {self.spTypes()}'
        return res[spType]

    def getIndByName(self, name):
        assert self.nameColumn is not None
        if isinstance(name, str): names = [name]
        else: names = name
        ind = []
        for n in names:
            i = np.where(self._params[self.nameColumn].to_numpy() == n)[0]
            assert len(i) == 1, f'Indexes of {n}: {i}. All names: {self._params[self.nameColumn].tolist()}'
            ind.append(i[0])
        if isinstance(name, str): return ind[0]
        else: return ind

    def getName(self, i):
        assert self.nameColumn is not None
        return self._params.loc[i, self.nameColumn]

    def setSpectrum(self, spectrum, ind=None, name=None, spType=None):
        """
        Set spectrum
        :param ind: index
        :param name: name (alternative to index)
        :param spectrum: spectrum
        :param spType: if spType==None and there are several spTypes, spectrum should be dict of spectra for all spTypes
        """
        if name is None: assert ind is not None
        else:
            assert ind is None
            ind = self.getIndByName(name)
        if isinstance(spectrum, dict):
            assert set(spectrum.keys()) == set(self._spectra.keys())
            for k in spectrum:
                assert isinstance(spectrum[k], np.ndarray), str(spectrum[k])
                assert len(spectrum[k]) == len(self._energy[k])
        else:
            spType = self.getDefaultSpType()
            assert isinstance(spectrum, np.ndarray), str(spectrum)
            assert len(spectrum) == len(self._energy[spType])

        if spType is None:
            if len(self._spectra) > 1:
                assert isinstance(spectrum, dict)
                for spType in spectrum:
                    if self.isCommonEnergy(spType):
                        self._spectra[spType].loc[ind] = spectrum[spType]
                    else:
                        self._spectra[spType][ind] = spectrum[spType]
                return
        if isinstance(spectrum, dict):
            assert len(spectrum) == 1
            spectrum = spectrum[list(spectrum.keys())[0]]
        if self.isCommonEnergy(spType):
            self._spectra[spType].loc[ind] = spectrum
        else:
            self._spectra[spType][ind] = spectrum

    def setParams(self, params):
        assert isinstance(params, pd.DataFrame)
        self._params = params
        self.paramNames = params.columns.to_numpy()

    def getParams(self): return self._params

    params = property(getParams, setParams)

    def getEnergy(self, spType=None):
        if spType is None: spType = self.getDefaultSpType()
        assert spType in self.spTypes(), f'SpType {spType} is unknown. Known: {self.spTypes()}'
        return self._energy[spType]

    def setEnergy(self, energy, spType=None):
        assert False, 'Do not set energy explicitly. It is done after setting spectra'

    energy = property(getEnergy, setEnergy)

    @property
    def meta(self):
        res = {}
        if self.nameColumn is not None: res['nameColumn'] = self.nameColumn
        if len(self.labels) > 0: res['labels'] = self.labels
        if len(self.labelMaps) > 0: res['labelMaps'] = self.labelMaps
        if len(self.features) > 0: res['features'] = self.features
        if self.userDefined is not None: res['userDefined'] = self.userDefined
        res['defaultSpType'] = self.defaultSpType
        return res

    def shiftEnergy(self, shift, spType=None, inplace=False):
        if spType is None: spType = self.getDefaultSpType()
        assert self.isCommonEnergy(spType)
        newEnergy = self._energy[spType] + shift
        sam = self if inplace else self.copy()
        sam._energy[spType] = newEnergy
        sam._spectra[spType].columns = ['e_' + str(e) for e in newEnergy]
        if not inplace: return sam

    def changeEnergy(self, newEnergy, spType=None, inplace=False, interpArgs=None):
        if spType is None: spType = self.getDefaultSpType()
        assert self.isCommonEnergy(spType)
        if interpArgs is None: interpArgs = {}
        oldEnergy = self._energy[spType]
        sam = self if inplace else self.copy()
        spectra = np.zeros((self.getLength(), len(newEnergy)))
        oldSpectra = sam._spectra[spType].to_numpy()
        for i in range(self.getLength()):
            spectra[i] = np.interp(newEnergy, oldEnergy, oldSpectra[i], **interpArgs)
        sam.setSpectra(spectra, energy=newEnergy, spType=spType)
        if not inplace: return sam

    @classmethod
    def readFolder(cls, folder):
        h = cls.getSavedSampleHash(folder)
        binFile = folder+os.sep+'binary_repr.pkl'
        if os.path.exists(binFile):
            try:
                sample, savedHash = utils.loadData(binFile)
                if savedHash == h:  # user didn't change anything
                    return sample
            except:
                warnings.warn(f"Can't read binary sample file {binFile}, but it exists. Different python versions can cause this. I use text format")
                pass
        sampleFiles = getSampleFiles(folder)
        def readSpectra(f):
            ext = os.path.splitext(f)[-1]
            newFormat = True
            if ext == '.csv':
                res = pd.read_csv(f, sep='\t', header=None).to_numpy().T
                res = utils.makeDataFrame(res[0], res[1:])
            elif ext == '.txt':  # old format
                res = pd.read_csv(f, sep=r'\s+')
                newFormat = False
            else:
                assert ext == '.json'
                res = utils.loadData(f)
            return res, newFormat
        files = sampleFiles['spectra']
        if len(files) == 1 and os.path.split(files[0])[-1][:8] == 'spectra.':
            spectra, newFormat = readSpectra(files[0])
        else:
            spectra = {}
            for f in files:
                base = os.path.split(f)[-1]
                assert '_spectra' in base, 'Incorrect spectra file name '+f
                spType = base[:-len('_spectra.csv')]
                spectra[spType], newFormat = readSpectra(f)
        metaFile = sampleFiles.get('meta', None)
        if metaFile is not None and os.path.exists(metaFile):
            meta = utils.loadData(metaFile)
            if 'labelMaps' in meta:
                for label in meta['labelMaps']:
                    mp = meta['labelMaps'][label]
                    keys = list(mp.keys())
                    if np.all([re.match(r'\d+', k) for k in keys]):
                        meta['labelMaps'][label] = {int(k): mp[k] for k in mp}
        else: meta = None
        paramFile = sampleFiles['params']
        sep = '\t' if newFormat else r'\s+'
        res = cls(pd.read_csv(paramFile, sep=sep), spectra, meta=meta)
        return res

    @classmethod
    def getSavedSampleHash(cls, folder):
        sampleFiles = getSampleFiles(folder)
        h = ''
        for t in sampleFiles:
            files = sampleFiles[t]
            if not isinstance(files, list):
                assert isinstance(files, str)
                files = [files]
            for f in files:
                with open(f) as fo: s = fo.read()
                h += str(utils.hash(s))
        return utils.hash(h)

    def saveToFolder(self, folder, oldFormat=False, plot=False, **plotSampleKws):
        def saveSpectra(spType, spectra, filename):
            if oldFormat:
                spectra.to_csv(filename, index=False, sep=' ')
            else:
                if self.isCommonEnergy(spType):
                    e = utils.getEnergy(spectra)
                    data = np.hstack((e.reshape(-1,1), spectra.to_numpy().T))
                    data = pd.DataFrame(data)
                    data.to_csv(filename, header=False, sep='\t', index=False)
                else:
                    spectra = [[s.x.tolist(), s.y.tolist()] for s in spectra]
                    filename = os.path.splitext(filename)[0]+'.json'
                    utils.saveData(spectra, filename)

        if os.path.exists(folder): shutil.rmtree(folder)
        if not os.path.exists(folder): os.makedirs(folder)
        with open(folder+os.sep+'readme.txt','w') as f:
            f.write('''Pyfitit sample consists of\n- parameters table (params.csv) where each row corresponds to one sample object\n- spectrum tables of different spectrum types (spectrumType_spectra.csv files), the first column contains energy/wavelength values, other columns are spectra (number of columns in spectrumType_spectra.csv files equals to the number of rows in params.csv)\n- meta information in JSON format:\n  * nameColumn - name of the parameter to use as index in the sample (names should be short to be pretty displayed on scatter plots!)\n  * labels - list of parameter names to predict\n  * features - list of parameter names to use as sample object features\n  * labelMaps - dict of dicts with label encoding in the format "stringLabelValue":number\n- sample in binary format (binary_repr.pkl), it is used for round-trip save/load floats (text format is not round-trip and modify floats)''')
        ext = 'txt' if oldFormat else 'csv'
        if len(self._spectra) == 1 and self.getDefaultSpType() == 'default':
            saveSpectra('default', self.spectra, folder+os.sep+f'spectra.{ext}')
        else:
            for spType in self._spectra:
                saveSpectra(spType, self._spectra[spType], folder + os.sep + f'{spType}_spectra.{ext}')
        sep = ' ' if oldFormat else '\t'
        self.params.to_csv(folder+os.sep+f'params.{ext}', sep=sep, index=False)
        self.folder = folder
        utils.saveData(self.meta, folder+os.sep+'meta.json')
        utils.saveData((self, self.getSavedSampleHash(folder)), folder+os.sep+'binary_repr.pkl')
        if plot: self.plot(folder, **plotSampleKws)

    def copy(self):
        return Sample(self._params, self._spectra, meta=self.meta)

    def addParam(self, paramGenerator=None, paramName='', paramData=None):
        """
        Add new parameters to sample.params
        :param paramGenerator: function(paramDict) to calculate new params (single or multiple)
        :param paramName: name or list of new param names
        :param project: to call moleculeConstructor(sample.params) and pass molecula to paramGenerator
        :param paramData: already calculated params - alternative to paramGenerator
        """
        assert (paramData is None) or (paramGenerator is None)
        assert paramName != ''
        if isinstance(paramName, str):  # need to construct one parameter
            paramName = [paramName]
        for pn in paramName:
            assert pn not in self.paramNames, f'Parameter {pn} already exists'
        n = self.params.shape[0]
        if paramGenerator is None and paramData is None:
            paramData = np.zeros((n, len(paramName)))
            paramData[:,:] = np.nan
        if paramData is None:
            newParam = np.zeros((n, len(paramName)))
            for i in range(n):
                params = {self.paramNames[j]:self.params.loc[i,self.paramNames[j]] for j in range(self.paramNames.size)}
                t = paramGenerator(params)
                assert len(t) == len(paramName)
                newParam[i] = t
            for p,j in zip(paramName, range(len(paramName))): self.params[p] = newParam[:,j]
        else:
            if not isinstance(paramData, np.ndarray): paramData = np.array(paramData)
            assert paramData.shape[0] == n, f'{paramData.shape[0]} != {n}'
            if len(paramData.shape) == 1: paramData = paramData.reshape(-1,1)
            assert len(paramData.shape) == 2, f'len({paramData.shape}) != 2'
            for j,pn in enumerate(paramName):
                self.params[pn] = paramData[:,j]
        self.paramNames = self.params.columns.values

    def delParam(self, paramName):
        assert self.params.shape[1]>1, 'Can\'t delete last parameter'
        if isinstance(paramName, str): paramName = [paramName]
        for p in paramName:
            assert p in self.paramNames, f'{p} not in paramNames: {self.paramNames}'
            del self.params[p]
            if self.nameColumn == p: self.nameColumn = None
            if p in self.labels: del self.labels[self.labels.index(p)]
            if p in self.features: del self.features[self.features.index(p)]
            if p in self.labelMaps: del self.labelMaps[p]
        self.paramNames = self.params.columns.to_numpy()

    def delRow(self, i, inplace=True):
        if inplace:
            sample = self
        else:
            sample = self.copy()
        if isinstance(i, (np.ndarray, pd.Series)) and i.dtype == bool:
            i = np.where(i)[0]
        sample.params.drop(i, inplace=True)
        sample.params.reset_index(inplace=True, drop=True)
        for spType in sample._spectra:
            if self.isCommonEnergy(spType):
                sample._spectra[spType].drop(i, inplace=True)
                sample._spectra[spType].reset_index(inplace=True, drop=True)
            else: del sample._spectra[spType][i]
        if not inplace: return sample

    def delRowByName(self, names, inplace=True):
        if isinstance(names, str): names = [names]
        i = [self.getIndByName(n) for n in names]
        res = self.delRow(i, inplace=inplace)
        if not inplace: return res

    def takeRows(self, ind):
        if isinstance(ind, list): ind = np.array(ind)
        if ind.dtype == bool: ind = np.where(ind)[0]
        if len(ind.shape) != 1:
            assert np.prod(ind.shape) == np.max(ind.shape)
            ind = ind.flatten()
        spectra = {}
        for spType in self.spTypes():
            if self.isCommonEnergy(spType):
                spectra[spType] = self._spectra[spType].loc[ind].reset_index(drop=True, inplace=False)
            else:
                spectra[spType] = [self._spectra[spType][i] for i in ind]
        p = self.params.loc[ind].reset_index(drop=True, inplace=False)
        sample = Sample(params=p, spectra=spectra, meta=self.meta, makeCommonEnergy=False)
        return sample

    def takeRowsByName(self, names):
        ind = [self.getIndByName(name) for name in names]
        return self.takeRows(ind)

    def unionWith(self, other, inplace=False):
        if other is None: return
        # checks
        assert isinstance(other, self.__class__)
        assert np.all(self.params.shape[1] == other.params.shape[1]), 'Params differ: self = '+str(self.paramNames)+' other = '+str(other.paramNames)
        assert self.spTypes() == other.spTypes(), f'{self.spTypes()} != {other.spTypes()}'
        for spType in self._spectra:
            assert self.isCommonEnergy(spType) == other.isCommonEnergy(spType)
            if self.isCommonEnergy(spType):
                assert np.all(self._spectra[spType].shape[1] == other._spectra[spType].shape[1]), f'spType={spType} {self._spectra[spType].shape[1]} != {other._spectra[spType].shape[1]}'
                assert np.all(self._energy[spType] == other._energy[spType])
        assert set(self.paramNames) == set(other.paramNames), 'Params differ: self = ' + str(self.paramNames) + ' other = ' + str(other.paramNames)
        assert self.labelMaps == other.labelMaps, f'{self.labelMaps} != {other.labelMaps}'

        # union
        if inplace: res = self
        else: res = self.copy()
        res.params = pd.concat((self.params, other.params), ignore_index=True)
        for spType in self._spectra:
            if self.isCommonEnergy(spType):
                res._spectra[spType] = pd.concat((self._spectra[spType], other._spectra[spType]), ignore_index=True)
            else:
                res._spectra[spType] += other._spectra[spType]
        res.folder = None
        res.check()
        if not inplace: return res

    def addSpectrumType(self, spectra, spType, energy=None, makeCommonEnergy=True, interpArgs=None):
        assert spType not in self.spTypes(), f'Spectrum type {spType} already exists'
        assert self.getDefaultSpType() != 'default', 'Rename default spType by renameSpType before adding new one'
        self.setSpectra(spectra=spectra, energy=energy, spType=spType, makeCommonEnergy=makeCommonEnergy, interpArgs=interpArgs)

    def addRow(self, spectrum=None, params=None):
        i = self.params.shape[0]
        if spectrum is not None:
            if len(self.spTypes()) > 1:
                assert isinstance(spectrum, dict)
            if not isinstance(spectrum, dict): spectrum = {self.getDefaultSpType(): spectrum}
            for spType in spectrum:
                sp = spectrum[spType]
                if isinstance(sp, utils.Spectrum):
                    if self.isCommonEnergy(spType):
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
            if self.isCommonEnergy(spType):
                self._spectra[spType].loc[i] = spectrum[spType]
            else:
                assert isinstance(spectrum[spType], utils.Spectrum)
                self._spectra[spType].append(spectrum[spType])
        if params is not None:
            if isinstance(params, pd.Series): params = params.to_dict()
            assert isinstance(params, dict)
            assert set(params.keys()) <= set(self.paramNames), 'Unknown param names: ' + str(set(self.paramNames) - set(params.keys()))
            if self.nameColumn is not None and self.nameColumn in params:
                assert params[self.nameColumn] not in self.params[self.nameColumn], f'Spectrum with name {params[self.nameColumn]} already exists'
        else: params = {}
        self.params.loc[i, :] = np.nan
        for p in params: self.params.loc[i, p] = params[p]

    def limit(self, energyRange, spType=None, inplace=True):
        if spType is None: spType = self.getDefaultSpType()
        assert spType in self.spTypes(), f'Spectrum type {spType} not in {self.spTypes()}'
        assert self.isCommonEnergy(spType)
        ind = (energyRange[0] <= self._energy[spType]) & (self._energy[spType] <= energyRange[1])
        energy = self._energy[spType][ind]
        assert len(energy) > 0, f'There are no energy points in the interval {energyRange}. Energy = {self._energy[spType]}'
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
            return Sample(self.params, newSpectra, newEnergy, meta=self.meta)

    def inverseLabelMaps(self):
        r = {}
        for l, labelMap in self.labelMaps.items():
            r[l] = {labelMap[k]: k for k in labelMap}
        return r

    def decode(self, label: str, labelMap=None, values=None):
        if isinstance(values, pd.Series): values = values.to_numpy()
        if values is not None and not isinstance(values,np.ndarray):
            values = np.array(values)
        inplace = values is None
        if labelMap is None:
            labelMap = self.labelMaps[label]
        else:
            assert label not in self.labelMaps
        assert len(labelMap) > 0
        if inplace: values = self.params[label].to_numpy()
        else: assert utils.isArray(values)
        values0 = values

        assert utils.is_numeric(values[0]), f'Couldn\'t decode {values}. Don\'t you try to decode already decoded values?'
        nan_presented = np.any(np.isnan(values))
        if nan_presented:
            good_ind = np.where(~np.isnan(values))[0]
            if isinstance(values, list): values = np.array(values)
            values = values[good_ind]
            if len(values) == 0:
                if inplace and label in self.labelMaps: del self.labelMaps[label]
                return values0  # all values are NaNs
        if isinstance(values[0], (float, np.float64, np.int64)):
            int_values = values.astype(int)
            assert np.all(int_values == values), f''
            values = int_values

        k = list(labelMap.keys())[0]
        typ = object if isinstance(k, str) else float
        r = np.zeros(len(values), typ)
        inverse = {labelMap[k]:k for k in labelMap}
        # print(values[0], inverse)
        for i in range(len(values)):
            r[i] = inverse[values[i]]
        if nan_presented:
            r1 = np.zeros(len(values0), typ)
            r1[:] = np.nan
            r1[good_ind] = r
            r = r1
        if inplace:
            self.params[label] = r
            if label in self.labelMaps:
                del self.labelMaps[label]
        else: return r

    def decodeAllLabels(self):
        for l in self.labelMaps: self.decode(l)

    def encode(self, label, labelMap=None):
        """Run LabelEncoder and stores result in params and labelMaps"""
        assert label in self.paramNames
        assert label not in self.labelMaps, f'Label {label} is already encoded'
        res = encode(self.params[label], labelMap=labelMap)
        if labelMap is None:
            self.params[label] = res[0]
            self.labelMaps[label] = res[1]
        else:
            self.params[label] = res
            self.labelMaps[label] = labelMap

    def checkEncodedLabel(self, label):
        assert isClassification(self.params, label)
        d: np.ndarray = self.params[label].to_numpy()
        d = d[~np.isnan(d)]
        d = d.astype(int)
        assert set(d) <= set(self.labelMaps[label].values()), f'For label {label} unknown label values detected: {set(d)}. Known: {set(self.labelMaps[label].values())}. If you didn\'t encode data, set encodeLabels=True'

    def calcLabelMaps(self):
        for label in self.labels:
            if isClassification(self.params, label):
                needEncoding = True
                d = self.params[label].to_numpy()
                if d.dtype == 'float64':
                    ud = np.unique(d[~np.isnan(d)])
                    if np.all(ud == np.arange(ud.size)):
                        needEncoding = False
                if needEncoding:
                    assert label not in self.labelMaps, f'Label {label} is already encoded'
                    self.encode(label)

    def splitUnknown(self, columnNames=None, returnInd=False) -> ('Sample', 'Sample'):
        """
        Divide sample into two parts: known (all columns are not NaN) and unknown (in each row there is NaN). Analyse only float64 columns
        :param columnNames: column name or list of names to analyse (default - all)
        :param returnInd: whether to return indexes of known and unknown rows
        :return: known, unknown
        """
        p = self.params
        if columnNames is None:
            columnsToAnalyse = p.select_dtypes(include=['float64'])
        else:
            if isinstance(columnNames, str): columnNames = [columnNames]
            columnsToAnalyse = p.loc[:, columnNames]
        nan = np.any(pd.isnull(columnsToAnalyse), axis=1)
        if np.all(nan):
            known, unknown = None, self.copy()
        elif not np.any(nan):
            known, unknown = self.copy(), None
        else:
            s = self._spectra
            known = self.takeRows(~nan)
            unknown = self.takeRows(nan)
        if returnInd:
            return known, unknown, np.where(~nan)[0], np.where(nan)[0]
        else:
            return known, unknown

    def plot(self, folder, colorParam=None, spType='all spTypes', plotIndividualParams=None, maxIndivPlotCount=None, plotSampleParams=None):
        """
        :param plotIndividualParams: dict params of plotToFile to plot individual spectra (not on sample)
        :param plotSampleParams: plotSample params and individual plot params if plotIndividualParams['plot on sample'] == True
        """
        if spType == 'all spTypes': spTypes = self.spTypes()
        else:
            assert spType in self.spTypes()
            spTypes = [spType]
        if plotSampleParams is None: plotSampleParams = {}
        plotSampleParams = copy.deepcopy(plotSampleParams)
        if not (set(plotSampleParams.keys()) <= set(self.spTypes())):
            plotSampleParams1 = {}
            for spType in spTypes:
                plotSampleParams1[spType] = copy.deepcopy(plotSampleParams)
                if colorParam is not None and 'cmap' not in plotSampleParams1[spType]:
                    plotSampleParams1[spType]['cmap'] = 'gist_rainbow'
            plotSampleParams = plotSampleParams1
        for spType in spTypes:
            if spType not in plotSampleParams: plotSampleParams[spType] = {}
        if isinstance(colorParam, str):
            colorParamData = self.params[colorParam].to_numpy()
            if colorParam in self.labelMaps:
                colorParamData = self.decode(colorParam, values=colorParamData)
            colorParam = colorParamData
        for spType in spTypes:
            ext = os.path.splitext(folder)[-1].lower()
            if len(spTypes) == 1 and ext != '' and ext in ['.jpg','.png','.svg'] and plotIndividualParams is None:
                filename = folder
            else: filename = folder + os.sep + f'plot_{spType}.png'
            plotting.plotSample(self._energy[spType], self._spectra[spType], fileName=filename, colorParam=colorParam, **plotSampleParams[spType])
        if plotIndividualParams is not None:
            assert isinstance(plotIndividualParams, dict)
            if not (set(plotIndividualParams.keys()) <= set(self.spTypes())) or len(plotIndividualParams) == 0:
                plotIndividualParams = {spType:copy.deepcopy(plotIndividualParams) for spType in self.spTypes()}
            for spType in spTypes:
                for i in range(self.getLength()):
                    name = utils.zfill(i,self.getLength()) if self.nameColumn is None else self.params.loc[i,self.nameColumn]
                    fileName = f'{folder}{os.sep}individ_{spType}{os.sep}{name}.png'
                    s = self.getSpectrum(i, spType=spType)
                    if plotIndividualParams[spType].get('plot on sample', False):
                        if colorParam is not None:
                            fileName = [fileName, f'{folder}{os.sep}individ_{spType}_sorted{os.sep}{colorParam[i]}_{name}.png']
                        plotting.plotSample(self._energy[spType], self._spectra[spType], fileName=fileName, colorParam=colorParam, highlight_inds=[i], **plotSampleParams[spType])
                    else:
                        plotting.plotToFile(s.x,s.y,'', fileName=fileName, **plotIndividualParams[spType])
                    if maxIndivPlotCount is not None and i>=maxIndivPlotCount: break

    def convertEnergyToWeights(self):
        e = self.energy
        assert e is not None, 'There is no common energy'
        de2 = e[2:]-e[:-2]
        w = np.insert(de2, 0, e[1]-e[0])
        w = np.append(w, e[-1]-e[-2])
        return w/np.sum(w)

    def normalize(self, paramName, meanStd=None, inplace=True):
        """
        :param paramName: Name of param (or list of param names)
        :param meanStd: tuple(mean, std) or dict {paramName: tuple(mean, std)}
        """
        sample = self if inplace else self.copy()
        if isinstance(paramName, str):
            paramName = [paramName]
            if meanStd is not None and not isinstance(meanStd, dict): meanStd = {'paramName': meanStd}
        if meanStd is None:
            meanStd = calcMeanStd(sample.params.loc[:,paramName])
        for p in paramName:
            sample.params[p] = normMeanStd(sample.params[p], *meanStd[p])
        if not inplace: return sample

    def isOrdinal(self, l):
        if l not in self.labelMaps: return isOrdinal(self.params, l)
        lm = self.labelMaps[l]
        for k in lm:
            if isinstance(k, str): return False
        return True

    def makeCommonEnergy(self, spType=None, inplace=False, interpArgs=None):
        """
        :params interpArgs: arguments of the np.interp: left=None, right=None, period=None
        """
        assert not self.isCommonEnergy(spType), 'Energy is already common'
        if inplace: s = self
        else: s = self.copy()
        s.setSpectra(spectra=self.getSpectra(spType), spType=spType, makeCommonEnergy=True, interpArgs=interpArgs)
        if not inplace: return s


readSample = Sample.readFolder


def fixDtypes(d:pd.DataFrame):
    assert isinstance(d, pd.DataFrame)
    d = d.infer_objects()
    for c in d.columns:
        if d.dtypes[c] == 'int64': d[c] = d[c].astype(float)
    return d


def encode(vector, labelMap=None):
    """Run LabelEncoder and returns dict: oldValue -> code as it is used in labelMaps. If labelEncoder is None - create one and return with label Maps
        :param labelMap: dict{humanValue->index}
    """
    assert len(vector.shape) == 1
    notNan = pd.notnull(vector)
    labelMap0 = labelMap
    if labelMap0 is None:
        unique = np.unique(vector[notNan])
        if isinstance(unique[0], (np.int64, float)):  # json doesn't like int64 dict keys
            assert np.all([c == int(c) for c in unique])
            labelMap = {int(c): i for i, c in enumerate(unique)}
        else:
            labelMap = {c: i for i, c in enumerate(unique)}
    else:
        assert isinstance(labelMap, dict)
        assert set(vector[notNan]) <= set(labelMap.keys()), f'Unknown label values detected: {set(vector[notNan])}. Known: {set(labelMap.keys())}'
    r = np.zeros(vector.size)
    r[:] = np.nan
    for i in np.where(notNan)[0]:
        r[i] = labelMap[vector[i]]
    if labelMap0 is None: return r, labelMap
    else: return r


def scoreFast(y, predictY):
    if len(y.shape) >=2 and np.sum(y.shape != 1) == 1: y = y.flatten()
    if len(predictY.shape) >=2 and np.sum(predictY.shape != 1) == 1: predictY = predictY.flatten()
    assert np.all(y.shape == predictY.shape), f'{y.shape} != {predictY.shape}'
    return sklearn.metrics.r2_score(y, predictY, multioutput='uniform_average')
    # if len(y.shape) == 1:
    #     u = np.mean((y - predictY)**2)
    #     v = np.mean((y - np.mean(y))**2)
    # else:
    #     u = np.mean(np.linalg.norm(y - predictY, axis=1, ord=2)**2)
    #     v = np.mean(np.linalg.norm(y - np.mean(y, axis=0).reshape([1,y.shape[1]]), axis=1, ord=2)**2)
    # if v == 0: return 0
    # return 1-u/v


def score(x,y,predictor):
    predictY = predictor(x)
    return scoreFast(y,predictY)


def confidenceInterval(valueName, confidenceLevel, **args):
    assert valueName in ['ratio', 'sigma', 'R2score', 'MAE', 'max']
    alpha = (1 - confidenceLevel) / 2
    if valueName == 'ratio':
        numerator, denominator = args['numerator'], args['denominator']
        a, b = numerator, denominator-numerator
        c1, c2 = scipy.stats.beta.ppf([alpha,1-alpha], a+1, b+1)
        return [c1,c2]
    elif valueName == 'sigma':
        rmse, n = args['rmse'], args['n']
        c1, c2 = scipy.stats.chi2.ppf([alpha, 1-alpha], n)
        rmse_a, rmse_b = np.sqrt(n / c2) * rmse, np.sqrt(n / c1) * rmse
        return [rmse_a, rmse_b]
    elif valueName =='R2score':
        R2score, n = args['R2score'], args['n']
        u2_div_v2 = 1 - R2score
        c1, c2 = scipy.stats.f.ppf([alpha, 1 - alpha], n, n)
        rs_a, rs_b = u2_div_v2 / c2, u2_div_v2 / c1
        rs_a, rs_b = 1 - rs_b, 1 - rs_a
        return [rs_a, rs_b]
    elif valueName == 'MAE':
        # |pred-true| ~ half-normal distribution of N(0,sigma^2) with mean = sigma*sqrt(2/pi), we need confidence interval for mean. It equals to confidence interval of RMSE*sqrt(2/pi)
        rmse_a, rmse_b = confidenceInterval('sigma', confidenceLevel, **args)
        tmp = np.sqrt(2/np.pi)
        return [rmse_a*tmp, rmse_b*tmp]
    elif valueName == 'max':
        # TODO: MAX - функция распределения максимума = произведению функций распределения (т.е. все просто возводится в степень!)
        # проблема: в предыдущих случаях мы интервалы для матожиданий оценок (по определению confidence интервал - это случайный интервал для неслучайной характеристики сл. величины. А тут для чего? Для матожидания максимума? Легче для медианы: sigma*sqrt(2)*erf^-1(sqrtn(0.5)). Тогда мы выразим оценку медианы через оценку sigma - она у нас уже есть!
        rmse_a, rmse_b = confidenceInterval('sigma', confidenceLevel, **args)
        n = args['n']
        tmp = np.sqrt(2)*scipy.special.erfinv(0.5**(1/n))
        return [rmse_a*tmp, rmse_b*tmp]


def calcAllMetrics(y_true, y_pred, classification:bool, confidenceLevel=0.95):
    assert len(y_pred.shape) == 1 or y_pred.shape[1] == 1
    if isinstance(y_true, (pd.Series,pd.DataFrame)): y_true = y_true.to_numpy()
    if isinstance(y_pred, (pd.Series,pd.DataFrame)): y_pred = y_pred.to_numpy()
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    assert len(y_pred) == len(y_true)
    n = len(y_true)
    if classification:
        if y_pred.dtype == float:
            assert np.all(y_pred == np.round(y_pred)), str(y_pred)
            y_pred = y_pred.astype(int)
        aba = sklearn.metrics.balanced_accuracy_score(y_true, y_pred, adjusted=True)
        acc = sklearn.metrics.accuracy_score(y_true, y_pred)
        res = {'adj.bal.acc': aba, 'accuracy': acc}
        a = np.sum(y_true==y_pred)
        intervals = {'accuracy': confidenceInterval('ratio', confidenceLevel, numerator=a, denominator=n)}
        # TODO: adj.bal.acc - нужен интервал для суммы двух бета-распределений
    else:
        r_score = scoreFast(y_true, y_pred)
        mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
        max = np.max(np.abs(y_true - y_pred))
        rmse = sklearn.metrics.mean_squared_error(y_true, y_pred, squared=False)
        res = {'R2-score': r_score, 'MAE': mae, 'MAX': max, 'RMSE': rmse}
        intervals = {
            'R2-score': confidenceInterval('R2score', confidenceLevel, R2score=r_score, n=n),
            'RMSE': confidenceInterval('sigma', confidenceLevel, rmse=rmse, n=n),
            'MAE': confidenceInterval('MAE', confidenceLevel, rmse=rmse, n=n),
            'MAX':confidenceInterval('max', confidenceLevel, rmse=rmse, n=n)
        }

    for n in intervals: res[f'{n} interval'] = intervals[n]
    return res


def score_cv(model, X, y, cv_count, returnPrediction=True, random_state=0):
    if isinstance(y, pd.Series): y = y.to_numpy().reshape(-1,1)
    if isinstance(X, pd.Series): X = X.to_numpy().reshape(-1,1)
    if len(y.shape) == 1: y = y.reshape(-1,1)
    if len(X.shape) == 1: X = X.reshape(-1,1)
    if cv_count < len(X)/4:
        cv = sklearn.model_selection.KFold(cv_count, shuffle=True, random_state=random_state)
    else:
        cv = sklearn.model_selection.LeaveOneOut()
    try:
        action = utils.disableCatchWarnings()
        with warnings.catch_warnings(record=True) as warn:
            pred = sklearn.model_selection.cross_val_predict(model, X, y, cv=cv)
        utils.restoreCatchWarnings(action)
    except Warning:
        pass
    res = calcAllMetrics(y, pred, isClassification(y))
    if returnPrediction: return res, pred
    else: return res


def logCumRankError(y_true, y_pred):
    """
    Measure to estimate ranking quality of distance-based ML methods. y_pred contains distance values. y_true - correct classes (true - 1, wrong - 0).
    y_true is sorted using y_pred as keys. Then sum(y_true[i]*ln(1+i), i=0..) is calculated
    """
    assert len(y_pred) == len(y_true)
    ind = np.argsort(y_pred)
    y_true = y_true[ind]
    i = np.arange(len(y_pred))
    # return np.sum(y_true*np.log(i+1))
    # return np.sum(y_true*np.sqrt(i))
    return np.sum(y_true*i)


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
        YColumnWeights = np.ones(Y.shape[1])
        YColumnWeights = YColumnWeights/np.sum(YColumnWeights)
    if Y.shape[1]==1 and YColumnWeights is not None:
        assert len(YColumnWeights) == 1
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
    else: rowWeights = np.ones(N)/N
    individualErrors = np.array([np.sqrt(np.sum(np.abs(Y[i] - predictedY[i])**2 * YColumnWeights)) for i in range(N)])
    if Y.shape[1]==1: YColumnWeights = None
    error = 1 - sklearn.metrics.r2_score(Y, predictedY, sample_weight=rowWeights, multioutput=YColumnWeights)
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


def calcMeanStd(dataframe):
    res = {}
    for name in dataframe.columns:
        res[name] = (np.mean(dataframe[name]), np.std(dataframe[name]))
    return res


def normMeanStd(x, mean, std):
    res = x-mean
    if std != 0: res /= std
    return res


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
        if hasattr(self.learner, 'classes_'): self.classes_ = self.learner.classes_
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


def cross_val_predict(method, X, y, cv=10, predictFuncName='predict'):
    if isinstance(cv, int):       
        kf = sklearn.model_selection.KFold(n_splits=cv, shuffle=True, random_state=0)
    else:
        kf = cv
    if type(X) is pd.DataFrame:
        X = X.to_numpy()
    if type(y) is pd.DataFrame:
        y = y.to_numpy()
    if predictFuncName in ['predict_proba', 'decision_function']:
        assert np.all(y == y.astype(int))
        n_classes = int(np.max(y)+1)
        assert len(np.unique(y)) == n_classes, f'{len(np.unique(y))} != {n_classes}'
        assert n_classes > 1, f'X.shape = {X.shape}'
        predictions = np.zeros((y.shape[0], n_classes))
    else:
        predictions = np.zeros(y.shape)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index,:], X[test_index,:]
        y_train, y_test = y[train_index,:], y[test_index,:]
        if y_train.shape[1] == 1:
            y_train = y_train.reshape(-1)
        method.fit(X_train, y_train)
        if predictFuncName == 'predict_proba':
            predictions[test_index,:] = method.predict_proba(X_test)
        elif predictFuncName == 'decision_function':
            predictions[test_index,:] = method.decision_function(X_test)
        else:
            predictions[test_index, :] = method.predict(X_test).reshape(test_index.size, -1)
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


def isOrdinal(data, column=None):
    """
    Float or int features are ordinal, but string feature is not.
    """
    if column is not None: data = data[column].to_numpy()
    else:
        if isinstance(data, list):
            data = np.array(data)
    if data.dtype == 'float64': return True
    assert data.dtype == 'object'
    ind = pd.notnull(data)
    for v in data[ind]:
        if isinstance(v,str): return False
    assert False, "object feature but no string values?"


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


def validateNormByRef(expSample:Sample, theorySample:Sample, normFunc, spType, plotNormValuesFileName=None):
    """
    Returns logCumRankError for ordering theorySample by normFunc to each exp spectrum. Also returns list of commonNames of theory and exp spectra
    :param normFunc: func(expSpectrum, theorySpectrum, expParams, theoryParams)
    """
    assert expSample.nameColumn is not None
    assert theorySample.nameColumn is not None
    commonNames = np.intersect1d(theorySample.params[theorySample.nameColumn], expSample.params[expSample.nameColumn])
    theoryOrder = {}; unk = 0; n = len(commonNames)
    for name in theorySample.params[theorySample.nameColumn]:
        if name in commonNames: theoryOrder[name] = np.where(commonNames == name)[0][0]
        else:
            theoryOrder[name] = n + unk
            unk += 2
    theoryNames = [None]*theorySample.getLength()
    for name in theorySample.params[theorySample.nameColumn]: theoryNames[theoryOrder[name]] = name
    qualities = np.zeros(n)
    all_norm_vals = np.zeros((n, theorySample.getLength()))
    for i,exp_name in enumerate(commonNames):
        i_exp = expSample.getIndByName(exp_name)
        exp_sp = expSample.getSpectrum(i_exp, spType)
        # we take all theory spectra, not only common
        norm_vals = np.zeros(theorySample.getLength())
        for j_theor in range(theorySample.getLength()):
            theorySpectrum = theorySample.getSpectrum(j_theor, spType)
            if isinstance(exp_sp, dict):
                assert set(exp_sp.keys()) == set(theorySpectrum.keys()), f'{set(exp_sp.keys())} != {set(theorySpectrum.keys())}'
            norm_vals[j_theor] = normFunc(exp_sp, theorySpectrum, expSample.params.loc[i_exp], theorySample.params.loc[j_theor])
            # if i % 2 == 0:
            #     norm_vals[j_theor] = 4*(1 - (theorySample.params.loc[j_theor,theorySample.nameColumn] == exp_name))
            theory_name = theorySample.params.loc[j_theor,theorySample.nameColumn]
            all_norm_vals[i,theoryOrder[theory_name]] = norm_vals[j_theor]
        true_classes = np.zeros(theorySample.getLength())
        true_classes[theorySample.getIndByName(exp_name)] = 1
        qualities[i] = logCumRankError(true_classes, norm_vals)
    if plotNormValuesFileName is not None:
        plotting.plotMatrix(all_norm_vals, ticklabelsX=theoryNames, ticklabelsY=commonNames, fileName=plotNormValuesFileName, xlabel='theory', ylabel='exp')
    return qualities, commonNames


def validateNormByClasses(expSample:Sample, theorySample:Sample, normFunc, classColumn, spType, plotNormValuesFileName=None):
    """
    Returns logCumRankError for ordering theorySample by normFunc to each exp spectrum. Classes of exp and theory spectra can be duplicate
    :param normFunc: func(expSpectrum, theorySpectrum, expParams, theoryParams)
    :param classColumn: name of column in sample to use as class value
    """
    assert classColumn in expSample.paramNames
    assert classColumn in theorySample.paramNames
    theoryNames = theorySample.params[classColumn]
    n = len(expSample)
    nt = theorySample.getLength()
    qualities = np.zeros(n)
    all_norm_vals = np.zeros((n, nt))
    for i in range(n):
        exp_name = expSample.params.loc[i,classColumn]
        exp_sp = expSample.getSpectrum(i, spType)
        norm_vals = np.zeros(nt)
        for j_theor in range(nt):
            theorySpectrum = theorySample.getSpectrum(j_theor, spType)
            if isinstance(exp_sp, dict):
                assert set(exp_sp.keys()) == set(theorySpectrum.keys()), f'{set(exp_sp.keys())} != {set(theorySpectrum.keys())}'
            norm_vals[j_theor] = normFunc(exp_sp, theorySpectrum, expSample.params.loc[i], theorySample.params.loc[j_theor])
            all_norm_vals[i,j_theor] = norm_vals[j_theor]
        true_classes = np.zeros(nt)
        true_classes[theoryNames == exp_name] = 1
        qualities[i] = logCumRankError(true_classes, norm_vals)
    if plotNormValuesFileName is not None:
        plotting.plotMatrix(all_norm_vals, ticklabelsX=theoryNames, ticklabelsY=expSample.params.loc[:,classColumn], fileName=plotNormValuesFileName, xlabel='theory', ylabel='exp')
    return qualities


class MyBaggingClassifier(sklearn.base.BaseEstimator):
    def __init__(self, estimator, max_samples=0.5, max_features=0.5, n_estimators=10, random_state=None):
        self.n_estimators = n_estimators
        self.estimator = estimator
        self.estimators = [copy.deepcopy(self.estimator) for i in range(self.n_estimators)]
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state if random_state is not None else time.time()
        self.classes = None
    # def get_params(self, deep=True):
    #     e = copy.deepcopy(self.estimator) if deep else self.estimator
    #     return dict(estimator=e, max_samples=self.max_samples, max_features=self.max_features, n_estimators=self.n_estimators)

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.classes_ = self.classes
        for i in range(self.n_estimators):
            # y1 = [1]
            # while len(np.unique(y)) != len(np.unique(y1)):
            X1, y1 = sklearn.utils.resample(X,y, replace=False, n_samples=round(len(X)*self.max_samples), stratify=y)
            self.estimators[i].fit(X1, y1)
        return self

    def predict_proba(self, X):
        pred = None
        for i in range(self.n_estimators):
            p = self.estimators[i].predict_proba(X)
            if pred is None: pred = p
            else: pred += p
        return pred/self.n_estimators

    def decision_function(self, X):
        return self.predict_proba(X)

    def predict(self, X):
        prob = self.predict_proba(X)
        res = np.zeros(len(X), dtype=type(self.classes[0]))
        for i in range(len(X)):
            res[i] = self.classes[np.argmax(prob[i])]
        return res


class FixedClassesClassifier(sklearn.base.ClassifierMixin):
    def __init__(self, estimator, classes):
        self.estimator = estimator
        self.classes = classes
        if np.any(sorted(classes) != classes):
            warnings.warn(f'Classes are not sorted. Do you sure? Classes: {classes}')
        self.fit_classes = None

    def fit(self, X, y):
        self.fit_classes = np.unique(y)
        assert set(self.fit_classes) <= set(self.classes), f'{self.fit_classes} is not subset of {self.classes}'
        self.estimator.fit(X,y)
        assert np.all(self.fit_classes == self.estimator.classes_)
        return self

    def predict_proba(self, X):
        try:
            p = self.estimator.predict_proba(X)
            decision_function = False
        except AttributeError:
            p = self.estimator.decision_function(X)
            decision_function =True
        if len(self.fit_classes) == 2 and decision_function:
            p = np.hstack(((-p).reshape(-1,1), p.reshape(-1,1)))
        if len(self.fit_classes) != len(self.classes) or np.any(self.fit_classes != self.classes):
            r = np.zeros((p.shape[0], len(self.classes)))
            for jf,c in enumerate(self.fit_classes):
                j = [t for t,el in enumerate(self.classes) if el==c][0]
                r[:,j] = p[:,jf]
            return r
        else: return p

    def predict(self, X):
        return self.estimator.predict(X)

    def decision_function(self, X):
        return self.predict_proba(X)


def getSampleFiles(folder):
    paramFile = utils.fixPath(folder + os.sep + 'params')
    paramFiles = glob.glob(paramFile + '.*')
    assert len(paramFiles) <= 1, 'Multiple params files detected: ' + str(paramFiles)
    assert len(paramFiles) > 0, 'Param file not found in the folder ' + folder
    paramFile = paramFiles[0]
    files = glob.glob(folder + os.sep + '*spectra.*')
    files = [f for f in files if os.path.splitext(f)[-1] in ['.txt', '.csv', '.json']]
    assert len(files) > 0, 'No spectrum files were found in the folder ' + folder
    res = {'params':paramFile, 'spectra':files}
    meta = folder + os.sep + 'meta.json'
    if os.path.exists(meta): res = {**res, 'meta':meta}
    return res


def pairwiseTransform(X,y, maxSize=None, randomSeed=None, pairwiseTransformType='binary'):
    assert len(X) == len(y)
    assert pairwiseTransformType in ['binary', 'numerical']
    assert y.shape[1] == 1
    n = len(X)
    ind = utils.comb_index(n,2,repetition=False)
    if maxSize is not None and len(ind) > maxSize:
        rng = np.random.default_rng(randomSeed)
        perm = rng.permutation(len(ind))
    else: perm = np.arange(len(ind))
    Xp, yp = [], []
    for ii in range(len(ind)):
        i = perm[ii]
        j1, j2 = ind[i]
        if y[j1] == y[j2]: continue
        dy = y[j1]-y[j2]
        if pairwiseTransformType == 'binary': dy = np.sign(dy)
        dX = X[j1] - X[j2]
        Xp.append(dX)
        yp.append(dy)
        Xp.append(-dX)
        yp.append(-dy)
        if len(yp) >= maxSize: break
    Xp, yp = map(np.asanyarray, (Xp, yp))
    return Xp, yp.reshape(-1,1)

def pairwiseCV(model, X, y, testRatio=0.1, cvTryCount=300, pairwiseTransformType='numerical', maxTrainSize=1000, maxTestSize=100, lossFunc='RMSE'):
    """
    Run cross validation for pairwise problem

    :param lossFunc: 'RMSE' for numerical pairwiseTransformType, 'accuracy' or 'AUC' - for binary
    :returns: loss
    """
    assert 0<testRatio<1
    if len(y.shape) == 1: y = y.reshape(-1,1)
    n = len(X)
    testSize = max(2, int(np.round(testRatio*n)))
    y_true, y_pred = np.array([]), np.array([])
    rng = np.random.default_rng(0)
    for cv_try in range(min(cvTryCount, scipy.special.comb(n,testSize))):
        test_index = rng.choice(n, testSize, replace=False)
        train_index = np.setdiff1d(np.arange(n), test_index)
        X_train, X_test = X[train_index,:], X[test_index,:]
        y_train, y_test = y[train_index,:], y[test_index,:]
        if len(np.unique(y_test)) == 1: continue
        Xp, yp = pairwiseTransform(X_train, y_train, maxSize=maxTrainSize, randomSeed=0, pairwiseTransformType=pairwiseTransformType)
        model.fit(Xp, yp)
        # print(t1-t0, t2-t1)
        Xp_test, yp_test = pairwiseTransform(X_test, y_test, maxSize=maxTestSize, randomSeed=0, pairwiseTransformType=pairwiseTransformType)
        y_true = np.append(y_true, yp_test.flatten())
        if pairwiseTransformType == 'numerical':
            y_pred = np.append(y_pred, model.predict(Xp_test).flatten())
        else:
            y_pred = np.append(y_pred, model.predict_proba(Xp_test)[:,1])
    if lossFunc == 'RMSE':
        assert pairwiseTransformType == 'numerical'
        return np.sqrt(np.mean((y_true-y_pred)**2))
    elif lossFunc == 'AUC':
        assert pairwiseTransformType == 'binary'
        return sklearn.metrics.roc_auc_score(y_true, y_pred)
    else:
        assert pairwiseTransformType == 'binary'
        assert lossFunc == 'accuracy', f'Unknown loss function {lossFunc}'
        y_pred1 = -np.ones(len(y_true))
        y_pred1[y_pred>=0.5] = 1
        return np.sum(y_true == y_pred1)/len(y_true)


def pairwiseRidgeCV(X, y, alphas=(0.1,1,10,100,1000,100*1000), testRatio=0.1):
    assert 0<testRatio<1
    losses = []
    for alpha in alphas:
        model = sklearn.linear_model.Ridge(alpha=alpha, fit_intercept=False, random_state=0)
        loss = pairwiseCV(model, X, y, testRatio=testRatio, cvTryCount=300, pairwiseTransformType='numerical', maxTrainSize=1000, maxTestSize=100, lossFunc='RMSE')
        losses.append(loss)
    best_i = np.argmin(losses)
    Xp, yp = pairwiseTransform(X, y, maxSize=10000, randomSeed=0, pairwiseTransformType='numerical')
    print('best alpha =', alphas[best_i], losses)
    model = sklearn.linear_model.Ridge(alpha=alphas[best_i], fit_intercept=False, random_state=0)
    model.fit(Xp, yp)
    return model


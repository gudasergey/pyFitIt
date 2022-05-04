from . import utils
utils.fixDisplayError()
import numpy as np
import pandas as pd
import sklearn, json, os, copy, gc, matplotlib, warnings
from . import plotting, ML, smoothLib, fdmnes
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt

allowedMethods = ['Ridge', 'Ridge Quadric', 'Extra Trees', 'RBF']
# if utils.isLibExists("lightgbm"):
#     import lightgbm as lgb
#     allowedMethods.append('LightGBM')
if utils.isLibExists("keras"):
    allowedMethods.append('NN')


def expectation(prob, paramRange, classNum):
    assert classNum == prob.shape[1], 'Invrease dataset size or decrease cross validation number. classNum = '+str(classNum)+' prob.shape = '+str(prob.shape)
    a = paramRange[0]; b = paramRange[1]
    expect = np.zeros(prob.shape[0]);
    for ii in range(classNum):
        border1 = a + (b-a)/classNum*ii; border2 = a + (b-a)/classNum*(ii+1)
        expect += 0.5*prob[:,ii]*(border2**2 - border1**2)
    return expect


def makeClasses(x, paramRange, classNum):
    a = paramRange[0]; b = paramRange[1]
    cl = np.floor((x-a)/(b-a)*classNum)
    cl[cl==classNum] = classNum-1
    return cl


def classCV(classifier, classNum, sample, paramInd, CVcount):
    kf = KFold(n_splits=CVcount, shuffle=True, random_state=0)
    X = sample.spectra
    paramName = sample.paramNames[paramInd]
    y = sample.params[paramName].values
    paramRange = [np.min(y), np.max(y)]
    y = makeClasses(y, paramRange, classNum)
    RMSE = np.zeros(CVcount); i = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)
        predProba1 = classifier.predict_proba(X_test)
        predProba = np.zeros((X_test.shape[0], classNum))
        for j in range(classNum):
            if j in classifier.classes_:
                classInd = np.where(classifier.classes_==j)[0][0]
                predProba[:,j] = predProba1[:,classInd]
            else:
                predProba[:,j] = 0
        expect = expectation(predProba, paramRange, classNum)
        RMSE[i] = np.sqrt( np.mean( (y_test-expect)**2 ) )
        i += 1
    return np.mean(RMSE[:i])


def chooseClassCount(classifier, sample, CVcount):
    n = sample.params.shape[1]
    bestClassCounts = np.zeros(n)
    for i in range(n):
        classNum = 2
        newError = classCV(classifier, classNum, sample, i, CVcount)
        lastError = newError+1
        while (newError < lastError) and (classNum<32):
            lastError = newError
            classNum = classNum*2
            newError = classCV(classifier, classNum, sample, i, CVcount)
        if newError < lastError: bestClassCounts[i] = classNum
        else: bestClassCounts[i] = classNum/2
    return bestClassCounts


# when estimator is already fitted
def directCrossValidationFast(geometryParamsTest, xanesTest, estimator):
    paramNames = geometryParamsTest.columns.values
    X_test = xanesTest.values
    y_test = geometryParamsTest.values
    prediction = estimator.predict(X_test)
    res = {}
    for i in range(len(paramNames)):
        relativeToConstantError = 1-ML.scoreFast(y_test[:,i], prediction[:,i])
        RMSE = np.sqrt( np.mean( (y_test[:,i]-prediction[:,i])**2 ) )
        MAE = np.mean( np.abs(y_test[:,i]-prediction[:,i]) )
        res[paramNames[i]] = {'relToConstPredError':relativeToConstantError, 'RMSE':RMSE, 'MAE':MAE}
    return res


def directCrossValidation(geometryParamsTrain, geometryParamsTest, xanesTrain, xanesTest, estimator, xyRanges={}):
    if len(xyRanges)>0: estimator.fit(xanesTrain, geometryParamsTrain, xyRanges=xyRanges)
    else: estimator.fit(xanesTrain, geometryParamsTrain)
    return directCrossValidationFast(geometryParamsTest, xanesTest, estimator)


def KFoldCrossValidation(regressor, sample, CVcount):
    kf = KFold(n_splits=CVcount, shuffle=True, random_state=0)
    X = sample.spectra;  y = sample.params
    minX = np.min(X, axis=0);  maxX = np.max(X, axis=0)
    minY = np.min(y, axis=0);  maxY = np.max(y, axis=0)
    xyRanges = {'minX':minX, 'maxX':maxX, 'minY':minY, 'maxY':maxY}
    res = None
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        part = directCrossValidation(y_train, y_test, X_train, X_test, regressor, xyRanges=xyRanges)
        if res is None: res = part
        else:
            for pName in res:
                for errName in res[pName]: res[pName][errName] += part[pName][errName]
    for pName in res:
        for errName in res[pName]: res[pName][errName] /= CVcount
    return res


def recommendedParams(name):
    if name == "Ridge":
        params = {'alphas': [0.01, 0.1, 1, 10, 100]}
        allParams = list(RidgeCV().get_params().keys())
    elif name == "Ridge Quadric":
        params = {'alphas': [0.01, 0.1, 1, 10, 100]}
        allParams = list(RidgeCV().get_params().keys())
    elif name == 'Extra Trees':
        params = {'n_estimators':100, 'random_state':0, 'min_samples_leaf':10}
        allParams = list(ExtraTreesRegressor().get_params().keys())
    elif name[:3] == "RBF":
        params = {'function':'linear', 'baseRegression': 'linear'}
        allParams = list(params.keys())
    elif name == 'LightGBM':
        params = {'num_leaves':31, 'learning_rate':0.02, 'n_estimators':100}
        allParams = list(lgb.LGBMRegressor().get_params().keys())
    elif name == 'NN':
        params = {'epochs':50, 'batch_size':32, 'showProgress':False}
        allParams = ['epochs', 'batch_size']
    else: assert False
    return params, allParams


def getMethod(name, params0={}):
    rparams, allParams = recommendedParams(name)
    params = {p: params0[p] for p in params0 if p in allParams}
    for p in rparams:
        if p not in params: params[p] = rparams[p]
    if name == "Ridge": regressor = RidgeCV(**params)
    elif name == "Ridge Quadric": regressor = ML.makeQuadric(RidgeCV(**params))
    elif name == 'Extra Trees': regressor = ExtraTreesRegressor(**params)
    elif name[:3] == "RBF": regressor = ML.RBF(**params)
    elif name == 'LightGBM': regressor = ML.makeMulti(lgb.LGBMRegressor(objective='regression', verbosity=-1, **params))
    elif name == 'NN': regressor = ML.makeMulti(ML.NeuralNetDirect(**params))
    else: assert False
    return regressor


def prepareSample(sample0, diffFrom, project, norm):
    sample = copy.deepcopy(sample0)
    sample.spectra = smoothLib.smoothDataFrame(project.FDMNES_smooth, sample.spectra, 'fdmnes', project.spectrum, project.intervals['fit_smooth'], norm=norm, folder=sample.folder)
    gc.collect()
    if diffFrom is not None:
        sample.spectra = (sample.spectra - diffFrom['spectrumBase'].intensity) * diffFrom['purity']
        gc.collect()
    return sample


def prepareDiffFrom(project, diffFrom, norm):
    diffFrom = copy.deepcopy(diffFrom)
    diffFrom['projectBase'].spectrum.intensity = np.interp(project.spectrum.energy, diffFrom['projectBase'].spectrum.energy, diffFrom['projectBase'].spectrum.intensity)
    diffFrom['projectBase'].spectrum.energy = project.spectrum.energy
    diffFrom['spectrumBase'], _ = smoothLib.funcFitSmoothHelper(project.defaultSmoothParams['fdmnes'], diffFrom['spectrumBase'], 'fdmnes', diffFrom['projectBase'], norm)
    return diffFrom


class Estimator:

    # diffFrom = {'projectBase':..., 'spectrumBase':..., 'purity':...}
    def __init__(self, name, exp, convolutionParams, normalize=True, CVcount=10, diffFrom=None, classifierParams={}, regressorParams={}, probabilityIntervals='auto'):
        if name not in allowedMethods:
            raise Exception('Unknown method name. You can use: '+str(allowedMethods)+'. Recommendation: RBF')
        self.exp = copy.deepcopy(exp)
        interval = self.exp.intervals['fit_geometry']
        ind = (self.exp.spectrum.energy >= interval[0]) & (self.exp.spectrum.energy <= interval[1])
        self.exp.spectrum.energy = self.exp.spectrum.energy[ind]
        self.exp.spectrum.intensity = self.exp.spectrum.intensity[ind]
        self.convolutionParams = copy.deepcopy(convolutionParams)
        if 'norm' in self.convolutionParams:
            self.norm = self.convolutionParams['norm']
            del self.convolutionParams['norm']
        else: self.norm = None
        for pName in self.convolutionParams:
            self.exp.defaultSmoothParams['fdmnes'][pName] = self.convolutionParams[pName]
        self.regressor = getMethod(name, regressorParams)
        self.classifier0 = ExtraTreesClassifier(**classifierParams)
        self.normalize = normalize
        if normalize:
            self.regressor = ML.Normalize(self.regressor, xOnly=False)
        self.CVcount = CVcount
        self.probabilityIntervals = probabilityIntervals
        assert CVcount >= 2
        self.diffFrom = copy.deepcopy(diffFrom)
        if diffFrom is not None:
            self.diffFrom = prepareDiffFrom(self.exp, diffFrom, self.norm)
            self.expDiff = copy.deepcopy(self.exp)
            self.expDiff.spectrum = utils.Spectrum(self.expDiff.spectrum.energy, self.exp.spectrum.intensity - self.diffFrom['projectBase'].spectrum.intensity)

    def prepareSpectrumForPrediction(self, xanes, smooth):
        if smooth:
            smoothed_xanes, _ = smoothLib.funcFitSmoothHelper(self.exp.defaultSmoothParams['fdmnes'], xanes, 'fdmnes', self.exp, self.norm)
            xanesAbsorb = smoothed_xanes.intensity
        else:  # xanes - is experimental data
            if self.diffFrom is None:
                xanes = copy.deepcopy(xanes)
                interval = self.exp.intervals['fit_geometry']
                ind = (xanes.energy >= interval[0]) & (xanes.energy <= interval[1])
                xanes.energy = xanes.energy[ind]
                xanes.intensity = xanes.intensity[ind]
                xanesAbsorb = xanes.intensity
                xanesAbsorb = np.interp(self.exp.spectrum.energy, xanes.energy, xanesAbsorb)
        if self.diffFrom is not None:
            if smooth: xanesAbsorb = (xanesAbsorb - self.diffFrom['spectrumBase'].intensity)*self.diffFrom['purity']
            else:
                xanesAbsorb = np.interp(self.exp.spectrum.energy, xanes.energy, xanes.intensity)
                xanesAbsorb = xanesAbsorb - self.diffFrom['projectBase'].spectrum.intensity
        if len(xanesAbsorb.shape) == 1: xanesAbsorb = xanesAbsorb.reshape(-1,1).T
        return xanesAbsorb

    def fit(self, sample):
        sample = prepareSample(sample, self.diffFrom, self.exp, self.norm)
        self.xanes_energy = sample.energy
        self.sample = sample
        cvRes = KFoldCrossValidation(self.regressor, sample, self.CVcount)
        print(self.CVcount,' cross validation of regression:')
        for i in range(len(sample.paramNames)):
            pName = sample.paramNames[i]
            print(pName, 'relToConstPredError = %5.3g RMSE = %5.3g' % (cvRes[pName]['relToConstPredError'], cvRes[pName]['RMSE']))

        self.regressor.fit(sample.spectra, sample.params)

        # probability approach
        self.classifiers = []
        self.paramRanges = []
        self.paramNames = sample.paramNames
        if self.probabilityIntervals == 'auto':
            classCounts = chooseClassCount(self.classifier0, sample, self.CVcount)
        else:
            classCounts = [self.probabilityIntervals]*sample.paramNames.size
        for j in range(sample.paramNames.size):
            paramName = sample.paramNames[j]
            y = sample.params[paramName]
            paramRange = np.array([np.min(y), np.max(y)])
            self.paramRanges.append(paramRange)
            y = makeClasses(y, paramRange, classCounts[j])
            classifier = copy.deepcopy(self.classifier0)
            classifier.fit(sample.spectra, y)
            self.classifiers.append(classifier)

    # if smooth=False then prediction is made for spectrum-expBase.spectrum (without multiplying by purity)
    # if smooth=True then prediction is made for (smooth(spectrum)-smooth(spectrumBase))*purity
    # calcXanes = {'local':True/False, /*for cluster - */ 'memory':..., 'nProcs':...}
    def predict(self, spectrum, folderToSaveResult, smooth=True, calcXanes = None):
        folderToSaveResult = utils.fixPath(folderToSaveResult)
        xanesAbsorb = self.prepareSpectrumForPrediction(spectrum, smooth)
        predRegr = self.regressor.predict(xanesAbsorb).reshape((self.paramNames.size,))
        for j in range(self.paramNames.size):
            paramName = self.paramNames[j]
            predProba = self.classifiers[j].predict_proba(xanesAbsorb)[0]
            print(paramName,'=',predRegr[j])
            plotting.plotDirectMethodResult(predRegr[j], predProba, paramName, self.paramRanges[j], folder=folderToSaveResult)
        for j in range(self.paramNames.size):
            pn = self.paramNames[j]
            pv = predRegr[j]; a,b = self.paramRanges[j]
            if (pv<a) or (pv>b): print('Warning: parameter '+pn+' is out of borders. Shrinking it')
            if pv<a: predRegr[j] = a
            if pv>b: predRegr[j] = b
        predByParamNames = {self.paramNames[j]:predRegr[j] for j in range(self.paramNames.size)}
        molecula = None
        if self.exp.moleculeConstructor is not None:
            molecula = self.exp.moleculeConstructor(predByParamNames)
            if molecula is None:
                warnings.warn("Can't construct molecula for predicter parameter set")
            else:
                molecula.export_xyz(folderToSaveResult+'/molecule.xyz')
                fdmnes.generateInput(molecula, **self.exp.FDMNES_calc, folder=folderToSaveResult+'/fdmnes')
        if calcXanes is not None:
            if 'inverseEstimator' in calcXanes:
                inverseEstimator = calcXanes['inverseEstimator']
                xanesPred = inverseEstimator.predict(predRegr.reshape(1, -1)).reshape(self.exp.spectrum.energy.size)
                xanesPred = utils.Spectrum(self.exp.spectrum.energy, xanesPred.reshape(xanesPred.size))
                if self.diffFrom is None:
                    plotting.plotToFolder(folderToSaveResult, self.exp, None, xanesPred, fileName='xanesApproximation')
                else:
                    plotting.plotToFolder(folderToSaveResult, self.expDiff, None, xanesPred, fileName='xanesDiffApproximation')
            if ('local' in calcXanes) and (molecula is not None):
                if calcXanes['local']: fdmnes.runLocal(folderToSaveResult+'/fdmnes')
                else: fdmnes.runCluster(folderToSaveResult+'/fdmnes', calcXanes['memory'], calcXanes['nProcs'])
                xanes = fdmnes.parseOneFolder(folderToSaveResult + '/fdmnes')
                smoothed_xanes, _ = smoothLib.funcFitSmoothHelper(self.exp.defaultSmoothParams['fdmnes'], xanes, 'fdmnes', self.exp, self.norm)
                with open(folderToSaveResult+'/args_smooth.txt', 'w') as f: json.dump(self.exp.defaultSmoothParams['fdmnes'], f)
                if self.diffFrom is None:
                    plotting.plotToFolder(folderToSaveResult, self.exp, None, smoothed_xanes, fileName='xanes')
                else:
                    plotting.plotToFolder(folderToSaveResult, self.exp, None, smoothed_xanes, fileName='xanes', append=[{'data':self.diffFrom['spectrumBase'].intensity, 'label':'spectrumBase'}, {'data':self.diffFrom['projectBase'].spectrum.intensity, 'label':'expBase'}])
                    smoothed_xanes.intensity = (smoothed_xanes.intensity - self.diffFrom['spectrumBase'].intensity)*self.diffFrom['purity']
                    # print(smoothed_xanes.intensity)
                    plotting.plotToFolder(folderToSaveResult, self.expDiff, None, smoothed_xanes, fileName='xanesDiff')

    def predictRDF(self, spectrum, folderToSaveResult, atoms=None, smooth=True, check=False, extraMolecules={}):
        # atoms can be list of indices or atom name (string)
        folderToSaveResult = utils.fixPath(folderToSaveResult)
        if not os.path.exists(folderToSaveResult): os.makedirs(folderToSaveResult)
        if not hasattr(self, 'sample'):
            raise Exception('You should train estimator first')
        sample = copy.deepcopy(self.sample)
        if check:
            spectrum = utils.Spectrum(sample.energy, sample.spectra.values[0])
            trueParams = sample.params.loc[0]
            trueParams = {sample.paramNames[j]: trueParams[j] for j in range(sample.params.shape[1])}
            sample.spectra.drop(0, axis=0, inplace=True)
            sample.spectra.reset_index(drop=True, inplace=True)
            sample.params.drop(0, axis=0, inplace=True)
            sample.params.reset_index(drop=True, inplace=True)
        regressor = copy.deepcopy(self.regressor)
        for i in range(sample.params.shape[0]):
            geom = sample.params.loc[i]
            geom = {sample.paramNames[j]: geom[j] for j in range(sample.params.shape[1])}
            m = self.exp.moleculeConstructor(geom)
            dists = m.getSortedDists(atoms)
            if i == 0: newParams = np.zeros((sample.params.shape[0], dists.size))
            newParams[i,:] = dists
        newNames = ['dist_to_' + str(i) for i in range(newParams.shape[1])]
        sample.params = pd.DataFrame(data=newParams, columns=newNames)
        sample.paramNames = newNames

        cvRes = KFoldCrossValidation(regressor, sample, 4)
        RMSE = np.array([cvRes[pName]['RMSE'] for pName in sample.paramNames])
        RMSE[RMSE <= 1e-3] = 1e-3
        relToConstPredError = np.array([cvRes[pName]['relToConstPredError'] for pName in sample.paramNames])

        regressor.fit(sample.spectra, sample.params)
        if check:
            xanesAbsorb = spectrum.intensity.reshape(1, -1)
        else:
            xanesAbsorb = self.prepareSpectrumForPrediction(spectrum, smooth)
        predictedDists = regressor.predict(xanesAbsorb).reshape(-1)
        arg = np.linspace(np.max([0,predictedDists[0]-3*RMSE[0]]), predictedDists[-1]+3*RMSE[-1], 100)
        arg = np.append(arg, predictedDists)
        arg = np.sort(arg)
        predictedRdf = np.zeros(arg.size)
        for i in range(predictedDists.size):
            print("r_{} = {:.3g} ± {:.4g}".format(i,predictedDists[i], RMSE[i]))
            predictedRdf += utils.gauss(arg, predictedDists[i], RMSE[i])/arg**2
        fig, ax = plotting.createfig(interactive=True)
        funcName = 'rdf'; funcParams = {'sigma':np.min(RMSE), 'atoms':atoms}
        for mname in extraMolecules:
            ax.plot(arg, getattr(extraMolecules[mname], funcName)(arg, **funcParams), lw=2, label=mname)
        ax.plot(arg, predictedRdf, lw=2, color='r', label='predicted ' + funcName)
        ax.plot(predictedDists, np.zeros(predictedDists.size), 'rP', ms=15, label='predicted dists')
        if check:
            m = self.exp.moleculeConstructor(trueParams)
            dists = m.getSortedDists(atoms)
            trueRdf = getattr(m, funcName)(arg, **funcParams)
            ax.plot(arg, trueRdf, lw=2, color='k', label='true ' + funcName)
            ax.plot(dists, np.zeros(dists.size), 'kP', ms=15, label='predicted dists')
        ax.legend()
        plotting.savefig(folderToSaveResult + os.sep + funcName + '.png', fig)
        plotting.closefig(fig, interactive=True)

        plotData = pd.DataFrame()
        # plotData['r'] = arg
        plotData['predicted_dists'] = predictedDists
        plotData['RMSE'] = RMSE
        plotData['relToConstPredError'] = relToConstPredError
        plotData.to_csv(folderToSaveResult + os.sep + funcName + '.csv', sep=' ', index=False)

    def predictMoleculeFunction(self, spectrum, folderToSaveResult, smooth=True, check=False, extraMolecules={}, funcName='rdf', valMin=0, valMax=1, valCount=20, funcParams={}, **otherParams):
        # default parameters
        if funcName not in ['rdf', 'adf']:
            raise Exception('Unknown function name. You can use: rdf, adf')
        folderToSaveResult = utils.fixPath(folderToSaveResult)
        if not os.path.exists(folderToSaveResult): os.makedirs(folderToSaveResult)
        if not hasattr(self, 'sample'):
            raise Exception('You should train estimator first')
        sample = copy.deepcopy(self.sample)
        if check:
            spectrum = utils.Spectrum(sample.energy, sample.spectra.values[0])
            trueParams = sample.params.loc[0]
            trueParams = {sample.paramNames[j]: trueParams[j] for j in range(sample.params.shape[1])}
            sample.spectra.drop(0, axis=0, inplace=True)
            sample.spectra.reset_index(drop=True, inplace=True)
            sample.params.drop(0, axis=0, inplace=True)
            sample.params.reset_index(drop=True, inplace=True)
        regressor = copy.deepcopy(self.regressor)
        arg = np.linspace(valMin, valMax, valCount)
        argName = 'r'
        if funcName == 'adf': argName = 'angle'
        newParams = np.zeros((sample.params.shape[0], valCount))
        for i in range(sample.params.shape[0]):
            geom = sample.params.loc[i]
            geom = {sample.paramNames[j]: geom[j] for j in range(sample.params.shape[1])}
            m = self.exp.moleculeConstructor(geom)
            newParams[i] = getattr(m, funcName)(arg, **funcParams)
        newNames = [argName + '_' + str(arg[i]) for i in range(valCount)]
        sample.params = pd.DataFrame(data=newParams, columns=newNames)
        sample.paramNames = newNames

        cvRes = KFoldCrossValidation(regressor, sample, 4)
        RMSE = np.array([cvRes[pName]['RMSE'] for pName in sample.paramNames])
        relToConstPredError = np.array([cvRes[pName]['relToConstPredError'] for pName in sample.paramNames])

        regressor.fit(sample.spectra, sample.params)
        if check:
            xanesAbsorb = spectrum.intensity.reshape(1, -1)
        else:
            xanesAbsorb = self.prepareSpectrumForPrediction(spectrum, smooth)
        predictedRdf = regressor.predict(xanesAbsorb).reshape(-1)
        fig, ax = plotting.createfig(interactive=True)
        ax.fill_between(arg, predictedRdf - RMSE, predictedRdf + RMSE, color='grey', label='RMSE region (by cv)')
        for mname in extraMolecules:
            ax.plot(arg, getattr(extraMolecules[mname], funcName)(arg, **funcParams), lw=2, label=mname)
        ax.plot(arg, predictedRdf, lw=2, color='r', label='predicted ' + funcName)
        if check:
            m = self.exp.moleculeConstructor(trueParams)
            trueRdf = getattr(m, funcName)(arg, **funcParams)
            ax.plot(arg, trueRdf, lw=2, color='k', label='true ' + funcName)
        ax.legend()
        plotting.savefig(folderToSaveResult + os.sep + funcName + '.png', fig)
        plotting.closefig(fig, interactive=True)

        plotData = pd.DataFrame()
        plotData[argName] = arg
        plotData['predicted_'+funcName] = predictedRdf
        plotData['RMSE'] = RMSE
        plotData['relToConstPredError'] = relToConstPredError
        plotData.to_csv(folderToSaveResult + os.sep + funcName + '.csv', sep=' ', index=False)

    # if smooth=False then prediction is made for spectrum-expBase.spectrum (without multiplying by purity)
    # if smooth=True then prediction is made for (smooth(spectrum)-smooth(spectrumBase))*purity
    def predictRDF_old(self, spectrum, folderToSaveResult, atoms, smooth=True, rMin=0.5, rMax=6, rCount=20, sigma=0.2, check=False, extraMolecules={}):
        self.predictMoleculeFunction(spectrum, folderToSaveResult, smooth=smooth, check=check, extraMolecules=extraMolecules, funcName='rdf', valMin=rMin, valMax=rMax, valCount=rCount, funcParams={'sigma':sigma, 'atoms':atoms})

    # if smooth=False then prediction is made for spectrum-expBase.spectrum (without multiplying by purity)
    # if smooth=True then prediction is made for (smooth(spectrum)-smooth(spectrumBase))*purity
    def predictAngleDF(self, spectrum, folderToSaveResult, atoms, smooth=True, angleCount=20, sigma=0.2, check=False, extraMolecules={}):
        self.predictMoleculeFunction(spectrum, folderToSaveResult, smooth=smooth, check=check, extraMolecules=extraMolecules, funcName='adf', valMin=0, valMax=180, valCount=angleCount, funcParams={'sigma':sigma, 'atoms':atoms})


# geometryParam can be a name or dict: {'type':'RDF' or 'AngleDF', 'value':.., 'params':{'sigma':.., 'atomName':..}}
def compareDifferentMethods(sampleTrain, sampleTest, energyPoint, geometryParam, project, diffFrom=None, normalize=True, CVcount=0, folderToSaveResult='directMethodsCompare'):
    folderToSaveResult = utils.fixPath(folderToSaveResult)
    if not np.array_equal(sampleTrain.energy, sampleTest.energy):
        raise Exception('sampleTrain and sampleTest have different energy counts')
    if not np.array_equal(sampleTrain.paramNames, sampleTest.paramNames):
        raise Exception('sampleTrain and sampleTest have different geometry parameters')
    if diffFrom is not None: diffFrom = prepareDiffFrom(project, diffFrom, norm=None)
    sampleTrain = copy.deepcopy(sampleTrain)
    sampleTest = copy.deepcopy(sampleTest)
    if not isinstance(geometryParam, dict):
        if geometryParam not in sampleTrain.paramNames: raise Exception('Unknown geometry parameter '+str(geometryParam))
    else:
        complexGeometryParam = geometryParam
        if complexGeometryParam['type'] not in ['RDF', 'AngleDF']: raise Exception('Unknown geometry parameter type' + str(complexGeometryParam['type']))
        if complexGeometryParam['type'] == 'RDF': funcName = 'rdf'
        else: funcName = 'adf'
        value = float(complexGeometryParam['value'])
        geometryParam = complexGeometryParam['type'] + ' ' + str(value)
        for sample in [sampleTrain, sampleTest]:
            newParam = np.zeros(sample.params.shape[0])
            for i in range(sample.params.shape[0]):
                geom = sample.params.loc[i]
                geom = {sample.paramNames[j]: geom[j] for j in range(sample.params.shape[1])}
                m = project.moleculeConstructor(geom)
                newParam[i] = getattr(m, funcName)(np.array([value]), **complexGeometryParam['params'])
            sample.params[geometryParam] = newParam
    sampleTrain = prepareSample(sampleTrain, diffFrom, project, norm=None)
    sampleTest = prepareSample(sampleTest, diffFrom, project, norm=None)
    if (energyPoint < sampleTrain.energy[0]) or (energyPoint > sampleTrain.energy[-1]):
        raise Exception('energyPoint doesn\'t belong to experiment energy interval ['+str(sampleTrain.energy[0])+'; '+str(sampleTrain.energy[-1])+']')
    energyColumn = sampleTrain.spectra.columns[np.argmin(np.abs(sampleTrain.energy-energyPoint))]
    ind = np.argsort(sampleTest.spectra[energyColumn].values)
    sampleTest.params = sampleTest.params.iloc[ind]
    sampleTest.spectra = sampleTest.spectra.iloc[ind]
    sampleTest.params.reset_index(drop=True, inplace=True)
    sampleTest.spectra.reset_index(drop=True, inplace=True)
    plotData = pd.DataFrame()
    if not os.path.exists(folderToSaveResult): os.makedirs(folderToSaveResult)
    for methodName in allowedMethods:
        method = getMethod(methodName)
        if normalize: method = ML.Normalize(method, xOnly=False)
        if CVcount >= 2:
            cvRes = KFoldCrossValidation(method, sampleTrain, CVcount)
            print('\n',methodName,'cross validation of regression:')
            for i in range(len(sampleTrain.paramNames)):
                pName = sampleTrain.paramNames[i]
                print(pName, 'relToConstPredError = %5.3g RMSE = %5.3g' % (cvRes[pName]['relToConstPredError'], cvRes[pName]['RMSE']))
        method.fit(sampleTrain.spectra, sampleTrain.params[geometryParam])
        predicted = method.predict(sampleTest.spectra).reshape(-1)
        plotData[methodName + '_' + geometryParam] = predicted
    plotData[energyColumn] = sampleTest.spectra[energyColumn].values
    plotData['exact'] = sampleTest.params[geometryParam]
    plotData.to_csv(folderToSaveResult+os.sep+'compareDifferentMethodsDirect.csv', sep=' ', index=False)

    fig, ax = plotting.createfig()
    for methodName in allowedMethods:
        ax.plot(plotData[energyColumn], plotData[methodName+'_'+geometryParam], label=methodName)
    ax.plot(plotData[energyColumn], plotData['exact'], label='exact', lw=2, color='k')
    ax.set_xlabel(energyColumn)
    ax.set_ylabel(geometryParam)
    ax.legend()
    plotting.savefig(folderToSaveResult+os.sep+'compareDifferentMethodsDirect.png', fig)
    # if matplotlib.get_backend() != 'nbAgg': plt.close(fig)  # - sometimes figure is not shown

    fig2, ax2 = plotting.createfig()
    for methodName in allowedMethods:
        ax2.plot(plotData[energyColumn], np.abs(plotData[methodName+'_'+geometryParam] - plotData['exact']), label=methodName)
    ax2.plot(plotData[energyColumn], np.zeros(plotData[energyColumn].size), label='exact', lw=2, color='k')
    ax2.legend()
    ax2.set_xlabel(energyColumn)
    ax2.set_ylabel('abs(' + geometryParam + '-exact)')
    plotting.savefig(folderToSaveResult + os.sep + 'compareDifferentMethodsDirect_delta.png', fig2)


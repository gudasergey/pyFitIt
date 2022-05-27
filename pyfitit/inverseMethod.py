from . import utils, mixture
utils.fixDisplayError()
import numpy as np
import pandas as pd
import sklearn, copy, os, json, sys, math, logging, traceback, itertools, gc, shutil, scipy, glob
from distutils.dir_util import copy_tree, remove_tree
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from shutil import copyfile
from . import plotting, ML, optimize, smoothLib, fdmnes, curveFitting, funcModel
from .funcModel import FuncModel


allowedMethods = ['Ridge', 'Ridge Quadric', 'Extra Trees', 'RBF']
if utils.isLibExists("lightgbm"):
    import lightgbm as lgb
    allowedMethods.append('LightGBM')

# sort by speed

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
        params = {'function':'linear', 'baseRegression': 'quadric', 'scaleX': True, 'removeDublicates':True}
        allParams = list(params.keys())
    elif name == 'LightGBM':
        params = {'num_leaves':31, 'learning_rate':0.02, 'n_estimators':100}
        allParams = list(lgb.LGBMRegressor().get_params().keys())
    else: raise Exception('Unknown method name. You can use: '+str(allowedMethods))
    return params, allParams


def getMethod(name, params0=None):
    if params0 is None: params0 = {}
    rparams, allParams = recommendedParams(name)
    params = {p: params0[p] for p in params0 if p in allParams}
    for p in rparams:
        if p not in params: params[p] = rparams[p]
    if name == "Ridge": regressor = RidgeCV(**params)
    elif name == "Ridge Quadric": regressor = ML.makeQuadric(RidgeCV(**params))
    elif name == 'Extra Trees': regressor = ExtraTreesRegressor(**params)
    elif name[:3] == "RBF": regressor = ML.RBF(**params)
    elif name == 'LightGBM': regressor = ML.makeMulti(lgb.LGBMRegressor(objective='regression', verbosity=-1, **params))
    else: raise Exception('Unknown method name. You can use: '+str(allowedMethods))
    return regressor


def prepareSample(sample0, diffFrom, proj, samplePreprocessor, smoothType):
    sample = copy.deepcopy(sample0)
    assert set(sample.paramNames) == set(proj.geometryParamRanges.keys()), 'Param names in geometryParamRanges of project:\n'+str(list(proj.geometryParamRanges.keys()))+'\ndoes not equal to dataset param names:\n'+str(sample.paramNames)
    for pn in sample.paramNames:
        assert utils.inside(sample.params[pn], proj.geometryParamRanges[pn]), 'Project param ranges don\'t correspond to sample'
    if isinstance(samplePreprocessor, dict):
        convolutionParams = samplePreprocessor
        sample.spectra = smoothLib.smoothDataFrame(convolutionParams, sample.spectra, smoothType, proj.spectrum, proj.intervals['fit_norm'], folder=sample.folder)
    else:
        sample = samplePreprocessor(sample)
        assert len(sample.energy) == sample.spectra.shape[1]
        assert np.all(sample.energy == proj.spectrum.energy), str(sample.energy)+'\n'+str(proj.spectrum.energy)+'\n'+str(len(sample.energy))+' '+str(len(proj.spectrum.energy))
    if diffFrom is not None:
        sample.setSpectra(spectra=(sample.spectra.to_numpy() - diffFrom['spectrumBase'].intensity) * diffFrom['purity'], energy=sample.energy)
    return sample


def prepareDiffFrom(samplePreprocessor, proj, diffFrom, smoothType):
    diffFrom = copy.deepcopy(diffFrom)
    bs = diffFrom['projectBase'].spectrum
    bs.intensity = np.interp(proj.spectrum.energy, diffFrom['projectBase'].spectrum.energy, diffFrom['projectBase'].spectrum.intensity)
    bs.energy = proj.spectrum.energy
    diffFrom['projectBase'].spectrum = bs
    if isinstance(samplePreprocessor, dict):
        convolutionParams = samplePreprocessor
        diffFrom['spectrumBase'], _ = smoothLib.smoothInterpNorm(convolutionParams, diffFrom['spectrumBase'], smoothType, diffFrom['projectBase'].spectrum, proj.intervals['fit_norm'])
    else:
        diffFrom['spectrumBase'] = samplePreprocessor(diffFrom['spectrumBase'])
    return diffFrom


class Estimator:
    def __init__(self, method, proj, samplePreprocessor=None, normalize=True, CVcount=10, folderToSaveCVresult='', folderToDebugSample='', diffFrom=None, smooth_type='fdmnes', **params):
        """
        Class for predicting spectra by params

        :param method: ML method name: 'Ridge', 'Ridge Quadric', 'Extra Trees', 'RBF'
        :param proj: project
        :param samplePreprocessor: function(sample)->sample or dict convolutionParams
        :param diffFrom: dict {'projectBase':..., 'spectrumBase':..., 'purity':...}
        """
        folderToSaveCVresult = utils.fixPath(folderToSaveCVresult)
        if method not in allowedMethods:
            raise Exception('Unknown method name. You can use: '+str(allowedMethods))
        self.proj = copy.deepcopy(proj)
        self.regressor = getMethod(method, params)
        if normalize: self.regressor = ML.Normalize(self.regressor, xOnly=False)
        self.normalize = normalize
        self.smooth_type = smooth_type
        if isinstance(samplePreprocessor, dict):
            convolutionParams = samplePreprocessor
            self.convolutionParams = {k:convolutionParams[k] for k in convolutionParams}
            if 'norm' in self.convolutionParams:
                self.norm = self.convolutionParams['norm']
                del self.convolutionParams['norm']
            else: self.norm = None
            for pName in self.convolutionParams:
                self.proj.defaultSmoothParams[smooth_type][pName] = self.convolutionParams[pName]
            if smooth_type == 'optical' and self.norm is not None:
                self.regressor = ML.SeparateNorm(self.regressor, normMethod='mean')
        self.samplePreprocessor = samplePreprocessor
        self.CVcount = CVcount
        assert CVcount >= 2
        self.folderToSaveCVresult = folderToSaveCVresult
        self.folderToDebugSample = folderToDebugSample
        self.diffFrom = copy.deepcopy(diffFrom)
        if diffFrom is not None:
            self.diffFrom = prepareDiffFrom(self.proj, diffFrom, self.norm)
            self.projDiff = copy.deepcopy(self.proj)
            self.projDiff.spectrum = utils.Spectrum(self.projDiff.spectrum.energy, self.proj.spectrum.intensity - self.diffFrom['projectBase'].spectrum.intensity)
        self.sample = None
        self.xanes_energy = None
        self.geometryParamRanges = None
        # interface for findGlobalL2NormMinimum
        self.name = self.proj.name
        self.paramNames = None  # order will be copyed from sample during fit
        self.expSpectrum = self.proj.spectrum
        self.paramRanges = self.proj.geometryParamRanges
        self.fitSpectrumInterval = self.proj.intervals['fit_geometry']
        self.plotSpectrumInterval = self.proj.intervals['plot']
        self.fitSpectrumNormInterval = self.proj.intervals['fit_norm']

    def fit(self, sample0):
        sample = prepareSample(sample0, self.diffFrom, self.proj, self.samplePreprocessor, self.smooth_type)
        if self.folderToDebugSample != '':
            print('Ploting sample to folder', self.folderToDebugSample)
            if os.path.exists(self.folderToDebugSample): shutil.rmtree(self.folderToDebugSample)
            os.makedirs(self.folderToDebugSample, exist_ok=True)
            for i in range(sample0.spectra.shape[0]):
                plotting.plotToFile(sample0.energy, sample0.spectra.values[i], str(i), self.folderToDebugSample+os.sep+'spectrum_'+utils.zfill(i,sample0.spectra.shape[0]))
                plotting.plotToFile(sample.energy, sample.spectra.values[i], str(i), self.folderToDebugSample+os.sep+'smoothed_' + utils.zfill(i,sample0.spectra.shape[0]))
        self.sample = copy.deepcopy(sample)
        self.xanes_energy = sample.energy
        self.geometryParamRanges = {}
        for pName in sample.paramNames:
            self.geometryParamRanges[pName] = [np.min(sample.params[pName]), np.max(sample.params[pName])]
        sample_cv = sample.limit(energyRange=self.proj.intervals['fit_geometry'], inplace=False)
        res, individualSpectrErrors, predictions = ML.crossValidation(self.regressor, sample_cv.params, sample_cv.spectra, CVcount=self.CVcount, YColumnWeights=sample_cv.convertEnergyToWeights())
        output = 'Inverse method relative to constant prediction error = %5.3g\n' % res
        print(output)
        if self.smooth_type == 'optical' and self.norm is not None:
            assert 'SeparateNorm' in self.regressor.name, self.regressor.name
            if self.normalize: sepNorm = self.regressor.learner
            else: sepNorm = self.regressor
            nsample_spectra, nrm = sepNorm.normalize(sample_cv.spectra.values)
            nsample = sample_cv.copy()
            nsample.setSpectra(nsample_spectra, energy=sample_cv.energy)
            sepNormLearner = sepNorm.learner
            if self.normalize: sepNormLearner = ML.Normalize(sepNormLearner, xOnly=False)
            res, _, _ = ML.crossValidation(sepNormLearner, nsample.params, nsample.spectra, CVcount=self.CVcount, YColumnWeights=nsample.convertEnergyToWeights())
            output2 = 'relToConstPredError on normalized sample = %5.3g\n' % res
            output += output2
            print(output2)
            sepNormNormLearner = sepNorm.normLearner
            if self.normalize: sepNormNormLearner = ML.Normalize(sepNormNormLearner, xOnly=False)
            res = ML.score_cv(sepNormNormLearner, sample_cv.params, nrm, self.CVcount, returnPrediction=False)
            res = 1-res
            output3 = 'relToConstPredError on norm values = %5.3g\n' % res
            output += output3
            print(output3)
        if self.folderToSaveCVresult != '':
            os.makedirs(self.folderToSaveCVresult, exist_ok=True)
            with open(self.folderToSaveCVresult+'/info.txt', 'w') as f: f.write(output)
            ind = np.argsort(individualSpectrErrors)
            n = individualSpectrErrors.size
            energy = sample_cv.energy

            def plotCVres(energy, trueXan, predXan, fileName):
                plotting.plotToFile(energy, trueXan, "True spectrum", energy, predXan, "Predicted spectrum", title=os.path.basename(fileName), fileName=fileName, save_csv=True)

            i = ind[n//2]
            plotCVres(energy, sample_cv.spectra.loc[i], predictions[i], self.folderToSaveCVresult+'/xanes_mean_error.png')
            i = ind[9*n//10]
            plotCVres(energy, sample_cv.spectra.loc[i], predictions[i], self.folderToSaveCVresult+'/xanes_max_0.9_error.png')
            i = ind[-1]
            plotCVres(energy, sample_cv.spectra.loc[i], predictions[i], self.folderToSaveCVresult+'/xanes_max_error.png')
        self.regressor.fit(sample.params, sample.spectra)
        self.paramNames = list(sample.paramNames)
        # calcParamStdDev(self)

    def predict(self, params):
        return self.regressor.predict(params)


def compareDifferentMethods(sampleTrain, sampleTest, energyPoint, geometryParam, project, diffFrom=None, CVcount=4, folderToSaveResult='inverseMethodsCompare'):
    folderToSaveResult = utils.fixPath(folderToSaveResult)
    if not np.array_equal(sampleTrain.paramNames, sampleTest.paramNames):
        raise Exception('sampleTrain and sampleTest have different geometry parameters')
    if geometryParam not in sampleTrain.paramNames:
        raise Exception('samples don\'t contain geometry parameter '+str(geometryParam))
    if diffFrom is not None: diffFrom = prepareDiffFrom(samplePreprocessor=project.FDMNES_smooth, smoothType='fdmnes', proj=project, diffFrom=diffFrom)
    sampleTrain = prepareSample(sampleTrain, diffFrom, project, samplePreprocessor=project.FDMNES_smooth, smoothType='fdmnes')
    sampleTest = prepareSample(sampleTest, diffFrom, project, samplePreprocessor=project.FDMNES_smooth, smoothType='fdmnes')
    if (energyPoint<sampleTrain.energy[0]) or (energyPoint>sampleTrain.energy[-1]):
        raise Exception('energyPoint doesn\'t belong to experiment energy interval ['+str(sampleTrain.energy[0])+'; '+str(sampleTrain.energy[-1])+']')
    energyColumn = sampleTrain.spectra.columns[np.argmin(np.abs(sampleTrain.energy-energyPoint))]
    ind = np.argsort(sampleTest.params[geometryParam].values)
    sampleTest.params = sampleTest.params.iloc[ind]
    sampleTest.spectra = sampleTest.spectra.iloc[ind]
    sampleTest.params.reset_index(drop=True, inplace=True)
    sampleTest.spectra.reset_index(drop=True, inplace=True)
    if not os.path.exists(folderToSaveResult): os.makedirs(folderToSaveResult)
    plotData = pd.DataFrame()
    for methodName in allowedMethods:
        method = getMethod(methodName)
        if CVcount >= 2:
            relToConstPredError, _, _ = ML.crossValidation(method, sampleTrain.params, sampleTrain.spectra, CVcount, YColumnWeights=sampleTrain.convertEnergyToWeights())
            print(methodName+' relative to constant prediction error (all energies) = %5.3g\n' % relToConstPredError, flush=True)
        method.fit(sampleTrain.params, sampleTrain.spectra[energyColumn])
        predicted = method.predict(sampleTest.params).reshape(-1)
        plotData[methodName+'_'+energyColumn] = predicted
    plotData[geometryParam] = sampleTest.params[geometryParam].values
    plotData['exact'] = sampleTest.spectra[energyColumn]
    plotData.to_csv(folderToSaveResult+os.sep+'compareDifferentMethodsInverse.csv', sep=' ', index=False)

    fig, ax = plotting.createfig(interactive=True)
    for methodName in allowedMethods:
        ax.plot(plotData[geometryParam], plotData[methodName+'_'+energyColumn], label=methodName)
    ax.plot(plotData[geometryParam], sampleTest.spectra[energyColumn], label='exact', lw=2, color='k')

    ax.legend()
    ax.set_xlabel(geometryParam)
    ax.set_ylabel(energyColumn)
    plotting.savefig(folderToSaveResult+os.sep+'compareDifferentMethodsInverse.png', fig)
    # if not utils.isJupyterNotebook(): plt.close(fig)  #notebooks also have limit - 20 figures # - sometimes figure is not shown
    # if matplotlib.get_backend() != 'nbAgg': plt.close(fig)

    fig2, ax2 = plotting.createfig(interactive=True)
    for methodName in allowedMethods:
        ax2.plot(plotData[geometryParam], np.abs(plotData[methodName+'_'+energyColumn]-plotData['exact']), label=methodName)
    ax2.plot(plotData[geometryParam], np.zeros(plotData[geometryParam].size), label='exact', lw=2, color='k')
    ax2.legend()
    ax2.set_xlabel(geometryParam)
    ax2.set_ylabel('abs('+energyColumn+'-exact)')
    plotting.savefig(folderToSaveResult+os.sep+'compareDifferentMethodsInverse_delta.png', fig2)


def makeDictFromVector(arg, paramNames):
    assert len(arg) == len(paramNames)
    return {paramNames[i]:arg[i] for i in len(arg)}


def makeVectorFromDict(params, paramNames, check=True):
    if check:
        assert set(params.keys()) == set(paramNames)
    return np.array([params[p] for p in paramNames])


def findGlobalL2NormMinimum(trysCount, estimator, folderToSaveResult, calcXanes=None, fixParams=None, contourMapCalcMethod='fast', plotContourMaps='all', extraPlotFuncContourMaps=None, normalizeMixtureToExperiment=False, gaussComponents=False):
    """
    Try several optimizations from random start point. Retuns sorted list of results and plot contours around minimum

    :param trysCount: number of attempts to find minimum
    :param estimator: instance of Estimator class to predict spectrum by parameters
    :param folderToSaveResult: folder to save graphs
    :param calcXanes: calcXanes = {'local':True/False, /*for cluster - */ 'memory':..., 'nProcs':...}
    :param fixParams: dict of paramName:value to fix
    :param contourMapCalcMethod: 'fast' - plot contours of the target function; 'thorough' - plot contours of the min of target function by all arguments except axes
    :param plotContourMaps: 'all' or list of 1-element or 2-elements lists of axes names to plot contours of target function
    :param extraPlotFuncContourMaps: user defined function to plot something on result contours: func(ax, axisNamesList, xminDict)
    :param normalizeMixtureToExperiment: True if we need to normalize spectrum of mixture when compare to experiment
    :param gaussComponents: True, if instead of spectrum(param0) you want to use gauss mixture of spectrum(param) with center in param0 (center and covariance matrix are fitted)
    :return: dict with keys 'value' (minimum rFactor value), 'x', 'spectrum'
    """
    return findGlobalL2NormMinimumMixture(trysCount, [estimator], folderToSaveResult, calcXanes=calcXanes, fixParams=fixParams, contourMapCalcMethod=contourMapCalcMethod, plotContourMaps=plotContourMaps, extraPlotFuncContourMaps=extraPlotFuncContourMaps, normalizeMixtureToExperiment=normalizeMixtureToExperiment, gaussComponents=gaussComponents)


def findGlobalL2NormMinimumMixture(trysCount, estimatorList, folderToSaveResult, calcXanes=None, fixParams=None, contourMapCalcMethod='fast', plotContourMaps='all', extraPlotFuncContourMaps=None, normalizeMixtureToExperiment=False, gaussComponents=False):
    """
    Try several optimizations from random start point. Retuns sorted list of results and plot contours around minimum

    :param trysCount: number of attempts to find minimum
    :param estimatorList: list of Estimator classes of mixture components
    :param folderToSaveResult: folder to save graphs
    :param calcXanes: calcXanes = {'local':True/False, /*for cluster - */ 'memory':..., 'nProcs':...}
    :param fixParams: dict of paramName:value to fix. Param names: projectName_paramName, concentration names - projectName
    :param contourMapCalcMethod: 'fast' - plot contours of the target function; 'thorough' - plot contours of the min of target function by all arguments except axes
    :param plotContourMaps: 'all' or list of 1-element or 2-elements lists of axes names to plot contours of target function
    :param extraPlotFuncContourMaps: user defined function to plot something on result contours: func(ax, axisNamesList, xminDict)
    :param normalizeMixtureToExperiment: True if we need to normalize spectrum of mixture when compare to experiment
    :param gaussComponents: True, if instead of spectrum(param0) you want to use gauss mixture of spectrum(param) with center in param0 (center and covariance matrix are fitted)
    :return: dict with keys 'value' (minimum value), 'x' (list of component parameter values lists), 'x1d' (flattened array of all param values), 'paramNames1d', 'concentrations', 'spectra', 'mixtureSpectrum'
    """

    oneComponent = len(estimatorList) == 1
    estimator0 = estimatorList[0]
    e0 = estimator0.expSpectrum.energy
    ind_geom = (estimator0.fitSpectrumInterval[0] <= e0) & (e0 <= estimator0.fitSpectrumInterval[1])
    e0_geom = e0[ind_geom]
    expXanesPure = estimator0.expSpectrum.intensity
    expXanes = expXanesPure if estimator0.diffFrom is None else estimator0.projDiff.spectrum.intensity
    expXanes_geom = expXanes[ind_geom]
    if fixParams is None: fixParams = {}
    folderToSaveResult = utils.fixPath(folderToSaveResult)
    if not os.path.exists(folderToSaveResult): os.makedirs(folderToSaveResult)

    def makeSpectraFunc(i):
        estimator = estimatorList[i]
        e = estimator.expSpectrum.energy

        def spectraFunc(arg):
            n = len(arg)
            if gaussComponents and fitGaussComponentParams:
                assert n % 2 == 0
                center = arg[:n//2].reshape(1,-1)
                sigma = arg[n//2:].reshape(1,-1)
                # assert np.all(sigma > 0), 'sigma = ' + str(sigma) + '   center = ' + str(center)
                sigma[sigma<1e-3] = 1e-3
                spectra = estimator.sample.spectra.to_numpy()
                params = estimator.sample.params.to_numpy()
                weights = np.sum(-0.5*(params-center)**2/sigma**2, axis=1)
                weights = weights - np.max(weights)  # div exp(weights) by max(exp(weights))
                weights = np.exp(weights).reshape(-1,1)
                assert np.sum(weights) > 0, f'{weights}  -0.5*(params-center)**2/sigma**2 = {-0.5*(params-center)**2/sigma**2}  sigma = {sigma}'
                weights /= np.sum(weights)
                xanesPred = np.sum(spectra*weights, axis=0)
            else:
                xanesPred = estimator.predict(arg.reshape(1,-1)).reshape(-1)
            assert len(e) == len(xanesPred), f"{len(e)} != {len(xanesPred)}"
            xanesPred = np.interp(e0, e, xanesPred)
            return xanesPred
        return spectraFunc
    spectraFuncs = [makeSpectraFunc(i) for i in range(len(estimatorList))]

    def makeMixture(spectraList, concentrations):
        mixtureSpectrum = spectraList[0]*concentrations[0]
        for i in range(1,len(estimatorList)):
            mixtureSpectrum += concentrations[i]*spectraList[i]
        if normalizeMixtureToExperiment:
            mixtureSpectrum = curveFitting.fit_by_regression(e0, expXanes, mixtureSpectrum, estimator0.fitSpectrumNormInterval)
        return mixtureSpectrum

    def distToExperiment(mixtureSpectrum, allArgs):
        assert len(mixtureSpectrum) == len(e0)
        rFactor = utils.rFactor(e0_geom, mixtureSpectrum[ind_geom], expXanes_geom)
        return rFactor

    def paramNames():
        if gaussComponents and fitGaussComponentParams:
            res = []
            for estimator in estimatorList:
                ps = estimator.paramNames.tolist()
                ps += [f'{p}_sigma' for p in ps]
                res.append(ps)
            return res
        else:
            return [estimator.paramNames for estimator in estimatorList]

    def bounds():
        b = []
        for estimator in estimatorList:
            pb = [estimator.paramRanges[p] for p in estimator.paramNames]
            if gaussComponents and fitGaussComponentParams:
                for p in estimator.paramNames:
                    r = estimator.paramRanges[p]
                    d = r[1]-r[0]
                    assert d > 0
                    pb.append([d*0.01, d])
            b.append(pb)
        return b

    componentNames = [estimator.name for estimator in estimatorList]
    for i in range(len(componentNames)):
        cn = componentNames[i]
        if componentNames.count(cn) > 1:
            k = 1
            for j in range(len(componentNames)):
                if componentNames[j] == cn:
                    componentNames[j] = cn+f'_{k}'
                    k += 1

    fitGaussComponentParams = True
    minimums = mixture.findGlobalMinimumMixture(distToExperiment, spectraFuncs, makeMixture, trysCount, bounds(), paramNames(), componentNames=componentNames, folderToSaveResult=folderToSaveResult, fixParams=fixParams, contourMapCalcMethod=contourMapCalcMethod, plotContourMaps=plotContourMaps, extraPlotFuncContourMaps=extraPlotFuncContourMaps)

    output = ''
    paramNames1d = minimums[0]['paramNames1d']
    for j in range(trysCount):
        output += str(minimums[j]['value'])+' '+optimize.arg2string(minimums[j]['x1d'], paramNames1d)+"\n"
    with open(folderToSaveResult+'/minimums.txt', 'w') as f: f.write(output)
    if trysCount > 1: print('Sorted results:')
    for j in range(trysCount):
        minimum = minimums[j]
        resultString = 'R-factor = {:.4g} '.format(minimum['value']) + ' '.join([paramNames1d[i]+'={:.2g}'.format(minimum['x1d'][i]) for i in range(len(paramNames1d))])
        print(resultString)
        strj = utils.zfill(j, trysCount)  # str(ir) !!!!!! - to have names sorted in same order as minimums
        if not oneComponent:
            for ip in range(len(estimatorList)):
                fileName = 'xanes_approx_'+strj if oneComponent else 'xanes_'+componentNames[ip]+'_approx_'+strj
                plotting.plotToFile(e0, minimum['spectra'][ip], 'theory', e0, expXanes, 'exp', title=resultString, fileName=folderToSaveResult+os.sep+fileName, xlim=estimator0.plotSpectrumInterval)
        # for one component mixture spectrum can be different from single component spectrum if normalizeMixtureToExperiment
        plotting.plotToFile(e0, minimum['mixtureSpectrum'], 'theory', e0, expXanes, 'exp', title=resultString, fileName=folderToSaveResult+os.sep+'xanes_mixture_approx_' + strj, xlim=estimator0.plotSpectrumInterval)

    minimum = minimums[0]
    concentrations = minimum['concentrations']
    spectraList = []
    for ip in range(len(estimatorList)):
        estimator = estimatorList[ip]
        if not hasattr(estimator, 'proj'): continue
        proj = estimator.proj
        if proj.moleculeConstructor is None: continue
        bestGeom = {}
        for p,j in zip(estimator.paramNames, range(len(estimator.paramNames))):
            bestGeom[p] = minimum['x'][ip][j]
        M = proj.moleculeConstructor(bestGeom)
        if hasattr(M, 'export_xyz'):
            M.export_xyz(folderToSaveResult+'/molecula_'+componentNames[ip]+'_best.xyz')
            fdmnes.generateInput(M, **proj.FDMNES_calc, folder=folderToSaveResult+'/fdmnes_'+componentNames[ip])

        if calcXanes is None: continue
        if calcXanes['local']: fdmnes.runLocal(folderToSaveResult+'/fdmnes_'+componentNames[ip])
        else: fdmnes.runCluster(folderToSaveResult+'/fdmnes_'+componentNames[ip], calcXanes['memory'], calcXanes['nProcs'])
        xanes = fdmnes.parseOneFolder(folderToSaveResult + '/fdmnes_' + componentNames[ip])
        smoothed_xanes, _ = smoothLib.funcFitSmoothHelper(estimator.convolutionParams, xanes, 'fdmnes', proj, estimator.norm)
        plotting.plotToFile(e0, smoothed_xanes, 'theory', e0, expXanesPure, 'exp', title=resultString, fileName=folderToSaveResult + os.sep + 'xanes_' + componentNames[ip] + '_best_minimum', xlim=estimator0.plotSpectrumInterval)
        if estimator.diffFrom is not None:
            smoothed_xanes.intensity = (smoothed_xanes.intensity - estimator.diffFrom['spectrumBase'].intensity)*estimator.diffFrom['purity']
            plotting.plotToFile(e0, smoothed_xanes, 'theory', e0, expXanes, 'exp', title=resultString, fileName=folderToSaveResult + os.sep + 'xanesDiff_'+componentNames[ip]+'_best_minimum', xlim=estimator0.plotSpectrumInterval)
        spectraList.append(smoothed_xanes)
    if calcXanes is not None:
        smoothed_xanesMixture = makeMixture(spectraList, concentrations)
        plotting.plotToFile(e0, smoothed_xanesMixture, 'theory', e0, expXanesPure, 'exp', title=resultString, fileName=folderToSaveResult + os.sep + 'xanes_mixture_best_minimum.png', xlim=estimator0.plotSpectrumInterval)
        if estimator0.diffFrom is not None:
            smoothed_xanesMixture = (smoothed_xanesMixture - estimator0.diffFrom['spectrumBase'].intensity) * estimator0.diffFrom['purity']
            plotting.plotToFile(e0, smoothed_xanesMixture, 'theory', e0, expXanes, 'exp', title=resultString, fileName=folderToSaveResult + os.sep + 'xanesDiff_mixture_best_minimum.png', xlim=estimator0.plotSpectrumInterval)
    if oneComponent:
        return {'value':minimum['value'], 'x':minimum['x'], 'spectrum':minimum['mixtureSpectrum']}
    else:
        return minimum


def findGlobalL2NormMinimumMixtureUniform(tryCount, spectrumCollection, estimatorList, concInterpPoints, folderToSaveResult, fixParams=None, fixConc=None, L2NormMap='fast', plotMaps='all'):
    """
    Fitting mixture uniformly for all spectra in a collection
    :param tryCount: try count to find global minimum
    :param spectrumCollection: collection of spectra, depending on some parameter
    :param estimatorList: inverse method estimators for mixture components
    :param concInterpPoints: interpolation points on parameter axis, used to interpolate concentraions and make their dependence on parameter smooth
    :param folderToSaveResult: folder to save result
    :param fixParams: dict. Geometry param names:values.  "Geometry param name" is "projectName_paramName"
    :param fixConc: dict. SpectrumCollectionParam:{projectName:concentraionValue,...}
    :param L2NormMap: 'fast' or 'thorough'
    :param plotMaps: list of 1d or 2d lists of names
    """

    assert len(estimatorList) > 1
    estimator0 = estimatorList[0]
    exp0 = estimator0.exp
    if fixConc is not None:
        for p in fixConc:
            assert p in concInterpPoints
    else: fixConc = {}

    spectrumCollection = copy.deepcopy(spectrumCollection)
    e = spectrumCollection.getSpectrumByParam(concInterpPoints[0]).energy
    ind = (exp0.intervals['fit_geometry'][0] <= e) & (e <= exp0.intervals['fit_geometry'][1])
    e0 = e[ind]
    spectrumCollection.spectra = spectrumCollection.spectra.loc[ind]
    spectrumCollection.spectra.reset_index(inplace=True, drop=True)

    def getParamName(estimator, paramName):
        assert paramName != ''
        return estimator.exp.name + '_' + paramName

    projectNames = []
    for estimator in estimatorList:
        projectNames.append(estimator.exp.name)

    if fixParams is None: fixParams = {}
    paramNames = []; paramNamesNotFixed = []; expNames = []
    for i_estimator in range(len(estimatorList)):
        estimator = estimatorList[i_estimator]
        expNames.append(estimator.exp.name)
        toAdd = copy.deepcopy(estimator.paramNames.tolist())
        for i in range(len(toAdd)):
            name = getParamName(estimator, toAdd[i])
            paramNames.append(name)
            if name not in fixParams: paramNamesNotFixed.append(name)
    assert set(fixParams.keys()) < set(paramNames), str(fixParams.keys())+'\n'+str(paramNames)
    assert len(np.unique(projectNames)) == len(projectNames), "Project names should be different!"
    assert len(np.unique(paramNames)) == len(paramNames), "Combinations of project names and param names should be different!"
    assert len(paramNames) == len(paramNamesNotFixed) + len(fixParams), 'paramNames = ' + str(paramNames) + '\nparamNamesNotFixed = ' + str(paramNamesNotFixed) + '\nfixParams = ' + str(fixParams)

    def getParamIndFullList(paramName):
        res = np.where(np.array(paramNames) == paramName)[0]
        assert len(res) == 1
        return res[0]

    def getProjectIndAndParam(paramName):
        assert paramName in paramNames
        for i in range(len(projectNames)):
            pn = projectNames[i]
            if paramName[:len(pn)] == pn:
                # check, if a param exists
                projectParams = [getParamName(estimatorList[i], p) for p in estimatorList[i].paramNames]
                j = np.where(np.array(projectParams) == paramName)[0]
                assert len(j) <= 1
                if paramName in projectParams: return i, estimatorList[i].paramNames[j[0]]
        assert False, "Couldn't find "+paramName+" in list of all params: "+str(paramNames)

    def getProjectInd(paramName):
        i, _ = getProjectIndAndParam(paramName)
        return i

    def getProjectParam(paramName):
        _, projectParam = getProjectIndAndParam(paramName)
        return projectParam

    def getProjectParamVector(fullParamList, estimatorInd):
        estimator = estimatorList[estimatorInd]
        geomArg = np.zeros(len(estimator.paramNames))
        for i in range(len(estimator.paramNames)):
            paramInd = getParamIndFullList(getParamName(estimator, estimator.paramNames[i]))
            geomArg[i] = fullParamList[paramInd]
        return geomArg

    def getXanesArray(arg):
        n = len(estimatorList)
        for i in range(n):
            estimator = estimatorList[i]
            geomArg = getProjectParamVector(arg, i)
            xanesPred = estimator.predict(geomArg.reshape(1,-1)).reshape(-1)
            xanesPred = np.interp(e0, estimator.exp.spectrum.energy, xanesPred)
            if estimator == estimatorList[0]:
                xanesArray = [xanesPred]
            else:
                xanesArray.append(xanesPred)
        return np.array(xanesArray)

    def L2normMixtureUniform(arg, returnConcentrations=False):
        xanesArray = getXanesArray(arg)
        n = len(estimatorList)
        concentrations = np.zeros((len(concInterpPoints),n))
        for param,i in zip(concInterpPoints, range(len(concInterpPoints))):
            expXanes = spectrumCollection.getSpectrumByParam(param).intensity
            if estimator.diffFrom is not None:
                expXanes -= np.interp(e0, estimator.diffFrom['projectBase'].spectrum.energy, estimator.diffFrom['projectBase'].spectrum.intensity)
            fixConcentrations = None if param not in fixConc else copy.deepcopy(fixConc[param])
            if fixConcentrations is not None:
                for expName in list(fixConcentrations.keys()):
                    fixConcentrations[expNames.index(expName)] = fixConcentrations.pop(expName)
            _, concentrations[i] = curveFitting.findConcentrations(e0, xanesArray, expXanes, fixConcentrations)
            # if abs(np.sum(concentrations[i])-1)>1e-2:
            #     print('Error: ',concentrations[i], fixConcentrations)
        manyParams = spectrumCollection.params
        manyPointsCount = len(manyParams)
        concentrations2 = np.zeros((manyPointsCount, n))
        for j in range(n):
            rbf_func = scipy.interpolate.Rbf(concInterpPoints, concentrations[:,j])
            concentrations2[:,j] = rbf_func(manyParams)
        rFactors = np.zeros(manyPointsCount)
        for i in range(manyPointsCount):
            xanesSum = np.sum(xanesArray*concentrations2[i].reshape(-1,1), axis=0)
            expXanes = spectrumCollection.getSpectrumByParam(manyParams[i]).intensity
            rFactors[i] = utils.integral(e0, (xanesSum - expXanes) ** 2) / utils.integral(e0, expXanes ** 2)
        mean_rFactor = np.sqrt(np.mean(rFactors**2))
        if returnConcentrations:
            return mean_rFactor, concentrations2, rFactors
        else:
            return mean_rFactor

    folderToSaveResult = utils.fixPath(folderToSaveResult)
    if not os.path.exists(folderToSaveResult): os.makedirs(folderToSaveResult)
    fmins = np.zeros(tryCount)
    geoms = [None] * tryCount
    geoms_partial = [None] * tryCount
    concentrations = [None] * tryCount
    rFactors = [None] * tryCount
    rand = np.random.rand

    def getArg0AndBounds():
        arg0 = []; bounds = []
        for p in paramNamesNotFixed:
            i = getProjectInd(p)
            a = estimatorList[i].exp.geometryParamRanges[getProjectParam(p)][0]
            b = estimatorList[i].exp.geometryParamRanges[getProjectParam(p)][1]
            arg0.append(rand() * (b - a) + a)
            bounds.append([a, b])
        return arg0, bounds

    def addFixedParams(arg1):
        assert len(paramNames) == len(arg1) + len(fixParams), 'paramNames = '+str(paramNames)+'\narg1 = '+str(arg1)+'\nfixParams = '+str(fixParams)
        arg = []
        i1 = 0; i = 0
        for p in paramNames:
            if p in fixParams:
                arg.append(fixParams[p])
            else:
                arg.append(arg1[i1])
                i1 += 1
            i += 1
        return np.array(arg)

    def L2norm1(arg1):
        return L2normMixtureUniform(addFixedParams(arg1))
    notFixedParams = [p for p in paramNames if p not in fixParams]

    for ir in range(tryCount):
        arg0, bounds = getArg0AndBounds()
        assert (len(bounds) == len(notFixedParams)) and (len(arg0) == len(bounds)), str(len(bounds))+' '+str(len(notFixedParams))+' '+str(len(arg0))
        fmins[ir], geoms_partial[ir] = optimize.minimize(L2norm1, arg0, bounds=bounds, fun_args=(), paramNames=notFixedParams, method='scipy')
        # method can violate bounds and constrains!
        for j in range(len(bounds)):
            g = geoms_partial[ir]
            geoms_partial[ir][j] = max(g[j], bounds[j][0])
            geoms_partial[ir][j] = min(g[j], bounds[j][1])
        geoms[ir] = addFixedParams(geoms_partial[ir])
        fmins[ir], concentrations[ir], rFactors[ir] = L2normMixtureUniform(geoms[ir], returnConcentrations=True)
        print('RMS R-factor = '+str(fmins[ir])+' '+optimize.arg2string(geoms[ir], paramNames), flush=True)

    ind = np.argsort(fmins)
    output = ''
    for ir in range(tryCount):
        j = ind[ir]
        output += str(fmins[j])+' '+optimize.arg2string(geoms[j], paramNames)+"\n"
    with open(folderToSaveResult+'/minimums.txt', 'w') as f: f.write(output)
    print('Sorted results:')
    for ir in range(tryCount):
        j = ind[ir]
        print('RMS R-factor = {:.4g}'.format(fmins[j]), ' '.join([paramNames[i]+'={:.2g}'.format(geoms[j][i]) for i in range(len(paramNames))]))
        graphs = ()
        for ip in range(len(estimatorList)):
            graphs += (spectrumCollection.params, concentrations[j][:, ip], estimatorList[ip].exp.name)
        graphs += (folderToSaveResult + '/concentrations_try_'+str(ir),)
        plotting.plotToFileAndSaveCsv(*graphs)

        plotting.plotToFileAndSaveCsv(spectrumCollection.params, rFactors[j], 'best r-factors', folderToSaveResult + '/r-factors_try_'+str(ir))

    j = ind[0]
    bestGeom_x_partial = geoms_partial[j]
    best_concentrations = concentrations[j]
    xanesArray = getXanesArray(bestGeom_x_partial)

    for i_param in range(spectrumCollection.params.size):
        param = spectrumCollection.params[i_param]
        xanes = np.sum(xanesArray*best_concentrations[i_param].reshape(-1,1), axis=0)
        xanes = utils.Spectrum(e0, xanes)
        estimator = estimatorList[0]
        exp = copy.deepcopy(estimator.exp)
        exp_spectrum = spectrumCollection.getSpectrumByParam(param)
        exp.spectrum = exp_spectrum
        if estimator.diffFrom is None:
            plotting.plotToFolder(folderToSaveResult, exp, None, xanes, fileName='xanes_approx_p='+str(param))
            np.savetxt(folderToSaveResult+'/xanes_approx_p='+str(param)+'.csv', [e0, exp_spectrum.intensity, xanes.intensity], delimiter=',')
        else:
            plotting.plotToFolder(folderToSaveResult, estimator.projDiff, None, xanes, fileName='xanes_approx_p=' + str(param))
            np.savetxt(folderToSaveResult +'/xanes_approx_p=' + str(param) +'.csv', [e0, estimator.projDiff.spectrum.intensity, xanes.intensity], delimiter=',')

    def plot1d(param):
        optimize.plotMap1d(param, L2norm1, bestGeom_x_partial, bounds=bounds, fun_args=(), paramNames=notFixedParams, optimizeMethod='scipy', calMapMethod=L2NormMap, folder=folderToSaveResult, postfix='_L2norm')

    def plot2d(param1, param2):
        optimize.plotMap2d([param1, param2], L2norm1, bestGeom_x_partial, bounds=bounds, fun_args=(), paramNames=notFixedParams, optimizeMethod='scipy', calMapMethod=L2NormMap, folder=folderToSaveResult, postfix='_L2norm')

    if plotMaps == 'all':
        for i in range(len(notFixedParams)):
            plot1d(i)
        for i1 in range(len(notFixedParams)):
            for i2 in range(i1+1,len(notFixedParams)):
                plot2d(i1,i2)
    else:
        assert isinstance(plotMaps, list)
        for params in plotMaps:
            assert isinstance(params, list)
            if len(params) == 1:
                assert params[0] in notFixedParams, params[0]+' is not in not fixed param list: '+str(notFixedParams)
                plot1d(params[0])
            else:
                assert len(params) == 2
                assert params[0] in notFixedParams, params[0] + ' is not in not fixed param list: ' + str(notFixedParams)
                assert params[1] in notFixedParams, params[1] + ' is not in not fixed param list: ' + str(notFixedParams)
                plot2d(params[0], params[1])


def parseFindGlobalL2NormMinimumMixtureResult(folder):
    with open(os.path.join(folder, 'minimums.txt'), 'r') as mf:
        line = mf.readline(10000).strip()
        minimum = float(line.split(' ')[0])
        paramsLine = line[line.index(' ') + 1:].strip()
        result = {'rFactor': minimum}
        spectraFileNames = glob.glob(f'{folder}/xanes_*_approx_*.png')
        spectraFileNames = [os.path.split(s)[1] for s in spectraFileNames]
        projectNames = []
        for s in spectraFileNames:
            i1 = s.find('_')
            i2 = s.rfind('_')
            assert i2 >= 0
            i2 = s.rfind('_', 0, i2-1)
            projectNames.append(s[i1+1:i2])
        projectNames = np.unique(projectNames)
        if len(projectNames) == 1 and projectNames[0] == 'mixture': projectNames = []
        params = {eq.split('=')[0]:float(eq.split('=')[1]) for eq in paramsLine.split('  ')}
        concentrations = {pn:params[pn] for pn in projectNames if pn in params}
        for pn in projectNames:
            if pn in params:
                del params[pn]
        result['concentrations'] = utils.dictToStr(concentrations)
        result['params'] = utils.dictToStr(params)
        return result


def multiFitHelper(estimators, combinations='all 1', findGlobalL2NormMinimumParams=None, outputFolder='multiFitResult'):
    """
        Fit all experiments by combinations of predicted spectra

        :param estimators: list of Estimator classes
        :param combinations: 'all 1', 'all 2', 'all 1,2', 'all 3', ... - means all combinations of the given size; or list of project/sample indices/ind_lists. Example: combinations = [[0,3], [1,3], 0, 2, [4,2]]
        :param findGlobalL2NormMinimumParams: dict
        :param outputFolder: working folder
        """
    n = len(estimators)
    if isinstance(combinations, str):
        assert combinations[:4] == 'all '
        all_types = [int(s) for s in combinations[4:].split(',')]
        trys = [i for i in range(n)]
        combinations = []
        for m in all_types:
            # combinations += list(itertools.combinations(trys, m))
            combinations += list(itertools.combinations_with_replacement(trys, m))
    names = [estimator.name for estimator in estimators]
    assert len(set(names)) == n, 'Duplicate project names: ' + str(names)
    results = []
    statFolder = f'{outputFolder}/statistics'
    os.makedirs(statFolder, exist_ok=True)
    for comb in combinations:
        if isinstance(comb, int): comb = [comb]
        nameComb = [names[ic] for ic in comb]
        suffix = '_'.join(nameComb)
        folderToSaveResult = f'{outputFolder}/fit_by_{suffix}'
        print(f'Start fitting by', nameComb, 'Save result in', outputFolder)
        ests = [estimators[ic] for ic in comb]
        findGlobalL2NormMinimumMixture(estimatorList=ests, folderToSaveResult=folderToSaveResult, **findGlobalL2NormMinimumParams)
        res = parseFindGlobalL2NormMinimumMixtureResult(folderToSaveResult)
        res['combination'] = suffix
        results.append(res)
        bestSpectrumFile = f'{res["rFactor"]:.4f}_{suffix}.png'
        f = sorted(glob.glob(os.path.join(folderToSaveResult, 'xanes_mixture_approx_0*.png')))
        if len(f) > 0:
            # mixture
            shutil.copyfile(f[0], os.path.join(statFolder, bestSpectrumFile))
        else:
            print('Can\'t find best spectrum graph in folder', folderToSaveResult)
    results = sorted(results, key=lambda r: r['rFactor'], reverse=False)
    with open(os.path.join(statFolder, '! statistics.csv'), 'w') as f:
        f.write("rFactor;combination;concentrations;params\n")
        for r in results:
            f.write(f"{r['rFactor']};{r['combination']};{r['concentrations']};{r['params']}\n")


def multiFit(projects, samples, expSpectrum, samplePreprocessors, combinations='all 1', constructInverseEstimatorParams=None, findGlobalL2NormMinimumParams=None, outputFolder='multiFitResult'):
    """
    Fit all experiments by combinations of given samples

    :param projects: list of projects
    :param samples: list of samples len(projects) == len(samples). Project names must be different!
    :param expSpectrum: exp spectrum
    :param samplePreprocessors: list of convolutionParams or funcs(sample) -> sample
    :param combinations: 'all 1', 'all 2', 'all 1,2', 'all 3', ... - means all combinations of the given size; or list of project/sample indices/ind_lists. Example: combinations = [[0,3], [1,3], 0, 2, [4,2]]
    :param constructInverseEstimatorParams: dict
    :param findGlobalL2NormMinimumParams: dict
    :param outputFolder: working folder
    """
    assert len(projects) == len(samples)
    n = len(projects)
    estimators = []
    for i in range(n):
        projects[i].spectrum = expSpectrum
        est = Estimator(proj=projects[i], samplePreprocessor=samplePreprocessors[i], folderToSaveCVresult=f'{outputFolder}/CVresults/{i}', **constructInverseEstimatorParams)
        est.fit(samples[i])
        estimators.append(est)
    multiFitHelper(estimators, combinations=combinations, findGlobalL2NormMinimumParams=findGlobalL2NormMinimumParams, outputFolder=outputFolder)


def multiFitFuncModels(models, combinations='all 1', optimizeParams=None, makeMixtureParams=None, outputFolder='multiFitResult'):
    """
        Fit all experiments by combinations of predicted spectra

        :param models: list of FuncModel classes
        :param combinations: 'all 1', 'all 2', 'all 1,2', 'all 3', ... - means all combinations of the given size; or list of project/sample indices/ind_lists. Example: combinations = [[0,3], [1,3], 0, 2, [4,2]]
        :param optimizeParams: dict. Params for FuncModel.optimize except folderToSaveResult
        :param makeMixtureParams: dict. Params for FuncModel.makeMixture (if mixtures are in combinations)
        :param outputFolder: working folder
        """
    if makeMixtureParams is None: makeMixtureParams = {}
    if optimizeParams is None: optimizeParams = {}
    if 'trysCount' not in optimizeParams: optimizeParams['trysCount'] = 1
    if 'optType' not in optimizeParams: optimizeParams['optType'] = 'min'
    n = len(models)
    if isinstance(combinations, str):
        assert combinations[:4] == 'all '
        all_types = [int(s) for s in combinations[4:].split(',')]
        trys = [i for i in range(n)]
        combinations = []
        for m in all_types:
            # combinations += list(itertools.combinations(trys, m))
            combinations += list(itertools.combinations_with_replacement(trys, m))
    names = [m.name for m in models]
    assert len(set(names)) == n, 'Duplicate project names: ' + str(names)
    result = None
    statFolder = f'{outputFolder}/statistics'
    os.makedirs(statFolder, exist_ok=True)
    for ic,comb in enumerate(combinations):
        if isinstance(comb, int): comb = [comb]
        nameComb = [names[ic] for ic in comb]
        suffix = '_'.join(nameComb)
        folderToSaveResult = f'{outputFolder}{os.sep}fit_by_{suffix}'
        print(f'Start fitting by', nameComb, 'Save result in', outputFolder)
        ests = [models[ic] for ic in comb]
        if len(ests) > 1: mix = FuncModel.makeMixture(ests, **makeMixtureParams)
        else: mix = ests[0]
        optimizeParams['folderToSaveResult'] = folderToSaveResult
        optValue, optParams = mix.optimize(**optimizeParams)

        if ic == 0:
            result = pd.DataFrame(columns=['func_value', 'combination', 'concentrations', 'params'])
        if len(ests) == 1:
            concentrations = np.nan
            other = optParams
        else:
            concentrations = ' '.join([f'{cn}={optParams[cn]:.2f}' for cn in mix.concNames])
            other = {pn:optParams[pn] for pn in optParams if pn not in mix.concNames}
        result.loc[ic] = {'func_value':optValue, 'combination':suffix, 'concentrations':concentrations, 'params':utils.dictToStr(other)}
        prefix = utils.zfill(0,optimizeParams['trysCount'])+'_'
        files = glob.glob(folderToSaveResult+os.sep+f'{prefix}*')
        for f in files:
            fn = os.path.split(f)[-1]
            newFn, ext = os.path.splitext(fn)
            newFn = newFn[len(prefix):]+'_'+suffix+ext
            shutil.copyfile(f, statFolder+os.sep+newFn)
    ascending = optimizeParams['optType'] == 'min'
    result.sort_values(by=['func_value'], inplace=True, ascending=ascending)
    result.to_excel(statFolder+os.sep+'statistics.xlsx', index=False)


def relativeToConstantPredictionError(yTrue, yPred, energy, weights=None):
    if isinstance(yTrue, pd.DataFrame): yTrue = yTrue.to_numpy()
    if isinstance(yPred, pd.DataFrame): yPred = yPred.to_numpy()
    if weights is None: weights = np.ones(yTrue.shape[0])/yTrue.shape[0]
    y_mean = np.mean(yTrue, axis=0)
    N = yTrue.shape[0]
    u = np.sum(np.array([utils.integral(energy, np.abs(yTrue[i] - yPred[i])**2) for i in range(N)])*weights)
    v = np.sum(np.array([utils.integral(energy, np.abs(yTrue[i] - y_mean)**2) for i in range(N)])*weights)
    return u / v


def calcParamStdDevHelper(geometryParamRanges, paramNames=None, numPointsAlongOneDim=3, trainedInverseModel=None, sample=None):
    assert trainedInverseModel is not None or sample is not None
    assert paramNames is not None or sample is not None
    if trainedInverseModel is None:
        trainedInverseModel = getMethod('Extra Trees')
        trainedInverseModel.fit(sample.params, sample.spectra)
    if paramNames is None: paramNames = sample.paramNames
    N = len(geometryParamRanges)
    assert len(paramNames) == N
    coords = [np.linspace(geometryParamRanges[p][0], geometryParamRanges[p][1], numPointsAlongOneDim) for p in paramNames]
    repeatedCoords = np.meshgrid(*coords)
    m = numPointsAlongOneDim**N
    params = np.zeros((m, N))
    for j in range(N):
        params[:,j] = repeatedCoords[j].flatten()
    xanes = trainedInverseModel.predict(params)
    stdDevXanes = np.std(xanes, axis=0)
    eps = 1e-5*np.mean(stdDevXanes)
    stdDevXanes[stdDevXanes<eps] = eps
    values = [np.unique(params[:,j]) for j in np.arange(N)]
    maxStdDev = {}
    meanStdDev = {}
    for paramInd in range(N):
        paramStdDevs = []
        values1 = copy.deepcopy(values)
        gc.collect()
        del values1[paramInd]
        otherParamInds = np.arange(N)
        otherParamInds = otherParamInds[otherParamInds!=paramInd]
        for otherParamValues in itertools.product( *values1 ):
            inds = np.array([True]*m)
            for j in range(N-1):
                inds = inds & (params[:,otherParamInds[j]]==otherParamValues[j])
            if np.sum(inds) != values[paramInd].size:
                print('Warning!!! np.sum(inds) != values[paramInd].size')
                otherParamValues = np.array(otherParamValues)
                for i in range(m):
                    if np.all(otherParamValues == params[i,otherParamInds]):
                        print(params[i,:])
                if np.sum(inds) == 0: continue
            stdDev = np.std(xanes[inds,:], axis=0)
            paramStdDevs.append(stdDev)
        paramStdDevs = np.array(paramStdDevs)
        maxStdDev[paramNames[paramInd]] = np.mean( np.max(paramStdDevs, axis=0) / stdDevXanes )
        meanStdDev[paramNames[paramInd]] = np.mean(  np.mean(paramStdDevs, axis=0) / stdDevXanes )
    return maxStdDev, meanStdDev


def calcParamStdDev(estimator, numPointsAlongOneDim=3, printResult=True):
    paramNames = estimator.paramNames
    maxStdDev, meanStdDev = calcParamStdDevHelper(geometryParamRanges=estimator.exp.geometryParamRanges, paramNames=paramNames, numPointsAlongOneDim=numPointsAlongOneDim, trainedInverseModel=estimator)
    if printResult:
        for name in paramNames:
            print('Relative StdDev for', name, ': max =', '%0.3f' % maxStdDev[name], 'mean =', '%0.3f' % meanStdDev[name])
    return maxStdDev, meanStdDev


def L2normExact(arg, **extraParams):
    exp = extraParams['exp']
    diffFrom = extraParams['diffFrom']
    calcParams = extraParams['calcParams']
    paramNames = list(exp.geometryParamRanges.keys()); paramNames.sort()
    geomArg = {arg[i]['paramName']:arg[i]['value'] for i in range(len(paramNames))}
    mol = exp.moleculeConstructor(geomArg)
    folder = fdmnes.generateInput(mol, **exp.FDMNES_calc)
    with open(folder+'/params.txt', 'w') as f: json.dump(geomArg, f)
    mol.export_xyz(folder+'/molecule.xyz')
    calcIsGood = False
    for tr in range(calcParams['fdmnesCalcMaxTryCount']):
        if calcParams['local']: fdmnes.runLocal(folder)
        else: fdmnes.runCluster(folder, calcParams['memory'], calcParams['nProcs'])
        try:
            xanes = fdmnes.parseOneFolder(folder)
        except Exception as e:
            print("Error in folder "+folder+" : ", sys.exc_info()[0])
            continue
        if abs(xanes.energy[-1] - float(exp.FDMNES_calc['Energy range'].split(' ')[-1]))<float(exp.FDMNES_calc['Energy range'].split(' ')[-2]) + 1e-6:
            calcIsGood = True
            break
        else: print('Wrong energy range in output file. Folder = '+folder)
    if calcIsGood:
        smoothed_xanes, _ = smoothLib.funcFitSmoothHelper(exp.defaultSmoothParams['fdmnes'], xanes, 'fdmnes', exp, extraParams['norm'])
        with open(folder+'/args_smooth.txt', 'w') as f: json.dump(exp.defaultSmoothParams['fdmnes'], f)
        ind = (exp.intervals['fit_geometry'][0]<=exp.spectrum.energy) & (exp.spectrum.energy<=exp.intervals['fit_geometry'][1])
        if diffFrom is None:
            plotting.plotToFolder(folder, exp, None, smoothed_xanes, fileName='spectrum', title=optimize.arg2string(arg))
            expXanes = exp.spectrum.intensity
            L2 = np.sqrt(utils.integral(exp.spectrum.energy[ind], (smoothed_xanes.intensity[ind]-expXanes[ind])**2))
        else:
            expDiffXanes = diffFrom['projDiff'].spectrum.intensity
            diffXanes = (smoothed_xanes.intensity - diffFrom['spectrumBase'].intensity)*diffFrom['purity']
            plotting.plotToFolder(folder, diffFrom['projDiff'], None, utils.Spectrum(exp.spectrum.energy, diffXanes), fileName='xanesDiff', title=optimize.arg2string(arg))
            L2 = np.sqrt(utils.integral(exp.spectrum.energy[ind], (diffXanes[ind]-expDiffXanes[ind])**2))
        with open(folder+'/L2_norm.txt', 'w') as f: json.dump(L2, f)
        return L2, folder
    else:
        raise Exception('Can\'t calculate spectrum for params '+json.dumps(geomArg)+". Folder: "+folder)


# diffFrom = {'projectBase':..., 'spectrumBase':..., 'purity':...}
# startPoint = 'center', 'random' or dict
# fdmnesCalcParams = {'fdmnesCalcMaxTryCount':??, 'local':True/False, 'memory':??, 'nProcs':??, 'radius','Quadrupole','Absorber','Green','Edge','cellSize' - optional}
def exact(exp, folderToSaveResult, fdmnesCalcParams, convolutionParams, minDeltaFunc=1e-5, startPoint='random', minCount=10, numThreads=1, diffFrom=None):
    folderToSaveResult = utils.fixPath(folderToSaveResult)
    if os.path.exists(folderToSaveResult): remove_tree(folderToSaveResult)
    os.makedirs(folderToSaveResult)
    exp = copy.deepcopy(exp)
    convolutionParams = copy.deepcopy(convolutionParams)
    if 'norm' in convolutionParams:
        norm = convolutionParams['norm'];
        del convolutionParams['norm']
    else: norm = None
    for pName in convolutionParams:
        setValue(exp.defaultSmoothParams['fdmnes'], pName, convolutionParams[pName])
    diffFrom = copy.deepcopy(diffFrom)
    if diffFrom is not None:
        diffFrom['projectBase'].spectrum.intensity = np.interp(exp.spectrum.energy, diffFrom['projectBase'].spectrum.energy, diffFrom['projectBase'].spectrum.intensity)
        diffFrom['projectBase'].spectrum.energy = exp.spectrum.energy
        diffFrom['projDiff'] = copy.deepcopy(exp)
        diffFrom['projDiff'].spectrum = utils.Spectrum(diffFrom['projDiff'].spectrum.energy, exp.spectrum.intensity - diffFrom['projectBase'].spectrum.intensity)
        diffFrom['spectrumBase'], _ = smoothLib.funcFitSmoothHelper(exp.defaultSmoothParams['fdmnes'], diffFrom['spectrumBase'], 'fdmnes', diffFrom['projectBase'], norm)
    fmins = np.zeros(minCount)
    geoms = [None]*minCount
    paramNames = list(exp.geometryParamRanges.keys()); paramNames.sort()
    rand = np.random.rand
    threadPool = ThreadPool(numThreads) if numThreads>1 else None
    if os.path.exists(folderToSaveResult+'/mins'): remove_tree(folderToSaveResult+'/mins')
    os.makedirs(folderToSaveResult+'/mins')
    folderNameSize = 1+math.floor(0.5+math.log(minCount,10))

    def calcOneMin(i):
        arg0 = []
        for p in paramNames:
            a = exp.geometryParamRanges[p][0]; b = exp.geometryParamRanges[p][1]
            if type(startPoint) is dict:
                assert (startPoint[p]>=a) and (startPoint[p]<=b)
                arg0.append(optimize.param(p, startPoint[p], [a,b], (b-a)/10, (b-a)/20))
            elif startPoint=='random':
                arg0.append(optimize.param(p, rand()*(b-a)+a, [a,b], (b-a)/10, (b-a)/20))
            elif startPoint=='center':
                arg0.append(optimize.param(p, (a+b)/2, [a,b], (b-a)/10, (b-a)/20))
        try:
            fmin, geom, folder = optimize.minimizePokoord(L2normExact, arg0, minDeltaFunc = minDeltaFunc, enableOutput = True, methodType = 'random', parallel=False, useRefinement=True, useGridSearch=False, returnTrace=False, extraValue=True, f_kwargs={'exp':exp, 'diffFrom':diffFrom, 'fdmnesCalcParams':fdmnesCalcParams, 'norm':norm})
        except Exception as e:
            print('There was error while search of min № '+str(i))
            logging.error(traceback.format_exc())
            return 1e10, arg0
        newFolderName = folderToSaveResult+'/mins/'+str(i).zfill(folderNameSize)
        copy_tree(folder, newFolderName)
        print(i,'fmin =', fmin, 'geom =', optimize.arg2string(geom))
        return fmin, geom
    if numThreads>1: minsInfo = threadPool.map(calcOneMin, range(minCount))
    else:
        minsInfo = []
        for i in range(minCount): minsInfo.append(calcOneMin(i))
    if os.path.exists(folderToSaveResult+'/pics'): remove_tree(folderToSaveResult+'/pics')
    os.makedirs(folderToSaveResult+'/pics')
    minFolders = os.listdir(folderToSaveResult+'/mins')
    minFolders.sort()
    for mf in minFolders:
        fmin, _ = minsInfo[int(mf)]
        picFile = folderToSaveResult+'/mins/'+mf+'/xanesDiff.png'
        if os.path.exists(picFile):
            copyfile(picFile, folderToSaveResult+'/pics/'+('%.3f' % fmin)+'_'+mf+'.png')


def buildEvolutionTrajectory(vertices, estimator, intermediatePointsNum, folderToSaveResult):
    folderToSaveResult = utils.fixPath(folderToSaveResult)
    os.makedirs(folderToSaveResult, exist_ok=True)
    assert isinstance(vertices[0][0], str), 'The first row must be parameter names'
    paramNames = vertices[0]
    assert estimator.paramNames is not None, 'You forget to fit estimator'
    assert set(estimator.paramNames) == set(paramNames), 'Param names in geometryParamRanges of project:\n'+str(estimator.paramNames)+'\ndoes not equal to dataset param names:\n'+str(paramNames)
    vertices = np.array(vertices[1:])
    ind = {estimator.paramNames[i]:i for i in range(len(paramNames))}
    ind = [ind[paramNames[i]] for i in range(len(paramNames))]
    vertices[:, ind] = vertices[:,range(len(paramNames))]
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

    trajectoryFull_df = pd.DataFrame(data=trajectoryFull, columns=estimator.paramNames)
    trajectoryFull_df.to_csv(folderToSaveResult+'/trajectory_params.txt', sep=' ', index=False)

    trajectory_df = pd.DataFrame(data=vertices, columns=estimator.paramNames)
    trajectory_df.to_csv(folderToSaveResult + '/trajectory_vertices_params.txt', sep=' ', index=False)

    prediction = estimator.predict(trajectoryFull_df.values)
    exp = estimator.projDiff if estimator.diffFrom is not None else estimator.exp
    energy = exp.spectrum.energy
    energyNames = ['e_'+str(e) for e in energy]
    prediction = pd.DataFrame(data=prediction, columns=energyNames)
    prediction.to_csv(folderToSaveResult+'/trajectory_xanes.txt', sep=' ', index=False)

    expData = pd.DataFrame(data=exp.spectrum.intensity.reshape(-1,1).T, columns=energyNames)
    expData.to_csv(folderToSaveResult+'/exp_xanes.txt', sep=' ', index=False)

    for i in range(Ntraj):
        geometryParams = {}
        for j in range(len(estimator.paramNames)):
            geometryParams[estimator.paramNames[j]] = trajectory_df.loc[i,estimator.paramNames[j]]
        molecula = estimator.exp.moleculeConstructor(geometryParams)
        molecula.export_xyz(folderToSaveResult+'/molecule'+str(i+1)+'.xyz')

    dM = np.max(exp.spectrum.intensity)-np.min(exp.spectrum.intensity)
    fig, ax = plotting.createfig()
    for i in range(prediction.shape[0]):
        p = prediction.loc[i]+dM/30*(i+3)
        if i % (intermediatePointsNum+1) == 0: ax.plot(energy, p, linewidth=2, c='r')
        else: ax.plot(energy, p, linewidth=1, c='b')
    ax.plot(energy, exp.spectrum.intensity, c='k', label="Experiment")
    plotting.savefig(folderToSaveResult+'/trajectory.png', fig)
    plotting.closefig(fig)

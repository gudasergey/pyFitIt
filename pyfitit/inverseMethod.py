from . import utils, mixture
utils.fixDisplayError()
import numpy as np
import pandas as pd
import sklearn, copy, os, json, sys, math, logging, traceback, itertools, gc, matplotlib, shutil, scipy
from distutils.dir_util import copy_tree, remove_tree
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.model_selection import KFold
from shutil import copyfile
from . import plotting, ML, optimize, smoothLib, fdmnes, sampling, curveFitting
import matplotlib.pyplot as plt

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
        params = {'function':'linear', 'baseRegression': 'quadric'}
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


def prepareSample(sample0, diffFrom, project, norm, smooth_type='fdmnes'):
    sample = copy.deepcopy(sample0)
    assert set(sample.paramNames) == set(project.geometryParamRanges.keys()), 'Param names in geometryParamRanges of project:\n'+str(list(project.geometryParamRanges.keys()))+'\ndoes not equal to dataset param names:\n'+str(sample.paramNames)
    sample.spectra = smoothLib.smoothDataFrame(project.defaultSmoothParams[smooth_type], sample.spectra, smooth_type, project.spectrum, project.intervals['fit_smooth'], norm=norm, folder=sample.folder)
    if diffFrom is not None:
        sample.spectra = (sample.spectra - diffFrom['spectrumBase'].intensity) * diffFrom['purity']
    return sample


def prepareDiffFrom(project, diffFrom, norm):
    diffFrom = copy.deepcopy(diffFrom)
    diffFrom['projectBase'].spectrum.intensity = np.interp(project.spectrum.energy, diffFrom['projectBase'].spectrum.energy, diffFrom['projectBase'].spectrum.intensity)
    diffFrom['projectBase'].spectrum.energy = project.spectrum.energy
    diffFrom['spectrumBase'], _ = smoothLib.funcFitSmoothHelper(project.defaultSmoothParams['fdmnes'], diffFrom['spectrumBase'], 'fdmnes', diffFrom['projectBase'], norm)
    return diffFrom


class Estimator:
    # diffFrom = {'projectBase':..., 'spectrumBase':..., 'purity':...}
    def __init__(self, method, exp, convolutionParams, normalize=True, CVcount=10, folderToSaveCVresult='', folderToDebugSample='', diffFrom=None, smooth_type='fdmnes', **params):
        folderToSaveCVresult = utils.fixPath(folderToSaveCVresult)
        if method not in allowedMethods:
            raise Exception('Unknown method name. You can use: '+str(allowedMethods))
        self.exp = copy.deepcopy(exp)
        interval = self.exp.intervals['fit_geometry']
        ind = (self.exp.spectrum.energy >= interval[0]) & (self.exp.spectrum.energy <= interval[1])
        self.exp.spectrum.energy = self.exp.spectrum.energy[ind]
        self.exp.spectrum.intensity = self.exp.spectrum.intensity[ind]
        self.convolutionParams = {k:convolutionParams[k] for k in convolutionParams}
        if 'norm' in self.convolutionParams:
            self.norm = self.convolutionParams['norm']
            del self.convolutionParams['norm']
        else: self.norm = None
        for pName in self.convolutionParams:
            self.exp.defaultSmoothParams[smooth_type][pName] = self.convolutionParams[pName]
        self.regressor = getMethod(method, params)
        self.normalize = normalize
        if normalize: self.regressor = ML.Normalize(self.regressor, xOnly=False)
        self.CVcount = CVcount
        assert CVcount>=2
        self.folderToSaveCVresult = folderToSaveCVresult
        self.folderToDebugSample = folderToDebugSample
        self.diffFrom = copy.deepcopy(diffFrom)
        if diffFrom is not None:
            self.diffFrom = prepareDiffFrom(self.exp, diffFrom, self.norm)
            self.expDiff = copy.deepcopy(self.exp)
            self.expDiff.spectrum = utils.Spectrum(self.expDiff.spectrum.energy, self.exp.spectrum.intensity - self.diffFrom['projectBase'].spectrum.intensity)
        self.smooth_type = smooth_type

    def fit(self, sample0):
        sample = prepareSample(sample0, self.diffFrom, self.exp, self.norm, self.smooth_type)
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

        res, stddevs, predictions = inverseCrossValidation(self.regressor, sample, self.CVcount)
        output = 'Inverse method relative to constant prediction error = %5.3g\n' % res['relToConstPredError']
        # output += 'L2 mean error = %5.3g\n' % res['meanL2']
        print(output)
        if self.folderToSaveCVresult != '':
            os.makedirs(self.folderToSaveCVresult, exist_ok=True)
            with open(self.folderToSaveCVresult+'/info.txt', 'w') as f: f.write(output)
            ind = np.argsort(stddevs)
            n = stddevs.size
            exp = self.exp if self.diffFrom is None else self.expDiff
            energy = exp.spectrum.energy

            def plotCVres(energy, trueXan, predXan, fileName):
                fig, ax = plt.subplots(figsize=plotting.figsize)
                ax.plot(energy, trueXan, label="True spectrum")
                ax.plot(energy, predXan, label="Predicted spectrum")
                ax.legend()
                fig.set_size_inches((16,9))
                ax.set_title(os.path.basename(fileName))
                plt.savefig(fileName, dpi=plotting.dpi)
                # if not utils.isJupyterNotebook(): plt.close(fig)  - notebooks also have limit - 20 figures
                if matplotlib.get_backend() != 'nbAgg': plt.close(fig)
                np.savetxt(fileName[:fileName.rfind('.')]+'.csv', [energy, trueXan, predXan], delimiter=',')
            i = ind[n//2]
            plotCVres(energy, sample.spectra.loc[i], predictions[i], self.folderToSaveCVresult+'/xanes_mean_error.png')
            i = ind[9*n//10]
            plotCVres(energy, sample.spectra.loc[i], predictions[i], self.folderToSaveCVresult+'/xanes_max_0.9_error.png')
            i = ind[-1]
            plotCVres(energy, sample.spectra.loc[i], predictions[i], self.folderToSaveCVresult+'/xanes_max_error.png')
        self.regressor.fit(sample.params, sample.spectra)
        self.paramNames = sample.paramNames
        # calcParamStdDev(self)

    def predict(self, params):
        return self.regressor.predict(params)

    def compareDifferentMethods_old(self, sample0, folderToSaveResult, verticesNum=10, intermediatePointsNum=10, calcExactInternalPoints=False):
        folderToSaveResult = utils.fixPath(folderToSaveResult)
        sample = self.prepareSample(sample0, self.diffFrom, self.exp, self.norm)
        fig, ax = plt.subplots(figsize=plotting.figsize)
        if not os.path.exists(folderToSaveResult): os.makedirs(folderToSaveResult)
        plotData = pd.DataFrame()
        for methodName in allowedMethods:
            method = getMethod(methodName)
            res, _, _ = inverseCrossValidation(method, sample, self.CVcount)
            print(methodName+' relative to constant prediction error = %5.3g\n' % res['relToConstPredError'], flush=True)
            plotData1, edgePoints, labels, curveParams = ML.getOneDimPrediction(method, sample.params, sample.spectra, verticesNum=verticesNum, intermediatePointsNum=intermediatePointsNum)
            plotData[methodName+'_'+labels[1]] = plotData1[1]
        plotData[labels[0]] = plotData1[0]
        for methodName in allowedMethods:
            ax.plot(plotData[labels[0]], plotData[methodName+'_'+labels[1]], label=methodName)
        ax.plot(edgePoints[0], edgePoints[1], 'r*', ms=10, label='exact edge points')
        if calcExactInternalPoints:
            exactFolder = folderToSaveResult+os.sep+'sample'
            calcFolder = exactFolder+os.sep+'calc'
            if not os.path.exists(exactFolder+os.sep+'params.txt'):
                if not os.path.exists(calcFolder):
                    os.makedirs(exactFolder); os.makedirs(calcFolder);
                    for i in range(curveParams.shape[0]):
                        params = {sample0.paramNames[j]:curveParams[i,j] for j in range(len(sample0.paramNames))}
                        m = self.exp.moleculeConstructor(params)
                        folder = calcFolder+os.sep+utils.zfill(i,curveParams.shape[0])
                        fdmnes.generateInput(m, folder=folder, **self.exp.FDMNES_calc)
                        geometryParamsToSave = [[sample0.paramNames[j], curveParams[i,j]] for j in range(len(sample0.paramNames))]
                        with open(folder+os.sep+'geometryParams.txt', 'w') as f: json.dump(geometryParamsToSave, f)
                sampling.calcSpectra('fdmnes', 'run-cluster', nProcs=6, memory=10000, calcSampleInParallel=10, folder=calcFolder, recalculateErrorsAttemptCount = 2, continueCalculation = True)
                sampling.collectResults(spectralProgram='fdmnes', folder=calcFolder, outputFolder=exactFolder)
            exactSample0 = ML.Sample.readFolder(exactFolder)
            exactSample = self.prepareSample(exactSample0, self.diffFrom, self.exp, self.norm)
            exact_x = exactSample.params.loc[:, exactSample.params.columns == labels[0]].values.reshape(-1)
            exact_y = exactSample.spectra.loc[:, exactSample.spectra.columns == labels[1]].values.reshape(-1)
            tmpind = np.argsort(exact_x)
            exact_x = exact_x[tmpind]
            exact_y = exact_y[tmpind]
            if exact_y.size == plotData.shape[0]:
                ax.plot(exact_x, exact_y, label='exact', lw=2, color='k') # - тоже неправильно будет строится, если поменялись длины
                plotData['exact'] = exact_y
                fig2, ax2 = plt.subplots(figsize=plotting.figsize)
                for methodName in allowedMethods:
                    ax2.plot(plotData[labels[0]], np.abs(plotData[methodName+'_'+labels[1]]-exact_y), label=methodName)
                ax2.plot(edgePoints[0], np.zeros(edgePoints[0].size), 'r*', ms=10, label='exact edge points')
                ax2.plot([edgePoints[0][0], edgePoints[0][-1]], [0,0], label='exact', lw=2, color='k')
                delta = (edgePoints[0][1]-edgePoints[0][0])/10
                # ax2.set_xlim(edgePoints[0][1]-delta, edgePoints[0][3]+delta)
                ax2.legend()
                ax2.set_xlabel(labels[0])
                ax2.set_ylabel('abs('+labels[1]+'-exact)')
                fig2.set_size_inches((16, 9))
                fig2.savefig(folderToSaveResult+os.sep+'compareDifferentMethodsInverse_delta.png', dpi=plotting.dpi)
            else:
                print('Length of exact data doesn\'t match verticesNum and intermediatePointsNum')

        # ax.set_xlim(edgePoints[0][1]-delta, edgePoints[0][3]+delta)
        ax.legend()
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        fig.set_size_inches((16, 9))
        plt.show()
        plotData.to_csv(folderToSaveResult+os.sep+'compareDifferentMethodsInverse.csv', sep=' ', index=False)
        fig.savefig(folderToSaveResult+os.sep+'compareDifferentMethodsInverse.png', dpi=plotting.dpi)
        # if not utils.isJupyterNotebook(): plt.close(fig)  #notebooks also have limit - 20 figures # - sometimes figure is not shown
        # if matplotlib.get_backend() != 'nbAgg': plt.close(fig)


def compareDifferentMethods(sampleTrain, sampleTest, energyPoint, geometryParam, project, diffFrom=None, CVcount=4, folderToSaveResult='inverseMethodsCompare'):
    folderToSaveResult = utils.fixPath(folderToSaveResult)
    if not np.array_equal(sampleTrain.paramNames, sampleTest.paramNames):
        raise Exception('sampleTrain and sampleTest have different geometry parameters')
    if geometryParam not in sampleTrain.paramNames:
        raise Exception('samples don\'t contain geometry parameter '+str(geometryParam))
    if diffFrom is not None: diffFrom = prepareDiffFrom(project, diffFrom, norm=None)
    sampleTrain = prepareSample(sampleTrain, diffFrom, project, norm=None)
    sampleTest = prepareSample(sampleTest, diffFrom, project, norm=None)
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
            res, _, _ = inverseCrossValidation(method, sampleTrain, CVcount)
            print(methodName+' relative to constant prediction error (all energies) = %5.3g\n' % res['relToConstPredError'], flush=True)
        method.fit(sampleTrain.params, sampleTrain.spectra[energyColumn])
        predicted = method.predict(sampleTest.params).reshape(-1)
        plotData[methodName+'_'+energyColumn] = predicted
    plotData[geometryParam] = sampleTest.params[geometryParam].values
    plotData['exact'] = sampleTest.spectra[energyColumn]
    plotData.to_csv(folderToSaveResult+os.sep+'compareDifferentMethodsInverse.csv', sep=' ', index=False)

    fig, ax = plt.subplots(figsize=plotting.figsize)
    for methodName in allowedMethods:
        ax.plot(plotData[geometryParam], plotData[methodName+'_'+energyColumn], label=methodName)
    ax.plot(plotData[geometryParam], sampleTest.spectra[energyColumn], label='exact', lw=2, color='k')

    ax.legend()
    ax.set_xlabel(geometryParam)
    ax.set_ylabel(energyColumn)
    fig.set_size_inches((16, 9))
    plt.show()
    fig.savefig(folderToSaveResult+os.sep+'compareDifferentMethodsInverse.png', dpi=plotting.dpi)
    # if not utils.isJupyterNotebook(): plt.close(fig)  #notebooks also have limit - 20 figures # - sometimes figure is not shown
    # if matplotlib.get_backend() != 'nbAgg': plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=plotting.figsize)
    for methodName in allowedMethods:
        ax2.plot(plotData[geometryParam], np.abs(plotData[methodName+'_'+energyColumn]-plotData['exact']), label=methodName)
    ax2.plot(plotData[geometryParam], np.zeros(plotData[geometryParam].size), label='exact', lw=2, color='k')
    ax2.legend()
    ax2.set_xlabel(geometryParam)
    ax2.set_ylabel('abs('+energyColumn+'-exact)')
    fig2.set_size_inches((16, 9))
    fig2.savefig(folderToSaveResult+os.sep+'compareDifferentMethodsInverse_delta.png', dpi=plotting.dpi)


def makeDictFromVector(arg, paramNames):
    assert len(arg) == len(paramNames)
    return {paramNames[i]:arg[i] for i in len(arg)}


def makeVectorFromDict(params, paramNames, check=True):
    if check:
        assert set(params.keys()) == set(paramNames)
    return np.array([params[p] for p in paramNames])


def findGlobalL2NormMinimum(trysCount, estimator, folderToSaveResult, calcXanes=None, fixParams=None, contourMapCalcMethod='fast', plotContourMaps='all', extra_plot_func=None):
    """
        Try several optimizations from random start point. Retuns sorted list of results and plot contours around minimum

        :param trysCount: number of attempts to find minimum
        :param estimator: instance of Estimator class to predict spectrum by parameters
        :param folderToSaveResult: folder to save graphs
        :param calcXanes: calcXanes = {'local':True/False, /*for cluster - */ 'memory':..., 'nProcs':...}
        :param fixParams: dict of paramName:value to fix
        :param contourMapCalcMethod: 'fast' - plot contours of the target function; 'thorough' - plot contours of the min of target function by all arguments except axes
        :param plotContourMaps: 'all' or list of 1-element or 2-elements lists of axes names to plot contours of target function
        :param extra_plot_func: user defined function to plot something on result contours: extra_plot_func(ax, axisNamesList)
        :return: dict with keys 'value' (minimum value), 'x'
        """

    return findGlobalL2NormMinimumMixture(trysCount, [estimator], folderToSaveResult, calcXanes=calcXanes, fixParams=fixParams, contourMapCalcMethod=contourMapCalcMethod, plotContourMaps=plotContourMaps, extra_plot_func=extra_plot_func)


def findGlobalL2NormMinimumMixture(trysCount, estimatorList, folderToSaveResult, calcXanes=None, fixParams=None, contourMapCalcMethod='fast', plotContourMaps='all', extra_plot_func=None):
    """
    Try several optimizations from random start point. Retuns sorted list of results and plot contours around minimum

    :param trysCount: number of attempts to find minimum
    :param estimatorList: list of Estimator classes of mixture components
    :param folderToSaveResult: folder to save graphs
    :param calcXanes: calcXanes = {'local':True/False, /*for cluster - */ 'memory':..., 'nProcs':...}
    :param fixParams: dict of paramName:value to fix. Param names: projectName_paramName, concentration names - projectName
    :param contourMapCalcMethod: 'fast' - plot contours of the target function; 'thorough' - plot contours of the min of target function by all arguments except axes
    :param plotContourMaps: 'all' or list of 1-element or 2-elements lists of axes names to plot contours of target function
    :param extra_plot_func: user defined function to plot something on result contours: extra_plot_func(ax, axisNamesList)
    :return: dict with keys 'value' (minimum value), 'x' (list of component parameter values lists), 'x1d' (flattened array of all param values), 'paramNames1d', 'concentrations', 'spectra', 'mixtureSpectrum'
    """

    oneComponent = len(estimatorList) == 1
    estimator0 = estimatorList[0]
    exp0 = estimator0.exp
    e0 = exp0.spectrum.energy
    ind = (exp0.intervals['fit_geometry'][0] <= e0) & (e0 <= exp0.intervals['fit_geometry'][1])
    e0 = e0[ind]
    if fixParams is None: fixParams = {}
    folderToSaveResult = utils.fixPath(folderToSaveResult)
    if not os.path.exists(folderToSaveResult): os.makedirs(folderToSaveResult)

    def makeSpectraFunc(i):
        estimator = estimatorList[i]
        exp = estimator.exp
        e = exp.spectrum.energy

        def spectraFunc(geomArg):
            xanesPred = estimator.predict(geomArg.reshape(1,-1)).reshape(-1)
            xanesPred = np.interp(e0, e, xanesPred)
            return xanesPred
        return spectraFunc
    spectraFuncs = [makeSpectraFunc(i) for i in range(len(estimatorList))]

    def makeMixture(spectraList, concentrations):
        mixtureSpectrum = spectraList[0]*concentrations[0]
        for i in range(1,len(estimatorList)):
            mixtureSpectrum += concentrations[i]*spectraList[i]
        return mixtureSpectrum

    def distToExperiment(mixtureSpectrum, allArgs):
        estimator = estimatorList[0]
        exp = estimator.exp
        e = exp.spectrum.energy
        ind = (exp.intervals['fit_geometry'][0] <= e) & (e <= exp.intervals['fit_geometry'][1])
        e0 = e[ind]
        expXanes = exp.spectrum.intensity if estimator.diffFrom is None else estimator.expDiff.spectrum.intensity
        expXanes = expXanes[ind]
        rFactor = utils.integral(e0, (mixtureSpectrum - expXanes) ** 2) / utils.integral(e0, expXanes ** 2)
        return rFactor

    paramNames = [estimator.paramNames.tolist() for estimator in estimatorList]
    bounds = []
    for estimator in estimatorList:
        bounds.append([estimator.exp.geometryParamRanges[p] for p in estimator.paramNames])
    componentNames = [estimator.exp.name for estimator in estimatorList]
    for i in range(len(componentNames)):
        cn = componentNames[i]
        if componentNames.count(cn) > 1:
            k = 1
            for j in range(len(componentNames)):
                if componentNames[j] == cn:
                    componentNames[j] = cn+f'_{k}'
                    k += 1
    minimums = mixture.findGlobalMinimumMixture(distToExperiment, spectraFuncs, makeMixture, trysCount, bounds,  paramNames, componentNames=componentNames, folderToSaveResult=folderToSaveResult, fixParams=fixParams, contourMapCalcMethod=contourMapCalcMethod, plotContourMaps=plotContourMaps, extra_plot_func=extra_plot_func)

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
        for ip in range(len(estimatorList)):
            estimator = estimatorList[ip]
            exp = estimator.exp
            xanes = utils.Spectrum(e0, minimum['spectra'][ip])
            fileName = 'xanes_approx_'+strj if oneComponent else 'xanes_'+exp.name+'_approx_'+strj
            if estimator.diffFrom is None:
                plotting.plotToFolder(folderToSaveResult, exp, None, xanes, fileName=fileName)
                np.savetxt(folderToSaveResult+os.sep+fileName+'.csv', [exp.spectrum.energy, exp.spectrum.intensity, xanes.intensity], delimiter=',')
            else:
                plotting.plotToFolder(folderToSaveResult, estimator.expDiff, None, xanes, fileName=fileName)
                np.savetxt(folderToSaveResult+os.sep+fileName+'.csv', [exp.spectrum.energy, estimator.expDiff.spectrum.intensity, xanes.intensity], delimiter=',')
        if not oneComponent:
            xanesMixture = utils.Spectrum(e0, minimum['mixtureSpectrum'])
            plotting.plotToFolder(folderToSaveResult, estimatorList[0].exp, None, xanesMixture, title=resultString, fileName='xanes_mixture_approx_' + strj)
            np.savetxt(folderToSaveResult + '/xanes_mixture_approx_' + strj + '.csv', [estimatorList[0].exp.spectrum.energy, estimatorList[0].exp.spectrum.intensity, xanesMixture.intensity], delimiter=',')

    minimum = minimums[0]
    concentrations = minimum['concentrations']
    for ip in range(len(estimatorList)):
        estimator = estimatorList[ip]
        exp = estimator.exp
        if exp.moleculeConstructor is None: continue
        bestGeom = {}
        for p,j in zip(estimator.paramNames, range(len(estimator.paramNames))):
            bestGeom[p] = minimum['x'][ip][j]
        M = exp.moleculeConstructor(bestGeom)
        if hasattr(M, 'export_xyz'):
            M.export_xyz(folderToSaveResult+'/molecula_'+exp.name+'_best.xyz')
            fdmnes.generateInput(M, **exp.FDMNES_calc, folder=folderToSaveResult+'/fdmnes_'+exp.name)

        if calcXanes is None: continue
        if calcXanes['local']: fdmnes.runLocal(folderToSaveResult+'/fdmnes_'+exp.name)
        else: fdmnes.runCluster(folderToSaveResult+'/fdmnes_'+exp.name, calcXanes['memory'], calcXanes['nProcs'])
        xanes = fdmnes.parse_one_folder(folderToSaveResult+'/fdmnes_'+exp.name)
        smoothed_xanes, _ = smoothLib.funcFitSmoothHelper(exp.defaultSmoothParams['fdmnes'], xanes, 'fdmnes', exp, estimator.norm)
        if estimator.diffFrom is None:
            plotting.plotToFolder(folderToSaveResult, exp, None, smoothed_xanes, fileName='xanes_'+exp.name+'_best_minimum')
            np.savetxt(folderToSaveResult+'/xanes_'+exp.name+'_best_minimum.csv', [exp.spectrum.energy, exp.spectrum.intensity, smoothed_xanes.intensity], delimiter=',')
        else:
            plotting.plotToFolder(folderToSaveResult, exp, None, smoothed_xanes, fileName='xanes_'+exp.name+'_best_minimum', append=[{'data':estimator.diffFrom['spectrumBase'].intensity, 'label':'spectrumBase'}, {'data':estimator.diffFrom['projectBase'].spectrum.intensity, 'label':'projectBase'}])
            np.savetxt(folderToSaveResult+'/xanes_'+exp.name+'_best_minimum.csv', [exp.spectrum.energy, exp.spectrum.intensity, smoothed_xanes.intensity], delimiter=',')
            smoothed_xanes.intensity = (smoothed_xanes.intensity - estimator.diffFrom['spectrumBase'].intensity)*estimator.diffFrom['purity']
            plotting.plotToFolder(folderToSaveResult, estimator.expDiff, None, smoothed_xanes, fileName='xanesDiff_'+exp.name+'_best_minimum')
            np.savetxt(folderToSaveResult+'/xanesDiff_best_minimum.csv', [exp.spectrum.energy, estimator.expDiff.spectrum.intensity, smoothed_xanes.intensity], delimiter=',')
        concentration = concentrations[ip]
        if estimator == estimatorList[0]:
            smoothed_xanesMixture = smoothed_xanes.clone()
            smoothed_xanesMixture.intensity *= concentration
        else:
            smoothed_xanesMixture.intensity += concentration * np.interp(smoothed_xanesMixture.energy, smoothed_xanes.energy, smoothed_xanes.intensity)
    if calcXanes is not None and not oneComponent:
        smoothed_xanesMixture.intensity /= np.sum(concentrations)
        plotting.plotToFolder(folderToSaveResult, exp0, None, smoothed_xanesMixture, fileName='xanes_mixture_best_minimum')
        np.savetxt(folderToSaveResult+'/xanes_mixture_best_minimum.csv', [exp0.spectrum.energy, exp0.spectrum.intensity, smoothed_xanesMixture.intensity], delimiter=',')
    if oneComponent:
        return {'value':minimum['value'], 'x':minimum['x']}
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
            plotting.plotToFolder(folderToSaveResult, estimator.expDiff, None, xanes, fileName='xanes_approx_p='+str(param))
            np.savetxt(folderToSaveResult+'/xanes_approx_p='+str(param)+'.csv', [e0, estimator.expDiff.spectrum.intensity, xanes.intensity], delimiter=',')

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


def inverseCrossValidation(estimator, sample, CVcount):
    if sample.spectra.shape[0] > 20:
        kf = KFold(n_splits=CVcount, shuffle=True, random_state=0)
    else:
        kf = sklearn.model_selection.LeaveOneOut()
    X = sample.params
    y = sample.spectra.to_numpy()
    prediction_spectra = np.zeros(y.shape)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index,:], y[test_index,:]
        estimator.fit(X_train, y_train)
        prediction_spectra[test_index] = estimator.predict(X_test)
    y_mean = np.mean(y, axis=0)
    N = y.shape[0]
    u = np.mean([utils.integral(sample.energy, (y[i] - prediction_spectra[i]) ** 2) for i in range(N)])
    v = np.mean([utils.integral(sample.energy, (y[i] - y_mean) ** 2) for i in range(N)])
    score = 1 - u / v
    stddevs = np.array([np.sqrt(utils.integral(sample.energy, (y[i] - prediction_spectra[i]) ** 2)) for i in range(N)])
    meanL2 = np.mean(stddevs)
    return {'relToConstPredError':1-score, 'meanL2':meanL2}, stddevs, prediction_spectra


def calcParamStdDev(estimator, numPointsAlongOneDim=3):
    geometryParamRanges = estimator.exp.geometryParamRanges
    N = len(geometryParamRanges)
    paramNames = estimator.paramNames
    coords = [np.linspace(geometryParamRanges[p][0], geometryParamRanges[p][1], numPointsAlongOneDim) for p in estimator.paramNames]
    repeatedCoords = np.meshgrid(*coords)
    m = numPointsAlongOneDim**N
    params = np.zeros((m, N))
    for j in range(N):
        params[:,j] = repeatedCoords[j].flatten()
    xanes = estimator.predict(params)
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
    for name in paramNames:
        print('Relative StdDev for', name, ': max =', '%0.3f' % maxStdDev[name], 'mean =', '%0.3f' % meanStdDev[name])


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
            xanes = fdmnes.parse_one_folder(folder)
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
        ind = (exp.intervals['fit_smooth'][0]<=exp.spectrum.energy) & (exp.spectrum.energy<=exp.intervals['fit_smooth'][1])
        if diffFrom is None:
            plotting.plotToFolder(folder, exp, None, smoothed_xanes, fileName='spectrum', title=optimize.arg2string(arg))
            expXanes = exp.spectrum.intensity
            L2 = np.sqrt(utils.integral(exp.spectrum.energy[ind], (smoothed_xanes.intensity[ind]-expXanes[ind])**2))
        else:
            expDiffXanes = diffFrom['expDiff'].spectrum.intensity
            diffXanes = (smoothed_xanes.intensity - diffFrom['spectrumBase'].intensity)*diffFrom['purity']
            plotting.plotToFolder(folder, diffFrom['expDiff'], None, utils.Spectrum(exp.spectrum.energy, diffXanes), fileName='xanesDiff', title=optimize.arg2string(arg))
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
        diffFrom['expDiff'] = copy.deepcopy(exp)
        diffFrom['expDiff'].spectrum = utils.Spectrum(diffFrom['expDiff'].spectrum.energy, exp.spectrum.intensity - diffFrom['projectBase'].spectrum.intensity)
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
    exp = estimator.expDiff if estimator.diffFrom is not None else estimator.exp
    energy = exp.spectrum.energy
    energyNames = ['e_'+str(e) for e in energy]
    prediction = pd.DataFrame(data=prediction, columns=energyNames)
    prediction.to_csv(folderToSaveResult+'/trajectory_xanes.txt', sep=' ', index=False)

    expData = pd.DataFrame(data=exp.spectrum.intensity.reshape(-1,1).T, columns=energyNames)
    expData.to_csv(folderToSaveResult+'/exp_xanes.txt', sep=' ', index=False)

    for i in range(Ntraj):
        geometryParams = {};
        for j in range(len(estimator.paramNames)):
            geometryParams[estimator.paramNames[j]] = trajectory_df.loc[i,estimator.paramNames[j]]
        molecula = estimator.exp.moleculeConstructor(geometryParams)
        molecula.export_xyz(folderToSaveResult+'/molecule'+str(i+1)+'.xyz')

    dM = np.max(exp.spectrum.intensity)-np.min(exp.spectrum.intensity)
    fig, ax = plt.subplots(figsize=plotting.figsize)
    for i in range(prediction.shape[0]):
        p = prediction.loc[i]+dM/30*(i+3)
        if i % (intermediatePointsNum+1) == 0: ax.plot(energy, p, linewidth=2, c='r')
        else: ax.plot(energy, p, linewidth=1, c='b')
    ax.plot(energy, exp.spectrum.intensity, c='k', label="Experiment")
    fig.set_size_inches((16,9))
    plt.savefig(folderToSaveResult+'/trajectory.png', dpi=plotting.dpi)
    plt.close(fig)

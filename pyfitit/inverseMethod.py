from . import utils
utils.fixDisplayError()
import numpy as np
import pandas as pd
import sklearn, copy, os, json, sys, math, logging, traceback, itertools, gc, matplotlib, shutil
from distutils.dir_util import copy_tree, remove_tree
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.model_selection import KFold
from shutil import copyfile
from .optimize import setValue
from . import plotting, ML, optimize, smoothLib, fdmnes, sampling
import matplotlib.pyplot as plt

allowedMethods = ['Ridge', 'Ridge Quadric', 'Extra Trees', 'RBF']
# if utils.isLibExists("lightgbm"):
#     import lightgbm as lgb
#     allowedMethods.append('LightGBM')

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
    else: raise Exception('Unknown method name. You can use: '+str(allowedMethods))
    return regressor


def prepareSample(sample0, diffFrom, project, norm):
    sample = copy.deepcopy(sample0)
    assert set(sample.params.columns.values) == set(project.geometryParamRanges.keys()), 'Param names in geometryParamRanges of experiment does not equal to dataset param names'
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
    def __init__(self, method, exp, convolutionParams, normalize=True, CVcount=10, folderToSaveCVresult='', folderToDebugSample='', diffFrom=None, **params):
        folderToSaveCVresult = utils.fixPath(folderToSaveCVresult)
        if method not in allowedMethods:
            raise Exception('Unknown method name. You can use: '+str(allowedMethods))
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
            setValue(self.exp.defaultSmoothParams['fdmnes'], pName, self.convolutionParams[pName])
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

    def fit(self, sample0):
        sample = prepareSample(sample0, self.diffFrom, self.exp, self.norm)
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
                fig, ax = plt.subplots()
                ax.plot(energy, trueXan, label="True spectrum")
                ax.plot(energy, predXan, label="Predicted spectrum")
                ax.legend()
                fig.set_size_inches((16,9))
                ax.set_title(os.path.basename(fileName))
                plt.savefig(fileName)
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
        fig, ax = plt.subplots()
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
                fig2, ax2 = plt.subplots()
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
                fig2.savefig(folderToSaveResult+os.sep+'compareDifferentMethodsInverse_delta.png')
            else:
                print('Length of exact data doesn\'t match verticesNum and intermediatePointsNum')

        # ax.set_xlim(edgePoints[0][1]-delta, edgePoints[0][3]+delta)
        ax.legend()
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        fig.set_size_inches((16, 9))
        plt.show()
        plotData.to_csv(folderToSaveResult+os.sep+'compareDifferentMethodsInverse.csv', sep=' ', index=False)
        fig.savefig(folderToSaveResult+os.sep+'compareDifferentMethodsInverse.png')
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

    fig, ax = plt.subplots()
    for methodName in allowedMethods:
        ax.plot(plotData[geometryParam], plotData[methodName+'_'+energyColumn], label=methodName)
    ax.plot(plotData[geometryParam], sampleTest.spectra[energyColumn], label='exact', lw=2, color='k')

    ax.legend()
    ax.set_xlabel(geometryParam)
    ax.set_ylabel(energyColumn)
    fig.set_size_inches((16, 9))
    plt.show()
    fig.savefig(folderToSaveResult+os.sep+'compareDifferentMethodsInverse.png')
    # if not utils.isJupyterNotebook(): plt.close(fig)  #notebooks also have limit - 20 figures # - sometimes figure is not shown
    # if matplotlib.get_backend() != 'nbAgg': plt.close(fig)

    fig2, ax2 = plt.subplots()
    for methodName in allowedMethods:
        ax2.plot(plotData[geometryParam], np.abs(plotData[methodName+'_'+energyColumn]-plotData['exact']), label=methodName)
    ax2.plot(plotData[geometryParam], np.zeros(plotData[geometryParam].size), label='exact', lw=2, color='k')
    ax2.legend()
    ax2.set_xlabel(geometryParam)
    ax2.set_ylabel('abs('+energyColumn+'-exact)')
    fig2.set_size_inches((16, 9))
    fig2.savefig(folderToSaveResult+os.sep+'compareDifferentMethodsInverse_delta.png')


def L2norm(arg, estimator):
    paramNames = estimator.paramNames
    exp = estimator.exp
    e = exp.spectrum.energy
    geomArg = np.zeros([1,len(paramNames)])
    for i in range(len(paramNames)): geomArg[0,i] = arg[i]
    xanesPred = estimator.predict(geomArg).reshape(e.size)
    ind = (exp.intervals['fit_geometry'][0] <= e) & (e <= exp.intervals['fit_geometry'][1])
    expXanes = exp.spectrum.intensity if estimator.diffFrom is None else estimator.expDiff.spectrum.intensity
    rFactor = utils.integral(e[ind], (xanesPred[ind]-expXanes[ind])**2) / utils.integral(e[ind], expXanes[ind]**2)
    return rFactor


# calcXanes = {'local':True/False, /*for cluster - */ 'memory':..., 'nProcs':...}
# L2NormMap='fast' or 'thorough'
def findGlobalL2NormMinimum(N, estimator, folderToSaveResult, calcXanes=None, fixParams=None, L2NormMap='fast'):
    if fixParams is None: fixParams = {}
    folderToSaveResult = utils.fixPath(folderToSaveResult)
    if not os.path.exists(folderToSaveResult): os.makedirs(folderToSaveResult)
    exp = estimator.exp
    fmins = np.zeros(N)
    geoms = [None]*N
    paramNames = estimator.paramNames
    arg0 = []; bounds = []
    rand = np.random.rand
    for p in paramNames:
        a = estimator.geometryParamRanges[p][0];
        b = estimator.geometryParamRanges[p][1]
        arg0.append(rand() * (b - a) + a)
        bounds.append([a, b])
    for ir in range(N):
        fmins[ir], geoms[ir] = optimize.minimize(L2norm, arg0, bounds, fun_args=(estimator,), paramNames=paramNames, method='scipy')
        print('R-factor = '+str(fmins[ir])+' '+optimize.arg2string(geoms[ir], paramNames), flush=True)

    ind = np.argsort(fmins)
    output = ''
    for ir in range(N):
        j = ind[ir]
        output += str(fmins[j])+' '+optimize.arg2string(geoms[j], paramNames)+"\n"
    with open(folderToSaveResult+'/minimums.txt', 'w') as f: f.write(output)
    with open(folderToSaveResult+'/args_smooth.txt', 'w') as f:  json.dump(estimator.convolutionParams, f)
    print('Sorted results:')
    for ir in range(N):
        j = ind[ir]
        print('R-factor = {:.4g}'.format(fmins[j]), ' '.join([paramNames[i]+'={:.2g}'.format(geoms[j][i]) for i in range(len(paramNames))]))
        strj = str(ir).zfill(1+math.floor(0.5+math.log(N,10))) # str(ir) !!!!!! - to have names dorted in same order as minimums
        geomArg = np.zeros([1,len(paramNames)])
        for i in range(len(paramNames)): geomArg[0,i] = geoms[j][i]
        xanesPred = estimator.predict(geomArg).reshape(exp.spectrum.energy.size)
        xanesPred = utils.Spectrum(exp.spectrum.energy, xanesPred.reshape(xanesPred.size))
        if estimator.diffFrom is None:
            plotting.plotToFolder(folderToSaveResult, exp, None, xanesPred, fileName='xanes_approx_'+strj)
            np.savetxt(folderToSaveResult+'/xanes_approx_'+strj+'.csv', [exp.spectrum.energy, exp.spectrum.intensity, xanesPred.intensity], delimiter=',')
        else:
            plotting.plotToFolder(folderToSaveResult, estimator.expDiff, None, xanesPred, fileName='xanes_approx_'+strj)
            np.savetxt(folderToSaveResult+'/xanes_approx_'+strj+'.csv', [exp.spectrum.energy, estimator.expDiff.spectrum.intensity, xanesPred.intensity], delimiter=',')
    j = ind[0]
    bestGeom_x = geoms[j]
    bestGeom = {paramNames[i]:bestGeom_x[i] for i in range(len(paramNames))}

    if exp.moleculeConstructor is None: return
    M = exp.moleculeConstructor(bestGeom)
    M.export_xyz(folderToSaveResult+'/molecula_best.xyz')
    fdmnes.generateInput(M, **exp.FDMNES_calc, folder=folderToSaveResult+'/fdmnes')

    if calcXanes is not None:
        if calcXanes['local']: fdmnes.runLocal(folderToSaveResult+'/fdmnes')
        else: fdmnes.runCluster(folderToSaveResult+'/fdmnes', calcXanes['memory'], calcXanes['nProcs'])
        xanes = fdmnes.parse_one_folder(folderToSaveResult+'/fdmnes')
        smoothed_xanes, _ = smoothLib.funcFitSmoothHelper(exp.defaultSmoothParams['fdmnes'], xanes, 'fdmnes', exp, estimator.norm)
        with open(folderToSaveResult+'/args_smooth.txt', 'w') as f: json.dump(exp.defaultSmoothParams['fdmnes'], f)
        if estimator.diffFrom is None:
            plotting.plotToFolder(folderToSaveResult, exp, None, smoothed_xanes, fileName='xanes_best_minimum')
            np.savetxt(folderToSaveResult+'/xanes_best_minimum.csv', [exp.spectrum.energy, exp.spectrum.intensity, smoothed_xanes.intensity], delimiter=',')
        else:
            plotting.plotToFolder(folderToSaveResult, exp, None, smoothed_xanes, fileName='xanes_best_minimum', append=[{'data':estimator.diffFrom['spectrumBase'].intensity, 'label':'spectrumBase'}, {'data':estimator.diffFrom['projectBase'].spectrum.intensity, 'label':'projectBase'}])
            np.savetxt(folderToSaveResult+'/xanes_best_minimum.csv', [exp.spectrum.energy, exp.spectrum.intensity, smoothed_xanes.intensity], delimiter=',')
            smoothed_xanes.intensity = (smoothed_xanes.intensity - estimator.diffFrom['spectrumBase'].intensity)*estimator.diffFrom['purity']
            plotting.plotToFolder(folderToSaveResult, estimator.expDiff, None, smoothed_xanes, fileName='xanesDiff_best_minimum')
            np.savetxt(folderToSaveResult+'/xanesDiff_best_minimum.csv', [exp.spectrum.energy, estimator.expDiff.spectrum.intensity, smoothed_xanes.intensity], delimiter=',')

    densityEstimator = ML.NNKCDE(estimator.sample.spectra.values, estimator.sample.params.values)

    def density(x):
        return -densityEstimator.predict(exp.spectrum.intensity.reshape(1, -1), x.reshape(1, -1), k=10, bandwidth=0.2)[0][0]
    for i in range(len(paramNames)):
        optimize.plotMap1d(i, density, bestGeom_x, bounds, paramNames=paramNames, optimizeMethod='scipy', calMapMethod=L2NormMap, folder=folderToSaveResult, postfix='_density')
        optimize.plotMap1d(i, L2norm, bestGeom_x, bounds, fun_args=(estimator,), paramNames=paramNames, optimizeMethod='scipy', calMapMethod=L2NormMap, folder=folderToSaveResult, postfix='_L2norm')
    for i1 in range(len(paramNames)):
        for i2 in range(i1+1,len(paramNames)):
            optimize.plotMap2d([i1,i2], density, bestGeom_x, bounds, paramNames=paramNames, optimizeMethod='scipy', calMapMethod=L2NormMap, folder=folderToSaveResult, postfix='_density')
            optimize.plotMap2d([i1,i2], L2norm, bestGeom_x, bounds, fun_args=(estimator,), paramNames=paramNames, optimizeMethod='scipy', calMapMethod=L2NormMap, folder=folderToSaveResult, postfix='_L2norm')


def inverseCrossValidationPart(geometryParamsTrain, geometryParamsTest, xanesTrain, xanesTest, estimator):
    X_train = geometryParamsTrain.values
    X_test = geometryParamsTest.values
    y_train = xanesTrain.values
    y_test = xanesTest.values

    e_names = xanesTrain.columns
    xanes_energy = np.array([float(e_names[i][2:]) for i in range(e_names.size)])

    estimator.fit(X_train, y_train)
    prediction = estimator.predict(X_test)
    y_test_mean = np.mean(y_test, axis=0)
    N = y_test.shape[0]
    u = np.mean([utils.integral(xanes_energy, (y_test[i] - prediction[i])**2) for i in range(N)])
    v = np.mean([utils.integral(xanes_energy, (y_test[i] - y_test_mean)**2) for i in range(N)])
    score = 1-u/v
    stddevs = [np.sqrt(utils.integral(xanes_energy, (y_test[i] - prediction[i])**2)) for i in range(N)]
    meanL2 = np.mean(stddevs)
    return {'errors':{'relToConstPredError':1-score, 'meanL2':meanL2}, 'predictions':prediction, 'stddevs':stddevs}


def inverseCrossValidation(estimator, sample, CVcount):
    kf = KFold(n_splits=CVcount, shuffle=True, random_state=0)
    X = sample.params
    y = sample.spectra
    res = None
    stddevs = np.zeros(X.shape[0])
    predictions = np.zeros(y.shape)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        part = inverseCrossValidationPart(X_train, X_test, y_train, y_test, estimator)
        if res is None: res = part['errors']
        else:
            for errName in res: res[errName] += part['errors'][errName]
        stddevs[test_index] = part['stddevs']
        predictions[test_index] = part['predictions']
    for errName in res: res[errName] /= CVcount
    return res, stddevs, predictions


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
    assert set(estimator.paramNames) == set(paramNames), 'Param names in geometryParamRanges of experiment does not equal to dataset param names'
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
    fig, ax = plt.subplots()
    for i in range(prediction.shape[0]):
        p = prediction.loc[i]+dM/30*(i+3)
        if i % (intermediatePointsNum+1) == 0: ax.plot(energy, p, linewidth=2, c='r')
        else: ax.plot(energy, p, linewidth=1, c='b')
    ax.plot(energy, exp.spectrum.intensity, c='k', label="Experiment")
    fig.set_size_inches((16,9))
    plt.savefig(folderToSaveResult+'/trajectory.png')
    plt.close(fig)

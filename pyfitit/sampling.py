import time
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
import copy, shutil, os, json, subprocess, threading
import pandas as pd
import matplotlib.pyplot as plt
from . import fdmnes, feff, adf, pyGDM, utils, ihs, w2auto, fdmnesTest, ML, plotting, adaptiveSampling, smoothLib, inverseMethod


knownPrograms = ['fdmnes', 'feff', 'adf', 'w2auto', 'fdmnesTest', 'pyGDM']


# ranges - dictionary with geometry parameters region {'paramName':[min,max], 'paramName':[min,max], ...}
# method - IHS, random, grid, line (in case of grid, sampleCount must be a dict of points count through each dimension)
# spectrCalcParams = {energyRange:..., radius:..., Green:True/False, Adimp=None} - for fdmnes
# spectrCalcParams = {RMAX:..., }
# lineEdges = {'start':{...}, 'end':{...}} - for method='line'
def generateInputFiles(ranges, moleculeConstructor, sampleCount, spectrCalcParams, spectralProgram='fdmnes', method='IHS', folder='sample', lineEdges=None, seed=0):
    if os.path.exists(folder): shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)
    paramNames = [k for k in ranges]
    paramNames.sort()
    N = len(paramNames)
    leftBorder = np.array([ranges[k][0] for k in paramNames])
    rightBorder = np.array([ranges[k][1] for k in paramNames])
    np.random.seed(seed)
    if method == 'IHS':
        points = (ihs.ihs(N, sampleCount, seed=seed) - 0.5) / sampleCount # row - is one point
        for j in range(N):
            points[:,j] = leftBorder[j] + points[:,j]*(rightBorder[j]-leftBorder[j])
    elif method == 'random':
        points = leftBorder + np.random.rand(sampleCount, N)*(rightBorder-leftBorder)
    elif method == 'line':
        if lineEdges is None:
            start = leftBorder; end = rightBorder
        else:
            start = np.array([lineEdges['start'][k] for k in paramNames]).reshape(1,-1)
            end = np.array([lineEdges['end'][k] for k in paramNames]).reshape(1,-1)
        points = start + np.linspace(0,1,sampleCount).reshape(-1,1)*(end-start)
    else:
        assert method == 'grid', 'Unknown method'
        assert (type(sampleCount) is dict) and (len(sampleCount) == N), 'sampleCount must be a dict of point count over dimensions of parameter space'
        coords = [np.linspace(leftBorder[j], rightBorder[j], sampleCount[paramNames[j]]) for j in range(N)]
        repeatedCoords = np.meshgrid(*coords)
        sz = np.prod([sampleCount[p] for p in paramNames])
        points = np.zeros((sz, N))
        for j in range(N): points[:,j] = repeatedCoords[j].reshape(-1)
    geometryParams = copy.deepcopy(ranges)
    for i in range(points.shape[0]):
        for j in range(N): geometryParams[paramNames[j]] = points[i,j]
        molecula = moleculeConstructor(geometryParams)
        if molecula is None: print("Can't construct molecula for parameters "+str(geometryParams)); continue
        folderOne = os.path.join(folder, utils.zfill(i,points.shape[0]))
        assert spectralProgram in knownPrograms, 'Unknown spectral program name: '+spectralProgram
        generateInput = getattr(globals()[spectralProgram], 'generateInput')
        generateInput(molecula, folder=folderOne, **spectrCalcParams)
        geometryParamsToSave = [[paramNames[j], points[i,j]] for j in range(N)]
        with open(os.path.join(folderOne,'geometryParams.txt'), 'w') as f: json.dump(geometryParamsToSave, f)
        print('folder=',folderOne, ' '.join([p+'={:.4g}'.format(geometryParams[p]) for p in geometryParams]))
        if hasattr(molecula, 'export_xyz'):
            molecula.export_xyz(folderOne+'/molecule.xyz')


def runUserDefined(cmd, folder = '.'):
    assert cmd != '', 'Specify command to run'
    proc = subprocess.Popen([cmd], cwd=folder, stdout=subprocess.PIPE, shell=True)
    stdoutdata, stderrdata = proc.communicate()
    if proc.returncode != 0:
        raise Exception('Error while executing "'+cmd+'" command. Stdout='+str(stdoutdata)+'\nStderr='+str(stderrdata))
    return stdoutdata


# runType = 'local', 'run-cluster', 'user defined'
def calcSpectra(spectralProgram='fdmnes', runType='local', runCmd='', nProcs=1, memory=5000, calcSampleInParallel=1, folder='sample', recalculateErrorsAttemptCount=0, continueCalculation=False):
    assert spectralProgram in knownPrograms, 'Unknown spectral program name: '+spectralProgram
    folders = os.listdir(folder)
    folders.sort()
    for i in range(len(folders)): folders[i] = os.path.join(folder, folders[i])

    def calculateXANES(folder):
        if runType == 'run-cluster':
            runCluster = getattr(globals()[spectralProgram], 'runCluster')
            runCluster(folder, memory, nProcs)
        elif runType == 'local':
            runLocal = getattr(globals()[spectralProgram], 'runLocal')
            runLocal(folder)
        elif runType == 'user defined':
            runUserDefined(runCmd, folder)
        else: assert False, 'Wrong runType'
    if calcSampleInParallel > 1: threadPool = ThreadPool(calcSampleInParallel)
    if continueCalculation:
        recalculateErrorsAttemptCount += 1
    else:
        if calcSampleInParallel > 1:
            threadPool.map(calculateXANES, folders)
        else:
            for i in range(len(folders)): calculateXANES(folders[i])
    if spectralProgram == 'pyGDM':
        return
    _, _, _, badFolders = parse_all_folders(folder, spectralProgram, printOutput=not continueCalculation)
    recalculateAttempt = 1
    while (recalculateAttempt <= recalculateErrorsAttemptCount) and (len(badFolders) > 0):
        if calcSampleInParallel > 1:
            threadPool.map(calculateXANES, badFolders)
        else:
            for i in range(len(badFolders)): calculateXANES(badFolders[i])
        _, _, _, badFolders = parse_all_folders(folder, spectralProgram)
        recalculateAttempt += 1


def collectResults(spectralProgram='fdmnes', folder='sample', outputFolder='.', printOutput=True):
    assert spectralProgram in knownPrograms, 'Unknown spectral program name: '+spectralProgram
    os.makedirs(outputFolder, exist_ok=True)
    df_xanes, df_params, goodFolders, badFolders = parse_all_folders(folder, spectralProgram, printOutput=printOutput)
    if df_xanes is None:
        raise Exception('There is no output extinction data in folder ' + folder)
    if isinstance(df_xanes, dict):
        for spType in df_xanes:
            df_xanes[spType].to_csv(os.path.join(outputFolder,f'{spType}_spectra.txt'), sep=' ', index=False)
    else:
        df_xanes.to_csv(os.path.join(outputFolder, 'spectra.txt'), sep=' ', index=False)
    df_params.to_csv(os.path.join(outputFolder, 'params.txt'), sep=' ', index=False)


class InputFilesGenerator:
    """Generates input folder with required content for a certain spectrum-calculating program (e.g. ADF, FDMNES)"""

    def __init__(self, ranges, paramNames, moleculeConstructor, spectrCalcParams, spectralProgram='fdmnes', folder='sample'):
        assert spectralProgram in ['fdmnes', 'feff', 'adf', 'w2auto', 'fdmnesTest', 'pyGDM'], 'Unknown spectral program name: ' + spectralProgram

        self.spectrCalcParams = spectrCalcParams
        self.spectralProgram = spectralProgram
        self.moleculeConstructor = moleculeConstructor
        self.ranges = ranges
        self.paramNames = paramNames
        self.folder = folder
        self.folderCounter = self.getFolderCount()

    def getFolderCount(self):
        os.makedirs(self.folder, exist_ok=True)
        subfolders = [f for f in os.listdir(self.folder) if os.path.isdir(os.path.join(self.folder, f))]
        return len(subfolders)

    def getFolderForPoint(self, x):
        # this would work correctly only if every passed x is unique
        # folder = '.'+os.path.sep+str(self.folderCounter)
        # if os.path.exists(self.folder):
        #     shutil.rmtree(self.folder)

        folder = self.tryGetFolderForPoint(x)
        if folder is not None:
            return folder

        os.makedirs(self.folder, exist_ok=True)
        geometryParams = {}
        N = len(self.paramNames)
        for j, name in enumerate(self.paramNames):
            geometryParams[name] = x[j]
        molecule = self.moleculeConstructor(geometryParams)
        if molecule is None:
            print("Can't construct molecule for parameters " + str(geometryParams))
            return None
        folderOne = os.path.join(self.folder, utils.zfill(self.folderCounter, 200000))
        generateInput = getattr(globals()[self.spectralProgram], 'generateInput')
        generateInput(molecule, folder=folderOne, **self.spectrCalcParams)
        geometryParamsToSave = [[self.paramNames[j], x[j]] for j in range(N)]
        with open(os.path.join(folderOne, 'geometryParams.txt'), 'w') as f:
            json.dump(geometryParamsToSave, f)
        print('folder=', folderOne, ' '.join([p + '={:.4g}'.format(geometryParams[p]) for p in geometryParams]))
        if hasattr(molecule, 'export_xyz'):
            molecule.export_xyz(folderOne + '/molecule.xyz')

        self.folderCounter += 1
        return folderOne

    def tryGetFolderForPoint(self, x):
        df_xanes, df_params, goodFolders, badFolders = parse_all_folders(self.folder, self.spectralProgram, printOutput=False)
        for folder in goodFolders + badFolders:
            if np.array_equal(x, loadParams(folder, self.spectralProgram)):
                return folder

        return None


class SpectrumCalculator(adaptiveSampling.CalculationProgram):

    def __init__(self, spectralProgram, inputGenerator, outputFolder, recalculateErrorsAttemptCount, smoothConfig):
        """

        :param spectralProgram:
        :param inputGenerator:
        :param outputFolder:
        :param recalculateErrorsAttemptCount:
        :param smoothConfig: dict with keys {'smoothParams', 'smoothType', 'expSpectrum', 'fitNormInterval', 'norm':None} or None (if we do not need to smooth)
        """
        self.recalculateErrorsAttemptCount = recalculateErrorsAttemptCount
        self.outputFolder = outputFolder
        self.input = inputGenerator
        self.spectralProgram = spectralProgram
        self.runType = None
        self.lock = threading.Lock()
        assert set(smoothConfig.keys()) >= {'smoothParams', 'smoothType', 'expSpectrum', 'fitNormInterval'}
        if 'norm' not in smoothConfig:
            if 'norm' in smoothConfig['smoothParams']:
                smoothConfig['norm'] = smoothConfig['smoothParams']['norm']
            else:
                smoothConfig['norm'] = None
        self.smoothConfig = smoothConfig

    def calculate(self, x):
        with self.lock:
            folder = self.input.getFolderForPoint(x)

        return self.calculateFolder(folder)

    def calculateFolder(self, folder):
        attemptsDone = 0
        while True:
            # checking if the folder already has good data
            isBad = self.checkIfBadFolder(folder)
            if not isBad or attemptsDone > self.recalculateErrorsAttemptCount:
                break
            if attemptsDone > 0:
                print(f'Folder {folder} is bad, recalculating')
            self.calculateSpectrum(folder)
            attemptsDone += 1
        if not isBad:
            with self.lock:
                print(f'Returning data from {folder} calculation iterations done: {attemptsDone}')
                ys, additionalData = self.parseAndCollect(folder)
            return ys, additionalData
        else:
            return None, None

    def configAll(self, runType, runCmd, nProcs, memory):
        self.runType = runType
        self.runCmd = runCmd
        self.nProcs = nProcs
        self.memory = memory

    def configUserDefined(self, runCmd):
        self.runType = 'user defined'
        self.runCmd = runCmd

    def configCluster(self, nProcs=1, memory=5000):
        self.runType = 'run-cluster'
        self.nProcs = nProcs
        self.memory = memory

    def configLocal(self):
        self.runType = 'local'

    def calculateSpectrum(self, folder):
        if self.runType == 'run-cluster':
            runCluster = getattr(globals()[self.spectralProgram], 'runCluster')
            runCluster(folder, self.memory, self.nProcs)
        elif self.runType == 'local':
            runLocal = getattr(globals()[self.spectralProgram], 'runLocal')
            runLocal(folder)
        elif self.runType == 'user defined':
            self.runUserDefined(self.runCmd, folder)
        else:
            assert False, 'Wrong runType'

    def runUserDefined(self, cmd, folder):
        import subprocess

        assert cmd != '', 'Specify command to run'
        proc = subprocess.Popen([cmd, str(folder)], cwd='.', stdout=subprocess.PIPE)
        proc.wait()
        if proc.returncode != 0:
            raise Exception('Error while executing "' + cmd + '" command')
        return proc.stdout.read()

    def checkIfBadFolder(self, folder):
        _, _, _, badFolders = parse_all_folders(self.input.folder, self.spectralProgram, printOutput=False)
        return folder in badFolders

    def parseAndCollect(self, folder):
        res = loadExistingSpectrum(self.spectralProgram, self.smoothConfig, folder)
        collectResults(self.spectralProgram, self.input.folder, self.outputFolder, printOutput=False)
        return res


def loadExistingSpectrum(spectralProgram, smoothConfig, folder):
    parse_method = getattr(globals()[spectralProgram], 'parse_one_folder')
    res = parse_method(folder)
    if smoothConfig is not None:
        exp = smoothConfig['expSpectrum']
        resSmoothed, _ = smoothLib.smoothInterpNorm(smoothConfig['smoothParams'], res,
                                                    smoothConfig['smoothType'], exp,
                                                    smoothConfig['fitNormInterval'],
                                                    smoothConfig['norm'])
    return resSmoothed.intensity, {'spectrum': res, 'folder': folder}


def loadParams(folder, spectralProgram):
    getParams = getattr(globals()[spectralProgram], 'getParams')
    _, x = getParams(os.path.join(folder, 'geometryParams.txt'))
    return x


def loadExistingXPoints(spectralProgram, folder):
    subfolders = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
    params = []
    for subfolder in subfolders:
        x = loadParams(subfolder, spectralProgram)
        params.append(x)

    return np.array(params)


def parse_all_folders(parentFolder, spectral_program, printOutput=True):
    """

    :param parentFolder: folder containing results
    :param spectral_program: one of fdmnes, fdmnesTest, adf, feff, pyGDM
    :param printOutput: whether parsing debug info should be printed
    :return: spectra dataframe, params dataframe, goodFolders list, badFolder list
    """

    def read_folders():
        """

        :return: dictionary { folder, parse_one_folder("parentFolder/folder") }
        """
        import traceback
        subfolders = [f for f in os.listdir(parentFolder) if os.path.isdir(os.path.join(parentFolder, f))]
        subfolders.sort()
        allXanes = {}
        for i in range(len(subfolders)):
            d = subfolders[i]
            try:
                res = parse_one_folder(os.path.join(parentFolder, d))
                allXanes[d] = res
                if res is None:
                    output.append('Can\'t read output in folder ' + d)
            except:
                output.append(traceback.format_exc())
                allXanes[d] = None

        return allXanes

    def separate_folders(allXanes):
        """

        :param allXanes: dictionary of parsed folders
        :return: array of good foler names, array of bad foler names
        """
        badFolders = []
        if spectral_program in ['fdmnes', 'fdmnesTest', 'adf']:
            energyCount = np.array([x.intensity.shape[0] for x in allXanes.values() if x is not None])
            maxEnergyCount = np.max(energyCount, initial=0)

        for d in allXanes:
            if allXanes[d] is None:
                badFolders.append(d)
                continue

            if spectral_program in ['fdmnes', 'fdmnesTest', 'adf'] and allXanes[d].intensity.shape[0] != maxEnergyCount:
                output.append(f'Error: in folder {d} there are less energies {allXanes[d].intensity.shape[0]}')
                badFolders.append(d)

        goodFolders = list(set(allXanes.keys()) - set(badFolders))
        goodFolders.sort()
        return goodFolders, badFolders

    def get_full_path_folders(folders):
        return [os.path.join(parentFolder, x) for x in folders]

    def create_dataframes(allXanes, goodFolders):
        """

        :param allXanes: dictionary of parsed folders
        :param goodFolders: list of good folders
        :return:
        """
        if len(goodFolders) == 0:
            output.append('None good folders')
            return None, None
        # get energies array
        allEnergies = np.array([allXanes[folder].energy for folder in goodFolders])
        n = len(goodFolders)
        if n == 1:
            allEnergies.reshape(1, -1)
        energies = allEnergies
        # make specific changes to energies
        if spectral_program in ['fdmnes', 'fdmnesTest'] and fdmnes.useEpsiiShift:
            energies = np.median(allEnergies, axis=0)
            energies = np.sort(energies)
            maxShift = np.max(allEnergies[:, 0]) - np.min(allEnergies[:, 0])
            output.append('Max energy shift between spectra: {:.2}'.format(maxShift))
        elif spectral_program == 'feff':
            if abs(float(allXanes[goodFolders[0]].values[0, 0])) < 0.00001:
                energies = allXanes[goodFolders[0]].iloc[1:, 0].ravel()
            else:
                energies = allXanes[goodFolders[0]].iloc[:, 0].ravel()
        elif spectral_program == 'adf':
            energies = allXanes[goodFolders[0]].loc[:, 'E'].ravel()
        elif spectral_program == 'pyGDM':
            energies = np.array(allEnergies)

        paramNames, _ = getParams(os.path.join(parentFolder, goodFolders[0], 'geometryParams.txt'))
        df_xanes = np.zeros([n, energies.size])
        df_params = np.zeros([n, len(paramNames)])
        for i in range(n):
            d = goodFolders[i]
            _, params = getParams(os.path.join(parentFolder, d, 'geometryParams.txt'))
            df_params[i, :] = np.array(params)
            # make specific spectrum changes
            if spectral_program in ['fdmnes', 'fdmnesTest'] and fdmnes.useEpsiiShift:
                df_xanes[i, :] = np.interp(energies, allXanes[d].energy, allXanes[d].intensity)
            elif spectral_program == 'feff':
                if abs(float(allXanes[d].values[0, 0])) < 0.00001:
                    df_xanes[i, :] = allXanes[d].iloc[1:, 1].ravel()
                else:
                    df_xanes[i, :] = allXanes[d].iloc[:, 1].ravel()
            elif spectral_program == 'adf':
                df_xanes[i, :] = allXanes[d].loc[:, 'ftot'].ravel()
            else:
                df_xanes[i, :] = allXanes[d].intensity
        df_xanes = pd.DataFrame(data=df_xanes, columns=['e_' + str(e) for e in energies])
        df_params = pd.DataFrame(data=df_params, columns=paramNames)
        return df_xanes, df_params

    parse_one_folder = getattr(globals()[spectral_program], 'parse_one_folder')
    getParams = getattr(globals()[spectral_program], 'getParams')
    output = []
    allXanes = read_folders()
    goodFolders, badFolders = separate_folders(allXanes)
    if spectral_program == 'pyGDM':
        df_abs, df_params = create_dataframes([x['abs'] for x in allXanes], goodFolders)
        df_ext, _ = create_dataframes([x['ext'] for x in allXanes], goodFolders)
        df_xanes = {'abs': df_abs, 'ext': df_ext}
    else:
        df_xanes, df_params = create_dataframes(allXanes, goodFolders)
    badFolders = get_full_path_folders(badFolders)
    goodFolders = get_full_path_folders(goodFolders)
    if printOutput:
        print(*output)
    return df_xanes, df_params, goodFolders, badFolders


def checkSampleIsGoodByCount(minPoints):
    return lambda dataset: len(dataset[0]) >= minPoints


class BadSpectrumInSampleError(Exception):
    pass


def convertToSample(dataset):
    xs, ys, additionalData = dataset
    spectra = []
    good = [i for i in range(len(additionalData)) if additionalData[i] is not None]
    if len(good) == 0: return None
    energyCount = np.array([additionalData[i]['spectrum'].intensity.shape[0] for i in good])
    maxEnergyCount = np.max(energyCount)
    if not np.all(energyCount == maxEnergyCount):
        raise BadSpectrumInSampleError('Bad spectra in sample. energyCount = '+str(energyCount))
    allEnergies = np.array([additionalData[i]['spectrum'].energy for i in good])
    n = len(good)
    if n == 1: allEnergies.reshape(1, -1)
    energies = np.median(allEnergies, axis=0)
    for i in good:
        spectrum = additionalData[i]['spectrum']
        interpolatedSpectrum = np.interp(energies, spectrum.energy, spectrum.intensity)
        spectra.append(interpolatedSpectrum)
    spectra = np.array(spectra)
    paramFile = additionalData[good[0]]['folder']+os.sep+'geometryParams.txt'
    if os.path.exists(paramFile):
        with open(paramFile, 'r') as f: params = json.load(f)
        paramNames = [p[0] for p in params]
        paramData = pd.DataFrame(data=xs[good, :], columns=paramNames)
    else:
        paramData = pd.DataFrame(data=xs[good, :])
    sample = ML.Sample(paramData, spectra, energies)
    foldersInds = [int(os.path.split(additionalData[good[i]]['folder'])[-1]) for i in range(len(good))]
    return sample, foldersInds


def checkSampleIsGoodByCVError(maxError, smoothConfig, estimator=None, minCountToCheckError=10, cvCount=10, debug=False, debugOutputFolder='debug', testSample=None):
    """
    Returns function to pass in sampleAdaptively as checkSampleIsGoodFunc argument
    :param maxError:
    :param smoothConfig: dict with keys {'smoothParams', 'smoothType', 'expSpectrum', 'fitNormInterval', 'norm':None, 'fitGeometryInterval'}, fitGeometryInterval - to limit fitting spectra
    :param minCountToCheckError:
    :param debug: plot error graphs
    :param debugOutputFolder:
    """
    assert set(smoothConfig.keys()) >= {'smoothParams', 'smoothType', 'expSpectrum', 'fitNormInterval'}
    assert set(smoothConfig.keys()) <= {'smoothParams', 'smoothType', 'expSpectrum', 'fitNormInterval', 'norm', 'fitGeometryInterval'}
    if 'norm' not in smoothConfig:
        if 'norm' in smoothConfig['smoothParams']:
            smoothConfig['norm'] = smoothConfig['smoothParams']['norm']
        else:
            smoothConfig['norm'] = None

    if estimator is None:
        estimator = inverseMethod.getMethod('RBF')

    if debug and os.path.exists(debugOutputFolder): shutil.rmtree(debugOutputFolder)

    def checkSampleIsGood(dataset):
        _, _, additionalData = dataset
        exp = copy.deepcopy(smoothConfig['expSpectrum'])
        if 'fitGeometryInterval' in smoothConfig:
            exp = exp.limit(smoothConfig['fitGeometryInterval'])
        sample, folderInds = convertToSample(dataset)
        lastSpInd = folderInds[-1]
        # print(len(dataset[0]), folderInds)
        if sample is None: return False
        spectra = smoothLib.smoothDataFrame(smoothConfig['smoothParams'], sample.spectra, smoothConfig['smoothType'], exp, smoothConfig['fitNormInterval'], smoothConfig['norm'])
        sample.setSpectra(spectra)
        n = sample.getLength()

        def plotSpectra(plotPrediction=True):
            for i,fi in enumerate(folderInds):
                fileName = f"{debugOutputFolder}{os.sep}spectrum_{fi:05d}.png"
                if os.path.exists(fileName): continue
                title = f'Spectrum {fi} (sample N {i})'
                if plotPrediction:
                    sample_loo = sample.copy()
                    sample_loo.delRow(i)
                    estimator.fit(sample_loo.params, sample_loo.spectra)
                    predicted = estimator.predict(sample.params.to_numpy()[i].reshape(1,-1))
                    title += f'. Predict by sample of {sample_loo.getLength()} spectra'
                    plotting.plotToFile(sample.energy, sample.spectra.to_numpy()[i], 'theory', sample.energy, predicted.reshape(-1), 'predicted', exp.energy, exp.intensity, 'exp', fileName=fileName, title=title, save_csv=False)
                else:
                    plotting.plotToFile(sample.energy, sample.spectra.to_numpy()[i], 'theory', exp.energy, exp.intensity, 'exp', fileName=fileName, title=title, save_csv=False)

        if n < minCountToCheckError:
            if debug: plotSpectra(plotPrediction=False)
            return False

        res = inverseMethod.inverseCrossValidation(estimator, sample, cvCount)
        relToConstPredErrorCV = res[0]['relToConstPredError']
        print(f'relToConstPredError: {relToConstPredErrorCV}')
        if debug: plotSpectra(plotPrediction=True)

        if testSample is not None:
            testSampleCopy = copy.deepcopy(testSample)
            spectraTest = smoothLib.smoothDataFrame(smoothConfig['smoothParams'], testSampleCopy.spectra,
                                                smoothConfig['smoothType'], exp, smoothConfig['fitNormInterval'],
                                                smoothConfig['norm'])
            testSampleCopy.setSpectra(spectraTest)
            estimator.fit(sample.params, sample.spectra.to_numpy())
            yPred = estimator.predict(testSampleCopy.params)
            relToConstPredErrorTest = inverseMethod.relativeToConstantPredictionError(yTrue=testSampleCopy.spectra.to_numpy(), yPred=yPred, energy=testSampleCopy.energy)
            print(f'relToConstPredErrorTest: {relToConstPredErrorTest}')
        if debug and len(sample.paramNames) >= 2:
            std = np.std(sample.spectra.to_numpy(), axis=0)
            j = np.argmax(std)
            # print('max spectrum std energy: ', sample.energy[j], 'all interval: ', sample.energy[0], sample.energy[-1])
            y = sample.spectra.to_numpy()[:,j]
            geometryParamRanges = {p:[np.min(sample.params[p]), np.max(sample.params[p])] for p in sample.paramNames}
            maxStdDev, meanStdDev = inverseMethod.calcParamStdDevHelper(geometryParamRanges, sample=sample)
            std_p = [maxStdDev[p] for p in sample.paramNames]
            ind = np.argsort(std_p)
            pn1 = sample.paramNames[ind[-1]]
            pn2 = sample.paramNames[ind[-2]]
            # sort in alphabetical order
            if pn1 > pn2: pn1, pn2 = pn2, pn1
            params = sample.params.loc[:,[pn1,pn2]].to_numpy()
            leftBorder = np.min(params, axis=0).reshape(-1)
            rightBorder = np.max(params, axis=0).reshape(-1)
            x1 = np.linspace(leftBorder[0], rightBorder[0], 50)
            x2 = np.linspace(leftBorder[1], rightBorder[1], 50)
            x1g, x2g = np.meshgrid(x1, x2)
            x1v = x1g.flatten()
            x2v = x2g.flatten()
            xsTest = np.dstack((x1v, x2v))[0]
            # rbf fails in case of equal param rows
            _, ind = np.unique([f"{params[i,0]}_{params[i,1]}" for i in range(len(params))], return_index=True)

            twoParamsEstimator = copy.deepcopy(estimator)
            twoParamsEstimator.fit(params[ind,:], y[ind])
            ysTest = twoParamsEstimator.predict(xsTest)
            y2d = ysTest.reshape(x1g.shape)

            fig, ax = plotting.createfig()
            cs1 = ax.contourf(x1, x2, y2d, cmap='jet')
            plt.colorbar(cs1)
            ax.set_xlabel(pn1)
            ax.set_ylabel(pn2)
            ax.scatter(params[:,0], params[:,1], s=30)
            for i in range(sample.spectra.to_numpy().shape[0]):
                ax.annotate(folderInds[i], (params[i,0], params[i,1]), fontsize=7)
            plotting.savefig(f"{debugOutputFolder}{os.sep}contour_{lastSpInd:05d}.png", fig)
            plotting.closefig(fig)

            relToConstPredError_s = "%.2g" % relToConstPredErrorCV

            plot_error = np.linalg.norm(res[2] - sample.spectra.to_numpy(), ord=np.inf, axis=1)
            me = np.max(plot_error)
            mes = '%.2g' % me
            fileName = utils.addPostfixIfExists(f"{debugOutputFolder}{os.sep}error_{lastSpInd:05d}.png")
            plotting.scatter(params[:,0], params[:,1], color=plot_error / me, colorMap='gist_rainbow_r', markersize=51, marker='s', marker_text=folderInds, title=f'Sample size={n}, max error = {mes}. relToConstPredError = {relToConstPredError_s}', xlabel=pn1, ylabel=pn2, fileName=fileName)

            if testSample is not None:
                fileNameTest = utils.addPostfixIfExists(f"{debugOutputFolder}{os.sep}errorTest_{lastSpInd:05d}.png")
                relToConstPredError_s = "%.2g" % relToConstPredErrorTest
                plot_error = np.linalg.norm(yPred - testSampleCopy.spectra.to_numpy(), ord=np.inf, axis=1)
                me = np.max(plot_error)
                mes = '%.2g' % me
                plotting.scatter(testSampleCopy.params[pn1], testSampleCopy.params[pn2], color=plot_error / me, markersize=51, marker='s', title=f'Sample size= {n}, max error = {mes}. relToConstPredError = {relToConstPredError_s}', xlabel=pn1, ylabel=pn2, fileName=fileNameTest)

        return (relToConstPredErrorCV if testSample is None else relToConstPredErrorTest) <= maxError
    return checkSampleIsGood


def ensureSettingsAreConsistent(settingsFileName, seed, paramRanges):
    with open(settingsFileName) as json_file:
        data = json.load(json_file)
        assert 'seed' in data, f'seed is not found in settings. Remove {settingsFileName} in working folder'
        seed = data['seed'] if seed is None else seed
        assert data['seed'] == seed, \
            f"Seed in working folder is inconsistent with the given one ({seed} vs {data['seed']})." \
            f"Set seed=None if you wish to continue calculation"

        assert 'paramRanges' in data, f'paramRanges is not found in settings. Remove {settingsFileName} in working folder'
        paramNames = [k for k in paramRanges]
        loadedParamNames = [k for k in data['paramRanges']]
        assert set(paramNames) == set(loadedParamNames), 'Inconsistent parameters. Remove working folder.'
        for name in paramNames:
            assert data['paramRanges'][name][0] <= paramRanges[name][0] and \
                   data['paramRanges'][name][1] >= paramRanges[name][1], \
                "Given parameter ranges are more strict than the loaded ones. Restore parameter ranges or clear working folder"

        return seed


def writeSettings(settingsFileName, seed, paramRanges):
    # ensuring seed is not empty, so we can save it
    seed = int(time.time()) if seed is None else seed

    with open(settingsFileName, 'w') as outfile:
        json.dump({
            'seed': seed,
            'paramRanges': paramRanges,
        }, outfile)

    return seed


def sampleAdaptively(paramRanges, moleculeConstructor, checkSampleIsGoodFunc, spectrCalcParams, spectralProgram='fdmnes', smoothConfig=None, workingFolder='sample', seed=None, outputFolder='sample_result', runConfiguration=None, adaptiveSamplerParams=None, settingsFileName='settings.json'):
    """
    Calculate sample adaptively
    :param paramRanges: dictionary with geometry parameters region {'paramName':[min,max], 'paramName':[min,max], ...}
    :param moleculeConstructor:
    :param checkSampleIsGoodFunc: function(dataset) for example lambda dataset: len(dataset[0]) >= 200 or use cross-validation error. dataset = (xs, ys, additionalData). You can use predefined functions: checkSampleIsGoodByCount and checkSampleIsGoodByCVError
    :param spectrCalcParams: see function generateInput in module correspondent to your spectralProgram
    :param spectralProgram: string, one of knownPrograms (see at the top of this file)
    :param smoothConfig: dict with keys {'smoothParams', 'smoothType', 'expSpectrum', 'fitNormInterval', 'norm':None}. None - means do not smooth
    :param workingFolder:
    :param seed: if None - get current timestamp
    :param outputFolder:
    :param continueCalculation: if true do not delete existing folders and continue sample calculation
    :param runConfiguration: dict, default = {'runType':'local', 'runCmd':'', 'nProcs':1, 'memory':5000, 'calcSampleInParallel':1, 'recalculateErrorsAttemptCount':0}
    :param adaptiveSamplerParams: dict, default = {'initialPoints':None, 'exploreNum':5, 'exploitNum':5, 'yWeightInNorm':1.0}
    :param settingsFileName: name of the persistently saved settings, like seed, paramRanges, etc.
    :return:
    """
    if adaptiveSamplerParams is None: adaptiveSamplerParams = {}
    if runConfiguration is None: runConfiguration = {}
    defaultRunConfig = {'runType':'local', 'runCmd':'', 'nProcs':1, 'memory':5000, 'calcSampleInParallel':1, 'recalculateErrorsAttemptCount':0}
    for param in defaultRunConfig:
        if param not in runConfiguration: runConfiguration[param] = defaultRunConfig[param]

    paramNames = [k for k in paramRanges]
    paramNames.sort()
    rangeValues = np.array([paramRanges[k] for k in paramNames])

    os.makedirs(workingFolder, exist_ok=True)
    initialPoints = None
    existingPoints = loadExistingXPoints(spectralProgram, workingFolder)
    if existingPoints.shape[0] > 0:
        initialPoints = existingPoints

    if settingsFileName in os.listdir(workingFolder):
        seed = ensureSettingsAreConsistent(workingFolder + os.sep + settingsFileName, seed, paramRanges)
    else:
        seed = writeSettings(workingFolder + os.sep + settingsFileName, seed, paramRanges)

    while True:
        try:
            sampler = adaptiveSampling.ErrorPredictingSampler(rangeValues, checkSampleIsGoodFunc=checkSampleIsGoodFunc,      seed=seed, initialPoints=initialPoints, **adaptiveSamplerParams)
            folderGen = InputFilesGenerator(rangeValues, paramNames, moleculeConstructor, spectrCalcParams, spectralProgram, workingFolder)
            func = SpectrumCalculator(spectralProgram, folderGen, outputFolder, runConfiguration['recalculateErrorsAttemptCount'], smoothConfig)
            func.configAll(runConfiguration['runType'], runConfiguration['runCmd'], runConfiguration['nProcs'], runConfiguration['memory'])
            orchestrator = adaptiveSampling.CalculationOrchestrator(func, runConfiguration['calcSampleInParallel'])
            generator = adaptiveSampling.DatasetGenerator(sampler, orchestrator)
            generator.generate()
            break
        except BadSpectrumInSampleError:
            print('There were bad spectra in sample during beginning of sampling. Restarting sampling process')


def convertSampleTo(sample, sampleType, sampleSize, featureRanges, seed=None):
    """
    Sample type converter. Mainly used to convert adaptive samples to uniform or IHS type
    :param sampleType: str 'IHS', 'uniform'
    :param featureRanges: dict what features to use for spectra prediction and their ranges
    :return: new sample with params only specified in featureRanges
    """
    paramNames = [k for k in featureRanges]
    paramNames.sort()
    N = len(paramNames)
    leftBorder = np.array([featureRanges[k][0] for k in paramNames])
    rightBorder = np.array([featureRanges[k][1] for k in paramNames])
    if seed is None: seed = int(time.time())
    rng = np.random.default_rng(seed)
    if sampleType == 'IHS':
        points = (ihs.ihs(N, sampleSize, seed=seed) - 0.5) / sampleSize  # row - is one point
        for j in range(N):
            points[:, j] = leftBorder[j] + points[:, j] * (rightBorder[j] - leftBorder[j])
    elif sampleType == 'random':
        points = leftBorder + rng.rand(sampleSize, N) * (rightBorder - leftBorder)
    estimator = ML.RBF()
    estimator.fit(sample.params.loc[:,paramNames], sample.spectra)
    newSpectra = estimator.predict(points)
    newParams = pd.DataFrame(data=points, columns=paramNames)
    return ML.Sample(newParams, newSpectra, energy=sample.energy)

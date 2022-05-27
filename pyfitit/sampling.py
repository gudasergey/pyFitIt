import time
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
import copy, shutil, os, json, subprocess, threading, itertools, shlex, traceback
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.ensemble

from . import fdmnes, feff, adf, pyGDM, utils, ihs, w2auto, fdmnesTest, ML, plotting, adaptiveSampling, smoothLib, inverseMethod, vasp_rdf_energy


knownPrograms = ['fdmnes', 'feff', 'adf', 'w2auto', 'fdmnesTest', 'pyGDM', 'vasp_rdf_energy']


def isKnown(name):
    return name in knownPrograms or name[:4] == 'feff'


def getInputGenerator(name):
    assert isKnown(name)
    if name[:4] == 'feff':
        version = name[4:]
        generateInput = lambda molecule,**p:  feff.generateInput(molecule, feffVersion=version, **p)
    else:
        generateInput = getattr(globals()[name], 'generateInput')
    return generateInput


def getParser(name):
    if name[:4] == 'feff': name = 'feff'
    parseOneFolder = getattr(globals()[name], 'parseOneFolder')
    return parseOneFolder


def getRunner(name, runType):
    if name[:4] == 'feff':
        version = name[4:]
        name = 'feff'
    if runType == 'run-cluster':
        run = getattr(globals()[name], 'runCluster')
    else:
        assert runType == 'local'
        run = getattr(globals()[name], 'runLocal')
        if name == 'feff':
            run0 = run
            run = lambda folder: run0(folder, feffVersion=version)
    return run


# ranges - dictionary with geometry parameters region {'paramName':[min,max], 'paramName':[min,max], ...}
# method - IHS, random, grid, line (in case of grid, sampleCount must be a dict of points count through each dimension)
# spectrCalcParams = {energyRange:..., radius:..., Green:True/False, Adimp=None} - for fdmnes
# spectrCalcParams = {RMAX:..., }
# lineEdges = {'start':{...}, 'end':{...}} - for method='line'
def generateInputFiles(ranges, moleculeConstructor, sampleCount, spectrCalcParams, spectralProgram='fdmnes', method='IHS', folder='sample', lineEdges=None, seed=0, debug=False):
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
        if molecula is None: print("Can't construct molecule for parameters "+str(geometryParams)); continue
        folderOne = os.path.join(folder, utils.zfill(i,points.shape[0]))
        generateInput = getInputGenerator(spectralProgram)
        generateInput(molecula, folder=folderOne, **spectrCalcParams)
        geometryParamsToSave = [[paramNames[j], points[i,j]] for j in range(N)]
        with open(os.path.join(folderOne,'params.txt'), 'w') as f: json.dump(geometryParamsToSave, f)
        if debug: print('folder=',folderOne, ' '.join([p+'={:.4g}'.format(geometryParams[p]) for p in geometryParams]))
        if hasattr(molecula, 'export_xyz'):
            molecula.export_xyz(folderOne+'/molecule.xyz')


def runUserDefined(cmd, folder='.'):
    assert cmd != '', 'Specify command to run'
    output, returncode = utils.runCommand(cmd, folder, outputTxtFile='output.txt')
    if returncode != 0:
        raise Exception('Error while executing "'+cmd+'" command. Output:\n'+output)
    return output


# runType = 'local', 'run-cluster', 'user defined'
def calcSpectra(spectralProgram='fdmnes', runType='local', runCmd='', nProcs=1, memory=5000, calcSampleInParallel=1, folder='sample', recalculateErrorsAttemptCount=0, continueCalculation=False):
    assert isKnown(spectralProgram), 'Unknown spectral program name: '+spectralProgram
    folders = os.listdir(folder)
    folders.sort()
    for i in range(len(folders)): folders[i] = os.path.join(folder, folders[i])

    def calculateXANES(folder):
        if runType == 'run-cluster':
            runCluster = getRunner(spectralProgram, runType)
            runCluster(folder, memory, nProcs)
        elif runType == 'local':
            runLocal = getRunner(spectralProgram, runType)
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
    _, _, _, badFolders = parseAllFolders(folder, spectralProgram, printOutput=not continueCalculation)
    recalculateAttempt = 1
    while (recalculateAttempt <= recalculateErrorsAttemptCount) and (len(badFolders) > 0):
        if calcSampleInParallel > 1:
            threadPool.map(calculateXANES, badFolders)
        else:
            for i in range(len(badFolders)): calculateXANES(badFolders[i])
        _, _, _, badFolders = parseAllFolders(folder, spectralProgram)
        recalculateAttempt += 1


def collectResults(spectralProgram='fdmnes', folder='sample', outputFolder='.', printOutput=True):
    if isinstance(spectralProgram, str):
        assert isKnown(spectralProgram), 'Unknown spectral program name: '+spectralProgram
    os.makedirs(outputFolder, exist_ok=True)
    df_spectra, df_params, goodFolders, badFolders = parseAllFolders(folder, spectralProgram, printOutput=printOutput)
    if df_spectra is None:
        # no good folders
        return df_spectra, df_params, goodFolders, badFolders
    if isinstance(df_spectra, dict):
        for spType in df_spectra:
            df_spectra[spType].to_csv(os.path.join(outputFolder,f'{spType}_spectra.txt'), sep=' ', index=False)
    else:
        df_spectra.to_csv(os.path.join(outputFolder, 'spectra.txt'), sep=' ', index=False)
    df_params.to_csv(os.path.join(outputFolder, 'params.txt'), sep=' ', index=False)
    return df_spectra, df_params, goodFolders, badFolders


class InputFilesGenerator:
    """Generates input folder with required content for a certain spectrum-calculating program (e.g. ADF, FDMNES)"""

    def __init__(self, ranges, paramNames, moleculeConstructor, spectrCalcParams, spectralProgram='fdmnes', folder='sample', debug=False):
        if isinstance(spectralProgram, str):
            assert spectralProgram in ['fdmnes', 'feff', 'feff6', 'feff8.5', 'adf', 'w2auto', 'fdmnesTest', 'pyGDM', 'vasp_rdf_energy'], 'Unknown spectral program name: ' + spectralProgram
        else:
            assert isinstance(spectralProgram, dict)
            assert set(spectralProgram.keys()) == {'generateInput', 'parseOneFolder', 'createDataframes'}

        self.spectrCalcParams = spectrCalcParams
        self.spectralProgram = spectralProgram
        self.moleculeConstructor = moleculeConstructor
        self.ranges = ranges
        self.paramNames = paramNames
        self.folder = folder
        self.folderCounter = self.getFolderCount()
        self.debug = debug

    def getFolderCount(self):
        os.makedirs(self.folder, exist_ok=True)
        subfolders = sorted([f for f in os.listdir(self.folder) if os.path.isdir(os.path.join(self.folder, f))])
        if len(subfolders) > 0: return int(subfolders[-1])+1
        return 0

    def getFolderForPoint(self, x):
        if self.debug: print('Trying get folder for point', x)
        folder = self.tryGetFolderForPoint(x)
        if folder is not None:
            if self.debug: print('Found existed folder', folder)
            return folder
        os.makedirs(self.folder, exist_ok=True)
        geometryParams = {}
        N = len(self.paramNames)
        for j, name in enumerate(self.paramNames):
            geometryParams[name] = x[j]
        folderOne = os.path.join(self.folder, utils.zfill(self.folderCounter, 200000))
        assert not os.path.exists(folderOne)
        if self.debug:
            print('The folder doesn\'t exist. Creating new', folderOne)
        if isinstance(self.spectralProgram, str):
            molecule = self.moleculeConstructor(geometryParams)
            if molecule is None:
                print("Can't construct molecule for parameters " + str(geometryParams))
                return None
            if hasattr(molecule, 'export_xyz'):
                molecule.export_xyz(folderOne + '/molecule.xyz')
            generateInput = getInputGenerator(self.spectralProgram)
            generateInput(molecule, folder=folderOne, **self.spectrCalcParams)
        else:
            generateInput = self.spectralProgram['generateInput']
            generateInput(geometryParams, folderOne)
        geometryParamsToSave = [[self.paramNames[j], x[j]] for j in range(N)]
        with open(os.path.join(folderOne, 'params.txt'), 'w') as f:
            json.dump(geometryParamsToSave, f)
        if self.debug: print('folder=', folderOne, ' '.join([p + '={:.4g}'.format(geometryParams[p]) for p in geometryParams]))
        self.folderCounter += 1
        return folderOne

    def tryGetFolderForPoint(self, x):
        #too slow:
        # df_xanes, df_params, goodFolders, badFolders = parse_all_folders(self.folder, self.spectralProgram, printOutput=False)
        for f in os.listdir(self.folder):
            folder = self.folder+os.sep+f
            if not os.path.isdir(folder): continue
            xf = np.array(loadParams(folder, self.spectralProgram))
            nrm = np.abs(x)+np.abs(xf)
            nrm[nrm==0] = 1
            if np.max(np.abs(x-xf)/nrm) < 1e-10:
                if utils.jobIsRunning(folder):
                    raise Exception(f'Running calculation detected in the folder {folder}. Stop it first')
                if os.path.exists(folder + os.sep + 'isRunning'): os.unlink(folder + os.sep + 'isRunning')
                return folder
        return None


class SpectrumCalculator(adaptiveSampling.CalculationProgram):

    def __init__(self, spectralProgram, inputGenerator, outputFolder, recalculateErrorsAttemptCount, samplePreprocessor, debug, lock):
        """

        :param spectralProgram:
        :param inputGenerator:
        :param outputFolder:
        :param recalculateErrorsAttemptCount:
        :param samplePreprocessor: function(sample)->sample or dict with keys {'smoothParams', 'smoothType', 'expSpectrum', 'fitNormInterval', 'norm':None, 'fitGeometryInterval'}. None - means do not smooth
        """
        self.recalculateErrorsAttemptCount = recalculateErrorsAttemptCount
        self.outputFolder = outputFolder
        self.input = inputGenerator
        self.spectralProgram = spectralProgram
        self.runType = None
        self.lock = lock
        self.samplePreprocessor = samplePreprocessor
        self.debug = debug

    def calculate(self, x):
        with self.lock:
            folder = self.input.getFolderForPoint(x)

        return self.calculateFolder(folder)

    def calculateFolder(self, folder):
        if self.debug: print('SpectrumCalculator: calculating folder', folder)
        # checking if the folder already has good data
        with self.lock:
            r = self.parseAndCollect(folder)
            if r is not None:
                if self.debug: print('SpectrumCalculator: we don\'t need to calc. The folder was already calculated', folder)
                return r
        attemptsDone = 0
        while True:
            if attemptsDone >= self.recalculateErrorsAttemptCount+1:
                if self.debug: print(f'Can\'t calculate folder {folder} after all attempts')
                return None, None
            if attemptsDone > 0:
                if self.debug: print(f'Folder {folder} is bad, recalculating')
            self.calculateSpectrum(folder)
            attemptsDone += 1
            with self.lock:
                r = self.parseAndCollect(folder)
                if r is not None:
                    if self.debug: print(f'Returning data from {folder} calculation iterations done: {attemptsDone}')
                    return r
        return None, None

    def configAll(self, runType, runCmd, nProcs, memory):
        self.runType = runType
        self.runCmd = runCmd
        self.nProcs = nProcs
        self.memory = memory

    def calculateSpectrum(self, folder):
        """
        Be careful, when add new spectrum calculators! Adaptive sampling runs calculation procedure in parallel with generation and parsing of folders.
        """
        if self.debug: print('Start calculating folder', folder)
        assert self.runType in ['run-cluster', 'local', 'user defined']
        open(folder+os.sep+'isRunning', 'a').close()
        try:
            if not isinstance(self.runCmd, str):
                assert callable(self.runCmd)
                self.runCmd(folder)
            elif self.runType == 'run-cluster':
                runCluster = getRunner(self.spectralProgram, self.runType)
                runCluster(folder, self.memory, self.nProcs)
            elif self.runType == 'local':
                if self.runCmd == '':
                    runLocal = getRunner(self.spectralProgram, self.runType)
                    runLocal(folder)
                else: runUserDefined(self.runCmd, folder)
            else: runUserDefined(self.runCmd, folder)
        except:
            print('Error while calculation folder', folder, ':\n', traceback.format_exc())
        os.remove(folder+os.sep+'isRunning')

    def parseAndCollect(self, folder):
        # for some cases energy is different in different folders, so we can't do loadExistingSpectrum
        # res = loadExistingSpectrum(self.spectralProgram, self.samplePreprocessor, folder)
        df_spectra, df_params, goodFolders, badFolders = collectResults(self.spectralProgram, self.input.folder, self.outputFolder, printOutput=False)
        if folder not in goodFolders:
            assert folder in badFolders
            return None
        i = goodFolders.index(folder)

        def getSp(df):
            e = utils.getEnergy(df)
            y = df.loc[i].to_numpy()
            return utils.Spectrum(e,y)
        if isinstance(df_spectra, dict):
            spectrum = {name:getSp(df) for name,df in df_spectra.values()}
        else: spectrum = getSp(df_spectra)
        if self.samplePreprocessor is not None:
            if isinstance(self.samplePreprocessor, dict):
                exp = self.samplePreprocessor['expSpectrum']
                if 'fitNormInterval' not in self.samplePreprocessor: self.samplePreprocessor['fitNormInterval'] = None
                resSmoothed, _ = smoothLib.smoothInterpNorm(smoothParams=self.samplePreprocessor['smoothParams'], spectrum=spectrum, smoothType=self.samplePreprocessor['smoothType'], expSpectrum=exp, fitNormInterval=self.samplePreprocessor['fitNormInterval'])
            else:
                resSmoothed = self.samplePreprocessor(spectrum)
        else:
            resSmoothed = spectrum
        if self.spectralProgram != 'vasp_rdf_energy':
            return resSmoothed.intensity, {'spectrum': spectrum, 'folder': folder}
        else:
            return df_params.loc[i,'energy'], {'folder': folder}


def getParams(fileName):
    with open(fileName, 'r') as f: params0 = json.load(f)
    return [p[0] for p in params0], [p[1] for p in params0]


def loadParams(folder, spectralProgram):
    f = os.path.join(folder, 'params.txt')
    if not os.path.exists(f):
        raise Exception(f'No params.txt in the folder {folder}. If it is due to previous sampling crash, remove folder, because sampling can\'t continue calculation')
    _, res = getParams(f)
    return res


def loadExistingXPoints(spectralProgram, folder):
    subfolders = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
    params = []
    for subfolder in subfolders:
        x = loadParams(subfolder, spectralProgram)
        params.append(x)

    return np.array(params)


def parseAllFolders(parentFolder, spectralProgram, printOutput=True):
    """
    Be careful, when add new spectrum parsers! Adaptive sampling runs calculation procedure in parallel with generation and parsing of folders. But generation and parsing are not parallel! The common error: calculation program is writing to a file, that is read by parser. It causes Segmentation fault. What to do?

    :param parentFolder: folder containing results
    :param spectralProgram: one of fdmnes, fdmnesTest, adf, feff, pyGDM
    :param printOutput: whether parsing debug info should be printed
    :return: spectra dataframe, params dataframe, goodFolders list, badFolder list
    """

    def read_folders():
        """

        :return: dictionary { folder, parseOneFolder("parentFolder/folder") }
        """
        import traceback
        subfolders = [f for f in os.listdir(parentFolder) if os.path.isdir(os.path.join(parentFolder, f))]
        subfolders.sort()
        allData = {}
        for i in range(len(subfolders)):
            d = subfolders[i]
            full_d = os.path.join(parentFolder, d)
            if os.path.exists(full_d+os.sep+'isRunning') or utils.jobIsRunning(full_d):
                allData[d] = None
            else:
                try:
                    res = parseOneFolder(full_d)
                    allData[d] = res
                    if res is None:
                        output.append('Can\'t read output in folder ' + d)
                except:
                    output.append(traceback.format_exc())
                    allData[d] = None

        return allData

    def separate_folders(allData):
        """

        :param allData: dictionary of parsed folders
        :return: array of good folder names, array of bad folder names
        """
        badFolders = []
        if spectralProgram in ['fdmnes', 'fdmnesTest']:
            energyCount = np.array([len(xanes.x) for xanes in allData.values() if xanes is not None])
            maxEnergyCount = np.max(energyCount, initial=0)

        for d in allData:
            if allData[d] is None:
                badFolders.append(d)
                continue

            if spectralProgram in ['fdmnes', 'fdmnesTest'] and len(allData[d].x) != maxEnergyCount:
                output.append(f'Error: in folder {d} there are less energies {len(allData[d].x)}')
                badFolders.append(d)

        goodFolders = list(set(allData.keys()) - set(badFolders))
        goodFolders.sort()
        return goodFolders, badFolders

    def get_full_path_folders(folders):
        return [os.path.join(parentFolder, x) for x in folders]

    def createDataframes(allData, goodFolders):
        """

        :param allData: dictionary of parsed folders
        :param goodFolders: list of good folders
        :return:
        """
        # get energies array
        allEnergies = np.array([allData[folder].x for folder in goodFolders])
        n = len(goodFolders)
        if n == 1:
            allEnergies.reshape(1, -1)
        # make specific changes to energies
        if spectralProgram in ['fdmnes', 'fdmnesTest'] and fdmnes.useEpsiiShift:
            energies = np.median(allEnergies, axis=0)
            energies = np.sort(energies)
            maxShift = np.max(allEnergies[:, 0]) - np.min(allEnergies[:, 0])
            output.append('Max energy shift between spectra: {:.2}'.format(maxShift))
        elif spectralProgram[:4] == 'feff':
            if abs(float(allData[goodFolders[0]].x[0])) < 0.00001:
                energies = allData[goodFolders[0]].x[1:]
            else:
                energies = allData[goodFolders[0]].x
        elif spectralProgram == 'adf':
            energies = allData[goodFolders[0]].loc[:, 'E'].ravel()
        elif spectralProgram == 'pyGDM':
            energies = allData[goodFolders[0]].x
        elif spectralProgram == 'vasp_rdf_energy':
            energies = allData[goodFolders[0]].x
        else:
            assert False
        assert np.all(energies[1:] >= energies[:-1]), f'Energies are not sorted!\n'+str(energies)
        paramNames, _ = getParams(os.path.join(parentFolder, goodFolders[0], 'params.txt'))
        df_spectra = np.zeros([n, energies.size])
        df_params = np.zeros([n, len(paramNames)])
        for i in range(n):
            d = goodFolders[i]
            _, params = getParams(os.path.join(parentFolder, d, 'params.txt'))
            df_params[i, :] = np.array(params)
            # make specific spectrum changes
            if spectralProgram in ['fdmnes', 'fdmnesTest'] and fdmnes.useEpsiiShift:
                df_spectra[i, :] = np.interp(energies, allData[d].x, allData[d].y)
            elif spectralProgram == 'adf':
                assert False, 'Это неправильно, нужно размазывать и потом интерполировать'
                df_spectra[i, :] = allData[d].loc[:, 'ftot'].ravel()
            else:
                df_spectra[i, :] = allData[d].y
        df_spectra = pd.DataFrame(data=df_spectra, columns=['e_' + str(e) for e in energies])
        df_params = pd.DataFrame(data=df_params, columns=paramNames)
        return df_spectra, df_params

    if isinstance(spectralProgram, str):
        parseOneFolder = getParser(spectralProgram)
    else:
        parseOneFolder = spectralProgram['parseOneFolder']
    output = []
    allData = read_folders()
    goodFolders, badFolders = separate_folders(allData)
    if len(goodFolders) == 0:
        output.append('None good folders')
        badFolders = get_full_path_folders(badFolders)
        return None, None, goodFolders, badFolders
    if not isinstance(spectralProgram, str):
        # custom
        df_spectra = spectralProgram['createDataframes'](allData, parentFolder, goodFolders)
        if isinstance(df_spectra, tuple):
            df_spectra, goodFolders, badFolders = df_spectra
        paramNames, _ = getParams(os.path.join(parentFolder, goodFolders[0], 'params.txt'))
        params = np.zeros([len(goodFolders), len(paramNames)])
        for i, d in enumerate(goodFolders):
            params[i, :] = np.array(getParams(os.path.join(parentFolder, d, 'params.txt'))[1])
        df_params = pd.DataFrame(data=params, columns=paramNames)
    elif spectralProgram == 'pyGDM':
        df_abs, df_params = createDataframes({f:d['abs'] for f,d in allData.items() if f in goodFolders}, goodFolders)
        df_ext, _ = createDataframes({f:d['ext'] for f,d in allData.items() if f in goodFolders}, goodFolders)
        df_spectra = {'abs': df_abs, 'ext': df_ext}
    elif spectralProgram == 'vasp_rdf_energy':
        allData1 = {}
        for f in allData:
            if allData[f] is None:
                allData1[f] = None
            else:
                allData1[f] = allData[f]['rdf']
        df_spectra, df_params = createDataframes(allData1, goodFolders)
        vasp_energies = []
        for f in goodFolders:
            vasp_energies.append(allData[f]['energy'])
        if df_params is not None:
            df_params['energy'] = vasp_energies
    else:
        df_spectra, df_params = createDataframes(allData, goodFolders)
    badFolders = get_full_path_folders(badFolders)
    goodFolders = get_full_path_folders(goodFolders)
    if printOutput:
        print(*output)
    # print(*output)
    return df_spectra, df_params, goodFolders, badFolders


def checkSampleIsGoodByCount(minPoints):
    return lambda dataset: len(dataset[0]) >= minPoints


class BadSpectrumInSampleError(Exception):
    pass


def convertToSample(dataset, spectralProgram):
    xs, ys, additionalData = dataset
    good = [i for i in range(len(additionalData)) if additionalData[i] is not None]
    if len(good) == 0: return None, None
    df_xanes, df_params, good, _ = parseAllFolders(parentFolder=os.path.split(additionalData[good[0]]['folder'])[0], spectralProgram=spectralProgram, printOutput=False)
    if len(good) == 0: return None, None
    sample = ML.Sample(df_params, df_xanes)
    foldersInds = [int(os.path.split(good[i])[-1]) for i in range(len(good))]
    return sample, foldersInds


def preprocessSample(sample, samplePreprocessor):
    if isinstance(samplePreprocessor, dict):
        smoothConfig = samplePreprocessor
        assert set(smoothConfig.keys()) >= {'smoothParams', 'smoothType', 'expSpectrum'}
        assert set(smoothConfig.keys()) <= {'smoothParams', 'smoothType', 'expSpectrum', 'fitNormInterval', 'norm', 'fitGeometryInterval'}
        if 'fitNormInterval' not in smoothConfig: smoothConfig['fitNormInterval'] = None
        exp = copy.deepcopy(smoothConfig['expSpectrum'])
        if 'fitGeometryInterval' in smoothConfig:
            exp = exp.limit(smoothConfig['fitGeometryInterval'])
        spectra = smoothLib.smoothDataFrame(smoothConfig['smoothParams'], sample.spectra, smoothConfig['smoothType'], exp, smoothConfig['fitNormInterval'])
        sample.setSpectra(spectra)
    else: sample = samplePreprocessor(sample)
    return sample


def plotSpectra(sample, folderInds, estimator, samplePreprocessor, debugOutputFolder, plotPrediction=True):
    if isinstance(samplePreprocessor, dict):
        exp = copy.deepcopy(samplePreprocessor['expSpectrum'])
        if 'fitGeometryInterval' in samplePreprocessor:
            exp = exp.limit(samplePreprocessor['fitGeometryInterval'])
        expPlot = (exp.energy, exp.intensity, 'exp')
    else: expPlot = tuple()
    for i, fi in enumerate(folderInds):
        fileName = f"{debugOutputFolder}{os.sep}spectrum_{fi:05d}.png"
        if os.path.exists(fileName): continue
        title = f'Spectrum {fi} (sample N {i})'
        if plotPrediction:
            sample_loo = sample.copy()
            sample_loo.delRow(i)
            estimator.fit(sample_loo.params, sample_loo.spectra)
            predicted = estimator.predict(sample.params.to_numpy()[i].reshape(1, -1))
            title += f'. Predict by sample of {sample_loo.getLength()} spectra'
            plotting.plotToFile(sample.energy, sample.spectra.to_numpy()[i], 'theory', sample.energy, predicted.reshape(-1), 'predicted', *expPlot, fileName=fileName, title=title, save_csv=False)
        else:
            plotting.plotToFile(sample.energy, sample.spectra.to_numpy()[i], 'theory', *expPlot, fileName=fileName, title=title, save_csv=False)


def plotFunctionHeatmap(estimator, x, y, paramNames, title, fileName, markerText=None):
    assert len(paramNames) == 2
    assert x.shape[1] == 2
    leftBorder = np.min(x, axis=0).reshape(-1)
    rightBorder = np.max(x, axis=0).reshape(-1)
    x1 = np.linspace(leftBorder[0], rightBorder[0], 50)
    x2 = np.linspace(leftBorder[1], rightBorder[1], 50)
    x1g, x2g = np.meshgrid(x1, x2)
    x1v = x1g.flatten()
    x2v = x2g.flatten()
    xsTest = np.dstack((x1v, x2v))[0]
    twoParamsEstimator = copy.deepcopy(estimator)
    twoParamsEstimator.fit(x, y)
    ysTest = twoParamsEstimator.predict(xsTest)
    y2d = ysTest.reshape(x1g.shape)

    fig, ax = plotting.createfig()
    cs1 = ax.contourf(x1, x2, y2d, cmap='jet')
    plt.colorbar(cs1)
    ax.set_xlabel(paramNames[0])
    ax.set_ylabel(paramNames[1])
    ax.scatter(x[:, 0], x[:, 1], s=30)
    if markerText is not None:
        assert len(markerText) == x.shape[0]
        for i in range(x.shape[0]):
            ax.annotate(markerText[i], (x[i, 0], x[i, 1]), fontsize=7)
    ax.set_title(title)
    plotting.savefig(fileName, fig)
    plotting.closefig(fig)


def getImportantParamPair(params,y):
    rf = inverseMethod.getMethod('Extra Trees')
    rf.fit(params, y)
    ind = np.argsort(rf.feature_importances_)
    paramNames = params.columns
    pn1 = paramNames[ind[-1]]
    pn2 = paramNames[ind[-2]]
    if pn1 > pn2: pn1, pn2 = pn2, pn1
    return pn1, pn2


def checkSampleIsGoodByCVError(maxError, samplePreprocessor, spectralProgram, estimator=None, minCountToCheckError=10, cvCount=10, debug=False, debugOutputFolder='debug', testSample=None, maxSampleSize=None):
    """
    Returns function to pass in sampleAdaptively as checkSampleIsGoodFunc argument
    :param maxError:
    :param samplePreprocessor: function(sample)->sample or dict smoothConfig with keys {'smoothParams', 'smoothType', 'expSpectrum', 'fitNormInterval', 'norm':None, 'fitGeometryInterval'}, fitGeometryInterval - to limit fitting spectra
    :param minCountToCheckError:
    :param debug: plot error graphs
    :param debugOutputFolder:
    """
    if samplePreprocessor is None: samplePreprocessor = lambda s: s
    if estimator is None:
        estimator = inverseMethod.getMethod('RBF')
    if debug and os.path.exists(debugOutputFolder): shutil.rmtree(debugOutputFolder)

    def checkSampleIsGood(dataset):
        sample, folderInds = convertToSample(dataset, spectralProgram)
        if sample is None:
            print('None good folders')
            return False
        if maxSampleSize is not None and sample.getLength() >= maxSampleSize: return True
        lastSpInd = folderInds[-1]
        sample = preprocessSample(sample, samplePreprocessor)
        n = sample.getLength()
        if n < minCountToCheckError:
            if debug: plotSpectra(sample, folderInds, estimator, samplePreprocessor, debugOutputFolder, plotPrediction=False)
            return False

        relToConstPredErrorCV, _, predictedSpectra = ML.crossValidation(estimator, sample.params, sample.spectra, cvCount, nonUniformSample=True, YColumnWeights=sample.convertEnergyToWeights())
        print(f'sampleSize = {n}  relToConstPredError = {relToConstPredErrorCV}')
        if debug: plotSpectra(sample, folderInds, estimator, samplePreprocessor, debugOutputFolder, plotPrediction=True)

        if testSample is not None:
            testSample1 = preprocessSample(testSample, samplePreprocessor)
            estimator.fit(sample.params, sample.spectra.to_numpy())
            yPred = estimator.predict(testSample1.params)
            relToConstPredErrorTest = inverseMethod.relativeToConstantPredictionError(yTrue=testSample1.spectra.to_numpy(), yPred=yPred, energy=testSample1.energy)
            print(f'relToConstPredErrorTest: {relToConstPredErrorTest}')
        if debug and len(sample.paramNames) >= 2:
            std = np.std(sample.spectra.to_numpy(), axis=0)
            j = np.argmax(std)
            # print('max spectrum std energy: ', sample.energy[j], 'all interval: ', sample.energy[0], sample.energy[-1])
            y = sample.spectra.to_numpy()[:,j]
            pn1, pn2 = getImportantParamPair(sample.params, y)
            params = sample.params.loc[:, [pn1, pn2]].to_numpy()
            plotFunctionHeatmap(estimator, x=params, y=y, paramNames=[pn1, pn2], title=f'Spectrum at energy = {sample.energy[j]:.0f}', fileName=f"{debugOutputFolder}{os.sep}contour_{lastSpInd:05d}.png", markerText=folderInds)
            relToConstPredError_s = "%.2g" % relToConstPredErrorCV

            plot_error = np.linalg.norm(predictedSpectra - sample.spectra.to_numpy(), ord=np.inf, axis=1)
            me = np.max(plot_error)
            mes = '%.2g' % me
            fileName = utils.addPostfixIfExists(f"{debugOutputFolder}{os.sep}error_{lastSpInd:05d}.png")
            plotting.scatter(params[:,0], params[:,1], color=plot_error / me, colorMap='gist_rainbow_r', marker='s', marker_text=folderInds, title=f'Sample size={n}, max error = {mes}. relToConstPredError = {relToConstPredError_s}', xlabel=pn1, ylabel=pn2, fileName=fileName)

            if testSample is not None:
                fileNameTest = utils.addPostfixIfExists(f"{debugOutputFolder}{os.sep}errorTest_{lastSpInd:05d}.png")
                relToConstPredError_s = "%.2g" % relToConstPredErrorTest
                plot_error = np.linalg.norm(yPred - testSample1.spectra.to_numpy(), ord=np.inf, axis=1)
                me = np.max(plot_error)
                mes = '%.2g' % me
                plotting.scatter(testSample1.params[pn1], testSample1.params[pn2], color=plot_error / me, marker='s', title=f'Sample size= {n}, max error = {mes}. relToConstPredError = {relToConstPredError_s}', xlabel=pn1, ylabel=pn2, fileName=fileNameTest)
        return (relToConstPredErrorCV if testSample is None else relToConstPredErrorTest) <= maxError
    return checkSampleIsGood


def checkSampleIsGoodByVASP(spectralProgram, estimator=None, minCountToCheckError=10, cvCount=10, debug=False, debugOutputFolder='debug', testSample=None, param_names=['theta', 'phi', 'r_pdc']):
    """
    Returns function to pass in sampleAdaptively as checkSampleIsGoodFunc argument
    :param minCountToCheckError:
    :param debug: plot error graphs
    :param debugOutputFolder:
    """
    if estimator is None:
        estimator = inverseMethod.getMethod('RBF')
    if debug and os.path.exists(debugOutputFolder): shutil.rmtree(debugOutputFolder)

    def checkSampleIsGood(dataset):
        sample, folderInds = convertToSample(dataset, spectralProgram)
        if sample is None:
            print('None good folders')
            return False
        lastSpInd = folderInds[-1]
        n = sample.getLength()
        if n < minCountToCheckError:
            if debug: plotSpectra(sample, folderInds, estimator, None, debugOutputFolder, plotPrediction=False)
            return False
        
        # Energy prediction by RDF
        relToConstPredErrorCV, pred = ML.score_cv(estimator, sample.spectra, sample.params['energy'], cv_count=cvCount, returnPrediction=True)
        print(f'relToConstPredError(by_RDF): {relToConstPredErrorCV}')
        if debug: plotSpectra(sample, folderInds, estimator, None, debugOutputFolder, plotPrediction=True)
        
        # Energy prediction by params
        relToConstPredErrorCV_params, pred_params = ML.score_cv(estimator, sample.params.loc[:, param_names], sample.params['energy'], cv_count=cvCount, returnPrediction=True)
        pred_params = pred_params.reshape(-1)
        print(f'relToConstPredError(by_params): {relToConstPredErrorCV_params}')
        # Plot scatters for predictions by params
        for pair in itertools.permutations(param_names, 2):#  [['theta','phi'], ['theta', 'r_pdc'], ['phi', 'r_pdc']]:
            p1 = sample.params.loc[:, pair[0]]
            p2 = sample.params.loc[:, pair[1]]
            plot_error = np.abs(pred_params - sample.params['energy'])
            me = np.max(plot_error)
            mes = '%.2g' % me
            relToConstPredError_params_s = "%.2g" % relToConstPredErrorCV_params
            fileName_params = utils.addPostfixIfExists(f"{debugOutputFolder}{os.sep}error_params_{lastSpInd:05d}.png")
            plotting.scatter(p1, p2, color=plot_error / me, colorMap='gist_rainbow_r', marker='s', marker_text=folderInds, title=f'Sample size={n}, max error = {mes}. relToConstPredError = {relToConstPredError_params_s}', xlabel=pair[0], ylabel=pair[1], fileName=fileName_params)
        
        if testSample is not None:
            X_test = testSample.iloc[:, :700].values
            y_test = testSample.loc[:, 'energy'].values
            estimator.fit(sample.spectra, sample.params['energy'])
            yPred = estimator.predict(X_test)
            relToConstPredErrorTest = ML.scoreFast(y_test, yPred)
            print(f'relToConstPredErrorTest: {relToConstPredErrorTest}')
        if debug and len(sample.paramNames) >= 2:
            y = sample.params['energy']
            pn1, pn2 = param_names[0], param_names[1] #'theta', 'phi'
            params = sample.params.loc[:, [pn1, pn2]].to_numpy()
            plotFunctionHeatmap(estimator, x=params, y=y, paramNames=[pn1, pn2], title=f'Energy', fileName=f"{debugOutputFolder}{os.sep}contour_{lastSpInd:05d}.png", markerText=folderInds)
            relToConstPredError_s = "%.2g" % relToConstPredErrorCV

            plot_error = np.abs(pred - sample.params['energy'])
            me = np.max(plot_error)
            mes = '%.2g' % me
            fileName = utils.addPostfixIfExists(f"{debugOutputFolder}{os.sep}error_{lastSpInd:05d}.png")
            plotting.scatter(params[:,0], params[:,1], color=plot_error / me, colorMap='gist_rainbow_r', marker='s', marker_text=folderInds, title=f'Sample size={n}, max error = {mes}. relToConstPredError = {relToConstPredError_s}', xlabel=pn1, ylabel=pn2, fileName=fileName)

            # Так как нам неизвестны параметры theta и phi для тестового множества, то отключаю построение графиков ниже
            #if testSample is not None:
                #fileNameTest = utils.addPostfixIfExists(f"{debugOutputFolder}{os.sep}errorTest_{lastSpInd:05d}.png")
                #relToConstPredError_s = "%.2g" % relToConstPredErrorTest
                #plot_error = np.abs(yPred - testSample.params['energy'])
                #me = np.max(plot_error)
                #mes = '%.2g' % me
                #plotting.scatter(testSample.params[pn1], testSample.params[pn2], color=plot_error / me, marker='s', title=f'Sample size= {n}, max error = {mes}. relToConstPredError = {relToConstPredError_s}', xlabel=pn1, ylabel=pn2, fileName=fileNameTest)
        return n >= 500
        #return (relToConstPredErrorCV if testSample is None else relToConstPredErrorTest) <= maxError
    return checkSampleIsGood


def sampleAdaptively(paramRanges, moleculeConstructor=None, spectrCalcParams=None, maxError=0.01, maxSampleSize=None, spectralProgram='fdmnes', samplePreprocessor=None, workingFolder='sample', seed=None, outputFolder=None, debugFolder=None, runConfiguration=None, adaptiveSamplerParams=None, settingsFileName='settings.json', debug=False):
    """
    Calculate sample adaptively
    :param paramRanges: dictionary with geometry parameters region {'paramName':[min,max], 'paramName':[min,max], ...}
    :param moleculeConstructor:
    :param maxError: CV-error to stop sampling
    :param maxSampleSize: max sample size to stop sampling
    :param spectrCalcParams: see function generateInput in module correspondent to your spectralProgram
    :param spectralProgram: string - one of knownPrograms (see at the top of this file), or dict{'generateInput':func(params,folder), 'parseOneFolder':func(folder), 'createDataframes':func(allData, parentFolder, goodFolders)->(df_spectra,df_params)}
    :param samplePreprocessor: Is applied to y before given it to adaptive sampler, also applied before plotting. Is NOT applied before saving sample. func(sample)->sample and func(spectrum)->spectrum (spectrum - output of parseOneFolder, sample - ML.Sample of output of parseAllFolders) or dict with keys {'smoothParams', 'smoothType', 'expSpectrum', 'fitNormInterval', 'norm':None, 'fitGeometryInterval'}.
    :param workingFolder:
    :param seed: if None - get current timestamp
    :param outputFolder:
    :param runConfiguration: dict, default = {'runType':'local', 'runCmd':'', 'nProcs':1, 'memory':5000, 'calcSampleInParallel':1, 'recalculateErrorsAttemptCount':0}, runCmd can be command string or function(workingFolder)
    :param adaptiveSamplerParams: dict, default = {'initialIHSDatasetSize':None}
    :param settingsFileName: name of the persistently saved settings, like seed, paramRanges, etc.
    :param debug: print debug info
    :return:
    """
    if outputFolder is None: outputFolder = workingFolder+'_result'
    if debugFolder is None: debugFolder = workingFolder + '_debug'
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
        seed = adaptiveSampling.ensureSettingsAreConsistent(workingFolder + os.sep + settingsFileName, seed, paramRanges)
    else:
        seed = adaptiveSampling.writeSettings(workingFolder + os.sep + settingsFileName, seed, paramRanges)
    if callable(maxError): checkSampleIsGoodFunc = maxError
    else: checkSampleIsGoodFunc = checkSampleIsGoodByCVError(maxError=maxError, samplePreprocessor=samplePreprocessor, spectralProgram=spectralProgram, minCountToCheckError=max(len(paramRanges)*2,10), debug=True, maxSampleSize=maxSampleSize, debugOutputFolder=debugFolder)

    while True:
        try:
            lock = threading.Lock()
            sampler = adaptiveSampling.ErrorPredictingSampler(rangeValues, checkSampleIsGoodFunc=checkSampleIsGoodFunc, seed=seed, initialDataset=initialPoints, **adaptiveSamplerParams, debug=debug)
            folderGen = InputFilesGenerator(rangeValues, paramNames, moleculeConstructor, spectrCalcParams, spectralProgram, workingFolder, debug=debug)
            func = SpectrumCalculator(spectralProgram, folderGen, outputFolder, runConfiguration['recalculateErrorsAttemptCount'], samplePreprocessor, debug, lock)
            func.configAll(runConfiguration['runType'], runConfiguration['runCmd'], runConfiguration['nProcs'], runConfiguration['memory'])
            orchestrator = adaptiveSampling.CalculationOrchestrator(func, lock, runConfiguration['calcSampleInParallel'], debug=debug)
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

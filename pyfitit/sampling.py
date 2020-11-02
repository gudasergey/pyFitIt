import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from distutils.dir_util import copy_tree, remove_tree
import copy, math, shutil, os, tempfile, json, glob, subprocess

from IPython.utils.syspathcontext import prepended_to_syspath
from scipy.optimize import minimize, basinhopping

from pyfitit.smoothLib import findEfermiOnRawSpectrum
from . import fdmnes, feff, adf, utils, ihs, w2auto, fdmnesTest, ML, pyGDM


knownPrograms = ['fdmnes', 'feff', 'adf', 'w2auto', 'fdmnesTest', 'pyGDM']


# ranges - dictionary with geometry parameters region {'paramName':[min,max], 'paramName':[min,max], ...}
# method - IHS, random, grid, line (in case of grid, sampleCount must be a dict of points count through each dimension)
# spectrCalcParams = {energyRange:..., radius:..., Green:True/False, Adimp=None} - for fdmnes
# spectrCalcParams = {RMAX:..., }
# lineEdges = {'start':{...}, 'end':{...}} - for method='line'
def generateInputFiles(ranges, moleculeConstructor, sampleCount, spectrCalcParams, spectralProgram='fdmnes', method='IHS', folder='sample', lineEdges = None, seed=0):
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
        sampleCount = sz
    prettyPoints = []
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
def calcSpectra(spectralProgram='fdmnes', runType='local', runCmd='', nProcs=1, memory=5000, calcSampleInParallel=1, folder='sample', recalculateErrorsAttemptCount = 0, continueCalculation = False):
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
    parse_all_folders = getattr(globals()[spectralProgram], 'parse_all_folders')
    _, _, badFolders = parse_all_folders(folder, printOutput=not continueCalculation)
    recalculateAttempt = 1
    while (recalculateAttempt <= recalculateErrorsAttemptCount) and (len(badFolders) > 0):
        if calcSampleInParallel > 1:
            threadPool.map(calculateXANES, badFolders)
        else:
            for i in range(len(badFolders)): calculateXANES(badFolders[i])
        _, _, badFolders = parse_all_folders(folder)
        recalculateAttempt += 1


def collectResults(spectralProgram='fdmnes', folder='sample', outputFolder='.', printOutput=True):
    assert spectralProgram in knownPrograms, 'Unknown spectral program name: '+spectralProgram
    os.makedirs(outputFolder, exist_ok=True)
    parse_all_folders = getattr(globals()[spectralProgram], 'parse_all_folders')
    df_xanes, df_params, _ = parse_all_folders(folder, printOutput=printOutput)
    if df_xanes is None:
        raise Exception('There is no output data in folder '+folder)
    df_xanes.to_csv(os.path.join(outputFolder,'spectra.txt'), sep=' ', index=False)
    df_params.to_csv(os.path.join(outputFolder,'params.txt'), sep=' ', index=False)


# ranges - dictionary with geometry parameters region {'paramName':[min,max], 'paramName':[min,max], ...}
# method - IHS, random, grid, line (in case of grid, sampleCount must be a dict of points count through each dimension)
# spectrCalcParams = {energyRange:..., radius:..., Green:True/False, Adimp=None} - for fdmnes
# spectrCalcParams = {RMAX:..., }
# lineEdges = {'start':{...}, 'end':{...}} - for method='line'
# krigingType = 'Gaussian Process', 'NNKCDE'
def krigingSampling(ranges, moleculeConstructor, relativeToConstantPredictionError,
    spectrCalcParams, spectralProgram='fdmnes', folder='sample', seed=0, krigingType='GP',
    runType='local', runCmd='', nProcs=1, memory=5000, calcSampleInParallel=1,
    recalculateErrorsAttemptCount=0, continueCalculation=False, outputFolder='.'):
    geometryParametersCount = len(ranges)
    startSampleSize = 12 # 2 ** geometryParametersCount
    parse_all_folders = getattr(globals()[spectralProgram], 'parse_all_folders')

    if not continueCalculation:
        if os.path.exists(folder): shutil.rmtree(folder)

    flagCalculateStartSample = not continueCalculation
    if continueCalculation:
        df_xanes, df_params, badFolders = parse_all_folders(folder, printOutput=not continueCalculation)
        if df_xanes.shape[0] < startSampleSize:
            flagCalculateStartSample = True

    if flagCalculateStartSample:
        generateInputFiles(ranges, moleculeConstructor, startSampleSize, spectrCalcParams, spectralProgram, method='IHS', folder=folder, seed=seed)
        calcSpectra(spectralProgram=spectralProgram, runType=runType, runCmd=runCmd, nProcs=nProcs, memory=memory, calcSampleInParallel=calcSampleInParallel,
                    folder=folder, recalculateErrorsAttemptCount=recalculateErrorsAttemptCount, continueCalculation=continueCalculation)

    df_xanes, df_params, badFolders = parse_all_folders(folder, printOutput=not continueCalculation)
    intensities = df_xanes.values
    e_names = df_xanes.columns
    energy = np.array([float(e_names[i][2:]) for i in range(e_names.size)])
    smoothed = []
    for intensity in intensities:
        efermi, smoothedXanes = findEfermiOnRawSpectrum(energy, intensity)
        smoothedXanes = smoothedXanes[energy >= efermi]
        smoothed.append(smoothedXanes[-1])

    params = df_params.values
    if krigingType == 'GP':
        model = ML.KrigingGaussianProcess(alpha=1e-10)
    elif krigingType == 'EGP':
        model = ML.KrigingEnhancedGaussianProcess(alpha=1e-10)
    elif krigingType == 'NNKCDE':
        model = ML.KrigingNNKCDE()
    elif krigingType == 'Joint x,y':
        model = ML.KrigingJointXY()
    else: assert False, 'Unknown kriging type'
    model.fit(params, np.array(smoothed).reshape(-1,1))

    # find max std point
    def predictError(x):
        _, sigma = model.predict(np.array([x]).reshape(-1,1))
        sigma = sigma - popravka
        return -sigma[0]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=plotting.figsize)
    for i in range(params.shape[0]):
        ax.plot(list(map(lambda x: x[0], params)), list(map(lambda x: x[-1], intensities)), 'ko')

    x = np.arange(0, 1, 0.001)
    y, s = model.predict(x.reshape(-1,1))
    popravka = np.min(s)
    s = s - popravka
    y = y.reshape(-1)
    s = s.reshape(-1)
    print(np.max(s))
    s = s / np.max(s) * (np.max(y) - np.min(y)) / 5

    # print(y.shape, s.shape)
    # print(np.add(y, s))
    ax.fill_between(x, y + s, y - s)
    ax.plot(x, y - s, 'green')
    ax.plot(x, y + s, 'green')
    fig1, ax1 = plt.subplots(figsize=plotting.figsize)
    ax1.plot(x, s)
    # ax.plot(xList, y, 'ro')

    x0 = params[0]
    x0 = np.mean(df_params.values, axis=0)
    res = basinhopping(predictError, x0, minimizer_kwargs={'bounds': [ranges[x] for x in df_params.columns]}, niter=1000)

    # ax = plt.axes(projection='3d')
    #
    ax.plot(res.x[0], model.predict(np.array([res.x]))[0][-1], 'ro')


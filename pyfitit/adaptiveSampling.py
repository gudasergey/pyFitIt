import numpy as np
import os
import threading
import shutil
import copy
from . import utils
import json
from scipy.spatial import distance
from scipy.optimize import minimize
from multiprocessing.dummy import Pool as ThreadPool
from scipy.optimize import basinhopping
from pyfitit.sampling import ihs
from .ML import RBFWrapper
from .sampling import collectResults

# global modules for sampling
from . import fdmnes, feff, adf, utils, ihs, w2auto, fdmnesTest, ML, pyGDM


class Sampler:

    def getInitialPoints(self):
        """
        :returns: collection of points that should be calculated and added to dataset before calling getNewPoint method
        """
        pass

    def getNewPoint(self, dataset):
        """
        :param dataset : collection of samples (x, y). If y is unknown at this point - y is None
        :returns : a new point x for dataset
        """
        pass

    def isGoodEnough(self, dataset):
        """
        :param dataset : collection of samples (x, y). If y is unknown at this point - y is None
        :returns : bool - whether given dataset meets some evaluation criterion for stopping
        """
        pass


class DatasetGenerator:
    """Generates dataset using sampler and orchestrator"""
    def __init__(self, sampler, orchestrator):
        self.orchestrator = orchestrator
        self.sampler = sampler
        self.xs = None
        self.ys = None

    def generate(self):
        """

        :returns: good enough dataset
        """
        self.orchestrator.run(self)
        return self.xs, self.ys

    def isDatasetReady(self):
        """
        Returns True if dataset is ready, otherwise - False
        :return: bool
        """
        return self.sampler.isGoodEnough((self.xs, self.ys))

    def getNewPoint(self):
        """

        :return: point X
        """
        return self.sampler.getNewPoint((self.xs, self.ys))

    def getInitialPoints(self):
        return self.sampler.getInitialPoints()

    def addResult(self, x, y):
        """
        Adds new pair (x, y) to the dataset
        :param x: point "x"
        :param y: corresponding "y"
        :return: index of added result
        """
        # print(f'New point {x}')
        self.xs = np.array([x]) if self.xs is None else np.append(self.xs, [x], axis=0)
        self.ys = np.array([y], dtype=object) if self.ys is None else np.append(self.ys, [y], axis=0)
        return self.xs.shape[0] - 1

    def updateResult(self, index, y):
        assert 0 <= index < self.ys.shape[0]
        self.ys[index] = y


class CalculationOrchestrator:
    """ Decides how to calculate dataset: parallel/serialized, whether failed points should be recalculated, etc.
    """
    def __init__(self, program, calcSampleInParallel=1):
        self.calcSampleInParallel = calcSampleInParallel
        self.program = program
        self.generator = None
        self.lock = threading.Lock()

    def run(self, datasetGenerator):
        self.generator = datasetGenerator

        if self.calcSampleInParallel > 1:
            self.runParallel()
        else:
            self.runSerialized()

    def runSerialized(self):
        while not self.generator.isDatasetReady():
            x = self.generator.getNewPoint()
            y = self.program.calculate(x)
            self.generator.addResult(x, y)

    def runParallel(self):
        # calculating initial dataset
        self.calculateInitial()

        print("Done initial")
        # calculating the rest of the points
        self.calculateLazy(self.pointSequence())

    def calculateInitial(self):
        from multiprocessing.dummy import Pool as ThreadPool
        threadPool = ThreadPool(self.calcSampleInParallel)
        threadPool.map(self.addResultCalculateAndUpdate, self.generator.getInitialPoints())

    def calculateLazy(self, points):
        from pyfitit.executors import LazyThreadPoolExecutor
        pool = LazyThreadPoolExecutor(self.calcSampleInParallel)
        results = pool.map(self.calculate, points)
        for index, y in results:
            self.generator.updateResult(index, y)

    def pointSequence(self):
        while not self.generator.isDatasetReady():
            x = self.generator.getNewPoint()
            index = self.generator.addResult(x, 'calculating')
            yield index, x

    def calculate(self, pointAndIndex):
        index, x = pointAndIndex
        y = self.program.calculate(x)

        # from datetime import datetime
        # dateTimeObj = datetime.now()
        # timestampStr = dateTimeObj.strftime("%H:%M:%S.%f")
        # print(f'[{timestampStr}]done calculation ' + str(x))

        return index, y

    def addResultCalculateAndUpdate(self, x):
        with self.lock:
            index = self.generator.addResult(x, 'calculating')

        y = self.program.calculate(x)

        # from datetime import datetime
        # dateTimeObj = datetime.now()
        # timestampStr = dateTimeObj.strftime("%H:%M:%S.%f")
        # print(f'[{timestampStr}]done calculation ' + str(x))
        with self.lock:
            self.generator.updateResult(index, y)


class CalculationProgram:
    """Does the actual calculation. Provided with point "x" returns corresponding "y"
    """

    def calculate(self, x):
        return None


"""
------------------------------------ Implementations ------------------------------------
"""


class DiscontinuousFunc(CalculationProgram):

    def __init__(self):
        self.xs = np.linspace(0, 5, 20)

    def calculate(self, x):
        xs = self.xs
        x1, x2 = x
        sin = np.sin(xs + x2)
        sinsq = sin * sin
        if x1 > 2.3:
            return sinsq * (xs / (x1 + x2))
        else:
            return sinsq * (xs / x1)


# class SciPyRbf():
#     def fit(self, x, y):
#         from scipy.interpolate import Rbf
#
#         data = np.append(x, y, axis=1)
#         self.instance = Rbf(*data.T)
#
#     def predict(self, x):
#         return self.instance(*x.T)


class IHSSampler(Sampler):

    def __init__(self, paramRanges, pointCount):
        self.pointCount = pointCount
        self.paramRanges = paramRanges
        self.leftBorder, self.rightBorder = self.getParamBorders()
        self.points = self.initialPoints()

    def getParamBorders(self):
        leftBorder = np.array([x[0] for x in self.paramRanges])
        rightBorder = np.array([x[1] for x in self.paramRanges])
        return leftBorder, rightBorder

    def initialPoints(self):
        seed = 0
        N = len(self.paramRanges)
        sampleCount = self.pointCount
        points = (ihs.ihs(N, sampleCount, seed=seed) - 0.5) / sampleCount  # row - is one point
        for j in range(N):
            points[:, j] = self.leftBorder[j] + points[:, j] * (self.rightBorder[j] - self.leftBorder[j])

        return points

    def isGoodEnough(self, dataset):
        xs, ys = dataset
        isGood = xs is not None and xs.shape[0] == self.pointCount
        return isGood

    def getNewPoint(self, dataset):
        xs, ys = dataset
        return self.points[0] if xs is None else self.points[xs.shape[0]]


class ErrorPredictingSampler(Sampler):

    def __init__(self, paramRanges, minError=0.1):
        self.minError = minError
        self.paramRanges = paramRanges
        self.leftBorder, self.rightBorder = self.getParamBorders()
        self.xs = None
        self.ys = None
        self.yPredictionModel = None
        self.predictedErrors = []
        self.predictedYs = []
        self.initial = np.append(self.initialPointsCorners(), self.initialPointsIHS(), axis=0)
        self.initialSize = self.initial.shape[0]
        self.initialMaxDist = None

    def maxDistanceInInitialYs(self):
        size = self.initialSize
        d = distance.cdist(self.ys[:size], self.ys[:size], 'euclidean')
        return np.amax(d)

    def initialPointsIHS(self):
        seed = 0
        N = len(self.paramRanges)
        sampleCount = N + 1
        points = (ihs.ihs(N, sampleCount, seed=seed) - 0.5) / sampleCount  # row - is one point
        for j in range(N):
            points[:, j] = self.leftBorder[j] + points[:, j] * (self.rightBorder[j] - self.leftBorder[j])

        return points

    def initialPointsCorners(self):
        # points = np.dstack(np.array(np.meshgrid(self.paramRanges)).reshape(len(self.paramRanges), -1))[0]
        return np.array([self.leftBorder, self.rightBorder])

    def getParamBorders(self):
        leftBorder = np.array([x[0] for x in self.paramRanges])
        rightBorder = np.array([x[1] for x in self.paramRanges])
        return leftBorder, rightBorder

    def getInitialPoints(self):
        return self.initial

    def getNewPoint(self, dataset):
        self.extractDataset(dataset)
        assert self.xs.shape[0] >= len(self.initial)

        return self.getFromMidPoints()
        # return self.getFromGlobalMaxError(dataset)

    def extractDataset(self, dataset):
        xs, ys = dataset
        # print(type(ys), ys)
        notNone = ~utils.isObjectArraysEqual(ys, 'calculating')
        # print('notNone=',notNone)
        self.xs = xs[notNone]
        self.ys = np.vstack(ys[notNone])
        self.fitModel()

        self.xs = np.array(xs)
        self.ys = np.array(ys)
        for i, y in enumerate(ys):
            if isinstance(y, str) and y == 'calculating':
                x = np.array([xs[i]])
                self.ys[i] = self.yPredictionModel.predict(x)

        self.ys = np.vstack(self.ys)

    def getFromMidPoints(self):
        xsSorted = np.sort(self.xs, axis=0)

        candidate = None
        maxError = None
        for i in range(10):
            idx = np.random.randint(self.xs.shape[0] - 1, size=self.xs.shape[1])
            halfs = (xsSorted[idx + 1, range(idx.shape[0])] + xsSorted[idx, range(idx.shape[0])]) / 2
            assert all([r[0] <= x <= r[1] for x, r in zip(halfs, self.paramRanges)]), halfs
            newError = self.getError(halfs)
            if maxError is None or newError > maxError:
                maxError = newError
                candidate = halfs

        self.predictedErrors.append(maxError)
        self.predictedYs.append(self.yPredictionModel.predict(np.array([candidate])))
        return candidate

    def fitModel(self):
        self.yPredictionModel = RBFWrapper()
        self.yPredictionModel.fit(self.xs, self.ys)

    def getFromGlobalMaxError(self, dataset):
        # those are expected to be np arrays
        self.extractDataset(dataset)
        self.fitModel()
        x0 = np.mean(self.paramRanges, axis=1)
        res = basinhopping(self.getNegativeError, x0, minimizer_kwargs={'bounds': self.paramRanges}, niter=5)
        return res.x

    def getY(self, x):
        for i, xKnown in enumerate(self.xs):
            if np.array_equal(xKnown, x):
                return self.ys[i]

        # we didn't find x in known points, so we predict it
        return self.yPredictionModel.predict(np.array([x]))

    def getError(self, x):

        delta = self.rightBorder - self.leftBorder
        if self.initialMaxDist is None:
            self.initialMaxDist = self.maxDistanceInInitialYs()
        M = self.initialMaxDist
        alfa = 1.0
        y = self.yPredictionModel.predict(np.array([x]))

        def getDist(x0, y0, x1, y1):
            xPart = np.linalg.norm((x0 - x1) / delta)
            return xPart + alfa * np.linalg.norm(y0 - y1) / M

        minDist = None
        for i in range(len(self.xs)):
            newDist = getDist(self.xs[i], self.ys[i], x, y)
            if minDist is None or newDist < minDist:
                minDist = newDist

        return minDist

    def getNegativeError(self, x0):
        return -self.getError(x0)

    def isGoodEnough(self, dataset):
        # return self.pointsBasedCheck(dataset)
        return self.errorBasedCheck(dataset)

    def errorBasedCheck(self, dataset):
        xs, ys = dataset
        if xs.shape[0] < self.initialSize:
            return False

        x = self.getNewPoint(dataset)
        error = self.getError(x)
        print(f'inaccuracy: {error}')
        return error <= self.minError

    def pointsBasedCheck(self, dataset):
        xs, ys = dataset
        if xs is None:
            return False
        points = xs.shape[0]
        # print(points)
        return points >= 500


class XanesInputGenerator:
    """Generates input folder with required content for a certain xanes-calculating program (e.g. ADF, FDMNES)"""
    def __init__(self, ranges, paramNames, moleculeConstructor, spectrCalcParams, spectralProgram='fdmnes', folder='sample'):

        assert spectralProgram in ['fdmnes', 'feff', 'adf', 'w2auto', 'fdmnesTest', 'pyGDM'], \
            'Unknown spectral program name: '+spectralProgram

        self.spectrCalcParams = spectrCalcParams
        self.spectralProgram = spectralProgram
        self.moleculeConstructor = moleculeConstructor
        self.ranges = ranges
        self.paramNames = paramNames
        self.folder = folder
        self.folderCounter = 1

    def getFolderForPoint(self, x):

        # this would work correctly only every passed x is unique
        # folder = '.'+os.path.sep+str(self.folderCounter)
        # if os.path.exists(self.folder):
        #     shutil.rmtree(self.folder)

        os.makedirs(self.folder, exist_ok=True)
        geometryParams = {}
        N = len(self.paramNames)
        for j, name in enumerate(self.paramNames):
            geometryParams[name] = x[j]
        molecule = self.moleculeConstructor(geometryParams)
        if molecule is None:
            print("Can't construct molecule for parameters "+str(geometryParams))
            return None
        folderOne = os.path.join(self.folder, utils.zfill(self.folderCounter, 200000))
        generateInput = getattr(globals()[self.spectralProgram], 'generateInput')
        generateInput(molecule, folder=folderOne, **self.spectrCalcParams)
        geometryParamsToSave = [[self.paramNames[j], x[j]] for j in range(N)]
        with open(os.path.join(folderOne, 'geometryParams.txt'), 'w') as f:
            json.dump(geometryParamsToSave, f)
        print('folder=', folderOne, ' '.join([p+'={:.4g}'.format(geometryParams[p]) for p in geometryParams]))
        if hasattr(molecule, 'export_xyz'):
            molecule.export_xyz(folderOne+'/molecule.xyz')

        self.folderCounter += 1
        return folderOne


class XanesCalculator(CalculationProgram):

    def __init__(self, spectralProgram, inputGenerator, outputFolder, recalculateErrorsAttemptCount):
        self.recalculateErrorsAttemptCount = recalculateErrorsAttemptCount
        self.outputFolder = outputFolder
        self.input = inputGenerator
        self.spectralProgram = spectralProgram
        self.runType = None
        self.lock = threading.Lock()

    def calculate(self, x):
        with self.lock:
            folder = self.input.getFolderForPoint(x)

        attemptsDone = 0
        while True:
            self.calculateXANES(folder)
            if not self.checkIfBadFolder(folder) or attemptsDone > self.recalculateErrorsAttemptCount:
                break
            attemptsDone += 1
            print(f'Folder {folder} is bad, recalculating')

        with self.lock:
            ys = self.parseAndCollect(folder)

        # TODO: parse folder and return result
        # TODO: how to interpolate xanes by common energy?
        return ys

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

    def calculateXANES(self, folder):
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
        parse_method = getattr(globals()[self.spectralProgram], 'parse_all_folders')
        _, _, badFolders = parse_method(self.input.folder, printOutput=False)
        return os.path.basename(os.path.normpath(folder)) in badFolders

    def parseAndCollect(self, folder):
        # TODO: process bad folders
        parse_method = getattr(globals()[self.spectralProgram], 'parse_one_folder')
        res = parse_method(folder)
        collectResults(self.spectralProgram, self.input.folder, self.outputFolder, printOutput=False)
        return res.intensity

# TODO: memory/processors settings for local
def sampleAdaptively(paramRanges, moleculeConstructor, maxError, spectrCalcParams, spectralProgram='fdmnes', workingFolder='sample', seed=0,
                     runType='local', runCmd='', nProcs=1, memory=5000, calcSampleInParallel=1, recalculateErrorsAttemptCount=0,
                     outputFolder='sample_result'):
    
    if os.path.exists(workingFolder):
        shutil.rmtree(workingFolder)

    paramNames = [k for k in paramRanges]
    paramNames.sort()
    rangeValues = np.array([paramRanges[k] for k in paramNames])

    sampler = ErrorPredictingSampler(rangeValues, maxError)
    folderGen = XanesInputGenerator(rangeValues, paramNames, moleculeConstructor, spectrCalcParams, spectralProgram, workingFolder)
    func = XanesCalculator(spectralProgram, folderGen, outputFolder, recalculateErrorsAttemptCount)
    func.configAll(runType, runCmd, nProcs, memory)
    orchestrator = CalculationOrchestrator(func, calcSampleInParallel)
    generator = DatasetGenerator(sampler, orchestrator)
    xs, ys = generator.generate()

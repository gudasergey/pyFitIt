import numpy as np
import time, threading, scipy, traceback, copy, os, json
from .ML import RBF, isFitted, crossValidation
from . import utils, ihs, geometry, plotting
from numpy.random import default_rng


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
        self.ys = []
        self.additionalData = []

    def generate(self):
        """

        :returns: good enough dataset
        """
        self.orchestrator.run(self)
        return self.xs, self.ys, self.additionalData

    def isDatasetReady(self):
        """
        Returns True if dataset is ready, otherwise - False
        :return: bool
        """
        return self.sampler.isGoodEnough((self.xs, self.ys, self.additionalData))

    def getNewPoint(self):
        """

        :return: point X
        """
        return self.sampler.getNewPoint((self.xs, self.ys, self.additionalData))

    def getInitialPoints(self):
        return self.sampler.getInitialPoints()

    def addResult(self, x, y, additionalData=None):
        """
        Adds new pair (x, y) to the dataset
        :param x: point "x"
        :param y: corresponding "y"
        :return: index of added result
        """
        # print(f'New point {x}')
        self.xs = np.array([x]) if self.xs is None else np.append(self.xs, [x], axis=0)
        # self.ys = np.array([y], dtype=object) if self.ys is None else np.append(self.ys, [y], axis=0)
        self.ys.append(y)
        self.additionalData.append(additionalData)
        self.sampler.updateDatasetPointInfo((self.xs, self.ys, self.additionalData), len(self.ys)-1)
        return self.xs.shape[0] - 1

    def updateResult(self, index, y, additionalData=None):
        assert 0 <= index < len(self.ys)
        self.additionalData[index] = additionalData
        self.ys[index] = y
        self.sampler.updateDatasetPointInfo((self.xs, self.ys, self.additionalData), index)


class CalculationOrchestrator:
    """ Decides how to calculate dataset: parallel/serialized, whether failed points should be recalculated, etc.
    """
    def __init__(self, program, lock, calcSampleInParallel=1, existingDatasetGetter=None, debug=False):
        self.calcSampleInParallel = calcSampleInParallel
        if callable(program):
            class Prog(CalculationProgram):
                def __init__(self, func):
                    self.func = func
                def calculate(self, x):
                    return self.func(x), None
            program = Prog(program)
        self.program = program
        self.generator = None
        self.existingDatasetGetter = existingDatasetGetter
        self.lock = lock  # lock must be common in orchestrator and program
        self.calcTimes = []
        self.debug = debug

    def run(self, datasetGenerator):
        self.generator = datasetGenerator

        self.loadExistingDataset()

        if self.calcSampleInParallel > 1:
            self.runParallel()
        else:
            self.runSerialized()

    def addResult(self, x, res):
        if isinstance(res, tuple):
            assert len(res) == 2
            self.generator.addResult(x, y=res[0], additionalData=res[1])
        else:
            self.generator.addResult(x, res)

    def runSerialized(self):
        if self.debug: print('Run serialized sampling')

        def calculateAndAdd(x):
            res = self.calcAndUpdateTime(x)
            self.addResult(x, res)

        if self.existingDatasetGetter is None:
            for x in self.generator.getInitialPoints():
                calculateAndAdd(x)
        while not self.generator.isDatasetReady():
            x = self.generator.getNewPoint()
            calculateAndAdd(x)

    def runParallel(self):
        if self.debug: print(f'Run parallel sampling by {self.calcSampleInParallel} threads')
        # calculating initial dataset
        if self.existingDatasetGetter is None:
            self.calculateInitial()

        # calculating the rest of the points
        if self.debug: print("Run lazy parallel execution")
        self.calculateLazy(self.pointSequence())

    def calculateInitial(self):
        if self.debug: print('Calculating initial sample...')
        from multiprocessing.dummy import Pool as ThreadPool
        threadPool = ThreadPool(self.calcSampleInParallel)
        threadPool.map(self.addResultCalculateAndUpdate, self.generator.getInitialPoints())
        threadPool.close()
        if self.debug: print("Done initial")
        if np.all(np.array(self.generator.ys) == None):
            raise Exception('No good points in the initial sample. Check sampling settings')

    def calculateLazy(self, points):
        from pyfitit.executors import LazyThreadPoolExecutor
        # from concurrent.futures import ThreadPoolExecutor as LazyThreadPoolExecutor
        pool = LazyThreadPoolExecutor(self.calcSampleInParallel)
        results = pool.map(self.calculate, points)
        for index, y, additionalData in results:
            with self.lock:
                self.generator.updateResult(index, y, additionalData)

    def processCalculatedFutures(self, done):
        for index, y, additionalData in [f.result() for f in done]:
            with self.lock:
                # print(f'adding result {index}')
                self.generator.updateResult(index, y, additionalData)

    def pointSequence(self):
        with self.lock:
            isReady = self.generator.isDatasetReady()
        while not isReady:
            with self.lock:
                # print('getting new point')
                x = self.generator.getNewPoint()
                index = self.generator.addResult(x, 'calculating')
            yield index, x
            with self.lock:
                isReady = self.generator.isDatasetReady()

    def calculate(self, pointAndIndex):
        index, x = pointAndIndex
        if self.debug: print('Calculating for x=',x)
        y, additionalData = self.calcAndUpdateTime(x)
        if self.debug: print('End calculating for x=', x, 'y=', y, 'additionalData=', additionalData)
        return index, y, additionalData

    def calcAndUpdateTime(self, x):
        t0 = time.time()
        res = self.program.calculate(x)
        dt = time.time() - t0
        with self.lock:
            self.calcTimes.append(dt)
            assert hasattr(self.generator.sampler, 'avgPointCalcTime')
            self.generator.sampler.avgPointCalcTime = np.mean(self.calcTimes)
        return res

    def addResultCalculateAndUpdate(self, x):
        with self.lock:
            index = self.generator.addResult(x, 'calculating')

        y, additionalData = self.calcAndUpdateTime(x)

        with self.lock:
            self.generator.updateResult(index, y, additionalData)

    def loadExistingDataset(self):
        if self.existingDatasetGetter is None:
            return

        dataset = self.existingDatasetGetter()
        assert len(dataset) > 0, 'Loaded dataset is empty. Try starting a fresh dataset generation.'
        X, Y = dataset[0], dataset[1]
        additional = [None]*len(X) if len(dataset) < 3 else dataset[2]
        for i in range(len(X)):
            self.addResult(X[i], (Y[i], additional[i]))


class CalculationProgram:
    """Does the actual calculation. Provided with point "x" returns corresponding "y" and additional data
    """

    def calculate(self, x):
        return None, None



"""
------------------------------------ Implementations ------------------------------------
"""


class Rosenbrock_2D(CalculationProgram):

    def calculate(self, xx):
        d = len(xx)
        sum = 0
        for ii in range(d - 1):
            xi = xx[ii]
            xnext = xx[ii+1]
            new = 100 * (xnext - xi ** 2) ** 2 + (xi - 1) ** 2
            if xx[1] > 1.5:
                new = new + 700 * xx[1] * xx[0]
            sum = sum + new

        return sum

        # x = xx[0]
        # y = xx[1]
        # a = 1. - x
        # b = y - x * x
        # res = a * a + b * b * 100
        # if xx[1] > 1.5:
        #     res = res + 700 * xx[1] * xx[0]
        # return res


class SHCamel_2D(CalculationProgram):

    def calculate(self, xx):
        x1 = xx[0]
        x2 = xx[1]

        term1 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
        term2 = x1 * x2
        term3 = (-4 + 4 * x2 ** 2) * x2 ** 2

        y = term1 + term2 + term3
        return y


class DiscontinuousFunc(CalculationProgram):

    def __init__(self):
        self.xs = np.linspace(0, 5, 20)

    def calculate(self, x):
        xs = self.xs
        x1, x2 = x
        # sin = np.sin(xs + (x2-0.5)*2)
        # sinsq = sin ** 20
        # if x1 > 2.3:
        #     return [sinsq * (xs / (x1 + x2)), None]
        # else:
        #     return [sinsq * (xs / x1), None]
        return [xs/(0.02+np.abs(x1-2.4)**2+np.abs(x2-1.5)**2), None]


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


class Ring:
    def __init__(self, size, defaultValue):
        self.data = np.zeros(size) + defaultValue
        self.ind = 0

    def add(self, v):
        self.data[self.ind] = v
        self.ind = (self.ind + 1) % len(self.data)


def gradient(x, yPredictionModel, paramRanges, yDist):
    h = 0.1
    n = len(x)
    grad = np.zeros(n)
    d = paramRanges[:,1] - paramRanges[:,0]
    assert np.all(d>0)
    y1 = yPredictionModel.predict(x.reshape(1, -1))
    for i in range(n):
        x1 = np.copy(x)
        x1[i] = x[i] + h*d[i]
        y2 = yPredictionModel.predict(x1.reshape(1,-1))
        grad[i] = yDist(y1, y2) / h
    return grad


def normalizeGradients(points, yPredictionModel, paramRanges, yDist, eps):
    grads = np.zeros(points.shape)
    for i in range(points.shape[0]):
        grads[i, :] = gradient(points[i], yPredictionModel, paramRanges, yDist)
        # print(grads[i])
    max_grad = np.max(grads, axis=0).reshape(1,-1)
    max_grad /= np.max(max_grad)
    # print(max_grad, np.sum(points))
    if len(max_grad[max_grad <= eps]) > 0:
        max_grad[max_grad <= eps] = np.min(max_grad[max_grad > eps])*0.1
    return points*max_grad, max_grad.reshape(-1)


def cross_val_predict(estimator, xs, ys, CVcount):
    import sklearn

    if xs.shape[0] > 20:
        kf = sklearn.model_selection.KFold(n_splits=CVcount, shuffle=True, random_state=0)
    else:
        kf = sklearn.model_selection.LeaveOneOut()
    prediction_spectra = np.zeros(ys.shape)
    for train_index, test_index in kf.split(xs):
        X_train, X_test = xs[train_index], xs[test_index]
        y_train, y_test = ys[train_index, :], ys[test_index, :]
        estimator.fit(X_train, y_train)
        prediction_spectra[test_index] = estimator.predict(X_test)

    return prediction_spectra


def voronoyTimeParams(dim):
    assert dim >= 2
    C = [1e-5, 4.5e-5, 2.5e-4, 2.5e-4, 2e-5, 1.4e-4, 1e-5, 1e-6, 5e-6, 1e-5, 2e-5]
    power = [1,1,1, 1.25, 2,2, 3, 4,4,4,4]
    if dim <= 12: return C[dim-2], power[dim-2]
    elif dim <= 20: return 1.7**(dim-13)*2.5e-6, 5
    else: return 1.6**(2*np.sqrt(dim-20))*6e-6, 6


def voronoyTime(dim, n):
    assert n > dim
    if dim == 1: return 0
    C, power = voronoyTimeParams(dim)
    return C*(n-dim)**power


def getMaxNForVoronoyTime(dim, dt):
    assert dim >= 1
    if dim == 1: return 1000**3
    C, power = voronoyTimeParams(dim)
    return max(int(0.5 + dim + (dt/C)**(1/power)), dim+1)


def scaleX(x, scaleGrad):
    if scaleGrad is None: return x
    if len(x.shape) == 1:
        assert len(scaleGrad.shape) == 1
        assert x.shape[0] == scaleGrad.shape[0]
    else:
        assert len(x.shape) == 2
        if len(scaleGrad.shape) == 1: scaleGrad = scaleGrad.reshape(1,-1)
        assert x.shape[1] == scaleGrad.shape[1]
        assert scaleGrad.shape[0] == 1
    return x*scaleGrad


def unscaleX(x, scaleGrad):
    if scaleGrad is None: return x
    if len(x.shape) == 1:
        assert len(scaleGrad.shape) == 1
        assert x.shape[0] == scaleGrad.shape[0]
    else:
        assert len(x.shape) == 2
        if len(scaleGrad.shape == 1): scaleGrad = scaleGrad.reshape(1,-1)
        assert x.shape[1] == scaleGrad.shape[1]
        assert scaleGrad.shape[0] == 1
    assert np.all(scaleGrad != 0)
    return x/scaleGrad


def fixOutPoints(x, paramRanges, throw=False):
    leftBorder = paramRanges[:, 0].reshape(-1)
    rightBorder = paramRanges[:, 1].reshape(-1)
    if throw:
        return np.array([x[i] for i in range(len(x)) if np.all(leftBorder <= x[i]) and np.all(x[i] <= rightBorder)])
    x = np.copy(x)
    for i in range(len(x)):
        vi = x[i]
        j = leftBorder > vi
        vi[j] = leftBorder[j]
        j = vi > rightBorder
        vi[j] = rightBorder[j]
        x[i] = vi
    return x


def randomDirection(dim, rng):
    vector = np.zeros(dim)
    while np.linalg.norm(vector) < 1e-3:
        vector = rng.normal(loc=0, size=dim)
    return vector / np.linalg.norm(vector)


def randomSample(paramRanges, rng, pointNum):
    dim = paramRanges.shape[0]
    randomPoints = rng.uniform(size=(pointNum, dim))
    a, b = paramRanges[:, 0], paramRanges[:, 1]
    ab = b - a
    randomPoints = randomPoints * ab.reshape(1, -1) + a.reshape(1, -1)
    return randomPoints


def voronoy(xNormed, scaleGrad=None, xInitial=None):
    """
    Calculate Voronoy vertices
    :param xNormed: normalized points (each row - point)
    :param scaleGrad: to denormalize voronoy vertices   vertices[i] / scaleGrad
    :param xInitial: to print in case of failed Voronoy
    :returns: denormalized vertices
    """
    if scaleGrad is None: scaleGrad = np.ones(xNormed.shape[1])
    if xNormed.shape[1] >= 2:
        try:
            vor = scipy.spatial.Voronoi(xNormed)
        except Exception as e:
            print('Voronoi fails for points:')
            if xInitial is not None: print(xInitial)
            else: print(xNormed)
            print('dim =', xNormed.shape[1], 'count =', xNormed.shape[0])
            return np.array([])
        vertices = np.array([unscaleX(vor.vertices[i],scaleGrad) for i in range(len(vor.vertices))])
    else:
        tmp = np.sort(xNormed.reshape(-1))
        vertices = ((tmp[1:] + tmp[:-1]) / 2).reshape(-1, 1)
    if len(vertices) == 0: return np.array([])
    return vertices


class ErrorPredictingSampler(Sampler):

    def __init__(self, paramRanges, estimator=None, initialIHSDatasetSize=None, initialDataset=None, optimizeLp=2, seed=None, checkSampleIsGoodFunc=None, xDist=None, yDist=None, normalizeGradientsGlobally=True, normalizeGradientsLocally=False, fixOut=False, samplerTimeConstraint='default', profilingInfoFile=None, samplingMethod='auto', trueFunc=None, debug=False):
        """

        :param paramRanges: list of pairs [leftBorder, rightBorder]
        :param initialIHSDatasetSize: size of initial dataset, generated by IHS
        :param initialDataset: starting dataset
        :param optimizeLp: p in (0, np.inf] to minimize error: ||yTrue - yPred||_Lp
        :param seed:
        :param checkSampleIsGoodFunc: function(dataset), which returns True when we should stop calculation. dataset = (xs, ys, additionalData)
        :param xDist: function(x0, xs) returns distance array from point x0 to each point of array xs
        :param yDist: function(y1, y2) returns distance between y1 and y2. If None use np.linalg.norm(y1-y2, ord=np.inf)
        :param samplerTimeConstraint: float or func(onePointCalcTime). Defines how much time can be used for new point sampling (None - no constraints). Default:  max(min(onePointCalcTime/10,6)*10, 0.1)
        :param samplingMethod: Defines two methods: how to choose candidates for error maximization and how to estimate error.
            dict. Example: {'candidates':'neighbors', 'errorEst':'model-cv'}. Use 'auto' for auto switching between methods.
            Candidate methods:
                'voronoi' - all voronoi vertices (if samplerTimeConstraint=None - for all sample, otherwise - for neighbors of max error points)
                'neighbors' - random candidates in neighbourhood of max error points
                'random' - take multiple random points and choose best (error is not used, distance to other points is not used)
            Error estimation methods:
                'exact' - use trueFunc for error estimate
                'model-cv' - use model CV-error estimate
                'gradient' - use gradient approximation and distance to estimate error. Model is not used
                'distance' - search the most remote point from neighbors
        """
        if isinstance(samplingMethod,str):
            assert samplingMethod == 'auto'
            samplingMethod = {'candidates':'neighbors', 'errorEst':'model-cv'}
            self.autoMethodChoice = True
        else: self.autoMethodChoice = False
        assert samplingMethod['candidates'] in ['voronoi', 'neighbors', 'random']
        assert samplingMethod['errorEst'] in ['exact', 'model-cv', 'gradient', 'distance']
        if samplingMethod['errorEst'] == 'distance':
            assert samplingMethod['candidates'] == 'random', 'Other than random candidates are impractical for distance error estimation'
        if samplingMethod['errorEst'] == 'exact':
            assert trueFunc is not None
        # assert samplingMethod in ['voronoi', 'random', 'random-remote', 'max-error-global', 'max-error-exact', 'max-error-neighbor']
        self.samplingMethod = samplingMethod
        # candidate methods, that can easily expand sample out of initial sample bounds. In other methods we add one point (using random-remote) to candidates on each sampling step
        self.canExpand = ['random']
        self.trueFunc = trueFunc
        assert checkSampleIsGoodFunc is not None, 'You should define checkSampleIsGoodFunc, for example\nlambda dataset: len(dataset[0]) >= 200\nOr use cross-validation error.\ndataset = (xs, ys, additionalData)'
        if seed is None:
            seed = int(time.time())
        self.rng = default_rng(seed=seed)
        if isinstance(paramRanges, list): paramRanges = np.array(paramRanges)
        self.paramRanges = paramRanges
        dim = len(paramRanges)
        self.leftBorder, self.rightBorder = self.getParamBorders()
        self.xs = np.zeros((0,len(paramRanges)))
        self.ys = np.zeros((0,0))
        self.notValidYsInd = np.array([])
        self.additionalData = None
        self.estimator = RBF() if estimator is None else estimator
        self.predictedYs = []
        self.predictedYsCv = None
        self.ysErrors = []
        self.initialIHSDatasetSize = initialIHSDatasetSize if initialIHSDatasetSize is not None else dim+1
        self.initial = self.mergeInitialWithExternal(initialDataset, seed)
        self.initialMaxDist = None
        self.checkSampleIsGoodFunc = checkSampleIsGoodFunc
        if yDist is None: yDist = lambda y1, y2: np.linalg.norm(y1-y2, ord=np.inf)
        self.yDist = yDist
        if xDist is None:
            xDist = self.xDistDefault
        else:
            assert not normalizeGradientsGlobally
            assert not normalizeGradientsLocally
        self.xDist = xDist
        if dim == 1: normalizeGradientsGlobally = False
        if samplerTimeConstraint is not None:
            if samplerTimeConstraint == 'default': samplerTimeConstraint = lambda onePointCalcTime: max(min(onePointCalcTime/10,6)*10, 0.1)
            if not callable(samplerTimeConstraint):
                assert isinstance(samplerTimeConstraint, int) or isinstance(samplerTimeConstraint, float)
                t = samplerTimeConstraint
                samplerTimeConstraint = lambda _: t
        self.settings = {
            'optimizeLp': optimizeLp,
            'normalizeGradientsGlobally': normalizeGradientsGlobally,
            'normalizeGradientsLocally': normalizeGradientsLocally,
            'gradEps': 1e-2,
            'fixOut': fixOut,
            'samplerTimeConstraint': samplerTimeConstraint
        }
        if self.settings['normalizeGradientsGlobally']: assert not self.settings['normalizeGradientsLocally']
        if self.settings['normalizeGradientsLocally']: assert not self.settings['normalizeGradientsGlobally']
        self.scaleGrad = None
        if self.settings['normalizeGradientsGlobally']:
            self.appendInitialByScalePoints()
        else:
            self.scaleGrad = 1/(self.paramRanges[:,1] - self.paramRanges[:,0])
        # time spent calculating the last cross_val_predict
        self.lastCvCalcTime = 0
        # time at which the last cross_val_predict was performed
        self.timeOfLastCv = 0
        # collection of time spans spent to generate every new point (for saving to profile info file)
        self.pointGeneratingTimes = []
        self.profilingInfoFile = profilingInfoFile
        self.avgPointCalcTime = None
        self.fitModelTimes = []
        self.errorCalcTimes = Ring(size=10, defaultValue=0)
        self.voronoyTimeMult = Ring(size=10, defaultValue=1)
        self.checkSampleIsGoodFuncTimes = Ring(size=100, defaultValue=0)
        self.scaleForDist = np.linalg.norm([x[1]-x[0] for x in paramRanges])
        self.debug = debug

    def mergeInitialWithExternal(self, external, seed):
        initial = np.append(self.initialPointsCorners(), self.initialPointsIHS(seed), axis=0)
        if external is None:
            return initial
        else:
            initial, _ = geometry.unique_mulitdim(np.concatenate((external, initial), axis=0))
            return initial

    def maxDistanceInInitialYs(self):
        size = self.initial.shape[0]
        maxDist = 0
        for i in range(size):
            for j in range(i + 1, size):
                dist = self.yDist(self.ys[i], self.ys[j])
                if dist > maxDist:
                    maxDist = dist
        return maxDist

    def initialPointsIHS(self, seed):
        N = len(self.paramRanges)
        sampleCount = self.initialIHSDatasetSize
        points = (ihs.ihs(N, sampleCount, seed=seed) - 0.5) / sampleCount  # row - is one point
        for j in range(N):
            points[:, j] = self.leftBorder[j] + points[:, j] * (self.rightBorder[j] - self.leftBorder[j])
        return points

    def initialPointsCorners(self):
        # points = np.dstack(np.array(np.meshgrid(self.paramRanges)).reshape(len(self.paramRanges), -1))[0]
        return np.array([self.leftBorder, self.rightBorder])

    def getInitialByScalePoints(self):
        n = len(self.leftBorder)
        newPoints = np.zeros((2 * n + 1, n))
        c = (self.leftBorder + self.rightBorder) / 2
        newPoints[0] = c
        for i in range(n):
            c1 = np.copy(c)
            c1[i] = self.leftBorder[i]
            newPoints[2 * i + 1, :] = c1
            c1[i] = self.rightBorder[i]
            newPoints[2 * i + 2, :] = c1
        return newPoints

    def appendInitialByScalePoints(self):
        newPoints = self.getInitialByScalePoints()
        self.initial = np.append(self.initial, newPoints, axis=0)
        self.initial, _ = geometry.unique_mulitdim(self.initial)

    def calculateScale(self):
        n = len(self.leftBorder)
        count = 2*n+1
        if self.xs.shape[0] < count: return
        px = self.getInitialByScalePoints()
        py = np.zeros((count, self.ys.shape[1]))
        # find initial points px to calculate scale in the calculated sample xs
        for i in range(count):
            dists = geometry.relDist(self.xs, px[i])
            j = np.argmin(dists)
            if dists[j] > 1e-6: return
            py[i] = self.ys[j]
        c = (self.leftBorder + self.rightBorder) / 2
        assert np.all(px[0] == c)
        grad = np.zeros(n)
        for i in range(n):
            h = (self.rightBorder[i]-self.leftBorder[i]) / 2
            assert h > 0
            grad[i] = np.max([self.yDist(py[2*i+2], py[0])/h, self.yDist(py[0], py[2*i+1])/h])
        grad = grad / np.max(grad)
        eps = self.settings['gradEps']
        if len(grad[grad <= eps]) > 0:
            grad[grad <= eps] = np.min(grad[grad > eps]) * 0.1
        self.scaleGrad = grad
        self.scaleForDist = np.linalg.norm(grad*(self.paramRanges[:,1]-self.paramRanges[:,0]).reshape(-1))

    def getParamBorders(self):
        return self.paramRanges[:,0], self.paramRanges[:,1]

    def getInitialPoints(self):
        m = len(self.initial)
        ind = np.arange(m)
        for i in range(m):
            assert not self.isDublicate(self.initial[i], self.initial[ind!=i, :]), f"{i}\n{self.initial[i]}\n{self.initial}"
            assert not self.isDublicate(self.initial[i]), 'initial =\n'+str(self.initial)+'\n\nxs =\n'+str(self.xs)
        return self.initial

    def extractDatasetPoint(self, dataset, index):
        def isValidY(value):
            if isinstance(value, str) and value == 'calculating':
                return False
            if value is None:
                return False
            return True

        xs, ys, additionalData = dataset
        assert np.all(xs[:len(self.xs), :] == self.xs)
        y_len = None
        for y in ys:
            if not isValidY(y): continue
            if y_len is None: y_len = utils.length(y)
            assert y_len == utils.length(y), f'Func values in different points have different dimensions: {y_len} != {utils.length(y)}'
        if index > len(self.ys)-1:
            assert not self.isDublicate(xs[index]), f'index={index} len(xs)={len(xs)} xs =\n' + str(xs)
            assert len(self.ys) == len(ys) - 1
            assert index == len(ys) - 1
            yDim = utils.length(ys[index]) if isValidY(ys[index]) else 0
            if yDim == 0: yDim = self.ys.shape[1]
            if self.ys.shape[1] == 0 and yDim > 0:
                self.ys = np.zeros((self.ys.shape[0], yDim))
                if self.ys.shape[0] > 0:
                    for i in range(self.ys.shape[0]): self.ys[i] = ys[index]
            self.ys = np.vstack((self.ys, np.zeros(yDim)))
        self.xs = np.copy(xs)
        if isValidY(ys[index]):
            if self.ys.shape[1] == 0:
                self.ys = np.zeros((self.ys.shape[0], utils.length(ys[index])))
            self.ys[index] = ys[index]
            self.notValidYsInd = self.notValidYsInd[self.notValidYsInd != index]
        else:
            if self.ys.shape[1] > 0:
                self.ys[index] = self.predict(xs[index].reshape(1,-1)).reshape(-1)
            if index not in self.notValidYsInd:
                self.notValidYsInd = np.append(self.notValidYsInd, index)
        assert len(self.xs) == len(self.ys)

    def updateDatasetPointInfo(self, dataset, index):
        self.extractDatasetPoint(dataset, index)
        if self.samplingMethod['errorEst'] == 'distance': return
        if index > len(self.ysErrors)-1:
            assert len(self.ysErrors) == len(self.ys) - 1
            assert index == len(self.ys) - 1
            if self.ys.shape[1] > 0:
                self.ysErrors.append(self.getError(self.xs[index, :], LOO=True))
            else:
                self.ysErrors.append(-1)
        else:
            if self.ys.shape[1] > 0:
                self.ysErrors[index] = self.getError(self.xs[index, :], LOO=True)
            else:
                self.ysErrors[index] = -1
        # update broken points
        if self.ys.shape[1] > 0:
            ind = [i for i in range(len(self.ys)) if self.ysErrors[i] < 0]
            for i in ind:
                self.ysErrors[i] = self.getError(self.xs[i, :], LOO=True)
        # update neighbour info
        if len(self.xs) > 1 and self.ys.shape[1] > 0:
            dists = self.xDist(self.xs[index], self.xs)
            # there is zero dist - we take min non zero dist
            minDist = np.max(np.partition(dists, 2-1)[:2])
            neighbInd = np.where(dists<=minDist*2)[0]
            assert index in neighbInd
            for i in neighbInd:
                if i == index: continue
                if i in self.notValidYsInd:
                    self.ys[i] = self.predict(self.xs[i, :].reshape(1,-1))
            for i in neighbInd:
                if i == index: continue
                self.ysErrors[i] = self.getError(self.xs[i, :], LOO=True)
            if self.samplingMethod['errorEst'] == 'model-cv':
                self.fitModel()

    def getMaxErrorPointInds(self):
        ysErrors = np.array(self.ysErrors)
        ind = np.argsort(ysErrors)[::-1]
        maxErrorValue = ysErrors[ind[0]]
        # error of ysError ~ ysError
        maxErrorCount = np.sum(2*ysErrors > maxErrorValue)  # number of points assumed to have max error
        assert maxErrorCount > 0, f'All points in starting dataset were not calculating due to errors:\nX='+str(self.xs)+'\nY='+str(self.ys)
        return ind[:maxErrorCount]

    def getNewPointCandidates(self, samplingMethod, maxPointNum):
        if samplingMethod['candidates'] == 'voronoi':
            if self.settings['samplerTimeConstraint'] is None:
                xNormed = scaleX(self.xs, self.scaleGrad)
                candidates = voronoy(xNormed=xNormed, scaleGrad=self.scaleGrad, xInitial=self.xs)
                candidates = fixOutPoints(candidates, self.paramRanges, throw=False)
            else:
                # get voronoy centers whose neighbourhoods do not intersect
                maxErrorPointInds = list(self.getMaxErrorPointInds())
                # print('voronoyCenterInds before optimization =', len(maxErrorPointInds))
                voronoyCenterInds = copy.deepcopy(maxErrorPointInds)
                while True:
                    dt = self.getAvailableTime() / 3 / len(voronoyCenterInds)
                    dt *= np.mean(self.voronoyTimeMult.data)
                    neighbCount = getMaxNForVoronoyTime(dim=len(self.paramRanges), dt=dt)
                    if neighbCount > self.xs.shape[0]:
                        neighbCount = self.xs.shape[0]
                        break
                    i_center = 0
                    newVoronoyCenterInds = voronoyCenterInds
                    while i_center < len(newVoronoyCenterInds):
                        center = self.xs[voronoyCenterInds[i_center]]
                        dists = self.xDist(center, self.xs)
                        neighbInds = np.argsort(dists)[:neighbCount]
                        # delete centers, included in neighbours
                        voronoyCenterInds1 = newVoronoyCenterInds[:i_center + 1]
                        for i_center1 in range(i_center + 1, len(newVoronoyCenterInds)):
                            if newVoronoyCenterInds[i_center1] not in neighbInds: voronoyCenterInds1.append(newVoronoyCenterInds[i_center1])
                        newVoronoyCenterInds = voronoyCenterInds1
                        i_center += 1
                    if len(newVoronoyCenterInds) >= len(voronoyCenterInds): break
                    voronoyCenterInds = newVoronoyCenterInds
                # print('voronoyCenterInds after optimization =', len(voronoyCenterInds))

                def runVoronoy(pointsNotScaled, dt, center=None):
                    t0 = time.time()
                    candidates = voronoy(scaleX(pointsNotScaled, self.scaleGrad), scaleGrad=self.scaleGrad, xInitial=pointsNotScaled)
                    dtCheck = time.time() - t0
                    if len(candidates) > 0:
                        if len(self.xs) > len(pointsNotScaled): self.voronoyTimeMult.add(dtCheck / dt)
                        candidates = fixOutPoints(candidates, self.paramRanges, throw=not self.settings['fixOut'])
                    if center is not None and len(candidates) > 0:
                        # sort by dist to center
                        dists = self.xDist(center, candidates)
                        ind = np.argsort(dists)
                        candidates = candidates[ind, :]
                    assert len(candidates.shape) == 2 or len(candidates) == 0, str(candidates)
                    return candidates

                if neighbCount == self.xs.shape[0]:
                    # we can build voronoy diagram for the whole sample
                    # print('run voronoy for the whole sample')
                    candidates = runVoronoy(self.xs, dt)
                else:
                    candidates = None
                    for i_center in voronoyCenterInds:
                        x0 = self.xs[i_center]
                        dists = self.xDist(x0, self.xs)
                        neighborsWithCenter = self.xs[np.argsort(dists)[:neighbCount], :]
                        if candidates is None: candidates = runVoronoy(neighborsWithCenter, dt, x0)
                        else:
                            newCand = runVoronoy(neighborsWithCenter, dt, x0)
                            if len(newCand) > 0:
                                candidates = np.append(candidates, newCand, axis=0)
                if len(candidates) > 0:
                    # filter out existing points
                    mask = [not self.isDublicate(x) for x in candidates]
                    candidates = candidates[mask, :]
                # sort candidates by dist to maxErrorPointInds
                cd = self.cdist(candidates, self.xs[maxErrorPointInds,:])
                min_dists = np.min(cd, axis=1)
                ind = np.argsort(min_dists)
                candidates = candidates[ind, :]
                candidates = candidates[:maxPointNum,:]
        elif samplingMethod['candidates'] == 'neighbors':
            maxErrorInds = self.getMaxErrorPointInds()
            maxErrorCount = len(maxErrorInds)
            candidates = np.zeros((maxPointNum, self.xs.shape[1]))
            # take points from neighbourhood of each max error point
            for i in range(maxPointNum):
                maxErrorPoint = self.xs[maxErrorInds[i % maxErrorCount]]
                closestNeighbor, closestDist = self.getClosestNeighbor(maxErrorPoint)
                assert closestDist > 0
                candidates[i] = self.getUniquePointFromRandomDirection(closestNeighbor, closestDist)
        elif samplingMethod['candidates'] == 'random':
            candidates = randomSample(self.paramRanges, self.rng, maxPointNum)
        else:
            assert False, f'Unknown sampling method {samplingMethod}'
        return candidates

    def getNewPointHelper(self, samplingMethod):
        if self.profilingInfoFile is not None:
            start_time = time.time()
        assert self.xs.shape[0] >= len(self.initial), f'{self.xs.shape[0]} < {len(self.initial)}'
        if self.xs.shape[0] >= len(self.initial) and self.settings['normalizeGradientsGlobally'] and self.scaleGrad is None:
            self.calculateScale()

        # if self.debug:
        #     if self.xs.shape[1] == 2 and len(self.xs) % 10 == 0:
        #         self.plotErrorMap(f'graphs/debug/{len(self.xs)}.png')

        if samplingMethod['errorEst'] == 'distance':
            assert samplingMethod['candidates'] == 'random', 'Other than random candidates are impractical for distance error estimation'
            randomPoints = randomSample(self.paramRanges, self.rng, self.xs.shape[0])
            d = self.cdist(randomPoints, self.xs)
            min_d = np.min(d, axis=1)
            ind = np.argmax(min_d)
            newPoint = randomPoints[ind]
        else:
            canPredictCount = self.getCanErrorCalcCount()
            candidates = self.getNewPointCandidates(samplingMethod, maxPointNum=canPredictCount)
            if len(candidates) < canPredictCount:  # it can happen only for Voronoy
                candidates1 = self.getNewPointCandidates(samplingMethod={'candidates':'neighbors', 'errorEst': samplingMethod['errorEst']}, maxPointNum=canPredictCount-len(candidates))
                candidates = np.append(candidates, candidates1, axis=0)
            if samplingMethod['candidates'] not in self.canExpand:
                candidates = np.append(candidates, self.getNewPointHelper(samplingMethod={'candidates':'random', 'errorEst':'distance'}).reshape(1,-1), axis=0)
            if samplingMethod['errorEst'] == 'exact':
                i_max = np.argmax([self.getError(candidate, trueY=self.trueFunc(candidate), calcTime=True) for candidate in candidates])
            else:
                i_max = np.argmax([self.getError(candidate, calcTime=True) for candidate in candidates])
            newPoint = candidates[i_max]

        if self.profilingInfoFile is not None:
            self.pointGeneratingTimes.append(time.time() - start_time)
            np.savetxt(self.profilingInfoFile, self.pointGeneratingTimes)
        assert not self.isDublicate(newPoint)
        return newPoint

    def getNewPoint(self, dataset):
        return self.getNewPointHelper(self.samplingMethod)

    def switchToNoModel(self):
        assert self.samplingMethod['errorEst'] == 'model-cv'
        assert self.autoMethodChoice
        self.samplingMethod['errorEst'] = 'gradient'
        self.errorCalcTimes.data[:] = 0
        print('Switch to gradient error estimation. Sample size =', len(self.xs))

    def getAvailableTime(self):
        if self.settings['samplerTimeConstraint'] is None: return np.inf
        if self.avgPointCalcTime is None: return 1
        return self.settings['samplerTimeConstraint'](self.avgPointCalcTime)

    def getCanErrorCalcCount(self):
        meanErrorCalcTime = np.mean(self.errorCalcTimes.data)
        if meanErrorCalcTime > 0:
            canErrorCalcCount = int(min(self.getAvailableTime() / 3 / meanErrorCalcTime, 1000))
            if canErrorCalcCount == 0: canErrorCalcCount = 1
        else:
            canErrorCalcCount = len(self.errorCalcTimes.data)
        if canErrorCalcCount < 10 and self.samplingMethod['errorEst']=='model-cv' and self.autoMethodChoice:
            self.switchToNoModel()
            canErrorCalcCount = len(self.errorCalcTimes.data)
        # print('sample size =', len(self.xs), 'canErrorCalcCount =', canErrorCalcCount)
        if self.debug: print('meanErrorCalcTime =', meanErrorCalcTime, 'available time =', self.getAvailableTime() / 3, 'canErrorCalcCount =', canErrorCalcCount)
        return canErrorCalcCount

    def getClosestNeighbor(self, x0):
        dists = self.xDist(x0, self.xs)
        sortedDistIndices = np.argsort(dists)
        closestNeighbor = self.xs[sortedDistIndices[1]]
        closestDist = dists[sortedDistIndices[1]]
        assert closestDist > 0
        return closestNeighbor, closestDist

    def isDublicateByDist(self, normedDist):
        return normedDist < 1e-8*self.scaleForDist

    def isDublicate(self, x_not_scaled, xs=None):
        if xs is None: xs = self.xs
        if len(xs) == 0: return False
        d = np.min(self.xDist(x_not_scaled, xs))
        return self.isDublicateByDist(d)

    def getUniquePointFromRandomDirection(self, origin, expectedDist):
        originNormed = scaleX(origin, self.scaleGrad)
        while True:
            direction = randomDirection(self.xs.shape[1], self.rng)
            candidate = direction * expectedDist * self.rng.uniform(low=0.5, high=2.5) + originNormed
            candidate = unscaleX(candidate, self.scaleGrad)
            left, right = self.getParamBorders()
            # candidate = np.clip(candidate, left, right) - results in too many points on boundaries
            if np.all(left <= candidate) and np.all(candidate <= right):
                if not self.isDublicate(candidate):
                    return candidate

    def fitModel(self):
        assert self.samplingMethod['errorEst'] == 'model-cv'
        fitModelTimes = np.array(self.fitModelTimes)
        nonZeroTimes = fitModelTimes[fitModelTimes>0]
        if len(nonZeroTimes) == 0: meanTime = 0
        else: meanTime = np.mean(nonZeroTimes[-5:])
        zeroInd = np.where(fitModelTimes == 0)[0]
        if len(zeroInd) > 0:
            lastZeroCount = len(fitModelTimes) - zeroInd[-1]
        else:
            lastZeroCount = 0
        meanTime = meanTime / (1 + lastZeroCount)
        if self.debug: print('fit model mean time =', meanTime, 'available time =', self.getAvailableTime()/3)
        if meanTime < self.getAvailableTime()/3:
            t0 = time.time()
            estCopy = copy.deepcopy(self.estimator)
            wasFitted = False
            try:
                ys = self.ys
                if self.ys.shape[1] == 1 and self.xs.shape[1] > 1: ys = self.ys.reshape(-1)
                self.estimator.fit(self.xs, ys)
                wasFitted = True
                dt = time.time() - t0
                self.fitModelTimes.append(dt)
                # print('fit model time =', dt)
            except:
                if len(self.xs) > len(self.paramRanges)+1:
                    print('Error while fitting model for points: x=', self.xs, '\ny=',self.ys)
                    print(traceback.format_exc())
                self.estimator = estCopy
                self.fitModelTimes.append(0)
            if wasFitted:
                assert isFitted(self.estimator), 'Ordinary ML.isFitted function doesn\'t correctly work. Provide correct function isFitted(estimator) in adaptive sampling arguments'
        else:
            self.fitModelTimes.append(0)

    def predict(self, x):
        if self.samplingMethod['errorEst'] in ['gradient', 'distance']:
            # predict by 1-NN
            dists = self.xDist(x.reshape(-1), self.xs)
            ind = np.where(dists>0)[0]
            if len(ind) == 0: return np.zeros((1,self.ys.shape[1]))
            i = np.argmin(dists[ind])
            return self.ys[ind[i]]
        else: assert self.samplingMethod['errorEst'] == 'model-cv'
        if isFitted(self.estimator):
            res = self.estimator.predict(x)
            return res
        else:
            return np.zeros((1,self.ys.shape[1]))

    def getError(self, x, LOO=False, trueY=None, calcTime=False):
        if self.samplingMethod['errorEst'] == 'exact':
            assert trueY is not None or LOO
        assert self.samplingMethod['errorEst'] != 'distance'
        if len(self.xs) <= 1: return 0
        t0 = time.time()
        dists = self.xDist(x, self.xs)
        ind = np.argsort(dists)
        sorted_dists = dists[ind]
        dim = len(x)
        if LOO:
            # there is zero dist
            minDist = sorted_dists[1]
            assert sorted_dists[0] <= 1e-8*minDist, 'There is no zero dist and LOO is True'
            assert not self.isDublicateByDist(normedDist=minDist), 'There are duplicates in sample: ind = '+str(ind[:2])+'\n'+str(self.xs[ind[0]])+'\n'+str(self.xs[ind[1]])
            y = self.ys[ind[0]] if trueY is None else trueY
        else:
            minDist = sorted_dists[0]
            y = self.predict(np.array([x])) if trueY is None else trueY
        if minDist == 0: result = 0
        else:
            neighbInd = np.where((0 < dists) & (dists <= minDist*2))[0]
            if len(neighbInd) <= 1: neighbInd = ind[1:3] if LOO else ind[:2]
            grad_y = [self.yDist(y, self.ys[i]) / self.xDist(x, self.xs[i]) for i in neighbInd]
            max_grad_y = np.max(grad_y)
            delta_y = max_grad_y * minDist
            if np.isinf(self.settings['optimizeLp']):
                result = delta_y
            else:
                # near breaks xs gathered in a crowd, so better to optimize Lp
                result = delta_y**self.settings['optimizeLp'] * sorted_dists[min(dim*2, len(dists)-1)]**dim
        dt = time.time() - t0
        if calcTime: self.errorCalcTimes.add(dt)
        return result

    def isGoodEnough(self, dataset):
        try:
            if self.debug: print('checkSampleIsGoodFunc mean time =', np.mean(self.checkSampleIsGoodFuncTimes.data), 'available time =', self.getAvailableTime())
            if np.mean(self.checkSampleIsGoodFuncTimes.data) > self.getAvailableTime():
                self.checkSampleIsGoodFuncTimes.add(0)
                isGood = False
            else:
                t0 = time.time()
                isGood = self.checkSampleIsGoodFunc(dataset)
                self.checkSampleIsGoodFuncTimes.add(time.time()-t0)
        except:
            print('There was error in function checkSampleIsGoodFunc')
            print(traceback.format_exc())
            print('I assume that the sample is not good yet')
            isGood = False
        return isGood

    def xDistDefault(self, x0, xs):
        assert len(x0.shape) == 1, str(x0.shape)
        if len(xs.shape) == 1: xs = xs.reshape(1,-1)
        assert len(xs.shape) == 2, str(xs.shape)
        assert len(x0) == xs.shape[1]
        xNormed = scaleX(x0, self.scaleGrad)
        xsNormed = scaleX(xs, self.scaleGrad)
        return np.linalg.norm(xsNormed - xNormed.reshape(1, -1), axis=1)

    def cdist(self, a, b):
        if len(a.shape) == 1: a = a.reshape(1,-1)
        if len(b.shape) == 1: b = b.reshape(1, -1)
        assert len(a.shape) == 2 and len(b.shape) == 2
        swap = False
        if len(a) > len(b):
            a,b = b,a
            swap = True
        res = np.zeros((len(a), len(b)))
        for i in range(len(a)):
            res[i] = self.xDist(a[i], b)
        if swap: return res.T
        else: return res

    def plotErrorMap(self, fileName):
        assert self.xs.shape[1] == 2
        xsNormed = scaleX(self.xs, self.scaleGrad)
        def func(xNormed):
            x = unscaleX(np.array(xNormed), self.scaleGrad)
            return self.getError(x)
        def plotMoreFunction(ax):
            ax.scatter(xsNormed[:,0], xsNormed[:,1], c=self.ys)
        prNormed = scaleX(self.paramRanges.T, self.scaleGrad)
        plotting.plotHeatMap(func, prNormed[:,0], prNormed[:,1], N1=100, N2=100, cmap='plasma', fileName=fileName, plotMoreFunction=plotMoreFunction)


def nextPoint(X, Y, bounds, **samplerParams):
    """
    Return next point x to add to the sample X

    :param X: 2D array (row - one point)
    :param Y: 1D or 2D array (row - one value, may be multidimentional)
    :param bounds: array of coordinate intervals [a,b]
    :returns: next point x (1D array)
    """
    if isinstance(X, list): X = np.array(X)
    if isinstance(Y, list): Y = np.array(Y)
    if len(X.shape) == 1:
        print('Assume, that X is column, but not row')
        X = X.reshape(-1,1)
    if len(Y.shape) == 1: Y = Y.reshape(-1, 1)
    if isinstance(bounds, list): bounds = np.array(bounds)
    if len(bounds.shape) == 1: bounds = bounds.reshape(1, 2)
    if 'samplerTimeConstraint' not in samplerParams:
        samplerParams['samplerTimeConstraint'] = 1
    initial = ErrorPredictingSampler(bounds, checkSampleIsGoodFunc=lambda d: True, **samplerParams).initial
    sampler = ErrorPredictingSampler(bounds, initialDataset=X, checkSampleIsGoodFunc=lambda dataset: len(dataset[0]) >= len(X), **samplerParams)

    def func(x):
        for i in range(len(X)):
            if x == X[i]: return Y[i]
        assert False, "Calculate function for all the initial points:\n"+str(initial)
    orchestrator = CalculationOrchestrator(func)
    generator = DatasetGenerator(sampler, orchestrator)
    generator.generate()
    return generator.getNewPoint()


def checkSampleIsGoodByCVError(maxError, estimator=None, minCountToCheckError=10, cvCount=10, maxSampleSize=None, debug=False):
    """
    Returns function to pass in sampleAdaptively as checkSampleIsGoodFunc argument
    :param maxError: max cv error for sample to be good
    :param estimator: sklearn-compatible estimator
    :param minCountToCheckError: min sample size to evaluate CV
    :param cvCount: CV count
    :param maxSampleSize: when sample size > maxSampleSize we assume that it is good
    :param debug: print debug info
    """
    if estimator is None:
        estimator = RBF(function='linear', baseRegression='quadric', scaleX=True, removeDublicates=True)

    def checkSampleIsGood(dataset):
        xs, ys, additionalData = dataset
        if len(xs) < minCountToCheckError: return False
        if maxSampleSize is not None and len(xs) >= maxSampleSize: return True
        res = crossValidation(estimator, xs, ys, cvCount, nonUniformSample=True)
        relToConstPredErrorCV = res[0]
        if debug: print(f'sampleSize = {len(xs)}  relToConstPredError = 1-R^2 = {relToConstPredErrorCV}')
        return relToConstPredErrorCV <= maxError
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


def minlpe(function, paramRanges, folder, maxError=0.01, maxSampleSize=None, samplingMethod='auto', samplerTimeConstraint='default', batchSize=1, threadingLock=None, seed=None, estimator=None, initialIHSDatasetSize=None, optimizeLp=2, xDist=None, yDist=None, normalizeGradientsGlobally=True, debug=False):
    """
    Run adaptive sampling
    :param function: function to sample. Argument and value are numpy arrays. Can return additional data: return value, additionalData
    :param paramRanges: list of pairs [leftBorder, rightBorder]
    :param folder: folder to store settings and result. If the folder is not empty, adaptive sampling would continue calculation. Delete the folder to start new sampling.
    :param maxError: CV-error to stop sampling. If it is callable i.e. function(dataset), it is assumed that it returns True when we should stop calculation. dataset = (xs, ys, additionalData)
    :param maxSampleSize: max sample size to stop sampling
    :param samplingMethod: Defines two methods: how to choose candidates for error maximization and how to estimate error.
        dict. Example: {'candidates':'neighbors', 'errorEst':'model-cv'}. Use 'auto' for auto switching between methods.
        Candidate methods:
            'voronoi' - all voronoi vertices (if samplerTimeConstraint=None - for all sample, otherwise - for neighbors of max error points)
            'neighbors' - random candidates in neighbourhood of max error points
            'random' - take multiple random points and choose best (error is not used, distance to other points is not used)
        Error estimation methods:
            'model-cv' - use model CV-error estimate
            'gradient' - use gradient approximation and distance to estimate error. Model is not used
            'distance' - search the most remote point from neighbors
    :param samplerTimeConstraint: float or func(onePointCalcTime). Defines how much time can be used for new point sampling (None - no constraints). Default:  max(min(onePointCalcTime/10,6)*10, 0.1)
    :param batchSize: batch size for parallel sampling
    :param threadingLock: threading.Lock() for parallel sampling. It should be unique for sampler and user function. You need it only when your function is unsafe
    :param seed: if None take current timestamp
    :param estimator: scikit-learn compatible estimator to estimate approximation error for CV-error estimation. Be careful: theory requires the estimator to be "local" i.e. estimator(X_{l+1}) differs from estimator(X_l) only in the neighbourhood of added point
    :param initialIHSDatasetSize: size of initial dataset, generated by IHS
    :param optimizeLp: p in (0, np.inf] to minimize error: ||yTrue - yPred||_Lp
    :param xDist: user defined distance for x. xDist(x0, xs) returns distance array from point x0 to each point of array xs. Default: np.linalg.norm(x0-xs)
    :param yDist: user defined value comparison. yDist(y1, y2) returns distance between two function values: y1 and y2. Default: np.linalg.norm(y1-y2, ord=np.inf)
    :param normalizeGradientsGlobally: normalize x to equalize partial derivatives of function
    :param debug: if True print debug info
    :return: (xs, ys, additionalData). x,y - 2d numpy arrays, each row is one point. additionalData - list of additional data returned by function
    """
    if threadingLock is None: threadingLock = threading.Lock()
    if estimator is None: estimator = RBF(function='linear', baseRegression='quadric', scaleX=True, removeDublicates=False)
    os.makedirs(folder, exist_ok=True)
    if os.path.exists(folder+os.sep+'dataset.json'):
        with open(folder+os.sep+'dataset.json') as json_file: existingDataset = json.load(json_file)
    else: existingDataset = None
    settingsFileName = folder+os.sep+'settings.json'
    paramRangesDict = {f'x{i+1}':paramRanges[i] for i in range(len(paramRanges))}
    if os.path.exists(settingsFileName):
        seed = ensureSettingsAreConsistent(settingsFileName, seed, paramRangesDict)
    else:
        seed = writeSettings(settingsFileName, seed, paramRangesDict)
    if callable(maxError): checkSampleIsGoodFunc = maxError
    else: checkSampleIsGoodFunc = checkSampleIsGoodByCVError(maxError=maxError, estimator=estimator, minCountToCheckError=max(len(paramRanges)*2,10), maxSampleSize=maxSampleSize, debug=debug)

    def checkSampleIsGoodFuncWrapper(dataset):
        if existingDataset is not None:
            assert len(dataset[0]) >= len(existingDataset[0]), f'{len(dataset[0])} < {len(existingDataset[0])}'
        utils.saveData(dataset, folder+os.sep+'dataset.json')
        return checkSampleIsGoodFunc(dataset)
    sampler = ErrorPredictingSampler(paramRanges, checkSampleIsGoodFunc=checkSampleIsGoodFuncWrapper, seed=seed, samplingMethod=samplingMethod, samplerTimeConstraint=samplerTimeConstraint, estimator=estimator, initialIHSDatasetSize=initialIHSDatasetSize, optimizeLp=optimizeLp, xDist=xDist, yDist=yDist, normalizeGradientsGlobally=normalizeGradientsGlobally, debug=debug)
    if existingDataset is not None and len(existingDataset[0]) >= len(sampler.initial):
        existingDatasetGetter = lambda: existingDataset
    else:
        existingDatasetGetter = None
        existingDataset = None
    orchestrator = CalculationOrchestrator(function, threadingLock, calcSampleInParallel=batchSize, existingDatasetGetter=existingDatasetGetter, debug=debug)
    generator = DatasetGenerator(sampler, orchestrator)
    xs, ys, additionalData = generator.generate()
    return xs, ys, additionalData

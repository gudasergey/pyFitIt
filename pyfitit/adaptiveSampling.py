import copy
import numpy as np
import time, threading, scipy, traceback
from scipy.spatial import distance
from scipy.optimize import basinhopping
from .ML import RBF, isFitted
from . import utils, ihs, geometry
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
        return self.xs, self.ys

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
    def __init__(self, program, calcSampleInParallel=1, existingDatasetGetter=None):
        self.calcSampleInParallel = calcSampleInParallel
        self.program = program
        self.generator = None
        self.existingDatasetGetter = existingDatasetGetter
        self.lock = threading.Lock()
        self.calcTimes = []

    def run(self, datasetGenerator):
        self.generator = datasetGenerator

        self.loadExistingDataset()

        if self.calcSampleInParallel > 1:
            self.runParallel()
        else:
            self.runSerialized()

    def addResult(self, x, res):
        if hasattr(res, "__len__"):
            self.generator.addResult(x, y=res[0], additionalData=res[1])
        else:
            self.generator.addResult(x, res)

    def runSerialized(self):

        def calculateAndAdd(x):
            res = self.calcAndUpdateTime(x)
            self.addResult(x, res)

        if self.existingDatasetGetter is None:
            for x in self.generator.getInitialPoints():
                calculateAndAdd(x)
                # call to plot user defined graphs
                # self.generator.isDatasetReady()

        while not self.generator.isDatasetReady():
            x = self.generator.getNewPoint()
            calculateAndAdd(x)

    def runParallel(self):
        # calculating initial dataset
        if self.existingDatasetGetter is None:
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
        # from concurrent.futures import ThreadPoolExecutor as LazyThreadPoolExecutor
        pool = LazyThreadPoolExecutor(self.calcSampleInParallel)
        results = pool.map(self.calculate, points)
        for index, y, additionalData in results:
            with self.lock:
                self.generator.updateResult(index, y, additionalData)

    def pointSequence(self):
        with self.lock:
            isReady = self.generator.isDatasetReady()
        while not isReady:
            with self.lock:
                x = self.generator.getNewPoint()
                index = self.generator.addResult(x, 'calculating')
            yield index, x
            with self.lock:
                isReady = self.generator.isDatasetReady()

    def calculate(self, pointAndIndex):
        index, x = pointAndIndex
        y, additionalData = self.calcAndUpdateTime(x)
        return index, y, additionalData

    def calcAndUpdateTime(self, x):
        t0 = time.time()
        res = self.program.calculate(x)
        dt = time.time() - t0
        with self.lock:
            self.calcTimes.append(dt)
            self.generator.sampler.avgCalcTime = np.mean(self.calcTimes)
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
        for x, res in dataset:
            self.addResult(x, res)


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


def getExploitByNeighbourVoronoi(xs, settings, paramRanges, yPredictionModel, yDist, fixOut, ysTrue, ysPred, scaleGrad=None):
    exploitNum = settings['exploitNum']
    if exploitNum <= 0: return []
    exploitNeighborsNum = xs.shape[1]
    predDist = []
    for i in range(ysTrue.shape[0]):
        predDist.append(yDist(ysTrue[i], ysPred[i]))
    exploitOrigins = xs[np.argsort(predDist)[-exploitNum:]]

    pointDists = distance.cdist(exploitOrigins, xs)
    exploitPoints = []
    for origin, dists in zip(exploitOrigins, pointDists):
        sorted_ind_dists = np.argsort(dists)
        # исправить!!!!! Точки кучкуются с одной стороны. Нужно поменять это на вызов функции getNeighbUniformByDirection(xs, кол-во)
        # которая работает по следующему алгоритму: берет количество соседей с запасом (все брать нельзя, чтобы понятие 'соседей сохранилось' - это настроечный параметр (множитель dim)). Потом по очереди удаляем из двух точек с минимальным косинусом ту, которая дальше от центра.
        originNeighborsWithCenter = xs[sorted_ind_dists[:exploitNeighborsNum * 5]]
        if settings['normalizeGradientsLocally']:
            originNeighborsWithCenterNormed, max_grad = normalizeGradients(originNeighborsWithCenter, yPredictionModel, paramRanges, yDist, settings['gradEps'])
        else:
            originNeighborsWithCenterNormed = scaleX(originNeighborsWithCenter, scaleGrad)
        vertices = voronoy(originNeighborsWithCenterNormed, paramRanges, fixOut=fixOut, scaleGrad=scaleGrad, xInitial=originNeighborsWithCenter)
        if len(vertices) == 0: continue
        vor_dists = np.linalg.norm(origin.reshape(1,-1)-vertices, axis=1)
        # take only Voronoi vertices close to the origin
        # это источник наших проблем. Если размерность = 1, то exploitNeighborsNum=1, и точки кучкуются. Может так случится, что они кучкуются с одной стороны от некоторой, а с другой - огромная пропасть!!! Т.е. мы уничтожаем преимущества, которые нам дают диаграммы Вороного.
        maxDist = dists[sorted_ind_dists[exploitNeighborsNum * 2]] if len(dists) >= exploitNeighborsNum * 2 + 1 else dists[-1]
        exploitPointCandidates = vertices[vor_dists <= maxDist, :]
        if len(exploitPointCandidates) == 0:
            exploitPointCandidates = [vertices[np.argmin(vor_dists)]]
        # take point, which is the most distant from existed in sample
        candidateDists = distance.cdist(exploitPointCandidates, xs)
        bestCand = exploitPointCandidates[np.argmax(np.min(candidateDists, axis=1))]
        exploitPoints.append(bestCand)
        # for v in vertices: exploitPoints.append(v)
    return exploitPoints


def calcDistsFromPointToSampleNormed(x0, xs, scaleGrad):
    xNormed = scaleX(x0, scaleGrad)
    xsNormed = scaleX(xs, scaleGrad)
    return np.linalg.norm(xsNormed - xNormed.reshape(1, -1), axis=1)


class ErrorPredictingSampler(Sampler):

    def __init__(self, paramRanges, estimator=None, initialPoints=None, exploreNum=5, exploitNum=5, optimizeLp=2, seed=None, checkSampleIsGoodFunc=None, yDist=None, normalizeGradientsGlobally=True, normalizeGradientsLocally=False, fixOut=False, samplerTimeConstraint='default', profilingInfoFile=None, samplingMethod='max-error-neighbor', trueFunc=None, noModel='auto'):
        """

        :param paramRanges: list of pairs [leftBorder, rightBorder]
        :param initialPoints: starting dataset
        :param exploreNum:
        :param exploitNum:
        :param optimizeLp: p in (0, np.inf] to minimize error: ||yTrue - yPred||_Lp
        :param seed:
        :param checkSampleIsGoodFunc: function(dataset), which returns True when we should stop calculation. dataset = (xs, ys, additionalData)
        :param yDist: function(y1, y2) returns distance between y1 and y2. If None use np.linalg.norm(y1-y2, ord=np.inf)
        :param samplerTimeConstraint: dict. Defines how much time can be used for new point sampling (None - no constraints). {'relative':0.3} - relative to the function calculation procedure. {'absolute':5} time in seconds for new point sampling (not including function calculation time). You can combine constrains together: {'relative':0.3, 'absolute':5, 'combine':'max' or 'min'}. If None - perform the most precise point selection. Default: {'relative':0.1, 'absolute':1, 'combine':'max'}
        :param noModel: True/False/'auto' - fast error calculation. 'auto' means auto switch for large sample
        """
        assert samplingMethod in ['voronoi', 'random', 'random-remote', 'max-error-global', 'max-error-exact', 'max-error-neighbor']
        self.samplingMethod = samplingMethod
        # methods, that can easily expand sample out of initial sample bounds
        self.canExpand = ['random', 'random-remote', 'max-error-global', 'max-error-exact']
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
        self.initial = self.mergeInitialWithExternal(initialPoints, seed)
        self.initialMaxDist = None
        self.checkSampleIsGoodFunc = checkSampleIsGoodFunc
        if yDist is None: yDist = lambda y1, y2: np.linalg.norm(y1-y2, ord=np.inf)
        self.yDist = yDist
        if dim == 1: normalizeGradientsGlobally = False
        if samplerTimeConstraint is not None:
            if isinstance(samplerTimeConstraint, str): samplerTimeConstraint = {'relative':0.1, 'absolute':1, 'combine':'max'}
            assert set(samplerTimeConstraint.keys()) <= {'relative', 'absolute', 'combine'}
            assert len(samplerTimeConstraint) in [1,3]
            if 'combine' in samplerTimeConstraint:
                assert len(samplerTimeConstraint) == 3
                assert samplerTimeConstraint['combine'] in ['min', 'max']
        self.settings = {
            'exploreNum': exploreNum,
            'exploitNum': exploitNum,
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
        self.trueErrorRatio = Ring(size=20, defaultValue=-1)
        self.scaleForDist = np.linalg.norm([x[1]-x[0] for x in paramRanges])
        if isinstance(noModel, str):
            assert noModel == 'auto'
            self.noModel = False
            self.canSwitchToNoModel = True
        else:
            self.noModel = noModel
            self.canSwitchToNoModel = False

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
        sampleCount = N + 1
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
            assert not self.isDublicate(self.initial[i], self.initial[ind!=i, :])
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
        if index > len(self.ys)-1:
            assert not self.isDublicate(xs[index]), f'index={index} len(xs)={len(xs)} xs =\n' + str(xs)
            assert len(self.ys) == len(ys) - 1
            assert index == len(ys) - 1
            yDim = ys[index].size if isValidY(ys[index]) else 0
            if yDim == 0: yDim = self.ys.shape[1]
            if self.ys.shape[1] == 0 and yDim > 0:
                self.ys = np.zeros((self.ys.shape[0], yDim))
                if self.ys.shape[0] > 0:
                    for i in range(self.ys.shape[0]): self.ys[i] = ys[index]
            self.ys = np.vstack((self.ys, np.zeros(yDim)))
        self.xs = np.copy(xs)
        if isValidY(ys[index]):
            if self.ys.shape[1] == 0:
                self.ys = np.zeros((self.ys.shape[0], ys[index].size))
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
        if index > len(self.ysErrors)-1:
            assert len(self.ysErrors) == len(self.ys) - 1
            assert index == len(self.ys) - 1
            if self.ys.shape[1] > 0:
                exactError = self.getError(self.xs[index, :], LOO=True)
                approx_y = self.predict(self.xs[index, :].reshape(1,-1))
                approxError = self.getError(self.xs[index, :], LOO=True, trueY=approx_y)
                if approxError > 0:
                    self.trueErrorRatio.add(exactError / approxError)
                self.ysErrors.append(exactError)
            else:
                self.ysErrors.append(-1)
        else:
            if self.ys.shape[1] > 0:
                exactError = self.getError(self.xs[index, :], LOO=True)
                self.trueErrorRatio.add(exactError / self.ysErrors[index])
                self.ysErrors[index] = exactError
            else:
                self.ysErrors[index] = -1
        # update broken points
        if self.ys.shape[1] > 0:
            ind = [i for i in range(len(self.ys)) if self.ysErrors[i] < 0]
            for i in ind:
                self.ysErrors[i] = self.getError(self.xs[i, :], LOO=True)
        # update neighbour info
        if len(self.xs) > 1 and self.ys.shape[1] > 0:
            dists = calcDistsFromPointToSampleNormed(self.xs[index], self.xs, self.scaleGrad)
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
            if not self.noModel:
                self.fitModel()

    def getMaxErrorPointInds(self):
        ysErrors = np.array(self.ysErrors)
        ind = np.argsort(ysErrors)[::-1]
        maxErrorValue = ysErrors[ind[0]]
        if any(self.trueErrorRatio.data > 0):
            trueErrorRatio = np.median(self.trueErrorRatio.data[self.trueErrorRatio.data > 0])
        else: trueErrorRatio = 1
        # error of ysError ~ ysError*trueErrorRatio
        maxErrorCount = np.sum(ysErrors + ysErrors * trueErrorRatio > maxErrorValue)  # number of points assumed to have max error
        return ind[:maxErrorCount]

    def getNewPointCandidates(self, samplingMethod, maxPointNum):
        if samplingMethod == 'voronoi':
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
                        dists = calcDistsFromPointToSampleNormed(center, self.xs, self.scaleGrad)
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
                        dists = calcDistsFromPointToSampleNormed(center, candidates, self.scaleGrad)
                        ind = np.argsort(dists)
                        candidates = candidates[ind, :]
                    return candidates

                if neighbCount == self.xs.shape[0]:
                    # we can build voronoy diagram for the whole sample
                    # print('run voronoy for the whole sample')
                    candidates = runVoronoy(self.xs, dt)
                else:
                    candidates = None
                    for i_center in voronoyCenterInds:
                        x0 = self.xs[i_center]
                        dists = calcDistsFromPointToSampleNormed(x0, self.xs, self.scaleGrad)
                        neighborsWithCenter = self.xs[np.argsort(dists)[:neighbCount], :]
                        if candidates is None: candidates = runVoronoy(neighborsWithCenter, dt, x0)
                        else: candidates = np.append(candidates, runVoronoy(neighborsWithCenter, dt, x0), axis=0)
                if len(candidates) > 0:
                    # filter out existing points
                    mask = [not self.isDublicate(x) for x in candidates]
                    candidates = candidates[mask, :]
                # sort candidates by dist to maxErrorPointInds
                candidatesNormed = scaleX(candidates, self.scaleGrad)
                maxErrorPointsNormed = scaleX(self.xs[maxErrorPointInds, :], self.scaleGrad)
                cd = distance.cdist(candidatesNormed, maxErrorPointsNormed)
                min_dists = np.min(cd, axis=1)
                ind = np.argsort(min_dists)
                candidates = candidates[ind, :]
                candidates = candidates[:maxPointNum,:]
        elif samplingMethod == 'max-error-neighbor':
            maxErrorInds = self.getMaxErrorPointInds()
            maxErrorCount = len(maxErrorInds)
            candidates = np.zeros((maxPointNum, self.xs.shape[1]))
            # take points from neighbourhood of each max error point
            for i in range(maxPointNum):
                maxErrorPoint = self.xs[maxErrorInds[i % maxErrorCount]]
                closestNeighbor, closestDist = self.getClosestNeighbor(maxErrorPoint)
                assert closestDist > 0
                candidates[i] = self.getUniquePointFromRandomDirection(closestNeighbor, closestDist)
        elif samplingMethod in ['max-error-global', 'max-error-exact']:
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

        if samplingMethod == 'random':
            newPoint = np.apply_along_axis(lambda x: self.rng.uniform(low=x[0], high=x[1], size=1), arr=self.getParamBorders(), axis=0)[0]
        elif samplingMethod == 'random-remote':
            randomPoints = randomSample(self.paramRanges, self.rng, self.xs.shape[0])
            d = distance.cdist(randomPoints, self.xs)
            min_d = np.min(d, axis=1)
            ind = np.argmax(min_d)
            newPoint = randomPoints[ind]
        # elif samplingMethod == 'voronoi':
        #     newPoint = self.getFromNeighbourVoronoi()
        else:
            canPredictCount = self.getCanErrorCalcCount()
            candidates = self.getNewPointCandidates(samplingMethod, maxPointNum=canPredictCount)
            if len(candidates) < canPredictCount:  # it can happen only for Voronoy
                candidates1 = self.getNewPointCandidates(samplingMethod='max-error-neighbor', maxPointNum=canPredictCount-len(candidates))
                candidates = np.append(candidates, candidates1, axis=0)
            if samplingMethod not in self.canExpand:
                candidates = np.append(candidates, self.getNewPointHelper(samplingMethod='random-remote').reshape(1,-1), axis=0)
            if samplingMethod == 'max-error-exact':
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
        assert not self.noModel
        assert self.canSwitchToNoModel
        self.noModel = True
        self.trueErrorRatio.data[:] = -1
        self.errorCalcTimes.data[:] = 0
        print('Switch to no model error estimation. Sample size =', len(self.xs))

    def getAvailableTime(self):
        maxTime = np.inf
        if self.settings['samplerTimeConstraint'] is None: return np.inf
        absoluteMaxTime = np.inf
        if 'absolute' in self.settings['samplerTimeConstraint']:
            absoluteMaxTime = self.settings['samplerTimeConstraint']['absolute']
            maxTime = absoluteMaxTime
        relMaxTime = np.inf
        if 'relative' in self.settings['samplerTimeConstraint']:
            rel = self.settings['samplerTimeConstraint']['relative']
            assert rel > 0
            if self.avgPointCalcTime is not None:
                relMaxTime = rel*self.avgPointCalcTime
                maxTime = relMaxTime
        if 'combine' in self.settings['samplerTimeConstraint']:
            comb = self.settings['samplerTimeConstraint']['combine']
            if comb == 'min':
                maxTime = min(absoluteMaxTime, relMaxTime)
            else:
                maxTime = max(absoluteMaxTime, relMaxTime)
        return maxTime

    def getCanErrorCalcCount(self):
        meanErrorCalcTime = np.mean(self.errorCalcTimes.data)
        if meanErrorCalcTime > 0:
            canErrorCalcCount = int(min(self.getAvailableTime() / 3 / meanErrorCalcTime, 1000))
            if canErrorCalcCount == 0: canErrorCalcCount = 1
        else:
            canErrorCalcCount = len(self.errorCalcTimes.data)
        if canErrorCalcCount < 10 and not self.noModel and self.canSwitchToNoModel:
            self.switchToNoModel()
            canErrorCalcCount = len(self.errorCalcTimes.data)
        # print('sample size =', len(self.xs), 'canErrorCalcCount =', canErrorCalcCount)
        return canErrorCalcCount

    def getClosestNeighbor(self, x0):
        dists = calcDistsFromPointToSampleNormed(x0, self.xs, self.scaleGrad)
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
        d = np.min(np.linalg.norm(scaleX(x_not_scaled, self.scaleGrad).reshape(1,-1) - scaleX(xs, self.scaleGrad), axis=1))
        return self.isDublicateByDist(d)

    def getUniquePointFromRandomDirection(self, origin, expectedDist):
        originNormed = scaleX(origin, self.scaleGrad)
        while True:
            direction = randomDirection(self.xs.shape[1], self.rng)
            candidate = direction * expectedDist * self.rng.uniform(low=0.5, high=2.5) + originNormed
            candidate = unscaleX(candidate, self.scaleGrad)
            left, right = self.getParamBorders()
            candidate = np.clip(candidate, left, right)
            if not self.isDublicate(candidate):
                return candidate

    def fitModel(self):
        assert not self.noModel
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
        if meanTime < self.getAvailableTime()/3:
            t0 = time.time()
            estCopy = copy.deepcopy(self.estimator)
            try:
                self.estimator.fit(self.xs, self.ys)
                dt = time.time() - t0
                self.fitModelTimes.append(dt)
                # print('fit model time =', dt)
            except:
                if len(self.xs) > len(self.paramRanges)+1:
                    print('Error while fitting model for points: x=', self.xs, '\ny=',self.ys)
                    print(traceback.format_exc())
                self.estimator = estCopy
                self.fitModelTimes.append(0)
        else:
            self.fitModelTimes.append(0)

    def predict(self, x):
        if self.noModel:
            # predict by 1-NN
            dists = calcDistsFromPointToSampleNormed(x, self.xs, self.scaleGrad)
            ind = np.where(dists>0)[0]
            if len(ind) == 0: return np.zeros((1,self.ys.shape[1]))
            i = np.argmin(dists[ind])
            return self.ys[ind[i]]
        if isFitted(self.estimator):
            res = self.estimator.predict(x)
            return res
        else:
            return np.zeros((1,self.ys.shape[1]))

    def getError(self, x, LOO=False, trueY=None, calcTime=False):
        if len(self.xs) <= 1: return 0
        t0 = time.time()
        dists = calcDistsFromPointToSampleNormed(x, self.xs, self.scaleGrad)
        ind = np.argsort(dists)
        sorted_dists = dists[ind]
        dim = len(x)
        # if calcTime and self.samplingMethod != 'max-error-exact' and self.noModel:
        #     firstNonZero = np.where(sorted_dists>0)[0][0]
        #     if trueY is not None:
        #         y0 = trueY
        #         x0 = x
        #     elif LOO:
        #         assert firstNonZero>0
        #         y0 = self.ys[ind[0]]
        #         x0 = self.xs[ind[0]]
        #     else:
        #         y0 = self.ys[ind[firstNonZero+1]]
        #         x0 = self.xs[ind[firstNonZero+1]]
        #     grad_y = self.yDist(self.ys[ind[firstNonZero]], y0) / np.linalg.norm(self.xs[ind[firstNonZero]]-x0)
        #     dt = time.time() - t0
        #     if calcTime: self.errorCalcTimes.add(dt)
        #     return (sorted_dists[firstNonZero] * grad_y)**self.settings['optimizeLp'] * sorted_dists[min(dim*2, len(dists)-1)]**dim
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
            grad_y = [self.yDist(y, self.ys[i]) / np.linalg.norm(x-self.xs[i]) for i in neighbInd]
            max_grad_y = np.max(grad_y)
            delta_y = max_grad_y * minDist
            if np.isinf(self.settings['optimizeLp']):
                result = delta_y
            else:
                # near breaks xs gathered in a crowd, so better to optimize Lp
                result = delta_y**self.settings['optimizeLp'] * sorted_dists[min(dim*2, len(dists)-1)]**dim
            # dy = [self.yDist(y, self.ys[i]) for i in neighbInd]
            # if np.isinf(self.settings['optimizeLp']):
            #     result = np.max(dy)
            # else:
            #     # near breaks xs gathered in a crowd, so better to optimize Lp
            #     result = np.max(dy)**self.settings['optimizeLp'] * sorted_dists[min(dim*2, len(dists)-1)]**dim
        dt = time.time() - t0
        if calcTime: self.errorCalcTimes.add(dt)
        return result

    def isGoodEnough(self, dataset):
        try:
            isGood = self.checkSampleIsGoodFunc(dataset)
        except:
            print('There was error in function checkSampleIsGoodFunc')
            print(traceback.format_exc())
            print('I assume that the sample is not good yet')
            isGood = False
        return isGood

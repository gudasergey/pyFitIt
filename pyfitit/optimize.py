import random, copy, json, scipy, scipy.optimize, warnings, os, itertools
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
import matplotlib.pyplot as plt
from . import utils, plotting


def arg2string(arg, paramNames=None):
    s = ''
    if paramNames is None:
        for a in arg:
            s += a['paramName'] + ('=%.5g' % a['value'])
            if a != arg[-1]: s += '  '
    else:
        for i in range(len(arg)):
            s += paramNames[i] + ('=%.5g' % arg[i])
            if i != len(arg)-1: s += '  '
    return s


def arg2csv(arg, sep=' '):
    s = ''
    for a in arg:
        s += '%.5g' % a['value']
        if a != arg[-1]: s += sep
    return s


def arg2json(arg):
    s = {}
    for a in arg:
        assert a['paramName'] not in s, 'Duplicate param '+a['paramName']
        s[a['paramName']] = a['value']
    return json.dumps(s)


def param(paramName, value, borders, step, minStep, isInt = False):
    a = borders[0]; b = borders[1]
    assert  a<=value and value<=b, 'Error, for '+paramName+' not true: '+str(a)+'<='+str(value)+'<='+str(b)
    return {'paramName':paramName, 'value':value, 'leftBorder':a, 'rightBorder':b, 'step':step, 'minStep':minStep, 'isInt':isInt}


class VectorPoint:
    def __init__(self, paramList):
        self.params = copy.deepcopy(paramList)
        self.check()

    def findParamByIndOrName(self, indOrName):
        if isinstance(indOrName, str): 
            paramName = indOrName
            count = 0
            for i in range(len(self.params)):
                if self.params[i]['paramName'] == paramName:
                    p = self.params[i]
                    count += 1
            if count == 0: return None
            assert count==1, "Multiple params: "+paramName
            return p
        else: return self.params[indOrName]

    def __getitem__(self, i):
        p = self.findParamByIndOrName(i)
        if p is None: return None
        if isinstance(i, str): return p['value']
        else: return p

    def __setitem__(self, i, value):
        assert value is not None
        p = self.findParamByIndOrName(i)
        assert p is not None, "param = "+str(i)+" not found"
        if isinstance(value, dict):
            assert value['value'] is not None
            for n in ['value', 'leftBorder', 'rightBorder', 'step', 'minStep', 'isInt']:
                p[n] = value[n]
        else: p['value'] = value
        self.check()

    def append(self, param):
        p = self.findParamByIndOrName(param['paramName'])
        if p is not None: self[param['paramName']] = param
        else: self.params.append(param)
        self.check()

    def increaseParamWithBounds(self, indOrName, toAdd):
        p = self.findParamByIndOrName(indOrName)
        assert p is not None
        for n in ['value', 'leftBorder', 'rightBorder']: p[n] += toAdd
        self.check()

    def setAndExpandInterval(self, i, value):
        assert value is not None
        p = self.findParamByIndOrName(i)
        assert p is not None, "param = "+str(i)+" not found"
        assert not isinstance(value, dict)
        p['value'] = value
        if value<p['leftBorder']: p['leftBorder'] = value
        if value>p['rightBorder']: p['rightBorder'] = value
        self.check()

    def __len__(self): return len(self.params)

    def __str__(self):
        return arg2string(self.params)

    def to_dict(self):
        return dict([[p['paramName'], p['value']] for p in self.params])

    def check(self):
        for i in range(len(self.params)):
            a = self.params[i]['leftBorder']
            b = self.params[i]['rightBorder']
            v = self.params[i]['value']
            # print(self.params[i])
            assert  a<=v and v<=b, 'Error, for '+self.params[i]['paramName']+' not true: '+str(a)+'<='+str(v)+'<='+str(b)

    # def __deepcopy__(self, memo):
    #     return VectorPoint(self.params)

    # def __copy__(self):
    #     return VectorPoint(self.params)


def minimizePokoordOneDim(f, lastFuncValOneDim, x0, a, b, step, minStep, minFuncDelta, isInteger, paramName, enableOutput):
    # проверяем на локальный минимум и зависимость от параметра
    # делаем минимальные шаги в обе стороны
    initialLastFuncValOneDim = lastFuncValOneDim
    centrF = lastFuncValOneDim
    argLeft = x0
    if argLeft-minStep >= a:
        argLeft = argLeft-minStep
        leftF = f(argLeft)
        if centrF[0] == leftF[0]:
            if enableOutput: print('Функция не зависит от аргумента '+paramName+' df = 0')
            fmin = centrF; xmin = x0; return fmin,xmin # функция не зависит от аргумента
    else: leftF = centrF
    if leftF[0] >= centrF[0]: # пытаемся сделать шаг вправо
        argRight = x0
        if argRight+minStep <= b:
            argRight = argRight+minStep
            rightF = f(argRight)
        else: rightF = centrF
        if rightF[0] < centrF[0] :
            kudaIdem = -1; newFuncValOneDim = rightF; newArgi = argRight #вправо
        else:
            if enableOutput: print('Неудача при попытках смещения аргумента '+paramName)
            fmin = centrF; xmin = x0; return fmin,xmin
    else:
        kudaIdem = 1; newFuncValOneDim = leftF; newArgi = argLeft #влево
    assert newFuncValOneDim <= initialLastFuncValOneDim, 'We should minimize function, but new='+str(newFuncValOneDim)+' init='+str(initialLastFuncValOneDim)
    # проверяем существенно ли изменение функции (будет ли функция меняться более чем на minFuncDelta при большом шаге)
    if (abs(newFuncValOneDim[0]-centrF[0]) / minStep) * step < minFuncDelta:
        fmin = centrF; xmin = x0;
        assert fmin[0]<=initialLastFuncValOneDim[0], 'We should minimize function'
        if enableOutput: print('Функция слабо зависит от аргумента '+paramName+' df = '+str(fmin[0]-initialLastFuncValOneDim[0]))
        return fmin,xmin #функция слабо зависит от аргумента
    # выбрали направление, начинаем шагать-минимизировать
    lastFuncValOneDim = (newFuncValOneDim[0]+10, None)
    while newFuncValOneDim[0] <= lastFuncValOneDim[0] :
        #если изменение значения функции - незначительное, останавливаемся, чтобы зря время на расчеты не тратить
        if lastFuncValOneDim[0]-newFuncValOneDim[0] < minFuncDelta : break
        lastFuncValOneDim = newFuncValOneDim
        lastArg = newArgi
        if ( (kudaIdem==1) and (newArgi-step>=a) ) or ( (kudaIdem==-1) and (newArgi+step<=b) ):
            newArgi = newArgi - kudaIdem*step
        else:
            if kudaIdem == 1: newArgi = a
            else: newArgi = b
            newFuncValOneDim = f(newArgi)
            break
        #дошли до левой границы
        newFuncValOneDim = f(newArgi)
    if newFuncValOneDim[0] <= lastFuncValOneDim[0] : #т.е. мы вышли так и не обнаружив отрезка минимума
        xmin = newArgi
        fmin = newFuncValOneDim
        if enableOutput:
            print('Не обнаружен отрезок с минимумом для '+paramName+' df = '+str(fmin[0]-initialLastFuncValOneDim[0]))
        assert fmin[0]<=initialLastFuncValOneDim[0],'We should minimize function'
        return fmin,xmin
    # запускаем метод золотого сечения для уточнения минимума
    if kudaIdem==1:
        leftx=newArgi; leftF=newFuncValOneDim;
        rightx=lastArg; rightF=lastFuncValOneDim;
    else:
        rightx=newArgi; rightF=newFuncValOneDim;
        leftx=lastArg; leftF=lastFuncValOneDim;
    lastFuncValOneDim = leftF if leftF[0] <= rightF[0] else rightF
    if lastFuncValOneDim[0]==leftF[0]: lastArg = leftx
    else: lastArg = rightx;
    phi=1.618 # константа в методе золотого сечения
    x1=rightx-(rightx-leftx)/phi;
    x2=leftx+(rightx-leftx)/phi;
    if isInteger:
        x1=round(x1);
        x2=round(x2);
        if (leftx==x1) or (rightx==x2): #дальше некуда делить
            xmin=lastArg; fmin=lastFuncValOneDim; #last здесь лучше чем new
            return fmin,xmin
    y1=f(x1); y2=f(x2);
    while True:

        if min([y1[0],y2[0]]) > lastFuncValOneDim[0]: break #в середине отрезка функция оказалась больше чем по краям => она очень немонотонно себя ведет
        lastFuncValOneDim = y1 if y1[0]<=y2[0] else y2
        if lastFuncValOneDim[0]==y1[0]: lastArg = x1
        else: lastArg = x2;

        if y1[0] <= y2[0]:
            rightx=x2; rightF=y2;
            x2=x1; y2=y1;
            x1=rightx-(rightx-leftx)/phi;
            if isInteger:
                x1=round(x1);
                if leftx==x1: #дальше некуда делить
                    xmin=x1; fmin=y1;
                    return fmin,xmin
            y1=f(x1);
        else:
            leftx=x1; leftF=y1;
            x1=x2; y1=y2;
            x2=leftx+(rightx-leftx)/phi;
            if isInteger:
                x2=round(x2);
                if (rightx==x2): #дальше некуда делить
                    xmin=x2; fmin=y2;
                    return fmin,xmin
            y2=f(x2);

        #условия выхода
        if (rightx-leftx<minStep): break;
        if abs(y1[0]-y2[0])<minFuncDelta: break;

    yVals=[leftF,y1,y2,rightF,lastFuncValOneDim]
    xVals=[leftx,x1,x2,rightx,lastArg]
    indind = np.argmin([yv[0] for yv in yVals])
    fmin = yVals[indind]
    assert fmin[0]<=initialLastFuncValOneDim[0], 'We should minimize function'
    xmin=xVals[indind]
    return fmin, xmin


def minimizePokoordHelper(f, arg0, minDeltaFunc = 0.001, enableOutput = False, methodType = 'seq', parallel = False, useGridSearch=False, returnTrace=True, f_kwargs={}):
    assert methodType in ['seq', 'random', 'greedy'], 'Unknown methodType'
    if parallel: assert methodType=='greedy', 'Parallel is allowed only for methodType=greedy'
    n = len(arg0)
    threadPool = ThreadPool(n) if parallel else None
    #random.seed(0) - нельзя делать иначе одинаковые результаты получается, а нам хочется проверить насколько минимум устойчив
    newFuncVal = f(arg0, **f_kwargs)
    trace = [[copy.deepcopy(arg0), newFuncVal[0]]]
    if enableOutput: print('Начальное значение функции F = ', newFuncVal[0], 'Extra =', newFuncVal[1])
    veryLastFuncVal = (newFuncVal[0]+100, None)
    newArg = copy.deepcopy(arg0)
    lastBestCoord = -1; newBestCoord = -1; useGridSearchCoords = [False]*n; gridSearchCount = 0
    while veryLastFuncVal[0]-newFuncVal[0] > minDeltaFunc:
        veryLastFuncVal = newFuncVal
        posledKoord = [i for i in range(n)] # последовательность координат - случайна
        if methodType=='random': random.shuffle(posledKoord)
        tries = [] # для methodType = 'greedy'
        def findMinForCoord(ind):
            tmpArg = copy.deepcopy(newArg)
            lastFuncVal = newFuncVal
            i = posledKoord[ind]
            if (methodType == 'greedy') and (i == newBestCoord):
                return {'xmin':tmpArg[i]['value'], 'fmin':newFuncVal}
            def f1(xi):  tmpArg[i]['value'] = xi; return f(tmpArg, **f_kwargs)
            if useGridSearch and (not useGridSearchCoords[ind]) and (gridSearchCount<=n):
                startArg = newArg[i]['value']
                leftBorder = newArg[i]['leftBorder']
                rightBorder = newArg[i]['rightBorder']
                step = newArg[i]['step']
                searchArgValues = np.arange(np.ceil((leftBorder-startArg)/step), np.floor((rightBorder-startArg)/step), 1)*step + startArg
                funcValues =[None]*searchArgValues.size
                for i_sav in range(searchArgValues.size):
                    sav = searchArgValues[i_sav]
                    if abs(sav-startArg)>step/100: funcValues[i_sav] = f1(sav)
                    else: funcValues[i_sav] = lastFuncVal
                i_greedSearch = np.argmin([fv[0] for fv in funcValues])
                i0_greedSearch = max([0,i_greedSearch-1])
                i1_greedSearch = min([i_greedSearch+1,funcValues.size-1])
                startArg = searchArgValues[i_greedSearch]
                lastFuncVal = funcValues[i_greedSearch]
                leftBorder = searchArgValues[i0_greedSearch]
                rightBorder = searchArgValues[i1_greedSearch]
                step = newArg[i]['minStep']
                if enableOutput: print('Grid search found best start value',newArg[i]['paramName'],'=',startArg,'fval =', lastFuncVal[0], 'Extra =', lastFuncVal[1])
            else:
                startArg = newArg[i]['value']
                leftBorder = newArg[i]['leftBorder']
                rightBorder = newArg[i]['rightBorder']
                step = newArg[i]['step']
            fmin, xmin = minimizePokoordOneDim(f1, lastFuncVal, startArg, leftBorder, rightBorder,\
                         step, newArg[i]['minStep'], minDeltaFunc/n, newArg[i]['isInt'], newArg[i]['paramName'], enableOutput)
            if enableOutput: print('Проработал метод золотого сечения. Fmin = ', fmin[0], 'Параметр: ', newArg[i]['paramName'], 'значение: ', xmin, 'Extra =', fmin[1])
            return {'xmin':xmin, 'fmin':fmin}
        if methodType != 'greedy':
            for ind in range(n):
                res = findMinForCoord(ind)
                useGridSearchCoords[ind] = True
                gridSearchCount += 1
                newFuncVal = res['fmin']
                newArg[posledKoord[ind]]['value'] = res['xmin']
                trace.append([copy.deepcopy(newArg), newFuncVal[0]])
        else:
            if parallel: tries = threadPool.map(findMinForCoord, range(n))
            else: tries = [findMinForCoord(ind) for ind in range(n)]
            # выбираем наилучшую координату
            newBestCoord = np.argmin([t['fmin'] for t in tries])
            useGridSearchCoords[newBestCoord] = True
            gridSearchCount += 1
            newArg[newBestCoord]['value'] = tries[newBestCoord]['xmin']
            newFuncVal = tries[newBestCoord]['fmin']
            if enableOutput: print('Выбрали лучшую координату: ', newArg[newBestCoord]['paramName'], 'значение: ', newArg[newBestCoord]['value'], 'Fmin = ', tries[newBestCoord]['fmin'])
            trace.append([copy.deepcopy(newArg), newFuncVal[0]])
        # если переменная - одна, то больше ничего не минимизируем
        if n == 1 : break
    resF = newFuncVal
    resArg = copy.deepcopy(newArg)
    return resF, resArg, trace

# возвращает пару: максимальное значение функции и точку максимума
# methodType = 'seq' - минимизируем по координатам в обычном порядке
# methodType = 'random' - минимизируем по координатам в случайном порядке
# methodType = 'greedy' - пытаемся минимизировать по всем координатам и выбираем наилучшую
# f_kwargs = именованные параметры, подаваемые в функцию f
# useGridSearch - для быстроменяющихся функций. Если да, то проверяет значения одномерной функции на сетке с шагом step и минимизирует на интервале +-step вокруг минимального значения
# useRefinement - уточнять ли поиск в окрестности найденного значения, делая выходы по разным направлениями и повторные минимизации
# returnTrace = False - возвращать ли последовательность точек оптимизации вместе со значениями функции в них
# extraValue = False - возвращает ли f tuple из value и дополнительной информации, которую нужно вернуть с минимумом
def minimizePokoord(f0, arg0, useRefinement=False, extraValue=False, **kwargs):
    kwargs = copy.deepcopy(kwargs)
    if not extraValue:
        def fExtra(arg, **kwargs): return f0(arg, **kwargs), None
        f = fExtra
    else: f = f0
    fmin, xmin, trace = minimizePokoordHelper(f, arg0, **kwargs)
    if kwargs['enableOutput']: print('fmin = ', fmin[0], 'xmin = ',arg2string(xmin), 'extra =', fmin[1])
    if useRefinement and (len(xmin)>1):
        if kwargs['enableOutput']: print('Starting refinement')
        kwargs['useGridSearch'] = False
        fminOld = (fmin[0]+kwargs['minDeltaFunc']*2, None)
        iterCount = 0
        while fminOld[0]-fmin[0]>=kwargs['minDeltaFunc']:
            fminOld = fmin
            xminOld = copy.deepcopy(xmin)
            xminOld1 = copy.deepcopy(xmin)
            for arg in xminOld1: arg['value'] = min([arg['rightBorder'], arg['value']+arg['step']])
            fmin1, xmin1, _ = minimizePokoordHelper(f, xminOld1, **kwargs)
            xminOld2 = copy.deepcopy(xmin)
            for arg in xminOld2: arg['value'] = max([arg['leftBorder'], arg['value']-arg['step']])
            fmin2, xmin2, _ = minimizePokoordHelper(f, xminOld2, **kwargs)
            if kwargs['enableOutput']: print('Refinement cycle. fminOld,fmin1,fmin2 = ',fmin[0],fmin1[0],fmin2[0])
            if fmin1[0] < fmin2[0]:
                if fmin1[0] < fmin[0]: fmin=fmin1; xmin=xmin1
            else:
                if fmin2[0] < fmin[0]: fmin=fmin2; xmin=xmin2
            if kwargs['enableOutput']: print('fmin = ', fmin[0], 'xmin = ', arg2string(xmin), 'extra =', fmin[1])
            iterCount += 1
            trace.append([copy.deepcopy(xmin), fmin[0]])
            if iterCount > 50:
                if kwargs['enableOutput']: print('Max iter count reached in refinement')
                break
            bestRelativeDelta = 0
            for i in range(len(xmin)):
                realtiveDelta = abs(xmin[i]['value'] - xminOld[i]['value'])/xmin[i]['step']
                if realtiveDelta > bestRelativeDelta: bestRelativeDelta = realtiveDelta
            if bestRelativeDelta < 0.1:
                if kwargs['enableOutput']: print('Too small delta in refinement')
                break
    if kwargs['returnTrace']:
        if extraValue: return fmin[0], xmin, trace, fmin[1]
        else: return fmin[0], xmin, trace
    else:
        if extraValue: return fmin[0], xmin, fmin[1]
        else: return fmin[0], xmin


# bounds - is a list of intervals [a,b]
def minimize(fun, x0, bounds, constraints=(), fun_args=None, paramNames=None, method='trust-constr'):
    """
    method = scipy minimize method
    """
    assert (len(constraints) == 0) or method == 'trust-constr'
    if paramNames is None: paramNames = ['x'+str(i) for i in range(len(x0))]
    if method == 'coord':
        def makeCoordArg(x,b,pNames):
            arg = []
            for i in range(len(x)):
                arg.append(param(pNames[i], (b[i][0] + b[i][1]) / 2, b[i], (b[i][1] - b[i][0]) / 10, (b[i][1] - b[i][0]) / 100))
            return arg

        def makeClassicX(arg):
            return [arg[i].value for i in range(len(arg))]
        arg0 = makeCoordArg(x0,bounds,paramNames)

        def fun1(arg, *f_args):
            return fun(makeClassicX(arg), *f_args)
        fmin, argmin = minimizePokoord(fun1, arg0, useRefinement=False, extraValue=False, f_args=fun_args, methodType='random')
        xmin = makeClassicX(argmin)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = scipy.optimize.minimize(fun, x0, args=fun_args, bounds=bounds, constraints=constraints, method=method)
        fmin, xmin = res.fun, res.x
    return fmin, xmin


def findGlobalMinimum(targetFunction, trysCount, bounds, constraints=None, fun_args=None, paramNames=None, folderToSaveResult='globalMinimumSearchResult', fixParams=None, contourMapCalcMethod='fast', plotContourMaps='all', extraPlotFunc=None, printOnline=True, method='trust-constr'):
    """
    Try several optimizations from random start point. Retuns sorted list of results and plot contours around minimum.
    
    :param targetFunction: function to minimize targetFunction(argsList, *fun_args)
    :param trysCount: number of attempts to find minimum
    :param bounds: list of 2-element lists with parameter bounds
    :param constraints: additional constrains for trust-constr scipy.optimize.minimize method
    :param fun_args: extra params of targetFunction (tuple)
    :param paramNames: list of target function argument names to use in plotting (not fixed and fixed)
    :param folderToSaveResult: all result graphs and log are saved here
    :param fixParams: dict of paramName:value to fix
    :param contourMapCalcMethod: 'fast' - plot contours of the target function; 'thorough' - plot contours of the min of target function by all arguments except axes
    :param plotContourMaps: 'all' or list of 1-element or 2-elements lists of axes names to plot contours of target function 
    :param extraPlotFunc: user defined function to plot something on result contours: func(ax, axisNamesList, xminDict)
    :return: sorted list of trysCount minimums of dicts {'value':.., 'x':...} 
    """

    n = len(bounds)
    if paramNames is None:
        paramNames = [f'x_{i}' for i in range(n)]
    if fixParams is None: fixParams = {}
    if fun_args is None: fun_args = tuple()
    fixInd = np.zeros(len(fixParams), dtype=int)
    fixVal = np.zeros(len(fixParams))
    if constraints is None: constraints = tuple()
    for p,j in zip(fixParams, range(len(fixParams))):
        assert p in paramNames, f'Fixed param {p} doesn\'t present in paramNames'
        i = paramNames.index(p)
        fixInd[j] = i
        fixVal[j] = fixParams[p]
        a,b = bounds[i]
        assert a <= fixParams[p] and fixParams[p] <= b, f'Fixed param {p} = {fixParams[p]} is out of bounds [{a}, {b}]'
    notFixedInd = np.setdiff1d(np.arange(n), fixInd)
    m = len(notFixedInd)  # dimension of not fixed variable space
    notFixedParamNames = np.array(paramNames)[notFixedInd]

    def getFullx(partial_x):
        full_x = np.zeros(n)
        full_x[notFixedInd] = partial_x
        full_x[fixInd] = fixVal
        return full_x

    def targetFunctionPartial(partial_x, *f_args):
        return targetFunction(getFullx(partial_x), *f_args)

    partial_constrains = tuple()
    for c in constraints:
        if isinstance(c, scipy.optimize.LinearConstraint):
            if len(c.A.shape) == 1: c.A = c.A.reshape(1,-1)
            x = np.zeros(n)
            x[fixInd] = fixVal
            d = c.A.dot(x)
            part_c = scipy.optimize.LinearConstraint(c.A[:,notFixedInd], c.lb-d, c.ub-d, c.keep_feasible)
        else:
            assert isinstance(c, scipy.optimize.NonlinearConstraint)
            part_c = scipy.optimize.NonlinearConstraint(lambda partial_x: c.fun(getFullx(partial_x), *fun_args), c.lb, c.up, keep_feasible=c.keep_feasible)
        partial_constrains += (part_c,)

    partialBounds = np.zeros((m, 2))
    for i,j in zip(notFixedInd, range(m)):
        partialBounds[j,:] = bounds[i]

    fmins = np.zeros(trysCount)
    xs = [None]*trysCount
    xs_partial = [None]*trysCount
    rng = np.random.default_rng(0)

    # start point source
    edgePoints = np.array(list(itertools.product(*([[0., 1.]] * m))))
    edgePoints[1], edgePoints[-1] = edgePoints[-1], edgePoints[1]
    perm = rng.permutation(len(edgePoints) - 2)
    edgePoints[2:] = edgePoints[2 + perm, :]
    eps = 0.01
    edgePoints[edgePoints == 0] += eps
    edgePoints[edgePoints == 1] -= eps
    currentPointInd = [0]

    def getNextPoint():
        if (currentPointInd[0] % 2 == 0) and (currentPointInd[0] // 2 < edgePoints.shape[0]):
            init_point = edgePoints[currentPointInd[0] // 2] * (partialBounds[:,1] - partialBounds[:,0]) + partialBounds[:,0]
        else:
            init_point = rng.random(m) * (partialBounds[:,1] - partialBounds[:,0]) + partialBounds[:,0]
        currentPointInd[0] += 1
        return init_point

    def getArg0AndBounds():
        partial_arg0 = getNextPoint()
        try_i = 0
        while not checkConstrains(partial_arg0, partial_constrains):
            partial_arg0 = getNextPoint()
            try_i += 1
            assert try_i < 10000, f'Too many attempts to find start point for optimization inside constrains\nLast try: {partial_arg0}'
        return partial_arg0, partialBounds

    for ir in range(trysCount):
        partial_arg0, partialBounds = getArg0AndBounds()
        fmins[ir], xs_partial[ir] = minimize(targetFunctionPartial, partial_arg0, bounds=partialBounds, constraints=partial_constrains, fun_args=fun_args, paramNames=notFixedParamNames, method=method)
        # method can violate bounds and constrains!
        for j in range(len(partialBounds)):
            g = xs_partial[ir]
            xs_partial[ir][j] = max(g[j], partialBounds[j][0])
            xs_partial[ir][j] = min(g[j], partialBounds[j][1])
        xs[ir] = getFullx(xs_partial[ir])
        # old = fmins[ir]
        fmins[ir] = targetFunction(xs[ir])
        ok, rate = checkConstrains(xs[ir], constraints, returnRate=True)
        if not ok: warnings.warn(f'Constrains are violated. Violation size = {rate}')
        if printOnline: print('targetFunction = '+str(fmins[ir])+' '+arg2string(xs[ir], paramNames), flush=True)

    ind = np.argsort(fmins)
    output = ''
    for ir in range(trysCount):
        j = ind[ir]
        output += str(fmins[j])+' '+arg2string(xs[j], paramNames)+"\n"
    if folderToSaveResult is not None:
        folderToSaveResult = utils.fixPath(folderToSaveResult)
        if not os.path.exists(folderToSaveResult): os.makedirs(folderToSaveResult)
        with open(folderToSaveResult+'/minimums.txt', 'w') as f: f.write(output)

        best_ind = ind[0]
        best_x_partial = xs_partial[best_ind]

        def plot1d(param):
            plotMap1d(param, targetFunctionPartial, best_x_partial, bounds=partialBounds, constraints=partial_constrains, fun_args=fun_args, paramNames=notFixedParamNames, optimizeMethod='Powell', calMapMethod=contourMapCalcMethod, folder=folderToSaveResult, postfix='_1d_target_func')

        def plot2d(param1, param2):
            plotMap2d([param1, param2], targetFunctionPartial, best_x_partial, bounds=partialBounds, constraints=partial_constrains, fun_args=(), paramNames=notFixedParamNames, optimizeMethod='Powell', calMapMethod=contourMapCalcMethod, folder=folderToSaveResult, postfix='_2d_target_func', extraPlotFunc=extraPlotFunc)

        if plotContourMaps == 'all':
            for i in range(len(notFixedParamNames)):
                plot1d(i)
            for i1 in range(len(notFixedParamNames)):
                for i2 in range(i1+1,len(notFixedParamNames)):
                    plot2d(i1,i2)
        else:
            assert isinstance(plotContourMaps, list)
            for params in plotContourMaps:
                assert isinstance(params, list)
                if len(params) == 1:
                    assert params[0] in notFixedParamNames, params[0]+' is not in not fixed param list: '+str(notFixedParamNames)
                    plot1d(params[0])
                else:
                    assert len(params) == 2
                    assert params[0] in notFixedParamNames, params[0] + ' is not in not fixed param list: ' + str(notFixedParamNames)
                    assert params[1] in notFixedParamNames, params[1] + ' is not in not fixed param list: ' + str(notFixedParamNames)
                    plot2d(params[0], params[1])
    result = [{'value': fmins[ind[i]], 'x':xs[ind[i]]} for i in range(trysCount)]
    return result


def checkConstrains(x, constraints, returnRate=False):
    result = True
    if len(constraints) == 0:
        if returnRate: return result, 0
        else: return result
    rate = np.zeros(len(constraints))
    for c in constraints:
        if isinstance(c, scipy.optimize.LinearConstraint):
            Ax = c.A.dot(x)
            result &= np.all((c.lb <= Ax) & (Ax <= c.ub))
        else:
            assert isinstance(c, scipy.optimize.NonlinearConstraint)
            Ax = c.fun(x)
            result &= np.all((c.lb <= Ax) & (Ax <= c.ub))
        rate = np.maximum(rate, Ax - c.lb)
        rate = np.maximum(rate, c.ub - Ax)
    rate = np.max(rate)
    rate = max(0, rate)
    if returnRate: return result, rate
    else: return result


def plotMap1d(axis, fun, xmin, bounds, constraints=(), fun_args=None, paramNames=None, optimizeMethod='Powell', N=None, calMapMethod='fast', folder='.', postfix=''):
    assert calMapMethod in ['fast', 'thorough']
    if isinstance(axis, str):
        assert axis in paramNames
        axisName = axis
        axisInd = np.where(np.array(paramNames) == axisName)[0][0]
    else:
        axisInd = axis
        axisName = paramNames[axisInd]
    if N is None: N = 100
    axisValues = np.linspace(bounds[axisInd][0], bounds[axisInd][1], N)
    dim = len(xmin)
    forEvaluation = np.zeros([N, dim])
    for i in range(N):
        forEvaluation[i] = np.copy(xmin)
        forEvaluation[i, axisInd] = axisValues[i]

    def addFixedParams(xPart, fixParamValue):
        x = np.zeros(dim)
        j = 0
        for i in range(dim):
            if i == axisInd:
                x[i] = fixParamValue
            else:
                x[i] = xPart[j]
                j += 1
        return x

    def minFun(xPart, fixParamValue):
        x = addFixedParams(xPart, fixParamValue)
        return fun(x) if fun_args is None else fun(x, *fun_args)

    def transformConstrain(c, fixParamValue):
        if isinstance(c, dict): # nonlinear constrain
            return lambda xPart: c(addFixedParams(xPart, fixParamValue))
        elif isinstance(c, scipy.optimize.LinearConstraint):
            coef = c.A[axisInd]
            A = np.concatenate((c.A[:axisInd],c.A[axisInd+1:]))
            lb = c.lb - coef*fixParamValue
            ub = c.ub - coef*fixParamValue
            return scipy.optimize.LinearConstraint(A, lb, ub, keep_feasible=c.keep_feasible)
        else: assert False, str(c)

    indOther = np.setdiff1d(np.arange(dim), [axisInd])
    arg0 = xmin[indOther]
    boundsOther = [bounds[i] for i in indOther]
    paramNamesOther = [paramNames[i] for i in indOther]

    funcValues = np.zeros(axisValues.shape)
    insideConstrains = np.ones(axisValues.shape, dtype=bool)
    for i in range(N):
        inside = checkConstrains(forEvaluation[i], constraints)
        if not inside:
            insideConstrains[i] = False
            continue
        if calMapMethod == 'fast':
            funcValues[i] = fun(forEvaluation[i]) if fun_args is None else fun(forEvaluation[i], *fun_args)
        else:
            constraints1 = tuple(transformConstrain(c, axisValues[i]) for c in constraints)
            funcValues[i], _ = minimize(minFun, arg0, boundsOther, constraints=constraints1, fun_args=(axisValues[i],), paramNames=paramNamesOther, method=optimizeMethod)
    axisValues = axisValues[insideConstrains]
    funcValues = funcValues[insideConstrains]
    fig, ax = plotting.createfig()
    ax.plot(axisValues, funcValues)
    fun_val = fun(xmin) if fun_args is None else fun(xmin, *fun_args)
    ax.plot(xmin[axisInd], fun_val, marker='o', markersize=10, color="red")
    
    plt.xlabel(axisName)
    plt.title(utils.wrap('Map 1d '+arg2string(xmin, paramNames), 100))
    plotting.savefig(folder + '/' + axisName + postfix + '.png', fig)
    plotting.closefig(fig)
    data2csv = np.vstack((axisValues, funcValues)).T
    np.savetxt(folder + '/' + axisName + postfix + '.csv', data2csv, delimiter=',')


# axes - a pair of param names or param indexes
# N = {paramName1:N1, paramName2:N2}
def plotMap2d(axes, fun, xmin, bounds, constraints=(), fun_args=None, paramNames=None, optimizeMethod='Powell', N=None, calMapMethod='fast', folder='.', postfix='', extraPlotFunc=None):
    assert calMapMethod in ['fast', 'thorough']
    if isinstance(axes[0], str):
        assert (axes[0] in paramNames) and (axes[1] in paramNames)
        axisNames = axes
        axisInds = [np.where(np.array(paramNames) == axisNames[0])[0][0], np.where(np.array(paramNames) == axisNames[1])[0][0]]
    else:
        axisInds = axes
        axisNames = [paramNames[axisInds[0]], paramNames[axisInds[1]]]
    if N is None: N = {axisNames[0]: 50, axisNames[1]: 50}
    NN = [N[axisNames[0]], N[axisNames[1]]]
    axisValues = [np.linspace(bounds[axisInds[i]][0], bounds[axisInds[i]][1], NN[i]) for i in [0,1]]
    param1mesh, param2mesh = np.meshgrid(axisValues[0], axisValues[1])
    dim = len(xmin)
    forEvaluation = np.zeros([NN[0] * NN[1], dim])
    k = 0
    centerPoint = np.copy(xmin)
    for i0 in range(NN[0]):
        for i1 in range(NN[1]):
            forEvaluation[k] = np.copy(centerPoint)
            forEvaluation[k, axisInds[0]] = param1mesh[i0, i1]
            forEvaluation[k, axisInds[1]] = param2mesh[i0, i1]
            k += 1

    def addFixedParams(xPart, axis1, axis2):
        x = np.zeros(dim)
        j = 0
        for i in range(dim):
            if i == axisInds[0]: x[i] = axis1
            elif i == axisInds[1]: x[i] = axis2
            else:
                x[i] = xPart[j]
                j += 1
        return x

    def minFun(xPart, axis1, axis2):
        x = addFixedParams(xPart, axis1, axis2)
        return fun(x) if fun_args is None else fun(x, *fun_args)

    def transformConstrain(c, axis1, axis2):
        if isinstance(c, dict): # nonlinear constrain
            return lambda xPart: c(addFixedParams(xPart, axis1, axis2))
        elif isinstance(c, scipy.optimize.LinearConstraint):
            coef = [c.A[axisInds[0]], c.A[axisInds[1]]]
            A = np.array([c.A[i] for i in range(len(c.A)) if i not in axisInds])
            lb = c.lb - coef[0]*axis1
            ub = c.ub - coef[1]*axis2
            return scipy.optimize.LinearConstraint(A, lb, ub, keep_feasible=c.keep_feasible)
        else: assert False, str(c)

    indOther = np.setdiff1d(np.arange(dim), axisInds)
    arg0 = xmin[indOther]
    boundsOther = [bounds[i] for i in indOther]
    paramNamesOther = [paramNames[i] for i in indOther]

    funcValues = np.zeros(param1mesh.shape)
    insideConstrains = np.ones(param1mesh.shape, dtype=bool)
    k = 0
    for i0 in range(NN[0]):
        for i1 in range(NN[1]):
            inside = checkConstrains(forEvaluation[k], constraints)
            if not inside:
                insideConstrains[i0, i1] = False
                continue
            if calMapMethod == 'fast':
                funcValues[i0, i1] = fun(forEvaluation[k]) if fun_args is None else fun(forEvaluation[k], *fun_args)
            else:
                constraints1 = tuple(transformConstrain(c, param1mesh[i0, i1], param2mesh[i0, i1]) for c in constraints)
                funcValues[i0, i1], _ = minimize(minFun, arg0, boundsOther, constraints=constraints1, fun_args=(param1mesh[i0, i1], param2mesh[i0, i1]), paramNames=paramNamesOther, method=optimizeMethod)
                # print(k,'done of',NN[0]*NN[1])
            k += 1
    # param1mesh = param1mesh[insideConstrains]   - contourf works only for 2D grid
    # param2mesh = param2mesh[insideConstrains]
    funcValues[~insideConstrains] = np.max(funcValues)
    fig, ax = plotting.createfig()
    CS = plt.contourf(param1mesh, param2mesh, funcValues, cmap='plasma')
    plt.clabel(CS, fmt='%2.2f', colors='k', fontsize=15, inline=False)
    ax.plot([xmin[axisInds[0]]], [xmin[axisInds[1]]], marker='o', markersize=5, color="red")
    if extraPlotFunc is not None:
        xminDict = {paramNames[i]:xmin[i] for i in range(len(xmin))}
        extraPlotFunc(ax, axisNames, xminDict)
    # plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel(axisNames[0])
    plt.ylabel(axisNames[1])
    plotTitle = utils.wrap('Map '+arg2string(xmin, paramNames), 100)
    ax.set_title(plotTitle)
    plt.title(plotTitle)
    plotting.savefig(folder + '/' + axisNames[0] + '_' + axisNames[1] + postfix + '_contour.png', fig)
    plotting.closefig(fig)
    # for cmap in ['inferno', 'spectral', 'terrain', 'summer']:
    cmap = 'inferno'
    fig, ax = plotting.createfig(subplot_kw={'projection':'3d'})
    # cmap = 'summer'
    ax.plot_trisurf(param1mesh.flatten(), param2mesh.flatten(), funcValues.flatten(), linewidth=0.2, antialiased=True, cmap=cmap)
    ax.view_init(azim=310, elev=40)
    plt.xlabel(axisNames[0])
    plt.ylabel(axisNames[1])
    plt.title(plotTitle)
    # plt.legend()
    plotting.savefig(folder + '/' + axisNames[0] + '_' + axisNames[1] + postfix + '.png', fig)
    plotting.closefig(fig)
    data2csv = np.vstack((param1mesh.flatten(), param2mesh.flatten(), funcValues.flatten())).T
    np.savetxt(folder + '/' + axisNames[0] + '_' + axisNames[1] + postfix + '.csv', data2csv, delimiter=',')

import random
import copy
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
import json
# -*- coding: utf-8 -*-

def arg2string(arg):
    s = ''
    for a in arg:
        s += a['paramName'] + ('=%.5g' % a['value'])
        if a != arg[-1]: s += '  '
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
    return {'paramName':paramName, 'value':value, 'leftBorder':borders[0], 'rightBorder':borders[1], 'step':step, 'minStep':minStep, 'isInt':isInt}

def value(args, paramName):
    for i in range(len(args)):
        if args[i]['paramName'] == paramName: return args[i]['value']
    return None

def setValue(args, paramName, value):
    for i in range(len(args)):
        if args[i]['paramName'] == paramName:
            args[i]['value'] = value
            return
    assert False, "paramName = "+paramName+" not found"

def minimizePokoordOneDim(f, lastFuncValOneDim, x0, a, b, step, minStep, minFuncDelta, isInteger, paramName, enableOutput):
    #проверяем на локальный минимум и зависимость от параметра
    #делаем минимальные шаги в обе стороны
    initialLastFuncValOneDim = lastFuncValOneDim
    centrF=lastFuncValOneDim
    argLeft = x0
    if argLeft-minStep >= a:
        argLeft = argLeft-minStep
        leftF = f(argLeft)
        if centrF == leftF:
            if enableOutput: print('Функция не зависит от аргумента '+paramName+' df = 0')
            fmin = centrF; xmin = x0; return fmin,xmin # функция не зависит от аргумента
    else: leftF = centrF
    if leftF >= centrF: #пытаемся сделать шаг вправо
        argRight = x0
        if argRight+minStep <= b:
            argRight = argRight+minStep
            rightF = f(argRight)
        else: rightF = centrF
        if rightF < centrF :
            kudaIdem = -1; newFuncValOneDim = rightF; newArgi = argRight #вправо
        else:
            if enableOutput: print('Неудача при попытках смещения аргумента '+paramName)
            fmin = centrF; xmin = x0; return fmin,xmin
    else:
        kudaIdem = 1; newFuncValOneDim = leftF; newArgi = argLeft #влево
    assert newFuncValOneDim <= initialLastFuncValOneDim, 'We should minimize function, but new='+str(newFuncValOneDim)+' init='+str(initialLastFuncValOneDim)
    # проверяем существенно ли изменение функции (будет ли функция меняться более чем на minFuncDelta при большом шаге)
    if (abs(newFuncValOneDim-centrF) / minStep) * step < minFuncDelta:
        fmin = centrF; xmin = x0;
        assert fmin<=initialLastFuncValOneDim, 'We should minimize function'
        if enableOutput: print('Функция слабо зависит от аргумента '+paramName+' df = '+str(fmin-initialLastFuncValOneDim))
        return fmin,xmin #функция слабо зависит от аргумента
    # выбрали направление, начинаем шагать-минимизировать
    lastFuncValOneDim = newFuncValOneDim+10
    while newFuncValOneDim <= lastFuncValOneDim :
        #если изменение значения функции - незначительное, останавливаемся, чтобы зря время на расчеты не тратить
        if lastFuncValOneDim-newFuncValOneDim < minFuncDelta : break
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
    if newFuncValOneDim <= lastFuncValOneDim : #т.е. мы вышли так и не обнаружив отрезка минимума
        xmin = newArgi
        fmin = newFuncValOneDim
        if enableOutput: print('Не обнаружен отрезок с минимумом для '+paramName+' df = '+str(fmin-initialLastFuncValOneDim))
        assert fmin<=initialLastFuncValOneDim,'We should minimize function'
        return fmin,xmin
    # запускаем метод золотого сечения для уточнения минимума
    if kudaIdem==1:
        leftx=newArgi; leftF=newFuncValOneDim;
        rightx=lastArg; rightF=lastFuncValOneDim;
    else:
        rightx=newArgi; rightF=newFuncValOneDim;
        leftx=lastArg; leftF=lastFuncValOneDim;
    lastFuncValOneDim = min([leftF, rightF])
    if lastFuncValOneDim==leftF: lastArg = leftx
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

        if min([y1,y2]) > lastFuncValOneDim: break #в середине отрезка функция оказалась больше чем по краям => она очень немонотонно себя ведет
        lastFuncValOneDim = min([y1,y2])
        if lastFuncValOneDim==y1: lastArg = x1
        else: lastArg = x2;

        if y1<=y2:
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
        if abs(y1-y2)<minFuncDelta: break;

    yVals=[leftF,y1,y2,rightF,lastFuncValOneDim]
    xVals=[leftx,x1,x2,rightx,lastArg]
    indind = np.argmin(yVals)
    fmin = yVals[indind]
    assert fmin<=initialLastFuncValOneDim, 'We should minimize function'
    xmin=xVals[indind]
    return fmin, xmin

def minimizePokoordHelper(f, arg0, minDeltaFunc = 0.001, enableOutput = False, methodType = 'seq', parallel = False, useGridSearch=False, returnTrace=True, f_kwargs={}):
    assert methodType in ['seq', 'random', 'greedy'], 'Unknown methodType'
    if parallel: assert methodType=='greedy', 'Parallel is allowed only for methodType=greedy'
    n = len(arg0)
    threadPool = ThreadPool(n) if parallel else None
    random.seed(0) # инициализируем начальное значение счетчика случайных чисел
    newFuncVal = f(arg0, **f_kwargs)
    trace = [[copy.deepcopy(arg0), newFuncVal]]
    if enableOutput: print('Начальное значение функции F = ', newFuncVal)
    veryLastFuncVal = newFuncVal+100
    newArg = copy.deepcopy(arg0)
    lastBestCoord = -1; newBestCoord = -1; useGridSearchCoords = [False]*n; gridSearchCount = 0
    while veryLastFuncVal-newFuncVal > minDeltaFunc:
        veryLastFuncVal = newFuncVal
        posledKoord = [i for i in range(n)] # последовательность координат - случайна
        if methodType=='random': random.shuffle(posledKoord)
        tries = [] # для methodType = 'greedy'
        def findMinForCoord(ind):
            tmpArg = copy.deepcopy(newArg)
            lastFuncVal = newFuncVal
            i = posledKoord[ind]
            if (methodType == 'greedy') and (i == newBestCoord): return {'xmin':tmpArg[i]['value'], 'fmin':newFuncVal}
            def f1(xi):  tmpArg[i]['value'] = xi; return f(tmpArg, **f_kwargs)
            if useGridSearch and (not useGridSearchCoords[ind]) and (gridSearchCount<=n):
                startArg = newArg[i]['value']
                leftBorder = newArg[i]['leftBorder']
                rightBorder = newArg[i]['rightBorder']
                step = newArg[i]['step']
                searchArgValues = np.arange(np.ceil((leftBorder-startArg)/step), np.floor((rightBorder-startArg)/step), 1)*step + startArg
                funcValues = np.zeros(searchArgValues.size)
                for i_sav in range(searchArgValues.size):
                    sav = searchArgValues[i_sav]
                    if abs(sav-startArg)>step/100: funcValues[i_sav] = f1(sav)
                    else: funcValues[i_sav] = lastFuncVal
                i_greedSearch = np.argmin(funcValues)
                i0_greedSearch = max([0,i_greedSearch-1])
                i1_greedSearch = min([i_greedSearch+1,funcValues.size-1])
                startArg = searchArgValues[i_greedSearch]
                lastFuncVal = funcValues[i_greedSearch]
                leftBorder = searchArgValues[i0_greedSearch]
                rightBorder = searchArgValues[i1_greedSearch]
                step = newArg[i]['minStep']
                if enableOutput: print('Grid search found best start value',newArg[i]['paramName'],'=',startArg,' fval =', lastFuncVal)
            else:
                startArg = newArg[i]['value']
                leftBorder = newArg[i]['leftBorder']
                rightBorder = newArg[i]['rightBorder']
                step = newArg[i]['step']
            fmin, xmin = minimizePokoordOneDim(f1, lastFuncVal, startArg, leftBorder, rightBorder,\
                         step, newArg[i]['minStep'], minDeltaFunc/n, newArg[i]['isInt'], newArg[i]['paramName'], enableOutput)
            if enableOutput: print('Проработал метод золотого сечения. Fmin = ', fmin, 'Параметр: ', newArg[i]['paramName'], 'значение: ', xmin)
            return {'xmin':xmin, 'fmin':fmin}
        if methodType != 'greedy':
            for ind in range(n):
                res = findMinForCoord(ind)
                useGridSearchCoords[ind] = True
                gridSearchCount += 1
                newFuncVal = res['fmin']
                newArg[posledKoord[ind]]['value'] = res['xmin']
                trace.append([copy.deepcopy(newArg), newFuncVal])
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
            trace.append([copy.deepcopy(newArg), newFuncVal])
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
def minimizePokoord(f, arg0, useRefinement=False, **kwargs):
    fmin, xmin, trace = minimizePokoordHelper(f, arg0, **kwargs)
    if kwargs['enableOutput']: print('fmin = ', fmin, 'xmin = ',arg2string(xmin))
    if useRefinement and (len(xmin)>1):
        if kwargs['enableOutput']: print('Starting refinement')
        kwargs['useGridSearch'] = False
        fminOld = fmin+kwargs['minDeltaFunc']*2
        iterCount = 0
        while fminOld-fmin>=kwargs['minDeltaFunc']:
            fminOld = fmin
            xminOld = copy.deepcopy(xmin)
            xminOld1 = copy.deepcopy(xmin)
            for arg in xminOld1: arg['value'] = min([arg['rightBorder'], arg['value']+arg['step']])
            fmin1, xmin1, _ = minimizePokoordHelper(f, xminOld1, **kwargs)
            xminOld2 = copy.deepcopy(xmin)
            for arg in xminOld2: arg['value'] = max([arg['leftBorder'], arg['value']-arg['step']])
            fmin2, xmin2, _ = minimizePokoordHelper(f, xminOld2, **kwargs)
            if kwargs['enableOutput']: print('Refinement cycle. fminOld,fmin1,fmin2 = ',fmin,fmin1,fmin2)
            if fmin1<fmin2:
                if fmin1<fmin: fmin=fmin1; xmin=xmin1
            else:
                if fmin2<fmin: fmin=fmin2; xmin=xmin2
            if kwargs['enableOutput']: print('fmin = ', fmin, 'xmin = ', arg2string(xmin))
            iterCount += 1
            trace.append([copy.deepcopy(xmin), fmin])
            if iterCount > 50:
                if kwargs['enableOutput']: print('Max iter count reached in refinement')
                break
            bestRelativeDelta = 0
            for i in range(len(xmin)):
                realtiveDelta = abs(xmin[i]['value'] - xminOld[i]['value'])/xmin[i]['step']
                if  realtiveDelta > bestRelativeDelta: bestRelativeDelta = realtiveDelta
            if bestRelativeDelta < 0.1:
                if kwargs['enableOutput']: print('Too small delta in refinement')
                break
    if kwargs['returnTrace']: return fmin, xmin, trace
    else: return fmin, xmin

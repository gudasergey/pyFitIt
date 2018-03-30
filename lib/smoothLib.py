import sys
import numpy as np
import pandas as pd
import minimize
from minimize import value, param, arg2string, arg2json
import random
import fdmnes
import math
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils
import copy
import time
import tempfile
import os
from shutil import copyfile
import json
from scipy import interpolate

# ============================================================================================================================
# ============================================================================================================================
# типы сглаживаний
# ============================================================================================================================
# ============================================================================================================================

def kernelCauchy(x, a, sigma): return sigma/2/math.pi/((x-a)**2+sigma**2/4)

def kernelGauss(x, a, sigma): return 1/sigma/math.sqrt(2*math.pi)*np.exp(-(x-a)**2/2/sigma**2)

def YvesWidth(e, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi):
    ee = (e-Efermi)/Ecent
    ee[ee==0] = 1e-5
    return Gamma_hole + Gamma_max*(0.5+1/math.pi*np.arctan( math.pi/3*Gamma_max/Elarg*(ee-1/ee**2) ))

def lam(e, group, alpha1, alpha2, alpha3):
    # return (28-4)/(2000-100)*(e-100) + 5 + 2000/e**2*(group/11)
    if e==0: e = 1e-5
    return 0.01263*e*(1+alpha1) + 3.74*(1+alpha2) + group*182/e**2*(1+alpha3)

def MullerWidth(e0, group, Gamma_hole, Efermi, alpha1, alpha2, alpha3):
    sigma = np.zeros(e0.shape)
    for i in range(e0.size):
        e = e0[i]-Efermi
        if e<=0: sigma[i] = 1e-5; continue
        h = 6.582119514e-16 # eV*s
        m = 0.5109989461e6 / 0.3e9**2  # eV/c^2, c=0.3e9 m/s
        Gx = Gamma_hole + 2*h*np.sqrt(2*e/m)/(lam(e, group, alpha1, alpha2, alpha3)*1e-10)
        if Gx<1e-5: Gx = 1e-5
        sigma[i] = Gx
    return sigma

def multi_piecewise_width(e, g0,e1,g1,e2,g2,e3,g3):
    sigma = np.zeros(e.shape) + g0
    sigma[(e1<e)&(e<=e2)] = g1
    sigma[(e2<e)&(e<=e3)] = g2
    sigma[e>e3] = g3
    sigma = simpleSmooth(e, sigma, 3)
    return sigma

def spline_width(e, Efermi, *g):
    n = len(g)
    append = [50, 70, 90, 120, 160, 210]
    na = len(append)
    ei = [i/(n-na)*50 for i in range(n-na)]
    ei = ei + append
    ei = np.array(ei) + Efermi
    g = np.array([g[i] for i in range(n)])
    sigma = np.zeros(e.shape)
    sigma[e<ei[0]] = g[0]
    sigma[e>ei[-1]] = g[-1]
    ind = (e>=ei[0]) & (e<=ei[-1])
    tck = interpolate.splrep(ei, g, k=3)
    sigma[ind] = interpolate.splev(e[ind], tck)
    sigma[sigma<=0] = 1e-5
    return sigma

def simpleSmooth(e, xanes, sigma):
    new_xanes = np.zeros(e.shape)
    for i in range(e.size):
        kern = kernelCauchy(e, e[i], sigma)
        norm = utils.integral(e, kern)
        if norm == 0: norm = 1
        new_xanes[i] = utils.integral(e, xanes*kern)/norm
    return new_xanes

def smooth_fdmnes(e, xanes, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi, folder):
    assert os.path.exists(folder+'/out.txt'), 'Can\'t find file out.txt with header and xanes in '+folder
    return fdmnes.smooth(folder, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi)

def smooth_my_fdmnes(e, xanes, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi):
    xanes = np.copy(xanes)
    xanes[e<Efermi] = 0
    new_xanes = np.zeros(e.shape)
    sigma = YvesWidth(e, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi)
    for i in range(e.size):
        kern = kernelCauchy(e, e[i], sigma[i])
        norm = utils.integral(e, kern)
        new_xanes[i] = utils.integral(e, xanes*kern)/norm
    return e, new_xanes

def smooth_Muller(e, xanes, group, Gamma_hole, Efermi, alpha1, alpha2, alpha3):
    xanes = np.copy(xanes)
    # xanes = simpleSmooth(e, xanes, Gamma_hole)
    xanes[e<Efermi] = 0
    new_xanes = np.zeros(e.shape)
    sigma = MullerWidth(e, group, Gamma_hole, Efermi, alpha1, alpha2, alpha3)
    for i in range(e.size):
        kern = kernelCauchy(e, e[i], sigma[i])
        norm = utils.integral(e, kern)
        new_xanes[i] = utils.integral(e, xanes*kern)/norm
        # print(e[i], sigma[i])
    # exit(0)
    return e, new_xanes

def smooth_linear_conv(e, xanes, Gamma_hole, Gamma_max, Efermi):
    xanes = np.copy(xanes)
    xanes[e<Efermi] = 0
    new_xanes = np.zeros(e.shape)
    sigma = Gamma_hole + (Gamma_max-Gamma_hole)*(e-e[0])/(e[-1]-e[0])
    sigma[sigma<=0] = 1e-3
    for i in range(e.size):
        kern = kernelCauchy(e, e[i], sigma[i])
        norm = utils.integral(e, kern)
        new_xanes[i] = utils.integral(e, xanes*kern)/norm
    return e, new_xanes

def smooth_piecewise(e0, xanes, Gamma_hole, Gamma_max, Ecent):
    eleft = np.linspace(e0[0]-10, e0[0]-(e0[1]-e0[0]), 10)
    xleft = np.zeros(eleft.shape)
    eright = np.linspace(e0[-1]+(e0[-1]-e0[-2]), e0[-1]+50, 10)
    xright = np.zeros(eleft.shape) + xanes[-1]
    e = np.hstack((eleft,e0,eright))
    xanes = np.hstack((xleft,xanes,xright))
    new_xanes = np.zeros(e0.shape)
    sigma = np.zeros(e0.shape) + Gamma_hole
    sigma[e0>Ecent] = Gamma_max
    sigma[sigma<=0] = 1e-3
    for i in range(e0.size):
        kern = kernelCauchy(e, e0[i], sigma[i])
        norm = utils.integral(e, kern)
        new_xanes[i] = utils.integral(e, xanes*kern)/norm
    return e0, new_xanes

def generalSmooth(e, xanes, sigma):
    eleft = np.linspace(e[0]-10, e[0]-(e[1]-e[0]), 10)
    xleft = np.zeros(eleft.shape)
    eright = np.linspace(e[-1]+(e[-1]-e[-2]), e[-1]+50, 10)
    xright = np.zeros(eleft.shape) + xanes[-1]
    e_new = np.hstack((eleft,e,eright))
    xanes = np.hstack((xleft,xanes,xright))
    new_xanes = np.zeros(e.shape)
    for i in range(e.size):
        kern = kernelCauchy(e_new, e[i], sigma[i])
        norm = utils.integral(e_new, kern)
        new_xanes[i] = utils.integral(e_new, xanes*kern)/norm
    return e, new_xanes

class DefaultSmoothParams:
    def __init__(self, shift):
        shiftParam = param('shift', shift, [shift-20, shift+20], 1, 0.25)
        self.params = {'fdmnes':\
            [param('Gamma_hole', 4, [0.1,8], 0.4, 0.1), param('Ecent', 26, [1,100], 3, 0.5),\
             param('Elarg', 39, [1,100], 5, 0.5), param('Gamma_max', 25, [5,25], 1, 0.2), param('Efermi', 1.6, [0,20], 2, 0.2)\
            ], 'my_fdmnes':\
            [param('Gamma_hole', 2.5, [0.1,8], 0.4, 0.1), param('Ecent', 30, [1,100], 3, 0.5),\
             param('Elarg', 70, [1,100], 5, 0.5), param('Gamma_max', 13, [5,25], 1, 0.2), param('Efermi', 0, [0,20], 2, 0.2)\
            ], 'linear':\
            [param('Gamma_hole', -1, [-20,20], 1, 0.1), param('Gamma_max', 22, [-30,50], 2, 0.5), param('Efermi', 0, [-10,10], 2, 0.2)\
            ], 'Muller':\
            [param('group', 8, [1,18], 1, 1), param('Gamma_hole', 1, [0.01,5], 0.02, 0.002), param('Efermi', 0, [-20,20], 1, 0.2),\
             param('alpha1', 0, [-0.999,2], 0.1, 0.01), param('alpha2', 0, [-0.9,2], 0.1, 0.01), param('alpha3', 0, [-0.9,2], 0.1, 0.01)\
            ], 'piecewise':\
            [param('Gamma_hole', 5, [0.1,10], 0.2, 0.03), param('Gamma_max', 20, [5,40], 2, 0.5), param('Ecent', 30, [-10,100], 2, 0.2)\
            ], 'multi_piecewise':\
            [param('g0', 0.3, [0.01,2], 0.03, 0.005), param('e1', 10, [0,25], 1, 0.3), param('g1', 5, [1,10], 0.2, 0.05),\
            param('e2', 30, [25,70], 2, 0.5), param('g2', 8, [1,25], 0.5, 0.05), param('e3', 100, [70,150], 5, 1), param('g3', 20, [5,50], 1, 0.05)\
            ], 'spline':\
            [param('Efermi', 1.2, [-20,20], 1, 0.1)\
            ]}
        n = 20
        for i in range(n): self.params['spline'].append( param('g_'+str(i), 1+i/n*35, [0,50], 0.5, 0.05) )
        self.shiftIsAbsolute = True
        self.search_shift_level = 0.25
        self.fdmnesSmoothHeader = ''
        for smoothType in self.params:
            self.params[smoothType].append(shiftParam)

    def __getitem__(self, smoothType):
        return self.params[smoothType]

    def getDict(self, smoothType):
        res = {}
        for a in self.params[smoothType]:
            res[a['paramName']] = a['value']
        return res

def getSmoothParams(arg, names):
    res = ()
    for name in names: res = res + (value(arg,name),)
    return res

def getSmoothWidth(smoothType, e, args):
    ps = DefaultSmoothParams(0)
    names = [p['paramName'] for p in ps[smoothType]]
    names.remove('shift')
    params = getSmoothParams(args, names)
    if (smoothType == 'my_fdmnes') or (smoothType == 'fdmnes'):
        sigma = YvesWidth(e, *params)
    elif smoothType == 'Muller':
        sigma = MullerWidth(e, *params)
    elif smoothType == 'multi_piecewise':
        sigma = multi_piecewise_width(e, *params)
    elif smoothType == 'spline':
        sigma = spline_width(e, *params)
    else: return None
    return sigma

def plotSmoothWidthToFolder(smoothType, e, args, folder):
    sigma = getSmoothWidth(smoothType, e, args)
    if sigma is None:
        print('Cant plot smooth width for smooth type '+smoothType)
        return
    fig, ax = plt.subplots()
    ax.plot(e, sigma)
    ax.set_ylim([0, 50])
    ax.set_xlabel("Energy")
    ax.set_ylabel("Width")
    fig.set_size_inches((16/3*2, 9/3*2))
    fig.savefig(folder+'/smooth_width.png')
    plt.close(fig)

# ============================================================================================================================
# ============================================================================================================================
# оптимизация параметров сглаживания
# ============================================================================================================================
# ============================================================================================================================
def funcFitSmooth(args, xanes, smoothType, exp, norm = None):
    smoothed_xanes, _ = funcFitSmoothHelper(args, xanes, smoothType, exp, norm)
    i = (exp.fit_intervals['smooth'][0]<=exp.xanes.energy) & (exp.xanes.energy<=exp.fit_intervals['smooth'][1])
    return np.sqrt(utils.integral(exp.xanes.energy[i], abs(smoothed_xanes.absorb[i]-exp.xanes.absorb[i])**2))

def funcFitSmoothHelper(args, xanes, smoothType, exp, norm = None):
    shift = value(args, 'shift')
        # t1 = time.time()
    if smoothType == 'fdmnes':
        if xanes.folder is None: xanes.folder = tempfile.mkdtemp(dir='./tmp', prefix='smooth_')
        if not os.path.exists(xanes.folder+'/out.txt'):
            xanes.save(xanes.folder+'/out.txt', exp.defaultSmoothParams.fdmnesSmoothHeader)
        Gamma_hole, Ecent, Elarg, Gamma_max, Efermi = getSmoothParams(args, ['Gamma_hole', 'Ecent', 'Elarg', 'Gamma_max', 'Efermi'])
        fdmnes_en1, res = smooth_fdmnes(xanes.energy, xanes.absorb, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi, folder=xanes.folder)
    elif smoothType == 'my_fdmnes':
        Gamma_hole, Ecent, Elarg, Gamma_max, Efermi = getSmoothParams(args, ['Gamma_hole', 'Ecent', 'Elarg', 'Gamma_max', 'Efermi'])
        fdmnes_en1, res = smooth_my_fdmnes(xanes.energy, xanes.absorb, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi)
    elif smoothType == 'linear':
        Gamma_hole, Gamma_max, Efermi = getSmoothParams(args, ['Gamma_hole', 'Gamma_max', 'Efermi'])
        fdmnes_en1, res = smooth_linear_conv(xanes.energy, xanes.absorb, Gamma_hole, Gamma_max, Efermi)
    elif smoothType == 'Muller':
        group, Efermi, Gamma_hole, alpha1, alpha2, alpha3 = getSmoothParams(args, ['group', 'Efermi', 'Gamma_hole', 'alpha1', 'alpha2', 'alpha3'])
        fdmnes_en1, res = smooth_Muller(xanes.energy, xanes.absorb, group, Gamma_hole, Efermi, alpha1, alpha2, alpha3)
    elif smoothType == 'piecewise':
        Gamma_hole, Gamma_max, Ecent = getSmoothParams(args, ['Gamma_hole', 'Gamma_max', 'Ecent'])
        fdmnes_en1, res = smooth_piecewise(xanes.energy, xanes.absorb, Gamma_hole, Gamma_max, Ecent)
    elif smoothType == 'multi_piecewise':
        sigma = getSmoothWidth(smoothType, xanes.energy, args)
        fdmnes_en1, res = generalSmooth(xanes.energy, xanes.absorb, sigma)
    elif smoothType == 'spline':
        sigma = getSmoothWidth(smoothType, xanes.energy, args)
        fdmnes_en1, res = generalSmooth(xanes.energy, xanes.absorb, sigma)
    else: error('Unknown smooth type '+smoothType)
        # t2 = time.time()
        # print("Smooth time=", t2 - t1)
    res, norm = utils.fit_to_experiment_by_norm_or_regression_mult_only(exp.xanes.energy, exp.xanes.absorb, exp.fit_intervals['norm'], fdmnes_en1, res, shift, norm)
    return utils.Xanes(exp.xanes.energy,res,xanes.folder), norm

def createArg(expList, smoothType, fixParamNames, commonParams):
    arg = []
    # добавляем общие параметры
    exp0Params = copy.deepcopy(expList[0].defaultSmoothParams[smoothType])
    if 'norm' in commonParams:
        exp0Params.append(param('norm', commonParams['norm'], [commonParams['norm']-0.02, commonParams['norm']+0.02], 0.0003, 0.0001))
    for pName in commonParams:
        assert (pName not in fixParamNames), 'Common params can not be fixed. Remove '+pName+' from common'
        found = [p for p in exp0Params if p['paramName']==pName]
        assert len(found)>0, 'Common param name '+pName+' not found in params'
        assert len(found) == 1, 'Duplicate param name '+pName+' in params'
        arg.append(found[0])
    # добавляем частные параметры
    for exp in expList:
        expParams = exp.defaultSmoothParams[smoothType]
        for p in expParams:
            pName = p['paramName']
            if (pName not in fixParamNames) and (pName not in commonParams):
                newParam = copy.deepcopy(p)
                newParam['paramName'] = exp.name+'_'+newParam['paramName']
                arg.append(newParam)
    return arg

def getOneArg(argsOfList, exp, smoothType):
    args = copy.deepcopy(exp.defaultSmoothParams[smoothType])
    for arg in args:
        # ищем параметр в числе общих
        pName = arg['paramName']
        found = [a for a in argsOfList if a['paramName']==pName]
        if len(found)>0:
            arg['value'] = found[0]['value']
        else:
            # ищем параметр в числе частных
            pName = exp.name+'_'+arg['paramName']
            found = [a for a in argsOfList if a['paramName']==pName]
            if len(found)>0:
                arg['value'] = found[0]['value']
    return args

def getNorm(normType, argsOfList, arg):
    if normType == 'variablePrivate': norm = None
    else:
        if normType == 'variableCommon': norm = value(argsOfList, 'norm')
        else:
            norm = value(arg, 'norm') # normType == 'fixed' => норма должна сидеть в параметрах по умолчанию
            assert norm is not None, 'When norm is fixed it must be included in deafaul smooth parameters'
    return norm

def funcFitSmoothList(argsOfList, expList, xanesList, smoothType, targetFunc, normType):
    l2 = []
    diffs = []
    es = []
    for j in range(len(expList)):
        exp = expList[j]
        arg = getOneArg(argsOfList, exp, smoothType)
        norm = getNorm(normType, argsOfList, arg)
        smoothed_xanes, normOut = funcFitSmoothHelper(arg, xanesList[j], smoothType, exp, norm)
        # print(normOut)
        i = (exp.fit_intervals['smooth'][0]<=exp.xanes.energy) & (exp.xanes.energy<=exp.fit_intervals['smooth'][1])
        diff = abs(smoothed_xanes.absorb[i]-exp.xanes.absorb[i])**2
        diffs.append(diff)
        es.append(exp.xanes.energy[i])
        partial_func_val = np.sqrt(utils.integral(exp.xanes.energy[i], diff))
        l2.append( partial_func_val )
    if len(expList) == 1: return l2[0]
    l2 = np.array(l2)
    if targetFunc == 'mean': return np.mean(l2)
    elif targetFunc == 'max(l2)': return np.max(l2)
    elif targetFunc == 'l2(max)':
        e = es[0]
        newDiffs = np.zeros([len(expList), e.size])
        newDiffs[0] = diffs[0]
        for j in range(1,len(expList)):
            newDiffs[j] = np.interp(e, es[j], diffs[j])
        maxDiff = np.max(newDiffs, axis=0)
        return np.sqrt(utils.integral(e, maxDiff))
    else: assert False, 'Unknown target func'

# fixParamNames - массив фиксируемых параметров (значения берутся из значений по умолчанию в экспериментах)
# commonParams0 - ассоциативный массив начальных значений общих параметров (интервалы поиска берутся из первого эксперимента)
# норма может быть: фиксированной (тогда она должна быть задана в параметрах каждого эксперимента), подбираемой: общей или частными
def fitSmooth(expList, xanesList0, smoothType = 'fdmnes', fixParamNames=[], commonParams0 = {}, targetFunc = 'max(l2)', plotTrace=False, crossValidationExp = None, crossValidationXanes = None, minimizeMethodType='seq', useGridSearch = True, useRefinement=True):
    xanesList = copy.deepcopy(xanesList0)
    if 'norm' in fixParamNames: normType = 'fixed'
    else:
        if 'norm' in commonParams0: normType = 'variableCommon'
        else: normType = 'variablePrivate'
    folder = tempfile.mkdtemp(dir='./tmp', prefix='smooth_')
    for i in range(len(expList)):
        os.makedirs(folder+'/'+expList[i].name)
        if xanesList[i].folder is not None:
            copyfile(xanesList[i].folder+'/out.txt', folder+'/'+expList[i].name+'/out.txt')
        xanesList[i].folder = folder+'/'+expList[i].name
        if xanesList[i].molecula is not None: xanesList[i].molecula.export_xyz(xanesList[i].folder+'/molecula.xyz')
    arg0 = createArg(expList, smoothType, fixParamNames, commonParams0)
    # 0.0001
    fmin, smooth_params, trace = minimize.minimizePokoord(funcFitSmoothList, arg0, minDeltaFunc = 1e-4, enableOutput = False, methodType=minimizeMethodType, parallel=False, useRefinement=useRefinement, useGridSearch=useGridSearch, returnTrace=True, f_kwargs={'expList':expList, 'xanesList':xanesList, 'smoothType': smoothType, 'targetFunc':targetFunc, 'normType':normType})
    with open(folder+'/func_smooth_value.txt', 'w') as f: json.dump(fmin, f)
    with open(folder+'/args_smooth.txt', 'w') as f: json.dump(smooth_params, f)
    with open(folder+'/args_smooth_human.txt', 'w') as f: f.write(minimize.arg2string(smooth_params))
    # выдаем предостережение, если достигли границы одного из параметров
    for p in smooth_params:
        d = p['rightBorder'] - p['leftBorder']
        if (abs(p['value']-p['leftBorder'])<=p['step']) or (abs(p['value']-p['rightBorder'])<=p['step']):
            print('Warning: parameter '+p['paramName']+'='+str(p['value'])+' is near border of domain ['+str(p['leftBorder'])+'; '+str(p['rightBorder'])+']')
    # считаем по отдельности размазку каждого xanes
    smoothed_xanes = []
    argsOfList = copy.deepcopy(smooth_params)
    for j in range(len(expList)):
        exp = expList[j]
        arg = getOneArg(argsOfList, exp, smoothType)
        norm = getNorm(normType, argsOfList, arg)
        fdmnes_xan, _ = funcFitSmoothHelper(arg, xanesList[j], smoothType, exp, norm)
        smoothed_xanes.append(fdmnes_xan)
        with open(folder+'/'+expList[j].name+'/args_smooth.txt', 'w') as f: json.dump(arg, f)
        with open(folder+'/'+expList[j].name+'/args_smooth_human.txt', 'w') as f: f.write(minimize.arg2string(arg))
        shift = value(arg,'shift')
        fdmnes.plotToFolder(folder+'/'+expList[j].name, exp, xanesList[j], fdmnes_xan, append = getSmoothWidth(smoothType, exp.xanes.energy-shift, arg))
        ind = (exp.fit_intervals['smooth'][0]<=exp.xanes.energy) & (exp.xanes.energy<=exp.fit_intervals['smooth'][1])
        partial_fmin = np.sqrt(utils.integral(exp.xanes.energy[ind], abs(fdmnes_xan.absorb[ind]-exp.xanes.absorb[ind])**2))
        with open(folder+'/'+expList[j].name+'/func_smooth_partial_value.txt', 'w') as f: json.dump(partial_fmin, f)
        plotSmoothWidthToFolder(smoothType, exp.xanes.energy[ind]-shift, arg, folder+'/'+expList[j].name)
    if plotTrace:
        privateFuncData = np.zeros([len(expList), len(trace)])
        norms = np.zeros([len(expList), len(trace)])
        for j in range(len(trace)):
            step = trace[j]
            argsOfList = step[0]
            for i in range(len(expList)):
                arg = getOneArg(argsOfList, expList[i], smoothType)
                norm = getNorm(normType, argsOfList, arg)
                privateFuncData[i,j] = funcFitSmooth(arg, xanesList[i], smoothType, expList[i], norm)
                _, norms[i,j] = funcFitSmoothHelper(arg, xanesList[i], smoothType, expList[i], norm)
        fig, ax = plt.subplots()
        steps = np.arange(1, len(trace)+1)
        targetFuncData = [step[1] for step in trace]
        for i in range(len(expList)): ax.plot(steps, privateFuncData[i], label=expList[i].name)
        ax.plot(steps, targetFuncData, label='target')
        if crossValidationExp is not None:
            crossValidationXanes = copy.deepcopy(crossValidationXanes)
            if smoothType == 'fdmnes':
                folder2 = folder+'/CV_'+crossValidationExp.name
                os.makedirs(folder2)
                if crossValidationXanes.folder is not None:
                    copyfile(crossValidationXanes.folder+'/out.txt', folder2+'/out.txt')
                crossValidationXanes.folder = folder2
            crossValidationData = []
            for step_i in range(len(trace)):
                step = trace[step_i]
                argsOfList = step[0]
                arg = getOneArg(argsOfList, crossValidationExp, smoothType)
                for a in arg:
                    if value(argsOfList, a['paramName']) is None:
                        mean = 0
                        for exp in expList: mean += value(argsOfList, exp.name+'_'+a['paramName'])
                        mean /= len(expList)
                        print('Warning: parameter '+a['paramName']+' is not common. Take mean for cross validation: '+str(mean))
                        a['value'] = mean
                fmin1 = funcFitSmooth(arg, crossValidationXanes, smoothType, crossValidationExp, np.mean(norms[:,step_i]))
                crossValidationData.append( fmin1 )
                if step_i == len(trace)-1:
                    with open(folder2+'/func_smooth_check_value.txt', 'w') as f: json.dump(fmin1, f)
                    with open(folder2+'/args_smooth.txt', 'w') as f: json.dump(arg, f)
                    fdmnes_xan, _ = funcFitSmoothHelper(arg, crossValidationXanes, smoothType, crossValidationExp, np.mean(norms[:,step_i]))
                    fdmnes.plotToFolder(folder2, crossValidationExp, crossValidationXanes, fdmnes_xan)
                    #print(np.mean(norms[:,step_i]))
            ax.plot(steps, crossValidationData, label='check')
        ax.set_xscale('log')
        ax.legend()
        fig.set_size_inches((16/3*2, 9/3*2))
        fig.savefig(folder+'/trace.png')
        plt.close(fig)
    return smoothed_xanes, smooth_params, fmin
# ============================================================================================================================
# ============================================================================================================================
# остальное
# ============================================================================================================================
# ============================================================================================================================

def deconvolve(e, xanes, smooth_params):
    xanes = utils.simpleSmooth(e, xanes, 1)
    Gamma_hole = minimize.value(smooth_params, 'Gamma_hole')
    Ecent = minimize.value(smooth_params, 'Ecent')
    Elarg = minimize.value(smooth_params, 'Elarg')
    Gamma_max = minimize.value(smooth_params, 'Gamma_max')
    Efermi = minimize.value(smooth_params, 'Efermi')
    n = e.size
    A = np.zeros([n,n])
    for i in range(n):
        #sigma = superWidth(e[i], Gamma_hole, Ecent, Elarg, Gamma_max, Efermi)
        sigma = 5
        kern = kernel(e, e[i], sigma)
        A[i] = kern/utils.integral(e, kern)
    res,_,_,s = np.linalg.lstsq(A,xanes,rcond=0.001)
    return res

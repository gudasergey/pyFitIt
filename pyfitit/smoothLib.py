from . import utils
utils.fixDisplayError()
import math, copy, os, json, hashlib, gc
from . import fdmnes, optimize, plotting, curveFitting
import numpy as np
import pandas as pd
from .optimize import param, arg2string, VectorPoint
import matplotlib.pyplot as plt
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
    w = Gamma_hole + Gamma_max*(0.5+1/math.pi*np.arctan( math.pi/3*Gamma_max/Elarg*(ee-1/ee**2) ))
    ind = e<Efermi
    w[ind] = Gamma_hole
    return w

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

def simpleSmooth(e, xanes, sigma, kernel='Cauchy'):
    new_xanes = np.zeros(e.shape)
    for i in range(e.size):
        if kernel == 'Cauchy':
            kern = kernelCauchy(e, e[i], sigma)
        elif kernel == 'Gauss':
            kern = kernelGauss(e, e[i], sigma)
        else: assert False, 'Unknown kernel name'
        norm = utils.integral(e, kern)
        if norm == 0: norm = 1
        new_xanes[i] = utils.integral(e, xanes*kern)/norm
    return new_xanes

# def smooth_fdmnes(e, xanes, exp_e, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi):
#     xanes = np.copy(xanes)
#     E_interval = e[-1] - e[0]
#     xanes[e<Efermi] = 0
#     new_xanes = np.zeros(exp_e.shape)
#     sigma = YvesWidth(exp_e, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi)
#     virtualStartEnergy = e[0]-E_interval; virtualEndEnergy = e[-1]+E_interval
#     norms = 1.0/math.pi*( np.arctan((virtualEndEnergy-exp_e)/sigma*2) - np.arctan((virtualStartEnergy-exp_e)/sigma*2) )
#     toAdd = 1.0/math.pi*( np.arctan((virtualEndEnergy-exp_e)/sigma*2) - np.arctan((e[-1]-exp_e)/sigma*2) ) * xanes[-1]
#     for i in range(exp_e.size):
#         kern = kernelCauchy(e, exp_e[i], sigma[i])
#         new_xanes[i] = (utils.integral(e, xanes*kern)+toAdd[i])/norms[i]
#     return exp_e, new_xanes

def smooth_fdmnes(e, xanes, exp_e, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi):
    # print(len(e), len(xanes), len(exp_e))
    xanes = np.copy(xanes)
    lastValueInd = xanes.size - int(xanes.size*0.05)
    lastValue = utils.integral(e[lastValueInd:], xanes[lastValueInd:])/(e[-1] - e[lastValueInd])
    E_interval = e[-1] - e[0]
    xanes[e<Efermi] = 0
    sigma = YvesWidth(exp_e, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi)
    virtualStartEnergy = e[0]-E_interval; virtualEndEnergy = e[-1]+E_interval
    norms = 1.0/math.pi*( np.arctan((virtualEndEnergy-exp_e)/sigma*2) - np.arctan((virtualStartEnergy-exp_e)/sigma*2) )
    toAdd = 1.0/math.pi*( np.arctan((virtualEndEnergy-exp_e)/sigma*2) - np.arctan((e[-1]-exp_e)/sigma*2) ) * lastValue
    max_memory = 2**32//4//8
    block_sz = max_memory//8//len(e) # max row count in kern
    block_count = (len(exp_e)+block_sz-1)//block_sz
    integr = np.zeros(len(exp_e))
    for bi in range(block_count):
        a,b = bi*block_sz,  min((bi+1)*block_sz, len(exp_e))
        exp_e_block = exp_e[a:b]
        sigma_block = sigma[a:b]
        kern = kernelCauchy(exp_e_block.reshape(-1,1), e.reshape(1,-1), sigma_block.reshape(-1,1))
        assert (kern.shape[0]==exp_e_block.size) and (kern.shape[1]==e.size)
        de = (e[1:]-e[:-1]).reshape(1,-1)
        f = xanes.reshape(1,-1) * kern
        integr[a:b] = np.sum((f[:,1:]+f[:,:-1])*de, axis=1)
    new_xanes = (0.5*integr + toAdd)/norms
    assert len(exp_e) == len(new_xanes)
    return exp_e, new_xanes

def smooth_fdmnes_notconv(e0, xanes0, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi):
    n0 = e0.size
    e,xanes = utils.expandEnergyRange(e0,xanes0)
    xanes[e<Efermi] = 0
    sigma = YvesWidth(e, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi)
    signals = np.zeros([e.size,e.size]) #столбец - сигнал от одного пика
    for i in range(e.size):
        if (i > 0) and (i<xanes.size-1): de = (e[i]-e[i-1])/2 + (e[i+1]-e[i])/2
        elif i==0:  de = e[1] - e[0]
        else: de = e[-1] - e[-2]
        kern = kernelCauchy(e, e[i], sigma[i])
        norm = utils.integral(e, kern)
        signals[:,i] = kern/norm*xanes[i]*de
    new_xanes = np.sum(signals, axis=1)
    return e0, new_xanes[n0:2*n0]


def smooth_adf(e0, xanes0, e, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi, reflect):
    # reflect spectrum
    if reflect:
        e0 = np.hstack((e0, e0[-1] + e0[-1]-np.flip(e0,0) ))
        fxanes0 = np.flip(xanes0,0); fxanes0[0] = 0
        xanes0 = np.hstack((xanes0, fxanes0))
    start = e0[0]
    sigma = YvesWidth(e0-start, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi)
    new_xanes = np.zeros(e.size)
    for i in range(e0.size):
        kern = kernelCauchy(e-start, e0[i]-start, sigma[i])
        new_xanes += kern*xanes0[i]
    return e, new_xanes


def smooth_fdmnes_multi(e, manyXanes, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi):
    xanes = np.copy(manyXanes)
    E_interval = e[-1] - e[0]
    xanes[:, np.where(e<Efermi)[0]] = 0
    sigma = YvesWidth(e, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi)
    virtualStartEnergy = e[0]-E_interval; virtualEndEnergy = e[-1]+E_interval
    norms = 1.0/math.pi*( np.arctan((virtualEndEnergy-e)/sigma*2) - np.arctan((virtualStartEnergy-e)/sigma*2) )
    norms = norms.reshape(1,-1)
    toAdd = 1.0/math.pi*( np.arctan((virtualEndEnergy-e)/sigma*2) - np.arctan((e[-1]-e)/sigma*2) )
    toAdd = toAdd.reshape(1,-1) * xanes[:,-1].reshape(-1,1)
    kern = kernelCauchy(e.reshape(-1,1), e.reshape(1,-1), sigma.reshape(-1,1))
    de = (e[1:]-e[:-1]).reshape((1,1,-1))
    f = np.expand_dims(xanes,1) * np.expand_dims(kern,0)
    new_xanes = (0.5*np.sum((f[:,:,1:]+f[:,:,:-1])*de, axis=2).reshape(manyXanes.shape) + toAdd)/norms
    del xanes
    gc.collect()
    return new_xanes


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
    def __init__(self, Efermi, shift):
        shiftParam = param('shift', shift, [shift-20, shift+20], 1, 0.25)
        efermiParam = param('Efermi', Efermi, [Efermi-20,Efermi+20], 2, 0.2)
        self.params = {'fdmnes':\
            [param('Gamma_hole', 1.5, [0.1,10], 0.4, 0.1), param('Ecent', 26, [1,100], 3, 0.5),\
             param('Elarg', 39, [1,100], 5, 0.5), param('Gamma_max', 15, [5,25], 1, 0.2), efermiParam\
            ], 'linear':\
            [param('Gamma_hole', -1, [-20,20], 1, 0.1), param('Gamma_max', 22, [-30,50], 2, 0.5), efermiParam\
            ], 'Muller':\
            [param('group', 8, [1,18], 1, 1), param('Gamma_hole', 1, [0.01,5], 0.02, 0.002), efermiParam,\
             param('alpha1', 0, [-0.999,2], 0.1, 0.01), param('alpha2', 0, [-0.9,2], 0.1, 0.01), param('alpha3', 0, [-0.9,2], 0.1, 0.01)\
            ], 'piecewise':\
            [param('Gamma_hole', 5, [0.1,10], 0.2, 0.03), param('Gamma_max', 20, [5,40], 2, 0.5), param('Ecent', 30, [-10,100], 2, 0.2)\
            ], 'multi_piecewise':\
            [param('g0', 0.3, [0.01,2], 0.03, 0.005), param('e1', 10, [0,25], 1, 0.3), param('g1', 5, [1,10], 0.2, 0.05),\
            param('e2', 30, [25,70], 2, 0.5), param('g2', 8, [1,25], 0.5, 0.05), param('e3', 100, [70,150], 5, 1), param('g3', 20, [5,50], 1, 0.05)\
            ], 'spline':\
            [efermiParam\
            ], 'optical': []}
        n = 20
        for i in range(n): self.params['spline'].append( param('g_'+str(i), 1+i/n*35, [0,50], 0.5, 0.05) )
        self.params['adf'] = copy.deepcopy(self.params['fdmnes'])
        for smoothType in self.params: self.params[smoothType] = VectorPoint(self.params[smoothType])
        self.shiftIsAbsolute = True
        self.search_shift_level = 0.25
        for smoothType in self.params:
            if smoothType != 'adf': self.params[smoothType].append(shiftParam)
            else: self.params[smoothType].append(param('shift', 0, [-20, 20], 1, 0.25))

    def __getitem__(self, smoothType):
        return self.params[smoothType]

    def getDict(self, smoothType):
        res = {}
        for a in self.params[smoothType]:
            res[a['paramName']] = a['value']
        return res


def findEfermiOnRawSpectrum(energy, intensity):
    smoothedTheorXanes = simpleSmooth(energy, intensity, 4)
    tmx = (np.min(smoothedTheorXanes) + np.max(smoothedTheorXanes)) / 2
    ind = np.where(smoothedTheorXanes > tmx)[0]
    tme = energy[ind[0]]
    return tme, smoothedTheorXanes


# spectrumType = 'fdmmnes' or 'adf'
def checkShift(expXanes, theorXanes, shift, spectrumType):
    smoothedTheorXanes = simpleSmooth(theorXanes.energy, theorXanes.intensity, 4)
    tmx = (np.min(smoothedTheorXanes) + np.max(smoothedTheorXanes)) / 2
    ind = np.where(smoothedTheorXanes > tmx)[0]
    tme = theorXanes.energy[ind[0]]
    if spectrumType == 'adf': tme = theorXanes.energy[0]
    emx = (np.min(expXanes.intensity) + np.max(expXanes.intensity)) / 2
    ind = np.where(expXanes.intensity > emx)[0]
    eme = expXanes.energy[ind[0]]
    shiftCheck = eme-tme
    if abs(shift-shiftCheck)>50: message = 'Warning: wrong shift detected. Recommend value near '+str(int(shiftCheck))
    else: message = ''
    return message, eme, tme


def getSmoothParams(arg, names):
    res = ()
    for name in names: res = res + (arg[name],)
    return res


def getSmoothWidth(smoothType, e, args):
    ps = DefaultSmoothParams(0,0)
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
    fig, ax = plt.subplots(figsize=plotting.figsize)
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
def funcFitSmooth(args, xanes, smoothType, exp, norm = None, fitDiffFrom=None):
    smoothed_xanes, _ = funcFitSmoothHelper(args, xanes, smoothType, exp, norm)
    i = (exp.intervals['fit_smooth'][0]<=exp.spectrum.energy) & (exp.spectrum.energy<=exp.intervals['fit_smooth'][1])
    if fitDiffFrom is None:
        return np.sqrt(utils.integral(exp.spectrum.energy[i], abs(smoothed_xanes.intensity[i]-exp.spectrum.intensity[i])**2))
    else:
        fitDiffFromExpXanes = fitDiffFrom['exp'].xanes
        fitDiffFromXanes = fitDiffFrom['xanes']
        fitDiffFromExpXanes_absorb = np.interp(exp.spectrum.energy, fitDiffFromExpXanes.energy, fitDiffFromExpXanes.intensity)
        fitDiffFrom_smoothed_xanes, _ = funcFitSmoothHelper(args, fitDiffFromXanes, smoothType, exp, norm)
        purity = value(args, 'purity')
        return np.sqrt(utils.integral(exp.spectrum.energy[i], (purity*(smoothed_xanes.intensity[i]-fitDiffFrom_smoothed_xanes.intensity[i]) - (exp.spectrum.intensity[i]-fitDiffFromExpXanes_absorb[i]))**2 ))


def funcFitSmoothHelper(smooth_params, spectrum, smoothType, exp, norm=None):
    return smoothInterpNorm(smooth_params, spectrum, smoothType, exp.spectrum, exp.intervals['fit_norm'], norm)


def smoothInterpNorm(smooth_params, spectrum, smoothType, exp_spectrum, fit_norm_interval, norm=None):
    shift = smooth_params['shift']
    spectrum_energy = spectrum.energy + shift
    # t1 = time.time()
    if (smoothType == 'fdmnes') or (smoothType == 'fdmnes_notconv') or (smoothType == 'adf'):
        Gamma_hole, Ecent, Elarg, Gamma_max, Efermi = smooth_params['Gamma_hole'], smooth_params['Ecent'], smooth_params['Elarg'], smooth_params['Gamma_max'], smooth_params['Efermi']
        if not fdmnes.useEpsiiShift: Efermi += shift
        if smoothType == 'fdmnes':
            fdmnes_en1, res = smooth_fdmnes(spectrum_energy, spectrum.intensity, exp_spectrum.energy, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi)
        elif smoothType == 'fdmnes_notconv':
            fdmnes_en1, res = smooth_fdmnes_notconv(spectrum_energy, spectrum.intensity, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi)
        else: # adf
            fdmnes_en1, res = smooth_adf(spectrum_energy, spectrum.intensity, exp_spectrum.energy, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi, smooth_params['reflect'])
    elif smoothType == 'linear':
        Gamma_hole, Gamma_max, Efermi = getSmoothParams(smooth_params, ['Gamma_hole', 'Gamma_max', 'Efermi'])
        fdmnes_en1, res = smooth_linear_conv(spectrum_energy, spectrum.intensity, Gamma_hole, Gamma_max, Efermi)
    elif smoothType == 'Muller':
        group, Efermi, Gamma_hole, alpha1, alpha2, alpha3 = getSmoothParams(smooth_params, ['group', 'Efermi', 'Gamma_hole', 'alpha1', 'alpha2', 'alpha3'])
        fdmnes_en1, res = smooth_Muller(spectrum_energy, spectrum.intensity, group, Gamma_hole, Efermi, alpha1, alpha2, alpha3)
    elif smoothType == 'piecewise':
        Gamma_hole, Gamma_max, Ecent = getSmoothParams(smooth_params, ['Gamma_hole', 'Gamma_max', 'Ecent'])
        fdmnes_en1, res = smooth_piecewise(spectrum_energy, spectrum.intensity, Gamma_hole, Gamma_max, Ecent)
    elif smoothType == 'multi_piecewise':
        sigma = getSmoothWidth(smoothType, spectrum_energy, smooth_params)
        fdmnes_en1, res = generalSmooth(spectrum_energy, spectrum.intensity, sigma)
    elif smoothType == 'spline':
        sigma = getSmoothWidth(smoothType, spectrum_energy, smooth_params)
        fdmnes_en1, res = generalSmooth(spectrum_energy, spectrum.intensity, sigma)
    elif smoothType == 'optical':
        fdmnes_en1 = spectrum.energy
        res = spectrum.intensity
    else: assert False, 'Unknown smooth type '+smoothType
    # t2 = time.time()
    # print("Smooth time=", t2 - t1)
    fit_norm_interval = copy.deepcopy(fit_norm_interval)
    if spectrum_energy[0] > fit_norm_interval[0]: fit_norm_interval[0] = spectrum_energy[0]
    if spectrum_energy[-1] < fit_norm_interval[-1]: fit_norm_interval[-1] = spectrum_energy[-1]
    res, norm = curveFitting.fit_to_experiment_by_norm_or_regression_mult_only(exp_spectrum.energy, exp_spectrum.intensity, fit_norm_interval, fdmnes_en1, res, 0, norm)
    # if smoothType == 'adf': print(norm)
    return utils.Spectrum(exp_spectrum.energy,res), norm


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
    return VectorPoint(arg)

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
            assert norm is not None, 'When norm is fixed it must be included in deafault smooth parameters'
    return norm

def funcFitSmoothList(argsOfList, expList, xanesList, smoothType, targetFunc, normType, fitDiffFrom):
    l2 = []
    diffs = []
    es = []
    if fitDiffFrom is not None:
        fitDiffFromExpXanes = fitDiffFrom['exp'].xanes
        fitDiffFromXanes = fitDiffFrom['xanes']
    for j in range(len(expList)):
        exp = expList[j]
        arg = getOneArg(argsOfList, exp, smoothType)
        norm = getNorm(normType, argsOfList, arg)
        smoothed_xanes, normOut = funcFitSmoothHelper(arg, xanesList[j], smoothType, exp, norm)
        # print(normOut)
        i = (exp.intervals['fit_smooth'][0]<=exp.spectrum.energy) & (exp.spectrum.energy<=exp.intervals['fit_smooth'][1])
        if fitDiffFrom is None:
            diff = abs(smoothed_xanes.intensity[i]-exp.spectrum.intensity[i])**2
        else:
            fitDiffFromExpXanes_absorb = np.interp(exp.spectrum.energy, fitDiffFromExpXanes.energy, fitDiffFromExpXanes.intensity)
            fitDiffFrom_smoothed_xanes, _ = funcFitSmoothHelper(arg, fitDiffFromXanes, smoothType, exp, norm)
            purity = value(arg, 'purity')
            diff = abs(purity*(smoothed_xanes.intensity[i]-fitDiffFrom_smoothed_xanes.intensity[i]) - (exp.spectrum.intensity[i]-fitDiffFromExpXanes_absorb[i]))**2
        diffs.append(diff)
        es.append(exp.spectrum.energy[i])
        partial_func_val = np.sqrt(utils.integral(exp.spectrum.energy[i], diff))
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
# fitDiffFrom = {'exp':exp, 'xanes':xanes}
def fitSmooth(expList, xanesList0, smoothType='fdmnes', fixParamNames=[], commonParams0={}, targetFunc='l2(max)', crossValidationExp=None, crossValidationXanes=None, fitDiffFrom=None, optimizeWithoutPlot=False, folder='result'):
    xanesList = copy.deepcopy(xanesList0)
    if 'norm' in fixParamNames: normType = 'fixed'
    else:
        if 'norm' in commonParams0: normType = 'variableCommon'
        else: normType = 'variablePrivate'
    if not optimizeWithoutPlot:
        for i in range(len(expList)):
            os.makedirs(folder+'/'+expList[i].name, exist_ok=True)
            xanesList[i].folder = folder+'/'+expList[i].name
            if xanesList[i].molecula is not None: xanesList[i].molecula.export_xyz(xanesList[i].folder+'/molecula.xyz')
    arg0 = createArg(expList, smoothType, fixParamNames, commonParams0)
    arg0_1 = [arg0[i]['value'] for i in range(len(arg0))]
    bounds = [[arg0[i]['leftBorder'], arg0[i]['rightBorder']] for i in range(len(arg0))]
    for i in range(len(arg0)): assert  bounds[i][0]<= arg0_1[i] and arg0_1[i] <= bounds[i][1]

    def funcFitSmoothList1(arg1, *params):
        arg = copy.deepcopy(arg0)
        for i in range(len(arg0)): arg[i]['value'] = arg1[i]
        res = funcFitSmoothList(arg, *params)
        return res

    fmin, smooth_params_vec = optimize.minimize(funcFitSmoothList1, arg0_1, bounds, fun_args=(expList, xanesList, smoothType, targetFunc, normType, fitDiffFrom), method='scipy')
    # print(fmin)
    smooth_params = copy.deepcopy(arg0)
    for i in range(len(arg0)): smooth_params[i]['value'] = smooth_params_vec[i]
    if optimizeWithoutPlot: return smooth_params
    with open(folder+'/func_smooth_value.txt', 'w') as f: json.dump(fmin, f)
    with open(folder+'/args_smooth.txt', 'w') as f: json.dump(smooth_params, f)
    with open(folder+'/args_smooth_human.txt', 'w') as f: f.write(optimize.arg2string(smooth_params))
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
        arg
        with open(folder+'/'+expList[j].name+'/args_smooth.txt', 'w') as f: json.dump(arg.to_dict(), f)
        with open(folder+'/'+expList[j].name+'/args_smooth_human.txt', 'w') as f: f.write(arg2string(arg))
        shift = value(arg,'shift')
        plotting.plotToFolder(folder+'/'+expList[j].name, exp, xanesList[j], fdmnes_xan, append = {'data':getSmoothWidth(smoothType, exp.spectrum.energy, arg), 'label':'smooth width', 'twinx':True}, fileName='xanes')
        if fitDiffFrom is not None:
            purity = value(arg, 'purity')
            fitDiffFromExpXanes = fitDiffFrom['exp'].xanes
            fitDiffFromXanes = fitDiffFrom['xanes']
            fitDiffFromExpXanes_absorb = np.interp(exp.spectrum.energy, fitDiffFromExpXanes.energy, fitDiffFromExpXanes.intensity)
            fitDiffFrom_smoothed_xanes, _ = funcFitSmoothHelper(arg, fitDiffFromXanes, smoothType, exp, norm)
            expDiff = copy.deepcopy(exp)
            expDiff.spectrum = utils.Spectrum(exp.spectrum.energy, exp.spectrum.intensity-fitDiffFromExpXanes.intensity)
            xanesDiff = utils.Spectrum(exp.spectrum.energy, purity*(np.interp(exp.spectrum.energy, xanesList[j].energy+shift, xanesList[j].intensity)-fitDiffFromExpXanes_absorb))
            smoothed_xanesDiff = utils.Spectrum(fdmnes_xan.energy, purity*(fdmnes_xan.intensity-fitDiffFrom_smoothed_xanes.intensity))
            plotting.plotToFolder(folder+'/'+expList[j].name, expDiff, xanesDiff, smoothed_xanesDiff, append = {'data':getSmoothWidth(smoothType, exp.spectrum.energy, arg), 'label':'smooth width', 'twinx':True}, fileName='xanesDiff')
        ind = (exp.intervals['fit_smooth'][0]<=exp.spectrum.energy) & (exp.spectrum.energy<=exp.intervals['fit_smooth'][1])
        if fitDiffFrom is None:
            partial_fmin = np.sqrt(utils.integral(exp.spectrum.energy[ind], abs(fdmnes_xan.intensity[ind]-exp.spectrum.intensity[ind])**2))
        else:
            partial_fmin = np.sqrt(utils.integral( exp.spectrum.energy[ind], (purity*(fdmnes_xan.intensity[ind]-fitDiffFrom_smoothed_xanes.intensity[ind]) - (exp.spectrum.intensity[ind]-fitDiffFromExpXanes_absorb[ind]))**2 ))
        with open(folder+'/'+expList[j].name+'/func_smooth_partial_value.txt', 'w') as f: json.dump(partial_fmin, f)
        # plotSmoothWidthToFolder(smoothType, exp.spectrum.energy[ind], arg, folder+'/'+expList[j].name)
    return smoothed_xanes, smooth_params, fmin
# ============================================================================================================================
# ============================================================================================================================
# остальное
# ============================================================================================================================
# ============================================================================================================================

def deconvolve(e, xanes, smooth_params):
    xanes = utils.simpleSmooth(e, xanes, 1)
    Gamma_hole = value(smooth_params, 'Gamma_hole')
    Ecent = value(smooth_params, 'Ecent')
    Elarg = value(smooth_params, 'Elarg')
    Gamma_max = value(smooth_params, 'Gamma_max')
    Efermi = value(smooth_params, 'Efermi')
    n = e.size
    A = np.zeros([n,n])
    for i in range(n):
        #sigma = superWidth(e[i], Gamma_hole, Ecent, Elarg, Gamma_max, Efermi)
        sigma = 5
        kern = kernel(e, e[i], sigma)
        A[i] = kern/utils.integral(e, kern)
    res,_,_,s = np.linalg.lstsq(A,xanes,rcond=0.001)
    return res


# параметры размазки и новый диапазон энергии беруться из эксперимента. norm - можно задавать, а можно писать None для автоопределения
# cacheStatus = True if read from cache
def smoothDataFrame(smoothParams, xanes_df, smoothType, exp_spectrum, fit_norm_interval, norm=None, folder=None, returnCacheStatus=False, energy=None):
    xanes_df_is_dataframe = isinstance(xanes_df, pd.DataFrame)
    if folder is not None:
        folder = utils.fixPath(folder)
        assert os.path.exists(folder+os.sep+'spectra.txt'), 'File spectra.txt doesn\'t exist in folder '+folder
        smoothFileName = folder+os.sep+'spectra_smooth.txt'
        smoothParamsFileName = folder+os.sep+'spectra_smooth_params.txt'
        smoothParams1 = {}
        for p in ['shift', 'Gamma_hole', 'Gamma_max', 'Ecent', 'Elarg', 'Efermi']:
            smoothParams1[p] = smoothParams[p]
        smoothParams = smoothParams1
        smoothParams['norm'] = norm
        smoothParams['useEpsii'] = fdmnes.useEpsiiShift
        with open(folder+os.sep+'spectra.txt', 'rb') as f:
            smoothParams['hash'] = hashlib.md5(f.read()).hexdigest()
        if os.path.isfile(smoothFileName) and os.path.isfile(smoothParamsFileName):
            with open(smoothParamsFileName, 'r') as f: cachedParams = json.load(f)
            smoothParamsCached = cachedParams['smoothParams']
            en = np.array(cachedParams['energy'])
            if 'exp' in cachedParams: exp_xanes = np.array(cachedParams['exp'])
            else: exp_xanes = np.array([0])
            equal = True
            if len(smoothParams) == len(smoothParamsCached):
                for p in smoothParams:
                    if smoothParams[p] != smoothParamsCached[p]: equal = False
                if not np.array_equal(en, exp_spectrum.energy): equal = False
                if not np.array_equal(exp_xanes, exp_spectrum.intensity): equal = False
            else: equal = False
            if equal:
                res = pd.read_csv(smoothFileName, sep=' ')
                if returnCacheStatus: return res, True
                else: return res
    if energy is None:
        energy = utils.getEnergy(xanes_df)
    if xanes_df_is_dataframe:
        xanes_df = xanes_df.to_numpy()
    # smoothed_xanes = funcFitSmoothHelperMulti(exp.defaultSmoothParams[smoothType], xanes_energy, xanes_df.values, exp, norm)
    smoothed_xanes = np.zeros([xanes_df.shape[0], exp_spectrum.energy.size])
    for k in range(smoothed_xanes.shape[0]):
        xanes = utils.Spectrum(energy, xanes_df[k,:])
        smoothed_xanes1, _ = smoothInterpNorm(smoothParams, xanes, smoothType, exp_spectrum, fit_norm_interval,  norm)
        smoothed_xanes[k] = smoothed_xanes1.intensity
    if xanes_df_is_dataframe:
        res = utils.makeDataFrame(exp_spectrum.energy, smoothed_xanes)
        if folder is not None:
            res.to_csv(smoothFileName, sep=' ', index=False)
            with open(smoothParamsFileName, 'w') as f: json.dump({'smoothParams':smoothParams, 'energy':exp_spectrum.energy.tolist(), 'exp':exp_spectrum.intensity.tolist()}, f)
        if returnCacheStatus:
            return res, False
        else:
            return res
    else: return smoothed_xanes, exp_spectrum.energy


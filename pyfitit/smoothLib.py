from . import utils
utils.fixDisplayError()
import math, copy, os, json, hashlib, gc, scipy, statsmodels.nonparametric.kernel_regression
from . import fdmnes, optimize, plotting, curveFitting
import numpy as np
import pandas as pd
from .optimize import param, arg2string, VectorPoint
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.font_manager import FontProperties


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


def simpleSmooth(e, xanes, sigma, kernel='Cauchy', new_e=None, sigma2percent=0.1, gaussWeight=0.2, assumeZeroInGaps=False, expandParams=None):
    """
    Smoothing
    :param e: argument (energy for xanes spectrum)
    :param xanes: function values (intensity for xanes spectrum)
    :param sigma: smooth width, scalar or vector of size same as new_e (or e if new_e=None)
    :param kernel: smooth kernel: 'Cauchy' 'Gauss' or 'C+G' with gaussWeight
    :param new_e: new argument for smooth result calculation (default - take e)
    :param sigma2percent: multiplier of sigma for Gauss kernel when kernel='C+G'
    :param gaussWeight: multiplier of Gauss kernel when kernel='C+G'
    :param assumeZeroInGaps: whether to assume, that spectrum = 0 between points (i.e. adf type smoothing)
    :param expandParams: params of utils.expandByReflection except e, xanes
    """
    assert len(e.shape) == 1
    assert len(xanes.shape) == 1
    sigma0 = sigma
    e0, xanes0 = e, xanes
    if expandParams is not None:
        e, xanes = utils.expandByReflection(e, xanes, **expandParams)
    # plotting.plotToFile(e,xanes,'expand', e0,xanes0,'init', fileName=f'debug.png')
    if new_e is None: new_e = e0
    new_xanes = np.zeros(new_e.shape)
    for i in range(new_e.size):
        sigma = sigma0[i] if isinstance(sigma0, np.ndarray) else sigma0
        if kernel == 'Cauchy':
            kern = kernelCauchy(e, new_e[i], sigma)
        elif kernel == 'Gauss':
            kern = kernelGauss(e, new_e[i], sigma)
        elif kernel == 'C+G':
            kern = kernelCauchy(e, new_e[i], sigma) + gaussWeight*kernelGauss(e, new_e[i], sigma*sigma2percent)
        else: assert False, 'Unknown kernel name'
        norm = 1 if assumeZeroInGaps else utils.integral(e, kern)
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
    assert xanes.size>=2
    xanes = np.copy(xanes)
    lastValueInd = xanes.size - int(xanes.size*0.05)
    lastValueInd = min(xanes.size-2, lastValueInd)
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
    """
    Smooth spectrum
    """
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


def generalSmooth(e, xanes, sigma, kernel='Cauchy'):
    eleft = np.linspace(e[0]-10, e[0]-(e[1]-e[0]), 10)
    xleft = np.zeros(eleft.shape)
    eright = np.linspace(e[-1]+(e[-1]-e[-2]), e[-1]+50, 10)
    xright = np.zeros(eleft.shape) + xanes[-1]
    e_new = np.hstack((eleft,e,eright))
    xanes = np.hstack((xleft,xanes,xright))
    new_xanes = np.zeros(e.shape)
    for i in range(e.size):
        if kernel == 'Cauchy':
            kern = kernelCauchy(e_new, e[i], sigma[i])
        elif kernel == 'Gauss':
            kern = kernelGauss(e_new, e[i], sigma[i])
        norm = utils.integral(e_new, kern)
        new_xanes[i] = utils.integral(e_new, xanes*kern)/norm
    return e, new_xanes


class DefaultSmoothParams:
    def __init__(self, Efermi, shift):
        shiftParam = param('shift', shift, [shift-20, shift+20], 1, 0.25)
        efermiParam = param('Efermi', Efermi, [Efermi-20,Efermi+20], 2, 0.2)
        self.params = {'fdmnes':\
            [param('Gamma_hole', 1.5, [0.1,10], 0.4, 0.1), param('Ecent', 26, [1,100], 3, 0.5),\
             param('Elarg', 39, [1,100], 5, 0.5), param('Gamma_max', 15, [5,50], 1, 0.2), efermiParam\
            ], 'simple_Gauss': [param('sigma', 1.5, [0.1,10], 0.1, 0.01)],
               'simple_Cauchy': [param('sigma', 1.5, [0.1,10], 0.1, 0.01)],
               'simple_C+G':\
            [param('sigma', 1.5, [0.1, 10], 0.1, 0.01), param('sigma2percent', 0.1, [0.01, 2], 0.1, 0.01), param('gaussWeight', 0.1, [0.01, 10], 0.1, 0.01)],
               'simple_Cauchy_then_Gauss': [param('sigma_G', 0.1, [0.01,2], 0.1, 0.01), param('sigma_C', 1.5, [0.1,10], 0.1, 0.01)],
               'linear':\
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


# searchLevel = 0.1 - close to min, 0.9 - close to max
def findEfermiOnRawSpectrum(energy, intensity, searchLevel=0.5):
    smoothedTheorXanes = simpleSmooth(energy, intensity, 4)
    tmx = np.min(smoothedTheorXanes)*(1-searchLevel) + np.max(smoothedTheorXanes)*searchLevel
    ind = np.where(smoothedTheorXanes < tmx)[0]
    tme = energy[ind[-1]]
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
    if abs(shift-shiftCheck) > 50:
        message = 'Warning: wrong shift detected. Recommend value near '+str(int(shiftCheck))
    else: message = ''
    return message, eme, tme


def getPreliminarySmoothParams(xanes):
    efermi, _ = findEfermiOnRawSpectrum(xanes.energy, xanes.intensity, searchLevel=0.1)
    efermi -= 5
    params = {'Efermi':efermi, 'Gamma_hole':2, 'Ecent':50, 'Elarg':50, 'Gamma_max':10, 'shift':0}
    energyInterval = [efermi, efermi+150]
    return params, energyInterval


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
    fig, ax = plotting.createfig()
    ax.plot(e, sigma)
    ax.set_ylim([0, 50])
    ax.set_xlabel("Energy")
    ax.set_ylabel("Width")
    plotting.savefig(folder+'/smooth_width.png', fig)
    plotting.close(fig)


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


def smoothInterpNorm(smoothParams, spectrum, smoothType, expSpectrum, fitNormInterval=None, norm=None, normType='multOnly'):
    assert smoothType in ['fdmnes', 'fdmnes_notconv', 'adf', 'simple_Gauss', 'simple_Cauchy', 'simple_Cauchy_then_Gauss', 'simple_C+G', 'linear', 'Muller', 'piecewise', 'multi_piecewise', 'spline', 'optical']
    if norm is None and 'norm' in smoothParams: norm = smoothParams['norm']
    if 'normFixType' in smoothParams: normType = smoothParams['normFixType']
    shift = smoothParams['shift']
    spectrum_energy = spectrum.energy + shift
    # t1 = time.time()
    if smoothType in ['fdmnes', 'fdmnes_notconv', 'adf']:
        Gamma_hole, Ecent, Elarg, Gamma_max, Efermi = smoothParams['Gamma_hole'], smoothParams['Ecent'], smoothParams['Elarg'], smoothParams['Gamma_max'], smoothParams['Efermi']
        if not fdmnes.useEpsiiShift: Efermi += shift
        if smoothType in ['fdmnes','fdmnes with linear norm']:
            fdmnes_en1, res = smooth_fdmnes(spectrum_energy, spectrum.intensity, expSpectrum.energy, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi)
        elif smoothType == 'fdmnes_notconv':
            fdmnes_en1, res = smooth_fdmnes_notconv(spectrum_energy, spectrum.intensity, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi)
        else: # adf
            fdmnes_en1, res = smooth_adf(spectrum_energy, spectrum.intensity, expSpectrum.energy, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi, smoothParams['reflect'])
    elif smoothType in ['simple_Gauss', 'simple_Cauchy', 'simple_Cauchy_then_Gauss', 'simple_C+G']:
        if smoothType == 'simple_Cauchy_then_Gauss':
            res = simpleSmooth(spectrum_energy, spectrum.intensity, smoothParams['sigma_C'], kernel='Cauchy')
            res = simpleSmooth(spectrum_energy, res, smoothParams['sigma_G'], kernel='Gauss')
        else:
            if smoothType == 'simple_C+G':
                res = simpleSmooth(spectrum_energy, spectrum.intensity, smoothParams['sigma'], kernel='C+G', sigma2percent=smoothParams['sigma2percent'], gaussWeight=smoothParams['gaussWeight'])
            else:
                res = simpleSmooth(spectrum_energy, spectrum.intensity, smoothParams['sigma'], kernel=smoothType[7:])
        fdmnes_en1 = spectrum_energy
    elif smoothType == 'linear':
        Gamma_hole, Gamma_max, Efermi = getSmoothParams(smoothParams, ['Gamma_hole', 'Gamma_max', 'Efermi'])
        fdmnes_en1, res = smooth_linear_conv(spectrum_energy, spectrum.intensity, Gamma_hole, Gamma_max, Efermi)
    elif smoothType == 'Muller':
        group, Efermi, Gamma_hole, alpha1, alpha2, alpha3 = getSmoothParams(smoothParams, ['group', 'Efermi', 'Gamma_hole', 'alpha1', 'alpha2', 'alpha3'])
        fdmnes_en1, res = smooth_Muller(spectrum_energy, spectrum.intensity, group, Gamma_hole, Efermi, alpha1, alpha2, alpha3)
    elif smoothType == 'piecewise':
        Gamma_hole, Gamma_max, Ecent = getSmoothParams(smoothParams, ['Gamma_hole', 'Gamma_max', 'Ecent'])
        fdmnes_en1, res = smooth_piecewise(spectrum_energy, spectrum.intensity, Gamma_hole, Gamma_max, Ecent)
    elif smoothType == 'multi_piecewise':
        sigma = getSmoothWidth(smoothType, spectrum_energy, smoothParams)
        fdmnes_en1, res = generalSmooth(spectrum_energy, spectrum.intensity, sigma)
    elif smoothType == 'spline':
        sigma = getSmoothWidth(smoothType, spectrum_energy, smoothParams)
        fdmnes_en1, res = generalSmooth(spectrum_energy, spectrum.intensity, sigma)
    elif smoothType == 'optical':
        fdmnes_en1 = spectrum.energy
        res = spectrum.intensity
    else: assert False, 'Unknown smooth type '+smoothType
    # t2 = time.time()
    # print("Smooth time=", t2 - t1)
    if fitNormInterval is None: fitNormInterval = [spectrum_energy[0], spectrum_energy[-1]]
    fitNormInterval = list(fitNormInterval)
    if spectrum_energy[0] > fitNormInterval[0]: fitNormInterval[0] = spectrum_energy[0]
    if spectrum_energy[-1] < fitNormInterval[-1]: fitNormInterval[-1] = spectrum_energy[-1]
    res, norm = curveFitting.fit_to_experiment_by_norm_or_regression(expSpectrum.energy, expSpectrum.intensity, fitNormInterval, fdmnes_en1, res, 0, norm, normType=normType)
    return utils.Spectrum(expSpectrum.energy, res), norm


def fitSmoothSimple(spectrum, smoothType, exp_spectrum, initialData=None, fixedParams=None, fit_smooth_interval=None, userBounds=None, plotFileName=None, printDebug=True, smoothInterpNormParams=None):
    """
    Find best smooth params
    :param spectrum: spectrum to smooth
    :param smoothType: string: 'fdmnes', 'adf', 'optical', ... (see DefaultSmoothParams)
    :param exp_spectrum:
    :param initialData: data to initialize DefaultSmoothParams (Efermi, shift)
    :param fixedParams: dict
    :param fit_smooth_interval:
    :param userBounds: dict - user defined intervals for smooth param search
    :param plotFileName: if None - do not plot
    :param printDebug:
    :param smoothInterpNormParams:
    :return: smooth_params, smoothed_spectrum, best_rFactor
    """
    if fixedParams is None: fixedParams = {}
    if userBounds is None: userBounds = {}
    if smoothInterpNormParams is None: smoothInterpNormParams = {}
    if 'norm' not in fixedParams: fixedParams['norm'] = None
    if initialData is None:
        if 'shift' in fixedParams: shift = fixedParams['shift']
        else:
            shift = 0
            _, eme, tme = checkShift(exp_spectrum, spectrum, shift, smoothType)
            shift = eme-tme
        initialData = {'Efermi': spectrum.energy[0], 'shift': shift}
    default = DefaultSmoothParams(initialData['Efermi'], initialData['shift'])
    params = default[smoothType]
    paramNames = [p['paramName'] for p in params]
    bounds0 = [[p['leftBorder'], p['rightBorder']] for p in params]
    bounds = []
    varParamNames = []
    for i,p in enumerate(paramNames):
        if p in fixedParams:
            v = fixedParams[p]
            a,b = bounds0[i]
            assert a <= v <= b, f'{p} not in [{a}; {b}]'
        else:
            if p in userBounds:
                assert userBounds[p][0] < userBounds[p][1]
                bounds.append(userBounds[p])
            else:
                bounds.append(bounds0[i])
            varParamNames.append(p)

    def makeParams(arg):
        smooth_params = {}
        i = 0
        for p in paramNames:
            if p in fixedParams: smooth_params[p] = fixedParams[p]
            else:
                smooth_params[p] = arg[i]
                i += 1
        return smooth_params

    def targetFunc(arg):
        smooth_params = makeParams(arg)
        sp, _ = smoothInterpNorm(smooth_params, spectrum, smoothType, exp_spectrum, **smoothInterpNormParams)
        if fit_smooth_interval is None:
            return utils.rFactor(exp_spectrum.energy, sp.intensity, exp_spectrum.intensity)
        else:
            a,b = fit_smooth_interval
            ind = (a <= exp_spectrum.energy) & (exp_spectrum.energy <= b)
            return utils.rFactor(exp_spectrum.energy[ind], sp.intensity[ind], exp_spectrum.intensity[ind])

    if len(varParamNames) > 0:
        x0 = [(b[0]+b[1])/2 for b in bounds]
        res = scipy.optimize.minimize(targetFunc, x0, bounds=bounds)
        fmin, xmin = res.fun, res.x
    else:
        xmin = []
        fmin = targetFunc(xmin)
    smooth_params = makeParams(xmin)
    if printDebug:
        for i in range(len(xmin)):
            a,b = bounds[i]
            d = b-a
            x = xmin[i]
            if abs(x-a)<0.01*d or abs(x-b)<0.01*d:
                print(f'Parameter {varParamNames[i]} = {x} is near bound [{a};{b}]')

    sp, _ = smoothInterpNorm(smooth_params, spectrum, smoothType, exp_spectrum, **smoothInterpNormParams)
    if plotFileName is not None:
        sigma = getSmoothWidth(smoothType, sp.energy, smooth_params)

        def plotSmoothWidth(ax):
            intervals = []
            colors = []
            if 'fit_norm_interval' in smoothInterpNormParams:
                intervals.append(smoothInterpNormParams['fit_norm_interval'])
                colors.append('blue')
            if fit_smooth_interval is not None:
                intervals.append(fit_smooth_interval)
                colors.append('red')
            ddd = 0
            font = FontProperties();
            font.set_weight('black');
            font.set_size(20)
            for color, interval in zip(colors, intervals):
                txt = ax.text(interval[0], ax.get_ylim()[0] + ddd, '[', color=color, verticalalignment='bottom', fontproperties=font)
                txt = ax.text(interval[1], ax.get_ylim()[0] + ddd, ']', color=color, verticalalignment='bottom', fontproperties=font)
                ddd += 0.05
            if smoothType in ['simple_Gauss', 'simple_Cauchy', 'simple_Cauchy_then_Gauss', 'simple_C+G']: return
            ax2 = ax.twinx()
            ax2.plot(exp_spectrum.energy, sigma, c='r', label='Smooth width')
            ax2.legend()

        title = 'rFactor = %.2g ' % fmin
        for p in smooth_params: title += (p+'=%.2g ' % smooth_params[p])
        sp0 = spectrum.clone()
        if 'shift' in smooth_params: sp0.energy += smooth_params['shift']
        mi = np.min(exp_spectrum.intensity)
        ma = np.max(exp_spectrum.intensity)
        dy = ma-mi
        plotting.plotToFile(sp0.energy, sp0.intensity, 'initial theory', exp_spectrum.energy, exp_spectrum.intensity, 'exp', sp.energy, sp.intensity, 'theory', fileName=plotFileName, plotMoreFunction=plotSmoothWidth, title=title, ylim=[mi-0.1*dy, ma+0.1*dy])
    return smooth_params, sp, fmin


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


def getNorm(normFixType, argsOfList, arg):
    if normFixType == 'variablePrivate': norm = None
    else:
        if normFixType == 'variableCommon': norm = argsOfList['norm']
        else:
            norm = arg['norm'] # normFixType == 'fixed' => норма должна сидеть в параметрах по умолчанию
            assert norm is not None, 'When norm is fixed it must be included in deafault smooth parameters'
    return norm


def funcFitSmoothList(argsOfList, expList, xanesList, smoothType, targetFunc, normFixType, fitDiffFrom):
    lp_str = 'l1' if 'l1' in targetFunc else 'l2'
    p = 1 if lp_str == 'l1' else 2
    lp = []
    diffs = []
    es = []
    if fitDiffFrom is not None:
        fitDiffFromExpXanes = fitDiffFrom['exp'].xanes
        fitDiffFromXanes = fitDiffFrom['xanes']
    for j in range(len(expList)):
        exp = expList[j]
        arg = getOneArg(argsOfList, exp, smoothType)
        norm = getNorm(normFixType, argsOfList, arg)
        smoothed_xanes, normOut = funcFitSmoothHelper(arg, xanesList[j], smoothType, exp, norm)
        # print(normOut)
        i = (exp.intervals['fit_smooth'][0]<=exp.spectrum.energy) & (exp.spectrum.energy<=exp.intervals['fit_smooth'][1])
        if fitDiffFrom is None:
            diff = abs(smoothed_xanes.intensity[i]-exp.spectrum.intensity[i])**p
        else:
            fitDiffFromExpXanes_absorb = np.interp(exp.spectrum.energy, fitDiffFromExpXanes.energy, fitDiffFromExpXanes.intensity)
            fitDiffFrom_smoothed_xanes, _ = funcFitSmoothHelper(arg, fitDiffFromXanes, smoothType, exp, norm)
            purity = value(arg, 'purity')
            diff = abs(purity*(smoothed_xanes.intensity[i]-fitDiffFrom_smoothed_xanes.intensity[i]) - (exp.spectrum.intensity[i]-fitDiffFromExpXanes_absorb[i]))**p
        diffs.append(diff)
        es.append(exp.spectrum.energy[i])
        partial_func_val = utils.integral(exp.spectrum.energy[i], diff)**(1/p)
        lp.append( partial_func_val )
    if len(expList) == 1: return lp[0]
    lp = np.array(lp)
    if targetFunc == 'mean': return np.mean(lp)
    elif targetFunc == f'max({lp_str})': return np.max(lp)
    elif targetFunc == f'{lp_str}(max)':
        e = es[0]
        newDiffs = np.zeros([len(expList), e.size])
        newDiffs[0] = diffs[0]
        for j in range(1,len(expList)):
            newDiffs[j] = np.interp(e, es[j], diffs[j])
        maxDiff = np.max(newDiffs, axis=0)
        return np.sqrt(utils.integral(e, maxDiff))
    else: assert False, 'Unknown target func '+targetFunc


# fixParamNames - массив фиксируемых параметров (значения берутся из значений по умолчанию в экспериментах)
# commonParams0 - ассоциативный массив начальных значений общих параметров (интервалы поиска берутся из первого эксперимента)
# норма может быть: фиксированной (тогда она должна быть задана в параметрах каждого эксперимента), подбираемой: общей или частными
# fitDiffFrom = {'exp':exp, 'xanes':xanes}
def fitSmooth(expList, xanesList0, smoothType='fdmnes', normType='multOnly', fixParamNames=None, commonParams0=None, targetFunc='l2(max)', crossValidationExp=None, crossValidationXanes=None, fitDiffFrom=None, optimizeWithoutPlot=False, folder='result'):
    if fixParamNames is None: fixParamNames = []
    if commonParams0 is None: commonParams0 = {}
    expNames = [exp.name for exp in expList]
    assert len(np.unique(expNames)) == len(expList), 'Duplicate project names!\n'+str(expNames)
    xanesList = copy.deepcopy(xanesList0)
    if 'norm' in fixParamNames: normFixType = 'fixed'
    else:
        if 'norm' in commonParams0: normFixType = 'variableCommon'
        else: normFixType = 'variablePrivate'
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

    fmin, smooth_params_vec = optimize.minimize(funcFitSmoothList1, arg0_1, bounds, fun_args=(expList, xanesList, smoothType, targetFunc, normFixType, fitDiffFrom), method='Powell')
    # print(fmin)
    smooth_params = copy.deepcopy(arg0)
    for i in range(len(arg0)): smooth_params[i]['value'] = smooth_params_vec[i]
    if optimizeWithoutPlot: return smooth_params
    with open(folder+'/func_smooth_value.txt', 'w') as f: json.dump(fmin, f)
    with open(folder+'/args_smooth.txt', 'w') as f: json.dump(smooth_params.__dict__, f)
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
        norm = getNorm(normFixType, argsOfList, arg)
        fdmnes_xan, _ = funcFitSmoothHelper(arg, xanesList[j], smoothType, exp, norm)
        smoothed_xanes.append(fdmnes_xan)
        with open(folder+'/'+expList[j].name+'/args_smooth.txt', 'w') as f: json.dump(arg.to_dict(), f)
        with open(folder+'/'+expList[j].name+'/args_smooth_human.txt', 'w') as f: f.write(arg2string(arg))
        shift = arg['shift']
        plotting.plotToFolder(folder+'/'+expList[j].name, exp, xanesList[j], fdmnes_xan, append = {'data':getSmoothWidth(smoothType, exp.spectrum.energy, arg), 'label':'smooth width', 'twinx':True}, fileName='xanes')
        if fitDiffFrom is not None:
            purity = arg['purity']
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
    assert len(exp_spectrum.energy) > 0
    assert norm is None or 'norm' not in smoothParams, f'norm = {norm}, smoothParams["norm"] = {smoothParams["norm"]}'
    smoothParams = {pn:smoothParams[pn] for pn in smoothParams}
    if norm is not None: smoothParams['norm'] = norm
    xanes_df_is_dataframe = isinstance(xanes_df, pd.DataFrame)
    if folder is not None:
        folder = utils.fixPath(folder)
        assert os.path.exists(folder+os.sep+'spectra.txt'), 'File spectra.txt doesn\'t exist in folder '+folder
        smoothFileName = folder+os.sep+'spectra_smooth.txt'
        smoothParamsFileName = folder+os.sep+'spectra_smooth_params.txt'
        smoothParams = copy.deepcopy(smoothParams)
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
        smoothed_xanes1, _ = smoothInterpNorm(smoothParams, xanes, smoothType, exp_spectrum, fit_norm_interval)
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


def removeNoise(energy, intensity, partCount=10, bw=None, debugPlotFile=None):
    """
    Use this function twice:
    1. bw=None: apply statsmodels.nonparametric.kernel_regression.KernelReg for bw estimation and save graph to debugPlotFile.
    2. bw=[[e1,bw1], [e2,bw2], ...] points to interpolate bw
    :param partCount: divide spectrum into parts
    :returns: clearIntensity
    """
    if bw is None:
        res = np.copy(intensity)
        n = len(energy) // partCount
        ebw, vbw = [], []
        for i in range((len(energy) + n - 1) // n):
            e, inten = energy[i * n:(i + 1) * n], intensity[i * n:(i + 1) * n]
            ebw.append(np.mean(e))
            kr = statsmodels.nonparametric.kernel_regression.KernelReg(inten, e, 'c')
            vbw.append(min(kr.bw, e[-1] - e[0]))
        if debugPlotFile is not None:
            plotting.plotToFile(ebw, vbw, 'bw', fileName=debugPlotFile)
    else:
        ebw = [t[0] for t in bw]
        vbw = [t[1] for t in bw]
        bw1 = np.interp(energy, ebw, vbw)
        _, res = generalSmooth(energy, intensity, bw1, kernel='Gauss')
        if debugPlotFile is not None:
            plotting.plotToFile(energy, bw1, 'bw', fileName=debugPlotFile)
    return res



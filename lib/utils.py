import parser
import numpy as np
import math
import random
import string

class Xanes:
    def __init__(self, energy, absorb, folder=None, molecula=None):
        assert folder != ''
        self.energy = energy
        self.absorb = absorb
        self.folder = folder
        self.molecula = molecula

    def save(self, fileName, header = ''):
        n = self.energy.size
        data = np.hstack((self.energy.reshape([n,1]), self.absorb.reshape([n,1])))
        np.savetxt(fileName, data, header=header, comments='')

def assignPosToPandasMol(pos):
    df, _ = parser.parse_mol('molecules/jz100548m_si_002_DSVsaved.mol2')
    atom_names = df['atom_type'].apply(lambda n: n[0:n.find('.')] if n.find('.')>=0 else n)
    i = (atom_names == 'N') | (atom_names == 'F') # TODO: выбирать самые тяжелые по атомному весу
    df = df.loc[i,].reset_index(drop=True)
    nitrogenCount = df.shape[0]-1
    newMol = df.copy(deep=True)
    Natom = pos.shape[0]//3
    for i in range(Natom):
        newMol.loc[i, ['x','y','z']] = pos[3*i:3*i+3]
    return newMol

def integral(x,y):
    my = (y[1:]+y[:-1])/2
    dx = x[1:]-x[:-1]
    return np.sum(my*dx)

def fit_arg_to_experiment(fdmnes_en, exp_e, fdmnes_xan, shift, lastValueNorm = False, interpolate = True):
    fdmnes_xan = np.copy(fdmnes_xan)
    fdmnes_en = fdmnes_en + shift
    # fdmnes_xan -= np.mean(fdmnes_xan[:10])
    if lastValueNorm:
        norm = np.mean(fdmnes_xan[-3:])
    else:
        norm = integral(exp_e, np.interp(exp_e, fdmnes_en, fdmnes_xan))
    fdmnes_xan /= norm
    if interpolate: fdmnes_xan = np.interp(exp_e, fdmnes_en, fdmnes_xan)
    return fdmnes_xan

def linearReg0(x,y):
    N = x.size
    sumX = np.sum(x)
    sumX2 = np.sum(x*x)
    sumY = np.sum(y)
    sumXY = np.sum(x*y)
    det = N*sumX2-sumX*sumX
    if det == 0:
        nn = abs(sumX)+abs(N)
        return [-sumX/nn, N/nn]
    return [(sumY*sumX2-sumXY*sumX)/det, (N*sumXY-sumX*sumY)/det]

# возвращает [b,a] из модели y=ax+b
def linearReg(x,y,de):
    N = np.sum(de)
    sumX = np.sum(x*de)
    sumX2 = np.sum(x*x*de)
    sumY = np.sum(y*de)
    sumXY = np.sum(x*y*de)
    det = N*sumX2-sumX*sumX
    if det == 0:
        nn = abs(sumX)+abs(N)
        return [-sumX/nn, N/nn]
    return [(sumY*sumX2-sumXY*sumX)/det, (N*sumXY-sumX*sumY)/det]


def fit_by_regression(exp_e, exp_xanes, fdmnes_xan, fitEnergyInterval):
    ind = (fitEnergyInterval[0]<=exp_e) & (exp_e<=fitEnergyInterval[1])
    e = exp_e[ind]
    ex = exp_xanes[ind]
    fx = fdmnes_xan[ind]
    mex = (ex[1:]+ex[:-1])/2
    mfx = (fx[1:]+fx[:-1])/2
    de = e[1:]-e[:-1]
    w = linearReg(mfx, mex, de)
    if w[1] < 0:
        w[0] = (np.sum(mex*de)-np.sum(mfx*de)) / np.sum(de)
        return fdmnes_xan + w[0]
    else:
        return w[1]*fdmnes_xan + w[0]

def linearReg_mult_only(x,y,de):
    N = np.sum(de)
    sumX2 = np.sum(x*x*de)
    sumXY = np.sum(x*y*de)
    return sumXY/sumX2

def fit_by_regression_mult_only(exp_e, exp_xanes, fdmnes_xan, fitEnergyInterval):
    ind = (fitEnergyInterval[0]<=exp_e) & (exp_e<=fitEnergyInterval[1])
    e = exp_e[ind]
    ex = exp_xanes[ind]
    fx = fdmnes_xan[ind]
    mex = (ex[1:]+ex[:-1])/2
    mfx = (fx[1:]+fx[:-1])/2
    de = e[1:]-e[:-1]
    w = linearReg_mult_only(mfx, mex, de)
    return w*fdmnes_xan

def fit_to_experiment_by_norm_or_regression_mult_only(exp_e, exp_xanes, fit_interval, fdmnes_en, fdmnes_xan, shift, norm = None):
    fdmnes_en = fdmnes_en + shift
    fdmnes_xan = np.interp(exp_e, fdmnes_en, fdmnes_xan)
    if norm is None:
        fdmnes_xan1 = fit_by_regression_mult_only(exp_e, exp_xanes, fdmnes_xan, fit_interval)
        norm = np.sum(fdmnes_xan)/np.sum(fdmnes_xan1)
        # print(norm)
        return fdmnes_xan1, norm
    else: return fdmnes_xan/norm, norm


def findNextMinimum(y, i0):
    i = i0
    n = y.size
    if i==0: i=1
    if i==n-1: return i
    while not ((y[i-1]>=y[i]) and (y[i]<=y[i+1])):
        i += 1
        if i>=n-1: return i
    return i

def findNextMaximum(y, i0):
    i = i0
    n = y.size
    if i==0: i=1
    if i==n-1: return i
    while not ((y[i-1]<=y[i]) and (y[i]>=y[i+1])):
        i += 1
        if i>=n-1: return i
    return i

def findExpEfermi(exp_e, exp_xanes, search_shift_level):
    ind = np.where(exp_xanes>=search_shift_level)[0][0]
    exp_Efermi_left = exp_e[ind]
    i = ind
    while exp_xanes[i]<=exp_xanes[i+1]: i += 1
    exp_Efermi_peak = exp_e[i]
    while exp_xanes[i]>=exp_xanes[i+1]: i += 1
    exp_Efermi_right = exp_e[i]
    return exp_Efermi_left, exp_Efermi_peak, exp_Efermi_right

def getInitialShift(exp_e, exp_xanes, fdmnes_en, fdmnes_xan, search_shift_level):
    maxVal0 = np.mean(exp_xanes[-3:])
    i0 = np.where(exp_xanes>=maxVal0*search_shift_level)[0][0]
    maxVal = np.mean(fdmnes_xan[-3:])
    ind = np.where(fdmnes_xan <= maxVal*search_shift_level)[0]
    i = ind[-1] if ind.size>0 else 0
    return exp_e[i0] - fdmnes_en[i]

def findAllMax(e, xanes, region = [], maxCount = -1, filterRightGreater = False):
    if len(region)==0: region = [e[0],e[-1]]
    ind = (xanes[:-2] <= xanes[1:-1]) & (xanes[1:-1] >= xanes[2:])
    ind = ind & (region[0]<=e[1:-1]) & (e[1:-1]<=region[1])
    res = e[1:-1][ind]
    ind = np.where(ind)[0]
    if maxCount>0:
        while res.size > maxCount:
            dr = res[1:]-res[:-1]
            i = np.argmin(dr)
            res = np.delete(res,i)
            ind = np.delete(ind,i)
    if filterRightGreater:
        n = res.size
        indFilt = np.array([False]*n)
        for i in range(n):
            xanVal = xanes[1:-1][ind[i]]
            indFilt[i] = np.all(xanVal>=xanes[1:-1][ind[i]+1:])
        res = res[indFilt]
    return res

def expandEnergyRange(e, xanes):
    n = e.size
    h0 = e[1]-e[0]
    h1 = e[-1]-e[-2]
    e0 = np.linspace(e[0]-n*h0, e[0]-h0,n)
    e1 = np.linspace(e[-1]+h1, e[-1]+n*h1, n)
    xanes0 = np.ones(e.size)*np.min(xanes)
    xanes1 = np.ones(e.size)*xanes[-1]
    return np.concatenate((e0,e,e1)), np.concatenate((xanes0,xanes,xanes1))

def randomString(N):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))

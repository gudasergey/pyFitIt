import parser, scipy, math, random, string, os, importlib, pathlib, matplotlib, ipykernel, urllib, json, traceback, copy, glob
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from notebook import notebookapp


class Spectrum:
    def __init__(self, energy, intensity, molecula=None, copy=False):
        if copy:
            self.energy = np.copy(energy)
            self.intensity = np.copy(intensity)
        else:
            self.energy = energy
            self.intensity = intensity
        self.molecula = molecula

    def save(self, fileName):
        folder = os.path.dirname(fileName)
        os.makedirs(folder, exist_ok=True)
        n = self.energy.size
        data = np.hstack((self.energy.reshape([n,1]), self.intensity.reshape([n,1])))
        np.savetxt(fileName, data, header='energy intensity', comments='')
        
    def clone(self):
        return Spectrum(self.energy, self.intensity, self.molecula, copy=True)


class Exafs:
    def __init__(self, k, chi):
        self.k = k
        self.chi = chi


def readSpectrum(fileName, skiprows=0, energyColumn=0, intensityColumn=1, separator=r'\s+', decimal="."):
    fileName = fixPath(fileName)
    data = pd.read_csv(fileName, sep=separator, decimal=decimal, skiprows = skiprows, header=None)
    if data.shape[1]<energyColumn:
        raise Exception('Data in file contains '+str(data.shape[1])+' only columns. But you specify energyColumn = '+str(energyColumn))
    if data.shape[1]<intensityColumn:
        raise Exception('Data in file contains '+str(data.shape[1])+' only columns. But you specify intensityColumn = '+str(intensityColumn))
    energy = data[data.columns[energyColumn]]
    if not is_numeric_dtype(energy):
        raise Exception('Data energy column is not numeric: '+str(energy))
    intensity = data[data.columns[intensityColumn]]
    if not is_numeric_dtype(intensity):
        raise Exception('Data energy column is not numeric: '+str(intensity))
    return Spectrum(energy.values, intensity.values)


def readExafs(fileName, skiprows=0, energyColumn=0, intensityColumn=1, separator=r'\s+', decimal="."):
    spectrum = readSpectrum(fileName, skiprows, energyColumn, intensityColumn, separator, decimal)
    return Exafs(spectrum.energy, spectrum.intensity)


def adjustSpectrum(s, maxSpectrumPoints, intervals):
    m = np.min([intervals['fit_norm'][0], intervals['fit_smooth'][0], intervals['fit_geometry'][0], intervals['plot'][0]])
    M = np.max([intervals['fit_norm'][1], intervals['fit_smooth'][1], intervals['fit_geometry'][1], intervals['plot'][1]])
    res = copy.deepcopy(s)
    ind = (res.energy>=m) & (res.energy<=M)
    res.energy = res.energy[ind]
    res.intensity = res.intensity[ind]
    if res.energy.size <= maxSpectrumPoints: return res
    var = np.cumsum(np.abs(res.intensity[1:]-res.intensity[:-1]))
    var = np.insert(var, 0, 0)
    var_edges = np.linspace(0,var[-1],maxSpectrumPoints)
    dists = scipy.spatial.distance.cdist(var.reshape(-1,1), var_edges.reshape(-1,1))
    ind = np.unique(np.argmin(dists, axis=0))
    res.energy = res.energy[ind]
    res.intensity = res.intensity[ind]
    return res


def integral(x,y):
    my = (y[1:]+y[:-1])/2
    dx = x[1:]-x[:-1]
    return np.sum(my*dx)


def norm_lp(x, y, p):
    return integral(x, np.abs(y)**p) ** (1/p)


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


def linearReg_mult_only(x,y,de):
    sumX2 = np.sum(x*x*de)
    sumXY = np.sum(x*y*de)
    if (sumX2 == 0) or np.isnan(sumX2) or np.isnan(sumXY): return 0
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
        s = np.sum(fdmnes_xan1)
        if s != 0: norm = np.sum(fdmnes_xan)/s
        else: norm = 0
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
    de = e[-1]-e[0]
    e0 = np.linspace(e[0]-de, e[0]-h0,n)
    e1 = np.linspace(e[-1]+h1, e[-1]+de, n)
    xanes0 = np.ones(e.size)*np.min(xanes)
    lastValueInd = xanes.size - int(xanes.size*0.05)
    lastValue = integral(e[lastValueInd:], xanes[lastValueInd:])/(e[-1] - e[lastValueInd])
    xanes1 = np.ones(e.size)*lastValue
    return np.concatenate((e0,e,e1)), np.concatenate((xanes0,xanes,xanes1))


def randomString(N):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))


def isJupyterNotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def this_notebook():
    """Returns the absolute path of the Notebook or None if it cannot be determined
    NOTE: works only when the security is token-based or there is also no password
    """
    connection_file = os.path.basename(ipykernel.get_connection_file())
    kernel_id = connection_file.split('-', 1)[1].split('.')[0]
    for srv in notebookapp.list_running_servers():
        try:
            if srv['token']=='' and not srv['password']:  # No token and no password, ahem...
                req = urllib.request.urlopen(srv['url']+'api/sessions')
            else:
                req = urllib.request.urlopen(srv['url']+'api/sessions?token='+srv['token'])
            sessions = json.load(req)
            for sess in sessions:
                if sess['kernel']['id'] == kernel_id:
                    return os.path.join(srv['notebook_dir'],sess['notebook']['path'])
        # except Exception as exc:
        #     print(traceback.format_exc())
        except:
            pass  # There may be stale entries in the runtime directory
    return None


def makePiramids(e, a, h):
    w = np.ones(e.size)
    de = e[1:]-e[:-1]
    minDeInd = np.argmin(de)
    while de[minDeInd] < 2*h:
        em = (e[minDeInd]*w[minDeInd] + e[minDeInd+1]*w[minDeInd+1]) / (w[minDeInd]+w[minDeInd+1])
        am = a[minDeInd] + a[minDeInd+1]
        e[minDeInd] = em; a[minDeInd] = am; w[minDeInd] = w[minDeInd]+w[minDeInd+1];
        e = np.delete(e, minDeInd+1); a = np.delete(a, minDeInd+1); w = np.delete(w, minDeInd+1)
        de = e[1:]-e[:-1]
        minDeInd = np.argmin(de)
    eNew = np.zeros(e.size*3)
    i = 1+np.arange(e.size)*3
    eNew[i] = e; eNew[i-1] = e-h; eNew[i+1] = e+h
    aNew = np.zeros(a.size*3)
    aNew[i] = a
    return eNew, aNew


def isLibExists(name):
    folder = os.path.dirname(os.path.realpath(__file__))
    if os.path.exists(folder+os.sep+name+'.py'): return True
    return importlib.util.find_spec(name) is not None


def fixPath(p):
  return str(pathlib.PurePath(p))


def fixDisplayError():
    if (os.name != 'nt') and ('DISPLAY' not in os.environ):
        matplotlib.use('Agg')


def zfill(i, n):
    return str(i).zfill(1+math.floor(0.5+math.log(n,10)))


def gauss(x,a,s):
    return 1/s/np.sqrt(2*np.pi)*np.exp(-(x-a)**2/2/s**2)


def findFile(folder, postfix, check_unique = True, ignoreSlurm=True):
    files = glob.glob(folder+os.sep+'*'+postfix)
    if ignoreSlurm:
        files = [f for f in files if f.find('slurm')<0]
    if check_unique:
        assert len(files)>0, 'Can\'t find file *'+postfix+' in folder '+folder
        assert len(files)==1, 'There are several files *'+postfix+' in folder '+folder
    if len(files)==1: return files[0]
    else: return None


def wrap(s, maxLineLen):
    if len(s) <= maxLineLen: return s
    i_last = 0
    i = i_last + maxLineLen
    res = ''
    while i < len(s):
        i1 = s.rfind(' ', i_last, i)
        if i1 == -1: i1 = i
        res += '\n' + s[i_last:i1]
        i_last = i1
        i = i_last + maxLineLen
    res += '\n' + s[i_last:]
    return res

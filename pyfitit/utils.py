import scipy, math, random, string, os, importlib, pathlib, matplotlib, ipykernel, urllib, json, copy, glob, numbers, itertools, pickle, time, re
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from notebook import notebookapp
from . import geometry


class Spectrum:
    def __init__(self, energy, intensity, molecula=None, copy=True):
        """Spectrum class

        :param energy: energy
        :param intensity: intensity
        :param molecula: molecule, defaults to None
        :param copy: copy params or assign as reference, defaults to False
        """
        assert len(energy) == len(intensity), f'{len(energy)} != {len(intensity)}'
        if copy:
            self.energy = np.copy(energy).reshape(-1)
            self.intensity = np.copy(intensity).reshape(-1)
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

    def limit(self, interval, inplace=False):
        e, inten = limit(interval, self.energy, self.intensity)
        if e[0] != interval[0] and self.energy[0] < interval[0] < self.energy[-1]:
            e = np.insert(e, 0, interval[0])
            inten = np.insert(inten, 0, np.interp(interval[0], self.energy, self.intensity))
        if e[-1] != interval[1] and self.energy[0] < interval[1] < self.energy[-1]:
            e = np.append(e, interval[1])
            inten = np.append(inten, np.interp(interval[1], self.energy, self.intensity))
        if inplace:
            self.energy = e
            self.intensity = inten
        else:
            return Spectrum(e, inten, self.molecula, copy=True)

    def changeEnergy(self, newEnergy, inplace=False):
        newInt = np.interp(newEnergy, self.energy, self.intensity)
        if inplace:
            self.energy = newEnergy
            self.intensity = newInt
        else:
            return Spectrum(newEnergy, newInt, copy=True)

    def val(self, energyValue):
        """
        Calculate spectrum value at specific energy
        """
        return np.interp(energyValue, self.energy, self.intensity)

    def inverse(self, intensity, select='min'):
        ind = np.where((self.intensity[:-1] - intensity) * (self.intensity[1:] - intensity) <= 0)[0]
        assert len(ind) > 0, f'Can\'t find inverse for value {intensity}'
        if select == 'min': i = ind[0]
        else:
            assert select == 'max'
            i = ind[-1]
        a,b,c = geometry.get_line_by_2_points(self.energy[i], self.intensity[i], self.energy[i+1], self.intensity[i+1])
        e,inten = geometry.get_line_intersection(a,b,c, 0,1,-intensity)
        return e

    def toExafs(self, Efermi, k_power):
        me = 2 * 9.109e-31  # kg
        h = 1.05457e-34  # J*s
        J = 6.24e18  # 1 J = 6.24e18 eV
        e, s = self.energy, self.intensity
        i = e >= Efermi
        k = np.sqrt((e[i] - Efermi) / J * me / h ** 2 / 1e20)  # J * kg /  (J*s)^2 = kg / (J * s^2) = kg / ( kg*m^2 ) = 1/m^2 = 1e-20 / A^2
        intSqr = (s[i] - 1) * k ** k_power
        return Exafs(k, intSqr)

    def __add__(self, other):
        e1 = max(self.energy[0], other.energy[0])
        e2 = min(self.energy[-1], other.energy[-1])
        ind1 = (e1 <= self.energy) & (self.energy <= e2)
        ind2 = (e1 <= other.energy) & (other.energy <= e2)
        if np.sum(ind1) > np.sum(ind2):
            e = self.energy[ind1]
            s = self.intensity[ind1] + np.interp(e, other.energy, other.intensity)
        else:
            e = other.energy[ind2]
            s = other.intensity[ind2] + np.interp(e, self.energy, self.intensity)
        return Spectrum(e, s)

    def __mul__(self, other):
        assert isinstance(other, numbers.Number)
        return Spectrum(self.energy, self.intensity*other, self.molecula)

    __rmul__ = __mul__

    def __truediv__(self, other):
        assert isinstance(other, numbers.Number)
        return Spectrum(self.energy, self.intensity/other, self.molecula)


class Exafs:
    def __init__(self, k, chi):
        self.k = k
        self.chi = chi

    def toXanes(self, Efermi, k_power):
        me = 2 * 9.109e-31  # kg
        h = 1.05457e-34  # J*s
        J = 6.24e18  # 1 J = 6.24e18 eV
        e = self.k**2 * J / me * h**2 * 1e20 + Efermi
        k = copy.deepcopy(self.k)
        k[k==1] = 1
        return Spectrum(e, self.chi/k**k_power + 1)

    def shift(self, dE, Efermi, inplace=False):
        xan = self.toXanes(Efermi, k_power=0)
        xan.energy += dE
        exafs = xan.toExafs(Efermi, k_power=0)
        if inplace:
            self.k, self.chi = exafs.k, exafs.chi
        else:
            return exafs

    def smooth(self, SO2, sigmaSquare, inplace=False):
        chi1 = self.chi * np.exp(-2 * self.k ** 2 * sigmaSquare) * SO2
        if inplace: self.chi = chi1
        else: return Exafs(self.k, chi1)


class SpectrumCollection:
    def __init__(self):
        self.labels = []
        self.params = None
        self.spectra = None
        self.energyColumn = 0

    def setSpectra(self, dataframe):
        self.spectra = dataframe
        # assuming that the first column is energy, and the rest are spectra
        self.labels = list(dataframe.columns[1:])

    def setLabel(self, index, label):
        if index < 0 or index >= len(self.labels):
            raise Exception('Wrong label: ' + index)
        self.labels[index] = label

    def getSpectrumByLabel(self, label):
        index = self.labels.index(label)
        energy = self.spectra[self.spectra.columns[self.energyColumn]]
        intensity = self.spectra[self.spectra.columns[index + 1]]
        return Spectrum(energy.values, intensity.values)

    def getSpectrumByParam(self, param):
        assert self.params is not None, "You need too call parseLabels first"
        assert param in self.params, 'Param = '+str(param)+' not known values: '+str(self.params)
        indexes = np.where(self.params == param)[0]
        assert len(indexes) == 1
        index = indexes[0]
        energy = self.spectra[self.spectra.columns[self.energyColumn]]
        intensity = self.spectra[self.spectra.columns[index + 1]]
        # print(energy)
        return Spectrum(energy.values, intensity.values)

    def interpolate(self, exp_e):
        dataframe = self.spectra
        newData = {'energy':exp_e}
        for column in dataframe.columns[1:]:
            newData[column] = np.interp(
                exp_e,
                dataframe[dataframe.columns[self.energyColumn]].values,
                dataframe[column].values)

        self.spectra = pd.DataFrame(newData)

    def parseLabels(self):
        self.params = np.array([float(l) for l in self.labels])
        assert len(np.unique(self.params)) == len(self.labels), "Duplicate param values!"


def readSpectra(fileName, header=True, skiprows=0, energyColumn=0, intensityColumns=None, separator=r'\s+', decimal="."):
    fileName = fixPath(fileName)
    data = pd.read_csv(fileName, sep=separator, decimal=decimal, skiprows=skiprows, header=0 if header else None)
    collection = SpectrumCollection()
    columnsToExport = []

    if data.shape[1] < energyColumn:
        raise Exception('Data in file contains '+str(data.shape[1])+' only columns. But you specify energyColumn = '+str(energyColumn))
    columnsToExport.append(data.columns[energyColumn])

    if intensityColumns is None:
        intensityColumns = list(range(energyColumn + 1, data.shape[1]))
    for column in intensityColumns:
        if data.shape[1] <= column:
            raise Exception('Data in file contains '+str(data.shape[1])+' only columns. But you specify intensityColumns = '+str(intensityColumns))
        columnsToExport.append(data.columns[column])

    # saving only desired spectra, first column - energy
    collection.setSpectra(data[columnsToExport])

    return collection


def readSpectrum(fileName, skiprows=0, energyColumn=0, intensityColumn=1, separator=r'\s+', decimal=".", guess=False):
    fileName = fixPath(fileName)
    if guess:
        with open(fileName, 'r') as f: lines = f.read()
        lines = lines.split('\n')
        lines = [l.strip() for l in lines]
        # delete empty lines
        lines = [l for l in lines if len(l) > 0]
        # delete comments
        lines = [l for l in lines if l[0] not in ['#', '!', '%']]
        # if last line contains ',' but not '.', than ',' is integer part feature
        if ',' in lines[-1] and '.' not in lines[-1]:
            lines = [l.replace(',', '.') for l in lines]
        # delete lines which are not numbers
        lines = [l for l in lines if re.match(r"^[\d.eE+\-\s]*$", l) is not None]
        assert len(lines) > 0, 'Unknown file format. Can\'t guess'
        numbers = re.findall(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', lines[-1])
        ncols = len(numbers)
        if ncols == 2:
            if ncols <= energyColumn:
                print(f'Warning: wrong energyColumn number {energyColumn} in readSpectrum. It was corrected to 0')
                energyColumn = 0
            if ncols <= intensityColumn:
                print(f'Warning: wrong intensityColumn number {intensityColumn} in readSpectrum. It was corrected to 1')
                intensityColumn = 1
        assert ncols > energyColumn
        assert ncols > intensityColumn
        result = []
        for i in range(len(lines)):
            numbers = re.findall(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', lines[i])
            # print(numbers)
            if len(numbers) != ncols:
                assert len(result) == 0, 'Rows with different number of values detected!'
                continue
            result.append(np.array([float(n) for n in numbers]))
        assert len(result) > 0, 'Unknown file format. Can\'t guess'
        result = np.array(result)
        result = Spectrum(result[:, energyColumn], result[:, intensityColumn])
    else:
        data = pd.read_csv(fileName, sep=separator, decimal=decimal, skiprows=skiprows, header=None)
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
        result = Spectrum(energy.values, intensity.values)
    e = result.energy
    assert np.all(e[1:] - e[:-1] >= 0)
    if np.any(e[1:] - e[:-1] == 0):
        print(f'Spectrum {fileName} contains duplicate energies. Calculate average')
        # calculate average for repeated energies
        ue, ind, counts = np.unique(e, return_counts=True, return_index=True)
        uin = result.intensity[ind]
        not_unique_ind = np.where(counts > 1)[0]
        for i in not_unique_ind:
            uin[i] = np.mean(result.intensity[e == ue[i]])
        result = Spectrum(ue, uin)
    return result



def readExafs(fileName, skiprows=0, energyColumn=0, intensityColumn=1, separator=r'\s+', decimal="."):
    spectrum = readSpectrum(fileName, skiprows, energyColumn, intensityColumn, separator, decimal)
    return Exafs(spectrum.energy, spectrum.intensity)


def adjustSpectrum(s, maxSpectrumPoints, intervals):
    m = np.min([intervals['fit_norm'][0], intervals['fit_smooth'][0], intervals['fit_geometry'][0], intervals['plot'][0]])
    M = np.max([intervals['fit_norm'][1], intervals['fit_smooth'][1], intervals['fit_geometry'][1], intervals['plot'][1]])
    res = copy.deepcopy(s)
    if m > M: return res
    ind = (res.energy>=m) & (res.energy<=M)
    res.energy = res.energy[ind]
    assert len(res.energy)>2, 'Too few energies are situated in the intervals ['+str(m)+', '+str(M)+']. Energy = '+str(s.energy)
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
    if isinstance(y, pd.DataFrame): y = y.to_numpy()
    if isinstance(y, pd.Series): y = y.to_numpy()
    assert len(x.shape) == 1
    dx = x[1:] - x[:-1]
    if len(y.shape) == 1:
        my = (y[1:]+y[:-1])/2
    else:
        assert len(y.shape) == 2
        assert y.shape[1] == len(x)
        # one spectrum - one row
        my = (y[:,1:] + y[:,:-1]) / 2
    return np.dot(my, dx)


def norm_lp(x, y, p):
    return integral(x, np.abs(y)**p) ** (1/p)


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


def findNearest(a, value, returnInd=False, direction='both', ignoreDirectionIfEmpty=False):
    """
    Find nearest value or index of nearest value
    :param a: numpy array
    :param value: scalar value
    :param returnInd: True/False
    :param direction: 'left', 'right' or 'both'
    :param ignoreDirectionIfEmpty: True/False
    :return: for empty array a (or left/right parts of a): nan if returnInd else None
    """
    assert np.isscalar(value)
    assert isinstance(a, np.ndarray)
    badRes = np.nan if returnInd else None
    if len(a) == 0: return badRes
    if direction == 'both':
        i = np.abs(a-value).argmin()
    else:
        if direction == 'left':
            ind = np.where(a <= value)[0]
        elif direction == 'right':
            ind = np.where(a >= value)[0]
        else: assert False, f'Unknown direction {direction}'
        if len(ind) == 0:
            if ignoreDirectionIfEmpty: i = np.abs(a - value).argmin()
            else: return badRes
        else:
            i = np.abs(a[ind] - value).argmin()
            i = ind[i]
    if returnInd: return i
    else: return a[i]


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
                    np = sess['notebook']['path']
                    if np[0] == '/': np = np[1:]
                    return os.path.join(srv['notebook_dir'], np)
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
    if p == '': return ''
    return str(pathlib.PurePath(p))


def fixDisplayError():
    if (os.name != 'nt') and ('DISPLAY' not in os.environ) and not isJupyterNotebook():
        matplotlib.use('Agg')


def zfill(i, n):
    assert n>0
    return str(i).zfill(1+math.floor(0.5+math.log(n,10)))


def gauss(x,a,s):
    return 1/s/np.sqrt(2*np.pi)*np.exp(-(x-a)**2/2/s**2)


def findFile(folder='.', postfix=None, mask=None, check_unique=True, ignoreSlurm=True):
    assert (mask is not None) or (postfix is not None)
    assert (mask is None) or (postfix is None)
    if postfix is not None: mask = f'*{postfix}'
    if folder != '.': mask = f'{folder}/{mask}'
    files = glob.glob(mask)
    if ignoreSlurm:
        files = [f for f in files if f.find('slurm')<0]
    if check_unique:
        assert len(files)>0, 'Can\'t find file '+mask
        assert len(files)==1, 'There are several files '+mask
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


def safePredict(estimator, x_for_predict, x_train, y_train):
    try:
        result = estimator.predict(x_for_predict)
    except:
        estimator.fit(x_train, y_train)
        result = estimator.predict(x_for_predict)
    return result


def profile(name, fun):
    start = time.time()
    fun()
    end = time.time()
    print(name, end - start)


def limit(interval,x,*ys):
    """Limit x and all y to the given interval of x
    """
    a,b = interval[0], interval[1]
    i = (a<=x) & (x<=b)
    res = (x[i],)
    for y in ys:
        assert len(y) == len(x)
        res += (y[i],)
    return res


def getEnergy(spectraDataframe):
    e_names = spectraDataframe.columns
    energy = np.array([float(e_names[i][2:]) for i in range(e_names.size)])
    return energy


def makeDataFrame(energy, spectra):
    return pd.DataFrame(data=spectra, columns=['e_' + str(e) for e in energy])


def makeDataFrameFromSpectraList(spectra, energy=None):
    assert isinstance(spectra, list)
    m = len(spectra)
    if energy is None:
        energies = np.array([])
        for s in spectra: energies = np.append(energies, s.energy)
        energies = np.sort(energies)
        max_count = np.max([len(s.energy) for s in spectra])
        step = len(energies) // max_count
        energy = np.unique(energies[::step])
    n = len(energy)
    sp_matr = np.zeros((m,n))
    for i,s in enumerate(spectra):
        sp_matr[i] = np.interp(energy, s.energy, s.intensity)
    return makeDataFrame(energy, sp_matr)


def isObjectArraysEqual(a, b):
    assert isinstance(a, np.ndarray) or isinstance(b, np.ndarray)
    if not isinstance(a, np.ndarray): return isObjectArraysEqual(b, a)
    n = len(a)
    if not isinstance(b, np.ndarray):
        b = np.array([b] * n, dtype=object)
    res = np.zeros(n, dtype=np.bool)
    for i in range(n):
        if isinstance(a[i], type(b[i])):
            res[i] = a[i] == b[i]
        else:
            res[i] = False
    return res


def comb_index(n, k, repetition=False):
    """
    :param n:
    :param k:
    :param repetition: if True include duplicate items in one combination
    :return: array of size C_n^k x k with all k-combinations (without duplicates) from np.arange(n)
    """
    count = scipy.special.comb(n, k, exact=True, repetition=repetition)
    if repetition:
        index = np.fromiter(itertools.chain.from_iterable(itertools.combinations_with_replacement(range(n), k)), int, count=count * k)
    else:
        index = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(n), k)), int, count=count*k)
    return index.reshape(-1, k)


def find_nearest_in_sorted_array(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1], idx-1
    else:
        return array[idx], idx


def rFactor(e, theory, exp):
    return integral(e, (theory - exp) ** 2) / integral(e, exp ** 2)


def save_pkl(obj, fileName):
    folder = os.path.split(fileName)[0]
    if folder != '' and not os.path.exists(folder):
        os.makedirs(folder)
    with open(fileName, 'wb') as f:
        pickle.dump(obj, f)


def load_pkl(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)


def appendToBeginningOfFile(fileName, s):
    if os.path.exists(fileName):
        with open(fileName, 'r') as f:
            content = f.read()
    else: content = ''
    content = s + content
    with open(fileName, 'w') as f:
        f.write(content)


def argrelmax(y, returnAll=False):
    """
    Analog of numpy.argrelmax to calculate argmax (global or all locals). Returns indices
    """
    y_prev = y[:-2]
    y_next = y[2:]
    y_cur = y[1:-1]
    ind = np.where((y_prev <= y_cur) & (y_cur >= y_next))[0]
    if len(ind) == 0:
        if returnAll: return np.nan, np.array([])
        else: return np.nan
    j = np.argmax(y_cur[ind])
    res = ind[j]+1
    assert y[res - 1] <= y[res] and y[res] >= y[res + 1], f'{y[res - 2]} {y[res - 1]} {y[res]} {y[res + 1]} {y[res + 2]}'
    if returnAll:
        # search for middles in segments of equal y values
        # print(ind)
        delta = ind[1:]-ind[:-1]
        all1 = ind[:-1][delta > 1]
        all2 = ind[1:][delta > 1]
        all = np.zeros(len(all1) + 1, dtype=int)
        all[:-1] = all1
        if len(all2) > 0:
            all[-1] = all2[-1]
            all[1:-1] = (all1[1:] + all2[:-1]) // 2
        else:
            all[-1] = (ind[len(all1)]+ind[-1]) // 2
        all += 1
        for r in all:
            assert y[r - 1] <= y[r] and y[r] >= y[r + 1], f'{y[r - 2]} {y[r - 1]} {y[r]} {y[r + 1]} {y[r + 2]}'
        return res, all
    else:
        return res


def addPostfixIfExists(fileName):
    if not os.path.exists(fileName): return fileName
    b = os.path.splitext(fileName)[0]
    ext = os.path.splitext(fileName)[1]
    i = 1
    while os.path.exists(f'{b}_{i}{ext}'):
        i += 1
    return f'{b}_{i}{ext}'


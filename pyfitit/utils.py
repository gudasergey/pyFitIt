import scipy, math, random, string, os, importlib, pathlib, matplotlib, ipykernel, urllib, json, copy, glob, numbers, itertools, pickle, time, re, warnings, sys, types, subprocess, shlex, jupytext, nbformat
from IPython.display import display, Javascript, HTML
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from notebook import notebookapp
from . import geometry


class Spectrum(object):
    __initialized = False
    def __init__(self, x, y, xName='energy', yName='intensity', copy=True, checkSorting=True):
        """Spectrum class

        :param energy: energy
        :param intensity: intensity
        :param copy: copy params or assign as reference, defaults to False
        """
        super(Spectrum, self).__init__()
        self.__initialized = False
        assert len(x) == len(y), f'{len(x)} != {len(y)}'
        self.checkSorting = checkSorting
        if not np.all(x[1:]>=x[:-1]):
            i = np.argsort(x)
            x = x[i]
            y = y[i]
            if checkSorting:
                raise Exception('Spectrum energy is not sorted')
            else:
                warnings.warn('Spectrum energy is not sorted. I sort it')
        if copy:
            self.x = np.copy(x).reshape(-1)
            self.y = np.copy(y).reshape(-1)
        else:
            self.x = x
            self.y = y
        self.xName = xName
        self.yName = yName
        self.__initialized = True

    def __getattr__(self, name):
        if name != '_Spectrum__initialized' and self.__initialized:
            if name == self.xName: return self.x
            if name == self.yName: return self.y
        return self.__getattribute__(name)

    def __setattr__(self, name, value):
        if name != '_Spectrum__initialized' and self.__initialized:
            if name == self.xName: self.x = value
            elif name == self.yName: self.y = value
            else: super(Spectrum, self).__setattr__(name, value)
        else:
            super(Spectrum, self).__setattr__(name, value)

    def save(self, fileName):
        folder = os.path.dirname(fileName)
        os.makedirs(folder, exist_ok=True)
        n = self.x.size
        data = np.hstack((self.x.reshape([n,1]), self.y.reshape([n,1])))
        np.savetxt(fileName, data, header='energy intensity', comments='')
        
    def clone(self):
        return Spectrum(self.x, self.y, xName=self.xName, yName=self.yName, copy=True, checkSorting=False)

    def limit(self, interval, inplace=False):
        e, inten = limit(interval, self.x, self.y)
        if e[0] != interval[0] and self.x[0] < interval[0] < self.x[-1]:
            e = np.insert(e, 0, interval[0])
            inten = np.insert(inten, 0, np.interp(interval[0], self.x, self.y))
        if e[-1] != interval[1] and self.x[0] < interval[1] < self.x[-1]:
            e = np.append(e, interval[1])
            inten = np.append(inten, np.interp(interval[1], self.x, self.y))
        if inplace:
            self.x = e
            self.y = inten
        else:
            return Spectrum(e, inten, xName=self.xName, yName=self.yName, copy=True)

    def changeEnergy(self, newEnergy, inplace=False):
        if self.checkSorting: assert np.all(newEnergy[1:]>=newEnergy[:-1])
        newInt = np.interp(newEnergy, self.x, self.y)
        if inplace:
            self.x = newEnergy
            self.y = newInt
        else:
            return Spectrum(newEnergy, newInt, xName=self.xName, yName=self.yName, copy=True, checkSorting=self.checkSorting)

    def shiftEnergy(self, shift, inplace=False):
        return self.changeEnergy(newEnergy=self.x+shift, inplace=inplace)

    def val(self, energyValue):
        """
        Calculate spectrum value at specific energy
        """
        return np.interp(energyValue, self.x, self.y)

    def inverse(self, y, select='min'):
        ind = np.where((self.y[:-1] - y) * (self.y[1:] - y) <= 0)[0]
        assert len(ind) > 0, f'Can\'t find inverse for value {y}'
        if select == 'min': i = ind[0]
        else:
            assert select == 'max'
            i = ind[-1]
        a,b,c = geometry.get_line_by_2_points(self.x[i], self.y[i], self.x[i+1], self.y[i+1])
        e,inten = geometry.get_line_intersection(a, b, c, 0, 1, -y)
        return e

    def toExafs(self, Efermi, k_power):
        me = 2 * 9.109e-31  # kg
        h = 1.05457e-34  # J*s
        J = 6.24e18  # 1 J = 6.24e18 eV
        e, s = self.x, self.y
        i = e >= Efermi
        k = np.sqrt((e[i] - Efermi) / J * me / h ** 2 / 1e20)  # J * kg /  (J*s)^2 = kg / (J * s^2) = kg / ( kg*m^2 ) = 1/m^2 = 1e-20 / A^2
        intSqr = (s[i] - 1) * k ** k_power
        # for j in range(len(k)):
        #     print(k[j], e[i][j])
        return Exafs(k, intSqr)

    def toTuple(self):
        return self.x, self.y

    def __add__(self, other):
        e1 = max(self.x[0], other.x[0])
        e2 = min(self.x[-1], other.x[-1])
        ind1 = (e1 <= self.x) & (self.x <= e2)
        ind2 = (e1 <= other.x) & (other.x <= e2)
        if np.sum(ind1) > np.sum(ind2):
            e = self.x[ind1]
            s = self.y[ind1] + np.interp(e, other.x, other.y)
        else:
            e = other.x[ind2]
            s = other.intensity[ind2] + np.interp(e, self.x, self.y)
        return Spectrum(e, s)

    def __mul__(self, other):
        assert isinstance(other, numbers.Number)
        return Spectrum(self.x, self.y*other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        assert isinstance(other, numbers.Number)
        return Spectrum(self.x, self.y/other)


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


def readSpectrum(fileName, skiprows=0, energyColumn=None, intensityColumn=None, separator=r'\s+', decimal=".", guess=True, gatherOp=None, xName='energy', yName='intensity'):
    """
    Read spectrum from file (columnwise). Try guess=True first

    :param gatherOp: 'average' or 'sum' - what to do with spetrum values for duplicate energies
    """
    fileName = fixPath(fileName)

    def fixColumnInd(ncols, energyColumn, intensityColumn):
        ext = os.path.splitext(fileName)[-1]
        if energyColumn is None:
            if ext == '.nor': energyColumn = 0
            else:
                if ncols > 2: print('energyColumn not set in readSpectrum. Use energyColumn=0')
                energyColumn = 0
        if intensityColumn is None:
            if ext == '.nor': intensityColumn = 3
            else:
                assert ncols == 2, f"intensityColumn not set in readSpectrum and ncols > 2. File: {fileName}"
                intensityColumn = 1
        if ncols == 2:
            if ncols <= energyColumn:
                print(f'Warning: wrong energyColumn number {energyColumn} in readSpectrum. It was corrected to 0')
                energyColumn = 0
            if ncols <= intensityColumn:
                print(f'Warning: wrong intensityColumn number {intensityColumn} in readSpectrum. It was corrected to 1')
                intensityColumn = 1
        assert ncols > energyColumn
        assert ncols > intensityColumn
        return energyColumn, intensityColumn
    if guess:
        with open(fileName, 'r') as f: lines = f.read()
        lines = lines.split('\n')
        #skip lines
        lines = lines[skiprows:]
        lines = [l.strip() for l in lines]
        # delete empty lines
        lines = [l for l in lines if len(l) > 0]
        # delete comments
        lines = [l for l in lines if l[0] not in ['#', '!', '%']]
        # if last line contains ',' but not '.', than ',' is integer part feature
        if ',' in lines[-1] and '.' not in lines[-1]:
            lines = [l.replace(',', '.') for l in lines]
        # check non-whitespace separators
        separators = [',', ';']
        if separators[0] in lines[-1]: assert separators[1] not in lines[-1], 'Two different separators are used. Can\'t guess'
        if separators[1] in lines[-1]: assert separators[0] not in lines[-1], 'Two different separators are used. Can\'t guess'
        for sep in separators:
            lines = [l.replace(sep, '\t') for l in lines]
        # delete lines which are not numbers
        lines = [l for l in lines if re.match(r"^[\d.eE+\-\s]*$", l) is not None]
        assert len(lines) > 0, 'Unknown file format. Can\'t guess'
        numbers = re.findall(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', lines[-1])
        ncols = len(numbers)
        energyColumn, intensityColumn = fixColumnInd(ncols, energyColumn, intensityColumn)
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
        result = Spectrum(result[:, energyColumn], result[:, intensityColumn], xName=xName, yName=yName)
    else:
        data = pd.read_csv(fileName, sep=separator, decimal=decimal, skiprows=skiprows, header=None)
        energyColumn, intensityColumn = fixColumnInd(data.shape[1], energyColumn, intensityColumn)
        energy = data[data.columns[energyColumn]]
        if not is_numeric_dtype(energy):
            raise Exception('Data energy column is not numeric: '+str(energy))
        intensity = data[data.columns[intensityColumn]]
        if not is_numeric_dtype(intensity):
            raise Exception('Data energy column is not numeric: '+str(intensity))
        result = Spectrum(energy.values, intensity.values, xName=xName, yName=yName)
    e = result.x
    assert np.all(e[1:] - e[:-1] >= 0)
    if np.any(e[1:] - e[:-1] == 0):
        if gatherOp is None:
            print(f'Spectrum {fileName} contains duplicate energies. Calculate average')
            gatherOp = 'average'
        result = Spectrum(*fixMultiValue(e,result.y, gatherOp=gatherOp), xName=xName, yName=yName)
    return result


def fixMultiValue(x, y, gatherOp):
    """
    For non-unique x returns x1,y1 with unique sorted x-values and gathered y

    :param x: x array
    :param y: y array
    :param gatherOp: 'mean' or 'sum'
    :return: x1,y1
    """
    assert gatherOp in ['mean', 'sum']
    x1, ind, counts = np.unique(x, return_counts=True, return_index=True)
    y1 = y[ind]
    not_unique_ind = np.where(counts > 1)[0]
    for i in not_unique_ind:
        if gatherOp == 'mean:':
            y1[i] = np.mean(y[x == x1[i]])
        else:
            y1[i] = np.sum(y[x == x1[i]])
    return x1,y1


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


def findAllMax(e, xanes, region=[], maxCount=-1, filterRightGreater=False):
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


def initPyfitit():
    styles = '''<style>
        .container { width:100% !important; }
        .output_area {display:inline-block !important; }
        .cell > .output_wrapper > .output {margin-left: 14ex;}
        .out_prompt_overlay.prompt {min-width: 14ex;}
        .fitBySlidersOutput {flex-direction:row !important; }
        /*.fitBySlidersOutput .p-Widget.jupyter-widgets-output-area.output_wrapper {display:none} - прячет ошибки*/
        .fitBySlidersOutput div.output_subarea {max-width: inherit}
        .fitBySlidersExafsOutput {flex-direction:row !important; }
        /*.fitBySlidersExafsOutput .p-Widget.jupyter-widgets-output-area.output_wrapper {display:none} - прячет ошибки*/
        .fitBySlidersExafsOutput div.output_subarea {max-width: inherit}
        .pcaOutput { display:block; }
        .pcaOutput div.output_subarea {max-width: inherit}
        .pcaOutput .output_subarea .widget-hbox { place-content: initial!important; }
        .pcaOutput > .output_area:nth-child(2) {float:left !important; }
        .pcaOutput > .output_area:nth-child(3) {display:block !important; }
        .pcaOutput > .output_area:nth-child(4) { /* width:100%; */ }
        .status { white-space: pre; }

        .fileBrowser {margin-left:14ex; display:block; }
        .fileBrowser div.p-Panel {display:block; }
        .fileBrowser button.folder { font-weight:bold; }
        .fileBrowser button.parentFolder { font-size:200%; }

        .widget-inline-hbox .widget-label {width:auto}
        </style>'''
    if isJupyterNotebook():
        display(HTML(styles))


def saveNotebook():
    if not isJupyterNotebook(): return
    display(Javascript('IPython.notebook.save_notebook();'))


def saveAsScript(fileName):
    if not isJupyterNotebook(): return
    fileName = fixPath(fileName)
    notebook_path = this_notebook()
    if notebook_path is None:
        print('Can\'t find notebook file. Do you use non standard connection to jupyter server? Save this file as script in main menu')
        return
    with open(notebook_path, 'r', encoding='utf-8') as fp:
        notebook = nbformat.read(fp, as_version=4)
    script = jupytext.writes(notebook, ext='.py', fmt='py')
    script_path = fileName
    if script_path[-3:] != '.py': script_path += '.py'
    with open(script_path, 'w', encoding='utf-8') as fp: fp.write(script)


def zfill(i, n):
    assert n>0
    return str(i).zfill(1+math.floor(0.5+math.log(n,10)))


def gauss(x,a,s):
    return 1/s/np.sqrt(2*np.pi)*np.exp(-(x-a)**2/2/s**2)


def findFile(folder='.', postfix=None, mask=None, check_unique=True, ignoreSlurm=True, returnAll=False):
    assert (mask is not None) or (postfix is not None)
    assert (mask is None) or (postfix is None)
    if postfix is not None: mask = f'*{postfix}'
    if folder != '.': mask = f'{folder}/{mask}'
    files = sorted(glob.glob(mask))
    if ignoreSlurm:
        files = [f for f in files if f.find('slurm')<0]
    if check_unique:
        assert len(files)>0, 'Can\'t find file '+mask
        assert len(files)==1, 'There are several files '+mask
    if len(files) == 1 and not returnAll: return files[0]
    else:
        if returnAll: return files
        else: return None


def wrap(s, maxLineLen, maxLineCount=None):
    if len(s) <= maxLineLen: return s
    i_last = 0
    i = i_last + maxLineLen
    lines = []
    while i < len(s):
        i1 = s.rfind(' ', i_last, i)
        if i1 == -1: i1 = i
        lines.append(s[i_last:i1])
        i_last = i1
        i = i_last + maxLineLen
    lines.append(s[i_last:])
    if maxLineCount is not None and len(lines) > maxLineCount:
        lines = lines[:maxLineCount]
        lines[-1] += ' ...'
    return '\n'.join(lines)


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
    assert np.all(energy[1:] >= energy[:-1]), f'Energies in data frame are not sorted!\n'+str(energy)
    return energy


def makeDataFrame(energy, spectra):
    assert np.all(energy[1:] >= energy[:-1]), f'Energies for new dataframe are not sorted!\n' + str(energy)
    return pd.DataFrame(data=spectra, columns=['e_' + str(e) for e in energy])


def makeDataFrameFromSpectraList(spectra, energy=None):
    assert isinstance(spectra, list)
    energy0 = energy
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
        if isinstance(s, np.ndarray):
            assert energy0 is not None
            assert len(s) == len(energy)
            sp_matr[i] = np.interp(energy, energy, s)
        else:
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


def rFactorSp(theory, exp, weight=None, p=2, sub1=False, interval=None):
    """
        :param p: power in Lp norm |theory-exp|^p/|exp|^p
        :param sub1: whether to divide in norm by |exp|^p or by |exp-1|^p
    """
    et, e = theory.energy, exp.energy
    yt, ye = theory.intensity, exp.intensity
    if interval is not None:
        ind = (interval[0]<=e) & (e<=interval[1])
        e, ye = e[ind], ye[ind]
    if len(et) != len(e) or np.any(et != e):
        yt = np.interp(e, et, yt)
    return rFactor(e, yt, ye, weight=weight, p=p, sub1=sub1)


def rFactor(e, theory, exp, weight=None, p=2, sub1=False):
    assert len(e) == len(theory), f'{len(e)} != {len(theory)}'
    assert len(e) == len(exp), f'{len(e)} != {len(exp)}'
    if weight is None: weight = np.ones(len(e))
    assert len(weight) == len(e)
    exp1 = exp-1 if sub1 else exp
    return integral(e, np.abs((theory - exp)*np.conj(theory - exp))**(p/2) * weight) / integral(e, np.abs(exp1*np.conj(exp1))**(p/2) * weight)


def save_pkl(obj, fileName):
    folder = os.path.split(fileName)[0]
    if folder != '' and not os.path.exists(folder):
        os.makedirs(folder)
    with open(fileName, 'wb') as f:
        pickle.dump(obj, f)


def load_pkl(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)


def saveData(obj, file_name):
    folder = os.path.split(file_name)[0]
    if folder != '' and not os.path.exists(folder):
        os.makedirs(folder)
    ext = os.path.splitext(file_name)[-1]
    if ext == '.pkl':
        with open(file_name, 'wb') as f:
            pickle.dump(obj, f)
    else:
        assert ext == '.json'
        with open(file_name, 'w') as f:
            json.dump(obj, f, cls=NumpyEncoder)


def loadData(file_name):
    ext = os.path.splitext(file_name)[-1]
    if ext == '.pkl':
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    else:
        assert ext == '.json'
        with open(file_name, 'r') as f:
            return json.load(f)


def cacheInFile(file_name, calc_func, is_changed_func=None):
    """
    Returns cached object or calculates new by calc_func and saves if cache doesn't exist or if is_changed_func(saved_obj)=True
    :param file_name: .json or .pkl file
    :param calc_func: func() -> obj
    :param is_changed_func: func(saved_obj)
    :return: cached or new object
    """
    if os.path.exists(file_name):
        saved = loadData(file_name)
        if is_changed_func is None:
            return saved
        else:
            if is_changed_func(saved):
                res = calc_func()
                saveData(res, file_name)
                return res
            else: return saved
    else:
        res = calc_func()
        saveData(res, file_name)
        return res


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


def makeBars(x, y, base=0):
    """
    Interpolate y by zero to make plot(x,y) like bar graph
    """
    i = np.argsort(x)
    x = x[i]
    y = y[i]
    n = len(x)
    dx = x[1:] - x[:-1]
    x = np.concatenate((x, x[1:] - dx / 5, x[:-1] + dx / 5))
    y = np.concatenate((y, np.zeros(n-1)+base, np.zeros(n-1)+base))
    i = np.argsort(x)
    x = x[i]
    y = y[i]
    return x,y


def is_numeric(obj):
    try:
        obj+obj, obj-obj, obj*obj, obj**obj, obj/obj
    except ZeroDivisionError:
        return True
    except Exception:
        return False
    else:
        return True


def length(x):
    if hasattr(x, "__len__"): return len(x)
    else: return 1   # x is scalar


def variation(x, y):
    diff = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    return integral((x[1:] + x[:-1]) / 2, np.abs(diff))


def expandByReflection(e, xanes, side='both', reflType='odd', stableReflMeanCount=1):
    """
    :param side: 'left', 'right', 'both'
    :param reflType: 'odd' or 'even'
    :param stableReflMeanCount: int, for odd reflection - reflect using mean of edge instead of one edge value
    """
    assert reflType in ['odd', 'even']
    assert side in ['left', 'right', 'both']
    e0, xanes0 = e, xanes
    assert np.all(e[1:]-e[:-1]>=0)
    rx = np.flip(xanes0)
    re = np.flip(e0)
    if side in ['left', 'both']:
        e = np.concatenate(( e0[0]-(re[:-1]-re[-1]), e))
        if reflType == 'even':
            xanes = np.concatenate((rx[:-1], xanes))
        else:
            mean = stableMean(xanes0[:stableReflMeanCount])
            xanes = np.concatenate((2*mean-rx[:-1], xanes))
    if side in ['right', 'both']:
        e = np.concatenate((e, e0[-1]-(re[1:]-re[0]) ))
        if reflType == 'even':
            xanes = np.concatenate((xanes, rx[1:]))
        else:
            mean = stableMean(rx[:stableReflMeanCount])
            xanes = np.concatenate((xanes, 2*mean-rx[1:]))
    return e, xanes


def stableMean(ar, throwCount=1):
    if throwCount <= 0: return np.mean(ar)
    if len(ar) <= 2*throwCount: return np.median(ar)
    sar = np.sort(ar)
    return np.mean(sar[throwCount:-throwCount])


def dict2str(d, precision=3):
    """
    Converts dict of floats to str
    """
    keys = sorted(list(d.keys()))
    res = ''
    for k in keys:
        res += f'{k}=' + f'%.{precision}g' % d[k]
        if k != keys[-1]: res += ' '
    return res


def inside(x, interval):
    return np.all((interval[0]<=x) & (x<=interval[1]))


class CacheInMemory:
    """
    Contains dataDict - dictionary with different data for speedup the calculations. Evaluate new data only if dependData was changed
    """
    def __init__(self):
        self.dataDict = {}

    def getHash(self, o):
        return hash(pickle.dumps(o))

    def getIfUpToDate(self, dataName, dependData):
        if dataName in self.dataDict:
            hash, data = self.dataDict[dataName]
            if hash == self.getHash(dependData):
                return data

        return None

    def getFromCacheOrEval(self, dataName, evalFunc, dependData):
        data = self.getIfUpToDate(dataName, dependData)
        if data is None:
            data = evalFunc()
            self.dataDict[dataName] = (self.getHash(dependData), data)

        return data


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def reloadModule(package_name):
    # get a reference to each loaded module
    loaded_package_modules = dict([
        (key, value) for key, value in sys.modules.items()
        if key.startswith(package_name) and isinstance(value, types.ModuleType)])

    # delete references to these loaded modules from sys.modules
    for key in loaded_package_modules:
        del sys.modules[key]

    # load each of the modules again;
    # make old modules share state with new modules
    for key in loaded_package_modules:
        # print 'loading %s' % key
        newmodule = __import__(key)
        oldmodule = loaded_package_modules[key]
        oldmodule.__dict__.clear()
        oldmodule.__dict__.update(newmodule.__dict__)


def reloadPyfitit():
    reloadModule('pyfitit')


def dictToStr(d, floatFormat="%.3g"):
    keys = sorted(list(d.keys()))
    s = ''
    for k in keys:
        v = floatFormat % d[k] if isinstance(d[k], float) else str(d[k])
        s += f' {k}={v}'
    return s.strip()


def runCommand(command, workingFolder='.', outputTxtFile='output.txt', env=None, input=None):
    """
    Run command inside the workingFolder. Writes output to outputTxtFile. If outputTxtFile=None, doesn't create the outputTxtFile.
    Returns output and command returncode
    """
    proc = subprocess.Popen(shlex.split(command), cwd=workingFolder, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, env=env)
    if isinstance(input, str): input = input.encode('utf-8')
    stdoutdata, stderrdata = proc.communicate(input=input)
    output = ''
    if stdoutdata is not None: output += stdoutdata.decode("utf-8")
    if stderrdata is not None: output += stderrdata.decode("utf-8")
    if output != '' and outputTxtFile is not None:
        with open(workingFolder+os.sep+outputTxtFile, 'w') as f: f.write(output)
    return output, proc.returncode


def jobIsRunning(folder):
    slurmFiles = findFile(folder=folder, mask='slurm-*.out', check_unique=False, ignoreSlurm=False, returnAll=True)
    if slurmFiles is None or len(slurmFiles) == 0: return False
    lastFile = slurmFiles[-1]
    id = int(os.path.splitext(os.path.split(lastFile)[-1])[0][6:])
    output, code = runCommand(f"squeue -j {id} -h")
    output = output.strip()
    if code != 0: return False
    if output != '' and str(id) in output: return True
    return False

import copy, os, json, jupytext, nbformat
from . import utils, smoothLib, optimize
# from importlib.machinery import SourceFileLoader, ExtensionFileLoader
from types import MethodType
import ipywidgets as widgets
from IPython.display import display, Javascript


def loadProject(projectFile, *params0, **params1):
    import importlib.util as _importlib_util
    projectFile = utils.fixPath(projectFile)
    name = utils.randomString(10)
    spec = _importlib_util.spec_from_file_location(name, projectFile)
    ProjectModule = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(ProjectModule)
    # ProjectModule = SourceFileLoader(name, projectFile).load_module()
    return ProjectModule.projectConstructor(*params0, **params1)


def createPartialProject(name = None, expSpectrum = None, intervals = None, fdmnesShift = 0, geometryParamRanges = None, moleculeConstructor = None):
    project = Project()
    project.name = name
    project.spectrum = expSpectrum
    if intervals is not None: project.intervals = intervals
    project.geometryParamRanges = geometryParamRanges
    if moleculeConstructor is not None:
        project.moleculeConstructor = MethodType(moleculeConstructor, project)
    if expSpectrum is not None:
        if intervals is None:
            a = expSpectrum.energy[0]; b = expSpectrum.energy[-1]
            project.intervals = {
              'fit_norm': [a, b],
              'fit_smooth': [a, b],
              'fit_geometry': [a, b],
              'plot': [a, b]
            }
        if fdmnesShift != 0: project.FDMNES_smooth['shift'] = fdmnesShift
        project.FDMNES_smooth['Efermi'] = expSpectrum.energy[0]
    return project


class LimitedDictClass(dict):
    def __init__(self, allowedKeys=None, userSetitemHook=None):
        super().__init__()
        if allowedKeys is None: allowedKeys = []
        self._keys = list(allowedKeys)
        for key in self._keys: self[key] = None
        self.userSetitemHook = userSetitemHook

    def __setitem__(self, key, val):
        if key not in self._keys: raise KeyError('Not existent key '+str(key))
        dict.__setitem__(self, key, val)
        if hasattr(self, 'userSetitemHook') and (self.userSetitemHook is not None):
            self.userSetitemHook(self, key, val)

    def __contains__(self, k):
        if k not in self._keys: return False
        if self[k] is None: return False
        return True

    def __iter__(self):
        return (k for k in self._keys if self[k] is not None)

    def __deepcopy__(self, memo):
        result = LimitedDictClass(allowedKeys = self._keys)
        for key in result._keys: result[key] = copy.deepcopy(self[key])
        result.userSetitemHook = self.userSetitemHook
        return result

    def __delitem__(self, k):
        dict.__setitem__(self, k, None)


def LimitedDictProperty(propertyName):
    hiddenPropertyName = '_'+propertyName
    def getter(self):
        return getattr(self, hiddenPropertyName)
    def setter(self, newDict):
        d = getattr(self, hiddenPropertyName)
        for k in newDict:
            d[k] = newDict[k]
    res = property(getter, setter)
    return res


class Project(object):
    intervalNames = ['fit_norm', 'fit_smooth', 'fit_geometry', 'plot', 'fit_exafs']
    fdmnesCalcNames = ['Energy range', 'Green', 'radius', 'Adimp', 'Quadrupole', 'Absorber', 'Edge', 'cellSize', 'electronTransfer', 'additional']
    fdmnesSmoothNames = ['Gamma_hole', 'Ecent', 'Elarg', 'Gamma_max', 'Efermi', 'shift', 'norm']
    opticalSmoothNames = ['shift', 'norm']
    
    # attributes of class type, but not of instances!!!
    intervals = LimitedDictProperty('intervals')
    FDMNES_calc = LimitedDictProperty('FDMNES_calc')
    FDMNES_smooth = LimitedDictProperty('FDMNES_smooth')
    optical_smooth = LimitedDictProperty('optical_smooth')

    def __init__(self):
        self.name = ''
        self._spectrum0 = None
        self._spectrum = None
        self._maxSpectrumPoints = 100
        self._useFullSpectrum = False

        def setitemHookIntervals(th, key, val):
            if key not in ['fit_norm', 'fit_smooth', 'fit_geometry', 'plot']: return
            self.spectrum = self._spectrum0

        self._intervals = LimitedDictClass(self.intervalNames, userSetitemHook=setitemHookIntervals)
        for interval in self.intervalNames: self._intervals[interval] = [1e6,-1e6]
        self.geometryParamRanges = None
        self._FDMNES_calc = LimitedDictClass(self.fdmnesCalcNames)
        self.FDMNES_calc = {'radius':5, 'Adimp':None, 'Quadrupole':False, 'Absorber':1, 'Green':False, 'Edge':'K', 'cellSize':1.0, 'electronTransfer':None, 'Energy range':'-15 0.02 8 0.1 18 0.5 30 2 54 3 117'}
        self.defaultSmoothParams = smoothLib.DefaultSmoothParams(0,0)

        def itemHookSmooth(smooth_type):
            def setitemHookSmooth(th, key, val):
                if val == '': return
                assert val is not None, 'key='+key+' val='+str(val)
                if key == 'shift':
                    self.defaultSmoothParams[smooth_type]['shift'] = optimize.param('shift', val, [val-20, val+20], 1, 0.25)
                elif key == 'norm':
                    self.defaultSmoothParams[smooth_type].params.append(optimize.param('norm', val, [val/10, val*10], val/10, val/10))
                elif key == 'Efermi':
                    self.defaultSmoothParams[smooth_type]['Efermi'] = optimize.param('Efermi', val, [val - 20, val + 20], 1, 0.25)
                else:
                    self.defaultSmoothParams[smooth_type][key] = val
            return setitemHookSmooth
        self._FDMNES_smooth = LimitedDictClass(self.fdmnesSmoothNames, userSetitemHook=itemHookSmooth('fdmnes'))
        self._optical_smooth = LimitedDictClass(self.opticalSmoothNames, userSetitemHook=itemHookSmooth('optical'))

    def spectrum_get(self): return self._spectrum
    def spectrum_set(self, s):
        self._spectrum0 = copy.deepcopy(s)
        if s is None: self._spectrum = None
        else:
            if self.useFullSpectrum: self._spectrum = self._spectrum0
            else:
                self._spectrum = utils.adjustSpectrum(s, self.maxSpectrumPoints, self.intervals)
                Efermi = self.defaultSmoothParams['fdmnes']['Efermi']
                if Efermi<s.energy[0]-200 or s.energy[0]>s.energy[-1]+200:
                    self.defaultSmoothParams['fdmnes']['Efermi'] = optimize.param('Efermi', s.energy[0]+10, [s.energy[0]-20, s.energy[-1]+20], 1, 0.25)
    spectrum = property(spectrum_get, spectrum_set)

    def maxSpectrumPoints_get(self): return self._maxSpectrumPoints
    def maxSpectrumPoints_set(self, msp):
        self._maxSpectrumPoints = msp
        self.spectrum = self._spectrum0
    maxSpectrumPoints = property(maxSpectrumPoints_get, maxSpectrumPoints_set)

    def useFullSpectrum_get(self): return self._useFullSpectrum
    def useFullSpectrum_set(self, ufs):
        self._useFullSpectrum = ufs
        self.maxSpectrumPoints = self.maxSpectrumPoints
    useFullSpectrum = property(useFullSpectrum_get, useFullSpectrum_set)

    def constructMoleculesForEdgePoints(self):
        params = list(self.geometryParamRanges.keys())
        params.sort()
        default = {}
        for pName in params:
            r = self.geometryParamRanges[pName]
            if (r[0]<=0) and (r[1]>=0): default[pName] = 0
            else: default[pName] = (r[0]+r[1])/2
        for pName in params:
            d = copy.deepcopy(default)
            d[pName] = self.geometryParamRanges[pName][0]
            M = self.moleculeConstructor(d)
            if M is None: print('M = None for', pName, '=', d[pName])
            else: M.export_xyz('mol_'+pName+'_'+str(d[pName])+'.xyz')
            d[pName] = self.geometryParamRanges[pName][1]
            M = self.moleculeConstructor(d)
            if M is None: print('M = None for', pName, '=', d[pName])
            else: M.export_xyz('mol_'+pName+'_'+str(d[pName])+'.xyz')


checkProjectData = type('', (), {})()
def checkProject(projectConstructor, checkProjectParameters = {}, checkMoleculaParameters = {}):
    global checkProjectData
    checkProjectData.params = {}
    checkProjectData.statusHTML = widgets.HTML(value='', placeholder='', description='')

    def setStatus(s):
        global checkProjectData
        checkProjectData.statusHTML.value = '<div class="status">'+s+'</div>'

    project = projectConstructor(**checkProjectParameters)

    def check(_):
        for attr in ['spectrum', 'moleculeConstructor']:
            if not hasattr(project, attr):
                print('The project property "'+attr+'" is not initialized')
                return
        molecule = project.moleculeConstructor(checkProjectData.params)
        molecule.export_xyz('molecule.xyz')
        setStatus('Saved successfully')

    controls = []
    o = 'horizontal'  # 'vertical'
    for pName in project.geometryParamRanges:
        p0 = project.geometryParamRanges[pName][0]; p1 = project.geometryParamRanges[pName][1]
        slider = widgets.FloatSlider(description=pName, min=p0, max=p1, step=(p1-p0)/30, value=(p0+p1)/2, orientation=o, continuous_update=False)
        if pName in checkMoleculaParameters: slider.value = checkMoleculaParameters[pName]
        checkProjectData.params[pName] = slider.value
        controls.append(slider)
    buttonCheck = widgets.Button(description="Generate molecule.xyz")
    buttonCheck.on_click(check)
    controls.append(buttonCheck)

    ui = widgets.VBox(tuple(controls) + (checkProjectData.statusHTML,))
    controlsDict = {}
    for c in controls:
        if hasattr(c, 'value'): controlsDict[c.description] = c

    def saveParams(**params):
        global checkProjectData
        for p in params: checkProjectData.params[p] = params[p]
    out = widgets.interactive_output(saveParams, controlsDict)
    display(ui, out)
    display(Javascript('$(this.element).addClass("fitBySlidersOutput");'))


def saveAsProject(fileName='project.py'):
    if not utils.isJupyterNotebook():
        print('Can\'t save, because the script is running not from Jupyter notebook system')
        return
    fileName = utils.fixPath(fileName)
    notebook_path = utils.this_notebook()
    with open(notebook_path, 'r', encoding='utf-8') as fp:
        notebook = nbformat.read(fp, as_version=4)
    project = jupytext.writes(notebook, ext='.py', fmt='py')
    project_path = fileName
    if project_path[-3:] != '.py': project_path += '.py'
    with open(project_path, 'w', encoding='utf-8') as fp: fp.write(postProcessProjectText(project))


def postProcessProjectText(project):
    endComment = '==============='

    endCommentIndex = project.find(endComment)
    if endCommentIndex < 0:
        return project

    # remove everythink after endComment
    project = project[:endCommentIndex + len(endComment)]

    getProjectFolderText = \
"""
import os
def getProjectFolder(): return os.path.dirname(os.path.realpath(__file__))
"""
    getProjectInsertionContext = "from pyfitit import *"
    contextPosition = project.find(getProjectInsertionContext)
    if contextPosition < 0:
        return project

    contextPosition = contextPosition + len(getProjectInsertionContext)
    project = project[:contextPosition] + getProjectFolderText + project[contextPosition:]
    return project

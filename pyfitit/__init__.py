import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
from . import utils
utils.fixDisplayError()
import numpy as np
import sys, tempfile, copy, json, warnings, traceback, gc, jupytext, nbformat, numbers
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, Javascript, HTML
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf, interp1d, CubicSpline
from matplotlib.font_manager import FontProperties
from types import MethodType
from sklearn.ensemble import ExtraTreesRegressor
# we need to list all libraries, otherwise python can't find them in other imports
from . import adf, directMethod, factorAnalysis, fdmnes, feff, fileBrowser, geometry

from . import inverseMethod, ML, molecule, optimize,  plotting, sampling, smoothLib

from .project import Project, loadProject, checkProject, saveAsProject, createPartialProject
from .molecule import Molecule, pi, norm, cross, dot, copy, normalize
from .utils import Spectrum, readSpectrum, readExafs
join = os.path.join

debug = False


def getProjectFolder(): return os.getcwd()


# uncomment to see warnings source line
if debug:
    warnings.filterwarnings('error')

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


def initPyfitit():
    if utils.isJupyterNotebook():
        display(HTML(styles))


initPyfitit()


def saveNotebook():
    if not utils.isJupyterNotebook(): return
    display(Javascript('IPython.notebook.save_notebook();'))


def loadProject(*p, **q):
    initPyfitit()
    return project.loadProject(*p,**q)


def parseFdmnesFolder(folder):
    spectrum = fdmnes.parse_one_folder(folder)
    return spectrum


def parseADFFolder(folder, makePiramids = False):
    spectrum, _ = adf.parse_one_folder(folder, makePiramids)
    return spectrum


parseFeffFolder = feff.parse_one_folder

# sampling
readSample = ML.Sample.readFolder
generateInputFiles = sampling.generateInputFiles
calcSpectra = sampling.calcSpectra
collectResults = sampling.collectResults
constructDirectEstimator = directMethod.Estimator
constructInverseEstimator = inverseMethod.Estimator


def saveAsScript(fileName):
    if not utils.isJupyterNotebook(): return
    fileName = utils.fixPath(fileName)
    notebook_path = utils.this_notebook()
    if notebook_path is None:
        print('Can\'t find notebook file. Do you use non standard connection to jupyter server? Save this file as script in main menu')
        return
    with open(notebook_path, 'r', encoding='utf-8') as fp:
        notebook = nbformat.read(fp, as_version=4)
    script = jupytext.writes(notebook, ext='.py', fmt='py')
    script_path = fileName
    if script_path[-3:] != '.py': script_path += '.py'
    with open(script_path, 'w', encoding='utf-8') as fp: fp.write(script)


compareDifferentInverseMethods = inverseMethod.compareDifferentMethods
compareDifferentDirectMethods = directMethod.compareDifferentMethods


# for PCA
def openFile(*p): return fileBrowser.openFile('openFile',*p)
calcSVD = factorAnalysis.calcSVD
MalinowskyParameters = factorAnalysis.MalinowskyParameters
plotTestStatistic = factorAnalysis.plotTestStatistic
recommendPCnumber = factorAnalysis.recommendPCnumber


def saveToFile(fileName, obj):
    fileName = utils.fixPath(fileName)
    folder = os.path.dirname(fileName)
    os.makedirs(folder, exist_ok=True)
    if isinstance(obj, np.ndarray):
        np.savetxt(fileName, obj, delimiter = ' ')
    elif isinstance(obj, pd.DataFrame):
        obj.to_csv(fileName, sep=' ', index=False)
    else:
        with open(fileName, 'w') as f: json.dump(obj, f)


# fields: params, spectrum, xanesSmoothed, xanesDiff, xanesDiffSmoothed, fig, ax
class fitBySliders:
    def setStatus(self, s):
        self.status = s
        self.statusHTML.value = '<div class="status">'+s+'</div>'

    def addToStatus(self, s):
        self.status += s+'<br>'
        self.setStatus(self.status)

    def __init__(self, sample, project, defaultParams=None, diffFrom=None, norm=None, methodParams=None, smoothType='fdmnes', extraSpectra=None):
        if defaultParams is None: defaultParams = {}
        if methodParams is None: methodParams = {}
        self.params = {}
        self.lastParams = {}
        self.statusHTML = widgets.HTML(value='', placeholder='', description='')
        self.status = ''
        self.xanes_sm = None
        project = copy.deepcopy(project)
        xanes_energy = sample.energy
        self.fig, self.ax = plt.subplots()
        if diffFrom is not None:
            diffFrom = copy.deepcopy(diffFrom)
            diffFrom['projectBase'].spectrum.intensity = np.interp(project.spectrum.energy, diffFrom['projectBase'].spectrum.energy, diffFrom['projectBase'].spectrum.intensity)
            diffFrom['projectBase'].spectrum.energy = project.spectrum.energy
            diffFrom['spectrumBase'].intensity = np.interp(xanes_energy, diffFrom['spectrumBase'].energy, diffFrom['spectrumBase'].intensity)
            diffFrom['spectrumBase'].energy = xanes_energy
        # normalize params
        geometryParamsMin = np.min(sample.params.values, axis=0)
        geometryParamsMax = np.max(sample.params.values, axis=0)
        geometryParams = 2*(sample.params-geometryParamsMin)/(geometryParamsMax-geometryParamsMin) - 1
        self.ax2 = None
        ax = self.ax
        exp_e = project.spectrum.energy

        def smoothAfterPrediction(geomArg, **params):
            if 'norm' in params: norm = params['norm']
            else: norm = None
            shift = params['shift']
            exp_xanes = project.spectrum.intensity
            # prediction
            absorbPrediction = estimator['nonSmoothed'].predict(geomArg)[0]
            # smoothing
            xanesPrediction = utils.Spectrum(xanes_energy, absorbPrediction)
            smoothedPredictionNormalized, norm1 = smoothLib.smoothInterpNorm(params, xanesPrediction, 'fdmnes', project.spectrum, project.intervals['fit_norm'], norm)
            smoothedPredictionNormalized = smoothedPredictionNormalized.intensity
            self.spectrumSmoothed = utils.Spectrum(exp_e, smoothedPredictionNormalized, copy=True)
            absorbPredictionNormalized = xanesPrediction.intensity / np.mean(xanesPrediction.intensity[-3:]) * np.mean(exp_xanes[-3:])
            self.spectrum = utils.Spectrum(xanes_energy+shift, absorbPredictionNormalized, copy=True)
            if diffFrom is not None:
                smoothedXanesBaseNormalized = smoothLib.smoothInterpNorm(params, diffFrom['spectrumBase'], 'fdmnes', diffFrom['projectBase'].spectrum, diffFrom['projectBase'].intervals['fit_norm'], norm)[0]
                smoothedXanesBaseNormalized = smoothedXanesBaseNormalized.intensity
                smoothedPredictionNormalized = (smoothedPredictionNormalized - smoothedXanesBaseNormalized)*params['purity']
                self.spectrumDiffSmoothed = utils.Spectrum(exp_e, smoothedPredictionNormalized, copy=True)
                absorbBaseNormalized = diffFrom['spectrumBase'].intensity / np.mean(diffFrom['spectrumBase'].intensity[-3:]) * np.mean(diffFrom['projectBase'].spectrum.intensity[-3:])
                absorbPredictionNormalized = (absorbPredictionNormalized - absorbBaseNormalized)*params['purity']
                self.spectrumDiff = utils.Spectrum(xanes_energy+shift, absorbPredictionNormalized, copy=True)
            return absorbPredictionNormalized, smoothedPredictionNormalized, norm1

        def smoothBeforePrediction(geomArg, **params):
            if 'norm' in params: norm = params['norm']
            else: norm = None
            shift = params['shift']
            cached = True
            isFast = self.xanes_sm is not None
            for p in ['shift', 'Gamma_hole', 'Gamma_max', 'Ecent', 'Elarg', 'Efermi']:
                if self.lastParams[p] != self.params[p]: isFast = False
            if debug: self.addToStatus('isFast smooth dataframe = '+str(isFast))
            if isFast: xanes_sm = self.xanes_sm
            else:
                xanes_sm, cached = smoothLib.smoothDataFrame(params, sample.spectra, 'fdmnes', project.spectrum, project.intervals['fit_smooth'], norm=norm, folder=sample.folder, returnCacheStatus=True)
                self.xanes_sm = xanes_sm
            if diffFrom is not None:
                smoothedXanesBase = smoothLib.smoothInterpNorm(params, diffFrom['spectrumBase'], 'fdmnes', diffFrom['projectBase'].spectrum, diffFrom['projectBase'].intervals['fit_norm'], norm)[0]
                smoothedXanesBase = smoothedXanesBase.intensity
                xanes_sm = (xanes_sm - smoothedXanesBase) * params['purity']
            isFast = cached
            for p in ['method', 'smooth before prediction']:
                if self.lastParams[p] != self.params[p]: isFast = False
            if debug: self.addToStatus('isFast fit estimator = '+str(isFast))
            if not isFast:
                estimator['smoothed'].fit(geometryParams.values, xanes_sm.values)
            smoothedPredictionNormalized = estimator['smoothed'].predict(geomArg)[0]
            absorbPredictionNormalized = np.zeros(xanes_energy.size)
            if diffFrom is None:
                self.spectrum = utils.Spectrum(xanes_energy+shift, absorbPredictionNormalized, copy=True)
                self.spectrumSmoothed = utils.Spectrum(exp_e, smoothedPredictionNormalized, copy=True)
            else:
                self.spectrum = utils.Spectrum(xanes_energy+shift, np.zeros(xanes_energy.size), copy=True)
                self.spectrumSmoothed = utils.Spectrum(exp_e, np.zeros(exp_e.size), copy=True)
                self.spectrumDiff = utils.Spectrum(xanes_energy+shift, absorbPredictionNormalized, copy=True)
                self.spectrumDiffSmoothed = utils.Spectrum(exp_e, smoothedPredictionNormalized, copy=True)
            gc.collect()
            return absorbPredictionNormalized, smoothedPredictionNormalized

        def smoothExtra(extraSpectra, **params):
            if 'norm' in params: norm = params['norm']
            else: norm = None
            extraSpectraSmoothed = []
            if extraSpectra is not None:
                for sp in extraSpectra:
                    smoothType = sp['smoothType']
                    assert smoothType in ['fdmnes', 'None', 'adf']
                    sp_sm = copy.deepcopy(sp)
                    if smoothType != 'None':
                        sp_sm['smoothed spectrum'] = smoothLib.smoothInterpNorm(params, sp['spectrum'], smoothType, project.spectrum, project.intervals['fit_norm'], norm)[0]
                    else:
                        sp_sm['smoothed spectrum'] = copy.deepcopy(sp_sm['spectrum'])
                    extraSpectraSmoothed.append(sp_sm)
            return extraSpectraSmoothed

        def plotXanes(**params):
            try:
                with warnings.catch_warnings(record=True) as warn:
                    for pName in params: self.params[pName] = params[pName]
                    shift = params['shift']
                    geomArg = np.array([params[pName] for pName in geometryParams.columns]).reshape([1,geometryParams.shape[1]])
                    geomArg = 2*(geomArg-geometryParamsMin)/(geometryParamsMax-geometryParamsMin) - 1

                    if params['smooth before prediction']:
                        absorbPredictionNormalized, smoothedPredictionNormalized = smoothBeforePrediction(geomArg, **params)
                        norm1 = None
                    else:
                        absorbPredictionNormalized, smoothedPredictionNormalized, norm1 = smoothAfterPrediction(geomArg, **params)
                    extraSpectraSmoothed = smoothExtra(extraSpectra, **params)
                    self.spectrumDiff = utils.Spectrum(xanes_energy+shift, absorbPredictionNormalized, copy=True)
                    # plotting
                    exp_xanes = project.spectrum.intensity
                    self.spectrumExp = utils.Spectrum(exp_e, exp_xanes, copy=True)
                    self.spectrumSmoothedDiff = utils.Spectrum(exp_e, smoothedPredictionNormalized, copy=True)
                    if diffFrom is not None:
                        expXanesBase = np.interp(exp_e, diffFrom['projectBase'].spectrum.energy, diffFrom['projectBase'].spectrum.intensity)
                        exp_xanes = exp_xanes - expXanesBase
                    self.spectrumExpDiff = utils.Spectrum(exp_e, exp_xanes, copy=True)
                    ax.clear()
                    if self.ax2 is not None: self.ax2.clear()
                    if params['not convoluted']:
                        ax.plot(self.spectrumDiff.energy, self.spectrumDiff.intensity, label='initial', color='orange')
                    ax.plot(self.spectrumSmoothedDiff.energy, self.spectrumSmoothedDiff.intensity, label='convolution', color='blue')
                    ax.plot(self.spectrumExpDiff.energy, self.spectrumExpDiff.intensity, c='k', label="Experiment")
                    for sp in extraSpectraSmoothed:
                        en = sp['smoothed spectrum'].energy
                        ax.plot(en, sp['smoothed spectrum'].intensity, label=sp['label'])
                    if params['smooth width']:
                        if self.ax2 is None: self.ax2 = ax.twinx()
                        Efermi = params['Efermi']
                        if not fdmnes.useEpsiiShift: Efermi += shift
                        smoothWidth = smoothLib.YvesWidth(exp_e, params['Gamma_hole'], params['Ecent'], params['Elarg'], params['Gamma_max'], Efermi)
                        self.ax2.plot(exp_e, smoothWidth, c='r', label='Smooth width')
                        self.ax2.legend()
                    else:
                        if self.ax2 is not None: self.ax2.remove(); self.ax2 = None
                    ax.set_xlim([params['energyRange'][0], params['energyRange'][1]])
                    if diffFrom is None: ax.set_ylim([0, np.max(exp_xanes)*1.2])
                    else:
                        d = np.max(exp_xanes)-np.min(exp_xanes)
                        ax.set_ylim([np.min(exp_xanes)-0.2*d, np.max(exp_xanes)+0.2*d])
                    font = FontProperties(); font.set_weight('black'); font.set_size(20)
                    txt = ax.text(project.intervals['fit_norm'][0], ax.get_ylim()[0], '[', color='green', verticalalignment='bottom', fontproperties=font)
                    txt.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
                    txt = ax.text(project.intervals['fit_norm'][1], ax.get_ylim()[0], ']', color='green', verticalalignment='bottom', fontproperties=font)
                    txt.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
                    ax.set_xlabel("Energy")
                    ax.set_ylabel("Intensity")
                    ax.legend(loc='upper right')
                    ind = (exp_e >= params['energyRange'][0]) & (exp_e <= params['energyRange'][1])
                    rFactor = utils.integral(exp_e[ind], (exp_xanes[ind]-smoothedPredictionNormalized[ind])**2) / utils.integral(exp_e[ind], exp_xanes[ind]**2)
                    if norm1 is None: info = 'R-factor = {:.4f}'.format(rFactor)
                    else: info = 'norm = {:.4g} R-factor = {:.4f}'.format(norm1, rFactor)
                    ax.text(0.98, 0.02, info, transform=ax.transAxes, horizontalalignment='right')
                    self.lastParams = copy.deepcopy(self.params)
                    if len(warn)>0:
                        status = 'Warning: '
                        for w in warn: status += str(w.message)+'\n'
                        self.addToStatus(status)
            except Exception as exc:
                self.addToStatus(traceback.format_exc())
            except Warning:
                self.addToStatus(traceback.format_exc())

        controls = []
        o = 'horizontal' # 'vertical'
        for pName in project.geometryParamRanges:
            p0 = project.geometryParamRanges[pName][0]; p1 = project.geometryParamRanges[pName][1]
            controls.append(widgets.FloatSlider(description=pName, min=p0, max=p1, step=(p1-p0)/30, value=(p0+p1)/2, orientation=o, continuous_update=False))
        if 'shift' in defaultParams: shift = defaultParams['shift']
        else: shift = optimize.value(project.defaultSmoothParams['fdmnes'], 'shift')
        # self.addToStatus('shift = '+str(shift))
        controls.append(widgets.FloatSlider(description='shift', min=shift-10.0, max=shift+10.0, step=0.3, value=shift, orientation=o, continuous_update=False))
        controls.append(widgets.FloatSlider(description='Gamma_hole', min=0.1, max=10, step=0.2, value=1, orientation=o, continuous_update=False))
        controls.append(widgets.FloatSlider(description='Ecent', min=1, max=100, step=1, value=50, orientation=o, continuous_update=False))
        controls.append(widgets.FloatSlider(description='Elarg', min=1, max=100, step=1, value=50, orientation=o, continuous_update=False))
        controls.append(widgets.FloatSlider(description='Gamma_max', min=1, max=100, step=1, value=15, orientation=o, continuous_update=False))
        if fdmnes.useEpsiiShift:
            Efermi = project.FDMNES_smooth['Efermi']
            controls.append(widgets.FloatSlider(description='Efermi', min=Efermi-20, max=Efermi+20, step=1, value=0, orientation=o, continuous_update=False))
        else:
            controls.append(widgets.FloatSlider(description='Efermi', min=-20, max=20, step=1, value=project.defaultSmoothParams['fdmnes']['Efermi'], orientation=o, continuous_update=False))
        if 'norm' in defaultParams:
            nrm = defaultParams['norm']
            controls.append(widgets.FloatSlider(description='norm', min=nrm/2, max=nrm*2, step=nrm/20, value=nrm, orientation=o, continuous_update=False))
        if diffFrom is not None:
            controls.append(widgets.FloatSlider(description='purity', min=0, max=1, step=0.05, value=diffFrom['purity'], orientation=o, continuous_update=False))

        def changedCheckBoxSmoothWidth(p):
            if p['name'] == '_property_lock': return
            if debug: self.addToStatus('changedCheckBoxSmoothWidth triggered')
            self.params['smooth width'] = p['new']
            plotXanes(**self.params)
        checkBoxSmoothWidth = widgets.Checkbox(description='smooth width', value=False)
        controls.append(checkBoxSmoothWidth)

        def changedCheckBoxNotConvoluted(p):
            if p['name'] == '_property_lock': return
            if debug: self.addToStatus('changedCheckBoxNotConvoluted triggered')
            self.params['not convoluted'] = p['new']
            plotXanes(**self.params)
        checkBoxNotConvoluted = widgets.Checkbox(description='not convoluted', value=True)
        controls.append(checkBoxNotConvoluted)

        def changedCheckBoxSmoothBeforePrediction(p):
            if p['name'] == '_property_lock': return
            if self.lastParams['smooth before prediction'] == p['new']: return
            if debug: self.addToStatus('changedCheckBoxSmoothBeforePrediction triggered')
            self.params['smooth before prediction'] = p['new']
            plotXanes(**self.params)
        checkBoxSmoothBeforePrediction = widgets.Checkbox(description='smooth before prediction', value=False)
        controls.append(checkBoxSmoothBeforePrediction)

        estimator = {'smoothed':None, 'nonSmoothed':None}
        allMethods = inverseMethod.allowedMethods
        def changedDropdownMethod(p, doNotPlot = False):
            if p['name'] == '_property_lock': return
            if not isinstance(p['new'], str): return
            method = p['new']
            if ('method' in self.lastParams) and (self.lastParams['method'] == method): return
            self.params['method'] = method
            if debug: self.addToStatus('changedDropdownMethod triggered')
            estimator['nonSmoothed'] = inverseMethod.getMethod(method, methodParams)
            estimator['smoothed'] = inverseMethod.getMethod(method, methodParams)
            if method == 'RBF':
                # this will trigger replot
                checkBoxSmoothBeforePrediction.value = True
                return
            if ('smooth before prediction' not in self.params) or not self.params['smooth before prediction']:
                estimator['nonSmoothed'].fit(geometryParams.values, sample.spectra.values)
            if not doNotPlot: plotXanes(**self.params)
        method = 'Extra Trees' if 'method' not in defaultParams else defaultParams['method']
        dropdownMethod = widgets.Dropdown(options=allMethods, value=method, description='method', disabled=False)
        controls.append(dropdownMethod)
        changedDropdownMethod({'new':controls[-1].value, 'name':None}, doNotPlot=True)

        e0 = project.spectrum.energy[0]-10; e1 = project.spectrum.energy[-1]+10
        v0 = project.intervals['fit_geometry'][0]; v1 = project.intervals['fit_geometry'][1]
        controls.append(widgets.FloatRangeSlider(description='energyRange', min=e0,max=e1,step=(e1-e0)/30,value=[v0,v1], orientation='horizontal', continuous_update=False))

        defaultExpSmooth = project.FDMNES_smooth
        for c in controls:
            if c.description in defaultParams:
                if c.description != 'method':
                    c.value = defaultParams[c.description]
            elif c.description in defaultExpSmooth:
                c.value = defaultExpSmooth[c.description]
            if hasattr(c, 'value'): self.params[c.description] = c.value
        for p in self.params:
            if isinstance(self.params[p], numbers.Number):
                self.lastParams[p] = self.params[p]+1

        # set observers after all controls initialization, becuase initialization triggers observers
        checkBoxSmoothWidth.observe(changedCheckBoxSmoothWidth)
        checkBoxNotConvoluted.observe(changedCheckBoxNotConvoluted)
        checkBoxSmoothBeforePrediction.observe(changedCheckBoxSmoothBeforePrediction)
        dropdownMethod.observe(changedDropdownMethod)

        ui = widgets.HBox(tuple(controls)+(self.statusHTML,))
        ui.layout.flex_flow = 'row wrap'
        ui.layout.justify_content = 'space-between'
        ui.layout.align_items = 'flex-start'
        ui.layout.align_content = 'flex-start'
        # outputWithFigure.layout.
        controlsDict = {}
        for c in controls: controlsDict[c.description] = c
        out = widgets.interactive_output(plotXanes, controlsDict)
        # out.layout.min_height = '400px'
        display(ui, out)
        display(Javascript('$(this.element).addClass("fitBySlidersOutput");'))


# fields: params, spectrum, xanesSmoothed, xanesDiff, xanesDiffSmoothed, fig, ax
class fitBySlidersMixture:
    def setStatus(self, s):
        self.status = s
        self.statusHTML.value = '<div class="status">'+s+'</div>'

    def addToStatus(self, s):
        self.status += s
        self.setStatus(self.status)

    # extraSpectra = [{'label': ..., 'spectrum': ..., 'smoothType': ..., 'smoothParams':{...}}, ...]  smoothType='None' for no smooth
    def __init__(self, sample_list, project_list, defaultParams_list=None, diffFrom_list=None, norm_list=None, methodParams_list=None, smoothType_list=None, extraSpectra=None):
        assert isinstance(sample_list, list) and isinstance(project_list, list)
        componentCount = len(sample_list)
        assert (len(project_list) == componentCount)
        if defaultParams_list is None: defaultParams_list = [{}]*componentCount
        if methodParams_list is None: methodParams_list = [{}]*componentCount
        if diffFrom_list is None: diffFrom_list = [None]*componentCount
        if norm_list is None: norm_list = [None] * componentCount
        if smoothType_list is None: smoothType_list = ['fdmnes'] * componentCount
        assert (len(defaultParams_list) == componentCount)
        assert (len(diffFrom_list) == componentCount)
        assert (len(norm_list) == componentCount)
        assert (len(methodParams_list) == componentCount)
        assert (len(smoothType_list) == componentCount)

        self.params = {}
        self.statusHTML = widgets.HTML(value='', placeholder='', description='')
        self.status = ''
        xanes_energy = [sample.energy for sample in sample_list]
        self.fig, self.ax = plt.subplots()
        for i in range(componentCount):
            diffFrom = diffFrom_list[i]
            if diffFrom is not None:
                diffFrom = copy.deepcopy(diffFrom)
                diffFrom['projectBase'].spectrum.intensity = np.interp(project_list[i].spectrum.energy, diffFrom['projectBase'].spectrum.energy, diffFrom['projectBase'].spectrum.intensity)
                diffFrom['projectBase'].spectrum.energy = project_list[i].spectrum.energy
                diffFrom['spectrumBase'].intensity = np.interp(xanes_energy, diffFrom['spectrumBase'].energy, diffFrom['spectrumBase'].intensity)
                diffFrom['spectrumBase'].energy = xanes_energy
                diffFrom_list[i] = diffFrom
        ax = self.ax

        def smoothAfterPrediction(geomArg, **params):
            self.spectrumSmoothed = [None]*componentCount
            self.spectrum = [None]*componentCount
            self.spectrumDiffSmoothed = [None]*componentCount
            self.spectrumDiff = [None]*componentCount
            absorbPredictionNormalized = [None]*componentCount
            smoothedPredictionNormalized = [None]*componentCount
            for i in range(componentCount):
                project = project_list[i]
                exp_e = project.spectrum.energy
                proj_name = project.name
                if proj_name+'_norm' in params: norm = params[proj_name+'_norm']
                else: norm = None
                shift = params[proj_name+'_shift']
                exp_xanes = project.spectrum.intensity
                # prediction
                absorbPrediction = estimator['nonSmoothed'][i].predict(geomArg[i])[0]
                # smoothing
                smooth_params = {}
                for p in ['shift', 'Efermi', 'Elarg', 'Ecent', 'Gamma_hole', 'Gamma_max']:
                    smooth_params[p] = params[proj_name+'_'+p]
                xanesPrediction = utils.Spectrum(xanes_energy[i], absorbPrediction)
                smoothedPredictionNormalized[i], _ = smoothLib.smoothInterpNorm(smooth_params, xanesPrediction, smoothType_list[i], project.spectrum, project.intervals['fit_norm'], norm)
                smoothedPredictionNormalized[i] = smoothedPredictionNormalized[i].intensity
                self.spectrumSmoothed[i] = utils.Spectrum(exp_e, smoothedPredictionNormalized[i], copy=True)
                absorbPredictionNormalized[i] = xanesPrediction.intensity / np.mean(xanesPrediction.intensity[-3:]) * np.mean(exp_xanes[-3:])
                self.spectrum[i] = utils.Spectrum(xanes_energy[i]+shift, absorbPredictionNormalized[i], copy=True)
                diffFrom = diffFrom_list[i]
                if diffFrom is not None:
                    smoothedXanesBaseNormalized = smoothLib.smoothInterpNorm(smooth_params, diffFrom['spectrumBase'], smoothType_list[i], diffFrom['projectBase'].spectrum, diffFrom['projectBase'].intervals['fit_norm'], norm)[0]
                    smoothedXanesBaseNormalized = smoothedXanesBaseNormalized.intensity
                    smoothedPredictionNormalized[i] = (smoothedPredictionNormalized[i] - smoothedXanesBaseNormalized)*params[proj_name+'_purity']
                    self.spectrumDiffSmoothed[i] = utils.Spectrum(exp_e, smoothedPredictionNormalized[i], copy=True)
                    absorbBaseNormalized = diffFrom['spectrumBase'].intensity / np.mean(diffFrom['spectrumBase'].intensity[-3:]) * np.mean(diffFrom['projectBase'].spectrum.intensity[-3:])
                    absorbPredictionNormalized[i] = (absorbPredictionNormalized[i] - absorbBaseNormalized)*params[proj_name+'_purity']
                    self.spectrumDiff[i] = utils.Spectrum(xanes_energy[i]+shift, absorbPredictionNormalized[i], copy=True)
            return absorbPredictionNormalized, smoothedPredictionNormalized

        def smoothBeforePrediction(geomArg, **params):
            self.spectrumSmoothed = [None] * componentCount
            self.spectrum = [None] * componentCount
            self.spectrumDiffSmoothed = [None] * componentCount
            self.spectrumDiff = [None] * componentCount
            absorbPredictionNormalized = [None] * componentCount
            smoothedPredictionNormalized = [None] * componentCount
            for i in range(componentCount):
                project = project_list[i]
                exp_e = project.spectrum.energy
                proj_name = project.name
                if proj_name + '_norm' in params:
                    norm = params[proj_name + '_norm']
                else:
                    norm = None
                shift = params[proj_name + '_shift']
                sample = sample_list[i]
                smooth_params = {}
                for p in ['shift', 'Efermi', 'Elarg', 'Ecent', 'Gamma_hole', 'Gamma_max']:
                    smooth_params[p] = params[proj_name + '_' + p]
                xanes_sm = smoothLib.smoothDataFrame(smooth_params, sample.spectra, smoothType_list[i], project.spectrum, project.intervals['fit_smooth'], norm=norm, folder=sample.folder)
                diffFrom = diffFrom_list[i]
                if diffFrom is not None:
                    smoothedXanesBase = smoothLib.smoothInterpNorm(params, diffFrom['spectrumBase'], smoothType_list[i], diffFrom['projectBase'].spectrum, diffFrom['projectBase'].intervals['fit_norm'], norm)[0]
                    smoothedXanesBase = smoothedXanesBase.intensity
                    xanes_sm = (xanes_sm - smoothedXanesBase) * params['purity']
                estimator['smoothed'][i].fit(sample_list[i].params.values, xanes_sm.values)
                smoothedPredictionNormalized[i] = estimator['smoothed'][i].predict(geomArg[i])[0]
                absorbPredictionNormalized[i] = np.zeros(xanes_energy[i].size)
                if diffFrom is None:
                    self.spectrum[i] = utils.Spectrum(xanes_energy[i]+shift, absorbPredictionNormalized[i], copy=True)
                    self.spectrumSmoothed[i] = utils.Spectrum(exp_e, smoothedPredictionNormalized[i], copy=True)
                else:
                    self.spectrum[i] = utils.Spectrum(xanes_energy[i]+shift, np.zeros(xanes_energy[i].size), copy=True)
                    self.spectrumSmoothed[i] = utils.Spectrum(exp_e, np.zeros(exp_e.size), copy=True)
                    self.spectrumDiff[i] = utils.Spectrum(xanes_energy[i]+shift, absorbPredictionNormalized[i], copy=True)
                    self.spectrumDiffSmoothed[i] = utils.Spectrum(exp_e, smoothedPredictionNormalized[i], copy=True)
            gc.collect()
            return absorbPredictionNormalized, smoothedPredictionNormalized

        def smoothExtra(extraSpectra, **params):
            extraSpectraSmoothed = []
            project = project_list[0]
            proj_name = project.name
            if extraSpectra is not None:
                for sp in extraSpectra:
                    smoothType = sp['smoothType']
                    assert smoothType in ['fdmnes', 'None', 'adf']
                    norm = None
                    if 'smoothParams' in sp:
                        smoothParams = sp['smoothParams']
                        if 'norm' in smoothParams:
                            norm = smoothParams['norm']
                    else:
                        smoothParams = {}
                        for p in ['shift', 'Efermi', 'Elarg', 'Ecent', 'Gamma_hole', 'Gamma_max']:
                            smoothParams[p] = params[proj_name + '_' + p]
                        if proj_name + '_norm' in params:
                            norm = params[proj_name + '_norm']
                    sp_sm = copy.deepcopy(sp)
                    if smoothType != 'None':
                        sp_sm['smoothed spectrum'] = smoothLib.smoothInterpNorm(smoothParams, sp['spectrum'], smoothType, project.spectrum, project.intervals['fit_norm'], norm)[0]
                    else:
                        sp_sm['smoothed spectrum'] = copy.deepcopy(sp_sm['spectrum'])
                    extraSpectraSmoothed.append(sp_sm)
            return extraSpectraSmoothed

        def plotXanes(**params):
            try:
                with warnings.catch_warnings(record=True) as warn:
                    for pName in params: self.params[pName] = params[pName]
                    geomArg = [None]*componentCount
                    for i in range(componentCount):
                        project = project_list[i]
                        proj_name = project.name
                        geomArg[i] = np.array([params[proj_name+'_'+pName] for pName in sample_list[i].params.columns]).reshape([1,sample_list[i].params.shape[1]])
                        params_i = {pName:params[proj_name+'_'+pName] for pName in sample_list[i].params.columns}
                        setattr(self, 'params'+str(i), params_i)
                    if params['smooth before prediction']:
                        absorbPredictionNormalized, smoothedPredictionNormalized = smoothBeforePrediction(geomArg, **params)
                    else:
                        absorbPredictionNormalized, smoothedPredictionNormalized = smoothAfterPrediction(geomArg, **params)
                    extraSpectraSmoothed = smoothExtra(extraSpectra, **params)

                    ax.clear()
                    self.spectrumExp = [None]*componentCount
                    self.spectrumSmoothedDiff = [None]*componentCount
                    self.spectrumExpDiff = [None]*componentCount
                    self.spectrumDiff = [None]*componentCount
                    conc = np.array([params[p.name] for p in project_list])
                    conc /= np.sum(conc)
                    for i in range(componentCount):
                        project = project_list[i]
                        proj_name = project.name
                        shift = params[proj_name+'_shift']
                        self.spectrumDiff[i] = utils.Spectrum(xanes_energy[i]+shift, absorbPredictionNormalized[i], copy=True)
                        # plotting
                        exp_xanes = project.spectrum.intensity
                        exp_e = project.spectrum.energy
                        self.spectrumExp[i] = utils.Spectrum(exp_e, exp_xanes, copy=True)
                        self.spectrumSmoothedDiff[i] = utils.Spectrum(exp_e, smoothedPredictionNormalized[i], copy=True)
                        diffFrom = diffFrom_list[i]
                        if diffFrom is not None:
                            expXanesBase = np.interp(exp_e, diffFrom['projectBase'].spectrum.energy, diffFrom['projectBase'].spectrum.intensity)
                            exp_xanes = exp_xanes - expXanesBase
                        self.spectrumExpDiff[i] = utils.Spectrum(exp_e, exp_xanes, copy=True)
                        if i == 0:
                            ax.plot(self.spectrumExpDiff[i].energy, self.spectrumExpDiff[0].intensity, c='k',
                                    label="Experiment")
                            font = FontProperties()
                            font.set_weight('black')
                            font.set_size(20)
                            txt = ax.text(project.intervals['fit_norm'][0], ax.get_ylim()[0], '[', color='green',
                                          verticalalignment='bottom', fontproperties=font)
                            txt.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
                            txt = ax.text(project.intervals['fit_norm'][1], ax.get_ylim()[0], ']', color='green',
                                          verticalalignment='bottom', fontproperties=font)
                            txt.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))

                    if params['not convoluted']:
                        en = self.spectrumDiff[0].energy
                        mixIntensities = np.array([np.interp(en, self.spectrumDiff[i].energy, self.spectrumDiff[i].intensity * conc[i]) for i in range(componentCount)])
                        mixIntensity = np.sum(mixIntensities, axis=0)
                        ax.plot(en, mixIntensity, label='mix_init', color='orange')

                    mixIntensities = [self.spectrumSmoothedDiff[i].intensity * conc[i] for i in range(componentCount)]
                    mixIntensity = [sum(i) for i in zip(*mixIntensities)]
                    ax.plot(self.spectrumSmoothedDiff[1].energy, mixIntensity, label='mix_conv', color='blue')

                    for sp in extraSpectraSmoothed:
                        en = sp['smoothed spectrum'].energy
                        ax.plot(en, sp['smoothed spectrum'].intensity, label=sp['label'])
                    ax.set_xlim([params['energyRange'][0], params['energyRange'][1]])
                    if diffFrom is None: ax.set_ylim([0, np.max(exp_xanes)*1.2])
                    else:
                        d = np.max(exp_xanes)-np.min(exp_xanes)
                        ax.set_ylim([np.min(exp_xanes)-0.2*d, np.max(exp_xanes)+0.2*d])
                    ax.set_xlabel("Energy")
                    ax.set_ylabel("Intensity")
                    ax.legend(loc='upper right')
                    ind = (exp_e >= params['energyRange'][0]) & (exp_e <= params['energyRange'][1])

                    exp_e0 = project_list[0].spectrum.energy
                    exp_xanes0 = project_list[0].spectrum.intensity
                    rFactor = utils.integral(exp_e[ind], (exp_xanes0[ind]-np.array(mixIntensity)[ind])**2) / utils.integral(exp_e0[ind], exp_xanes0[ind]**2)
                    ax.text(0.98, 0.02, ('c = ['+('{:.2f},'*componentCount)+'] R-factor = {:.4f}').format(*(conc.tolist()), rFactor), transform=ax.transAxes, horizontalalignment='right')
                    # ax.text(0.98, 0.02, ('c = [' + ('{:.2f},' * componentCount)[:-1] + ']').format(*(conc.tolist())), transform=ax.transAxes, horizontalalignment='right')
                    gc.collect()
                    if len(warn)>0:
                        status = 'Warning: '
                        for w in warn: status += str(w.message)+'\n'
                        self.setStatus(status)
            except Exception as exc:
                self.setStatus(traceback.format_exc())
            except Warning:
                self.setStatus(traceback.format_exc())

        controls = []
        o = 'horizontal' # 'vertical'
        for i in range(componentCount):
            project = project_list[i]
            proj_name = project.name
            controls.append(widgets.FloatSlider(description=proj_name, min=0.01, max=1, step=0.03, value=0.5, orientation=o, continuous_update=False))
        for i in range(componentCount):
            project = project_list[i]
            proj_name = project.name
            for pName in project.geometryParamRanges:
                p0 = project.geometryParamRanges[pName][0]; p1 = project.geometryParamRanges[pName][1]
                controls.append(widgets.FloatSlider(description=proj_name+'_'+pName, min=p0, max=p1, step=(p1-p0)/30, value=(p0+p1)/2, orientation=o, continuous_update=False))
        for i in range(componentCount):
            project = project_list[i]
            proj_name = project.name
            defaultParams = defaultParams_list[i]
            if 'shift' in defaultParams: shift = defaultParams['shift']
            else: shift = optimize.value(project.defaultSmoothParams['fdmnes'], 'shift')
            controls.append(widgets.FloatSlider(description=proj_name+'_'+'shift', min=shift-10.0, max=shift+10.0, step=0.3, value=shift, orientation=o, continuous_update=False))
            controls.append(widgets.FloatSlider(description=proj_name+'_'+'Gamma_hole', min=0.1, max=10, step=0.2, value=1, orientation=o, continuous_update=False))
            controls.append(widgets.FloatSlider(description=proj_name+'_'+'Ecent', min=1, max=100, step=1, value=50, orientation=o, continuous_update=False))
            controls.append(widgets.FloatSlider(description=proj_name+'_'+'Elarg', min=1, max=100, step=1, value=50, orientation=o, continuous_update=False))
            controls.append(widgets.FloatSlider(description=proj_name+'_'+'Gamma_max', min=1, max=100, step=1, value=15, orientation=o, continuous_update=False))
            if fdmnes.useEpsiiShift:
                controls.append(widgets.FloatSlider(description=proj_name+'_'+'Efermi', min=np.min(project.spectrum.energy), max=np.max(project.spectrum.energy), step=1, value=project.defaultSmoothParams['fdmnes']['Efermi'], orientation=o, continuous_update=False))
            else:
                controls.append(widgets.FloatSlider(description=proj_name+'_'+'Efermi', min=-20, max=20, step=1, value=project.defaultSmoothParams['fdmnes']['Efermi'], orientation=o, continuous_update=False))
            if 'norm' in defaultParams:
                nrm = defaultParams['norm']
                controls.append(widgets.FloatSlider(description=proj_name+'_'+'norm', min=nrm/2, max=nrm*2, step=nrm/20, value=nrm, orientation=o, continuous_update=False))
            if diffFrom is not None:
                controls.append(widgets.FloatSlider(description=proj_name+'_'+'purity', min=0, max=1, step=0.05, value=diffFrom['purity'], orientation=o, continuous_update=False))

        def changedCheckBoxNotConvoluted(p):
            if p['name'] == '_property_lock': return
            self.params['not convoluted'] = p['new']
            plotXanes(**self.params)
        controls.append(widgets.Checkbox(description='not convoluted', value=True)); controls[-1].observe(changedCheckBoxNotConvoluted)

        def changedCheckBoxSmoothBeforePrediction(p):
            if p['name'] == '_property_lock': return
            self.params['smooth before prediction'] = p['new']
            plotXanes(**self.params)
        controls.append(widgets.Checkbox(description='smooth before prediction', value=False)); controls[-1].observe(changedCheckBoxSmoothBeforePrediction)

        estimator = {'smoothed':None, 'nonSmoothed':None}
        allMethods = inverseMethod.allowedMethods
        def changedDropdownMethod(p, doNotPlot = False):
            if p['name'] == '_property_lock': return
            if not isinstance(p['new'], str): return
            method = p['new']
            estimator['nonSmoothed'] = [None]*componentCount
            estimator['smoothed'] = [None]*componentCount
            for i in range(componentCount):
                estimator['nonSmoothed'][i] = inverseMethod.getMethod(method, methodParams_list[i])
                estimator['nonSmoothed'][i] = ML.Normalize(estimator['nonSmoothed'][i], xOnly=True)
                estimator['smoothed'][i] = inverseMethod.getMethod(method, methodParams_list[i])
                estimator['smoothed'][i] = ML.Normalize(estimator['smoothed'][i], xOnly=True)
                estimator['nonSmoothed'][i].fit(sample_list[i].params.values, sample_list[i].spectra.values)
            if not doNotPlot: plotXanes(**self.params)
        method = 'Extra Trees'
        controls.append(widgets.Dropdown(options=allMethods, value=method, description='method', disabled=False))
        controls[-1].observe(changedDropdownMethod)
        changedDropdownMethod({'new':controls[-1].value, 'name':None}, doNotPlot=True)

        e0 = project.spectrum.energy[0]-10; e1 = project.spectrum.energy[-1]+10
        v0 = project.intervals['fit_geometry'][0]; v1 = project.intervals['fit_geometry'][1]
        controls.append(widgets.FloatRangeSlider(description='energyRange', min=e0,max=e1,step=(e1-e0)/30,value=[v0,v1], orientation='horizontal', continuous_update=False))
        for i in range(componentCount):
            project = project_list[i]
            proj_name = project.name
            defaultExpSmooth = project.defaultSmoothParams.getDict(smoothType_list[i])
            for c in controls:
                paramName = c.description
                j = paramName.find(proj_name)
                if j<0: continue
                paramName = paramName[j+len(proj_name)+1:]
                if paramName in defaultParams_list[i]:
                    c.value = defaultParams_list[i][paramName]
                elif paramName in defaultExpSmooth:
                    c.value = defaultExpSmooth[paramName]
                if hasattr(c, 'value'): self.params[c.description] = c.value
        ui = widgets.HBox(tuple(controls)+(self.statusHTML,))
        ui.layout.flex_flow = 'row wrap'
        ui.layout.justify_content = 'space-between'
        ui.layout.align_items = 'flex-start'
        ui.layout.align_content = 'flex-start'
        # outputWithFigure.layout.
        controlsDict = {}
        for c in controls: controlsDict[c.description] = c
        out = widgets.interactive_output(plotXanes, controlsDict)
        # out.layout.min_height = '400px'
        display(ui, out)
        display(Javascript('$(this.element).addClass("fitBySlidersOutput");'))



class fitBySlidersExafs:
    def __init__(self, sample, project, defaultParams={}):
        self.params = {}
        fitBySlidersExafsParams = self.params
        isML = isinstance(sample, ML.Sample)
        if isML:
            xanes = sample.spectra
            geometryParams = sample.params
            # normalize params
            geometryParamsMin = np.min(geometryParams.values, axis=0)
            geometryParamsMax = np.max(geometryParams.values, axis=0)
            geometryParams = 2*(geometryParams-geometryParamsMin)/(geometryParamsMax-geometryParamsMin) - 1
            # machine learning estimator training
            # ML.makeQuadric(RidgeCV(alphas=[0.01,0.1,1,10,100])) ExtraTreesRegressor(n_estimators = 200)
            estimator = inverseMethod.getMethod("Extra Trees") #ExtraTreesRegressor(n_estimators = 200, random_state=0)
            estimator.fit(geometryParams.values, xanes.values)
            e_names = xanes.columns
            k0 = np.array([float(e_names[i][2:]) for i in range(e_names.size)])
        else:
            exafs = sample
            k0 = exafs.k
        self.fig, axarray = plt.subplots(nrows=2)
        ax = axarray[0]
        ax1 = axarray[1]
        def relative_to_constant_prediction_error(n): # n - partitions count via cross-validation
            def partition(x, i, size):
                return np.vstack((x[0:i * size], x[(i + 1) * size:])), x[i*size:(i+1)*size]
            if isML:
                geom_params = sample.params.values
                # normalizing params
                geom_params_min = np.min(geom_params, axis=0)
                geom_params_max = np.max(geom_params, axis=0)
                geom_params = 2 * (geom_params - geom_params_min) / (geom_params_max - geom_params_min) - 1
                a, b = project.intervals['fit_exafs']
                ind = (sample.energy >= a) & (sample.energy <= b)
                sample_spectra = sample.spectra.values * sample.energy ** 2
                sample_count = sample_spectra.shape[0]
                if 2 * n > sample_count:
                    return None
                part_size = sample_count // n
                final_prediction = np.zeros((2, sample_spectra.shape[1]))
                for i in range(n):
                    # machine learning estimator training
                    estimator1 = ExtraTreesRegressor(n_estimators = 200, random_state=0)
                    training_params, test_params = partition(geom_params, i, part_size)
                    training_spectra, _ = partition(sample_spectra, i, part_size)
                    estimator1.fit(training_params, training_spectra)
                    final_prediction = np.vstack((final_prediction, estimator1.predict(test_params)))
                final_prediction = final_prediction[2:]
                k = project.exafs.k
                u = np.mean(np.mean((sample_spectra[:part_size * n, ind] - final_prediction[:,ind]) ** 2, axis=1))
                mean_sample_spectra = np.mean(sample_spectra, axis=0)
                v = np.mean(np.mean((sample_spectra[:part_size*n,ind] - mean_sample_spectra[ind]) ** 2, axis=1))
                return u / v
            else:
                return None
        def plotXanes(**params):
            for pName in params: fitBySlidersExafsParams[pName] = params[pName]
            if isML:
                geomArg = np.array([params[pName] for pName in geometryParams.columns]).reshape([1,geometryParams.shape[1]])
                geomArg = 2*(geomArg-geometryParamsMin)/(geometryParamsMax-geometryParamsMin) - 1
            p = params['Power of k']
            exp_k0 = project.exafs.k
            exp_k = np.linspace(exp_k0[0], exp_k0[-1], exp_k0.size)
            exp_xanes = np.interp(exp_k, exp_k0, project.exafs.chi)
            exp_xanes = exp_xanes * exp_k ** p
            if isML:
                PredictedSpectr = estimator.predict(geomArg)[0]
            else:
                PredictedSpectr = np.copy(exafs.chi)
            dE = params['dE']; me = 2*9.11e-31; h = 4.1e-15
            tmp = k0**2 - me*dE/h**2; tmp[tmp<0] = 0
            k = np.sqrt(tmp)
            PredictedSpectr *= k**p*np.exp(-2*k**2*params['sigma^2'])
            S02 = params['S0^2']
            PredictedSpectrFitted = np.interp(exp_k, k, PredictedSpectr)*S02
            self.exafs = PredictedSpectrFitted
            #plotting
            ax.clear();
            ax.plot(exp_k, PredictedSpectrFitted, label='Approximation', color='blue')
            ax.plot(exp_k, exp_xanes, c='k', label="Experiment")
            ax.set_xlim([params['k_range'][0], params['k_range'][1]])
            #ax.set_ylim([0, np.max(exp_xanes)*1.2])
            ax.set_xlabel("k")
            ax.set_ylabel("chi")
            ax.legend(loc='upper right')
            ind = (exp_k >= params['k_range'][0]) & (exp_k <= params['k_range'][1])
            rFactor = utils.integral(exp_k[ind], (exp_xanes[ind] - PredictedSpectrFitted[ind])**2) / utils.integral(exp_k[ind], exp_xanes[ind]**2)
            ax.text(0.98, 0.02, 'R-factor = %.4f' % rFactor, transform=ax.transAxes, horizontalalignment='right')
            # plotting fourier transform
            def fourier_transform(k, chi, kmin, kmax, A):
                w = np.ones(k.shape)
                i = k < kmin + A
                w[i] = 0.5 * (1 - np.cos(np.pi * (k[i] - kmin) / A))
                w[k < kmin] = 0
                i = k > kmax - A
                w[i] = 0.5 * (1 - np.cos(np.pi * (k[i] - kmax + A) / A))
                w[k > kmax] = 0
                M = k.size
                delta = (k[-1] - k[0]) / M
                m = np.arange(0, M // 2)
                wm = 2 * np.pi * m / M / delta
                #print('chi: ' + str(chi.size) + '\n')
                ft = delta * np.exp(complex(0, 1) * wm * k[0]) * np.fft.fft(chi * w)[:M // 2]
                rbfi = Rbf(wm, np.abs(ft))  # radial basis function interpolator instance
                freqw = np.linspace(wm[0], wm[-1], 500)
                rbfft = rbfi(freqw)   # interpolated values
                return freqw, rbfft, wm, ft

                #freqw = np.linspace(wm[0], wm[-1], 500)
                #f = interp1d(wm, np.abs(ft), kind='linear')
                #print(wm.shape, ft.shape, freqw.shape, f(freqw).shape)
                #return freqw, f(freqw), wm, ft

                #freqw  = np.linspace(wm[0], wm[-1], 500)
                #cs = CubicSpline(wm, ft)
                #return freqw, cs(freqw), wm, ft
                #return wm, ft
            R_exp, ft_exp, R_exp_no_interp, ft_exp_no_interp = fourier_transform(exp_k, exp_xanes, params['k_range'][0], params['k_range'][1], params['A'])
            R_pr, ft_pr, _, _ = fourier_transform(exp_k, PredictedSpectrFitted, params['k_range'][0], params['k_range'][1], params['A'])
            ax1.clear()
            ax1.plot(R_pr, np.abs(ft_pr), label='Approximation FT', color='blue')
            ax1.plot(R_exp, np.abs(ft_exp), c='k', label="Experiment FT")
            ax1.plot(R_exp_no_interp, np.abs(ft_exp_no_interp), 'o', label='Approximation FT', color='red')
            ax1.legend()
            ax1.set_xlim(0, 10)
        controls = []
        o = 'horizontal' # 'vertical'
        if isML:
            for pName in project.geometryParamRanges:
                p0 = project.geometryParamRanges[pName][0]; p1 = project.geometryParamRanges[pName][1]
                controls.append(widgets.FloatSlider(description=pName, min=p0, max=p1, step=(p1-p0)/30, value=(p0+p1)/2, orientation=o, continuous_update=False))
        e0 = project.exafs.k[0]; e1 = project.exafs.k[-1]
        e0 = max(e0, k0[0]); e1 = min(e1, k0[-1])
        v0 = project.intervals['fit_exafs'][0]; v1 = project.intervals['fit_exafs'][1]
        v0 = max(e0,v0); v1 = min(e1,v1)
        controls.append(widgets.FloatRangeSlider(description='k_range', min=e0, max=e1, step=(e1-e0)/30, value=[v0,v1], orientation=o, continuous_update=False))
        controls.append(widgets.IntSlider(value=2, min=0, max=5, step=1, description='Power of k',continuous_update=False,orientation=o))
        controls.append(widgets.FloatSlider(description='sigma^2', min=0, max=0.01, step=2e-4, value=0, orientation=o, continuous_update=False, readout_format='.4f'))
        controls.append(widgets.FloatSlider(description='S0^2', min=0, max=2, step=0.03, value=1, orientation=o, continuous_update=False))
        controls.append(widgets.FloatSlider(description='dE', min=-40, max=40, step=1, value=0, orientation=o, continuous_update=False))
        controls.append(widgets.FloatSlider(description='A', min=0, max=4, step=0.1, value=1, orientation=o, continuous_update=False))
        for c in controls:
            if c.description in defaultParams: c.value = defaultParams[c.description]
            fitBySlidersExafsParams[c.description] = c.value
        ui = widgets.HBox(tuple(controls))
        ui.layout.flex_flow = 'row wrap'
        ui.layout.justify_content = 'space-between'
        ui.layout.align_items = 'flex-start'
        ui.layout.align_content = 'flex-start'
        # outputWithFigure.layout.
        controlsDict = {}
        for c in controls: controlsDict[c.description] = c
        out = widgets.interactive_output(plotXanes, controlsDict)
        # out.layout.min_height = '400px'
        display(ui, out)
        display(Javascript('$(this.element).addClass("fitBySlidersExafsOutput");'))


class fitSmooth:
    def setStatus(self, s):
        self.status = s
        self.statusHTML.value = '<div class="status">'+s+'</div>'

    def addToStatus(self, s):
        self.status += s
        self.setStatus(self.status)

    def __init__(self, spectrum, project0, defaultParams={}, diffFrom=None, norm=None, smoothType='fdmnes', extraSpectra=None):
        self.params = {}
        self.statusHTML = widgets.HTML(value='', placeholder='', description='')
        self.status = ''

        self.project = copy.deepcopy(project0)
        project = self.project
        fitBySlidersParams = self.params
        xanes_energy = spectrum.energy
        self.fig, ax = plt.subplots(); self.ax2 = None
        exp_e = project.spectrum.energy
        exp_xanes = project.spectrum.intensity

        def smooth(extraSpectra, **params):
            if 'norm' in params: norm = params['norm']
            else: norm = None
            xanes_sm, norm1 = smoothLib.smoothInterpNorm(params, spectrum, smoothType, project.spectrum, project.intervals['fit_norm'], norm)
            xanes_sm = xanes_sm.intensity
            absorbPredictionNormalized = spectrum.intensity / np.mean(spectrum.intensity[-3:]) * np.mean(exp_xanes[-3:])
            extraSpectraSmoothed = {}
            if extraSpectra is not None:
                for sp in extraSpectra:
                    extraSpectraSmoothed[sp] = smoothLib.smoothInterpNorm(params, extraSpectra[sp], smoothType, project.spectrum, project.intervals['fit_norm'], norm)[0]
            if diffFrom is not None:
                smoothedXanesBase, _ = smoothLib.smoothInterpNorm(params, diffFrom['xanesBase'], smoothType, diffFrom['projectBase'].spectrum, project.intervals['fit_norm'], norm)
                smoothedXanesBase = smoothedXanesBase.intensity
                xanes_sm = (xanes_sm - smoothedXanesBase) * params['purity']
                absorbBaseNormalized,_ = diffFrom['xanesBase'].intensity  / np.mean(diffFrom['xanesBase'].intensity) * np.mean(exp_xanes[-3:])
                absorbPredictionNormalized = (absorbPredictionNormalized - absorbBaseNormalized)*params['purity']
            return absorbPredictionNormalized, xanes_sm, extraSpectraSmoothed, norm1

        def adjustFitIntervals(shift):
            e = np.sort(np.hstack((xanes_energy+shift, project.spectrum.energy)).flatten())
            for fiName in ['plot', 'fit_norm', 'fit_smooth']:
                fi0 = project0.intervals[fiName]
                fi = project.intervals[fiName]
                if fi0[0] < e[0]:
                    if fiName == 'plot': fi[0] = e[0]-10
                    else: fi[0] = e[0]
                if fi0[0] > e[-1]: # extraordinary case
                    fi[0] = e[0]
                if fi0[1] > e[-1]:
                    if fiName == 'plot': fi[1] = e[-1]+10
                    else: fi[1] = e[-1]
                if fi0[1] < e[0]: # extraordinary case
                    fi[1] = e[-1]
                # if fiName == 'plot': print('plot int:', fi)
                # assert fi[0]<fi[1]

        def plotXanes(**params):
            # self.setStatus('')
            try:
                with warnings.catch_warnings(record=True) as warn:
                    for pName in params: fitBySlidersParams[pName] = params[pName]
                    for fi in ['plot', 'fit_norm', 'fit_smooth']:
                        project.intervals[fi] = list(params['energyRange'])
                    shift = params['shift']
                    adjustFitIntervals(shift)
                    absorbPredictionNormalized, smoothedPredictionNormalized, extraSpectraSmoothed, norm1 = smooth(extraSpectra, **params)
                    self.spectrum = utils.Spectrum(spectrum.energy+shift, absorbPredictionNormalized, copy=True)
                    #plotting
                    exp_xanes = project.spectrum.intensity
                    if diffFrom is not None:
                        expXanesBase = np.interp(exp_e, diffFrom['projectBase'].spectrum.energy, diffFrom['projectBase'].spectrum.intensity)
                        exp_xanes = exp_xanes - expXanesBase
                    self.spectrumSmoothed = utils.Spectrum(exp_e, smoothedPredictionNormalized, copy=True)
                    self.spectrumExp = utils.Spectrum(exp_e, exp_xanes, copy=True)
                    ax.clear()
                    if self.ax2 is not None: self.ax2.clear()
                    if params['not convoluted']:
                        ax.plot(self.spectrum.energy, self.spectrum.intensity, label='initial', color='orange')
                    ax.plot(self.spectrumSmoothed.energy, self.spectrumSmoothed.intensity, label='convolution', color='blue')
                    ax.plot(self.spectrumExp.energy, self.spectrumExp.intensity, c='k', label="Experiment")
                    for sp in extraSpectraSmoothed:
                        ax.plot(exp_e, extraSpectraSmoothed[sp].intensity, label=sp)
                    if params['smooth width']:
                        if self.ax2 is None: self.ax2 = ax.twinx()
                        if smoothType == 'fdmnes':
                            Efermi = params['Efermi']
                            if not fdmnes.useEpsiiShift: Efermi += shift
                            smoothWidth = smoothLib.YvesWidth(exp_e, params['Gamma_hole'], params['Ecent'], params['Elarg'], params['Gamma_max'], Efermi)
                        elif smoothType == 'adf':
                            smoothWidth = smoothLib.YvesWidth(exp_e-exp_e[0], params['Gamma_hole'], params['Ecent'], params['Elarg'], params['Gamma_max'], params['Efermi'])
                        else: assert False, 'Unknown width'
                        e_smoothWidth = exp_e # -exp_e[0]+spectrum.energy[0]+shift if smoothType == 'adf' else exp_e
                        self.ax2.plot(e_smoothWidth, smoothWidth, c='r', label='Smooth width')
                        self.ax2.legend()
                    else:
                        if self.ax2 is not None: self.ax2.remove(); self.ax2 = None
                    i = 1
                    ymin = params['ylim'][0]; ymax = params['ylim'][1]
                    ax.set_xlim([params['energyRange'][0], params['energyRange'][1]])
                    ax.set_ylim([ymin, ymax])
                    font = FontProperties(); font.set_weight('black'); font.set_size(20)
                    colors = {'plot':'green', 'fit_norm':'blue', 'fit_smooth':'red'}
                    ddd = 0
                    for fi in ['plot', 'fit_norm', 'fit_smooth']:
                        txt = ax.text(project.intervals[fi][0], ax.get_ylim()[0]+ddd, '[', color=colors[fi], verticalalignment='bottom', fontproperties=font)
                        txt = ax.text(project.intervals[fi][1], ax.get_ylim()[0]+ddd, ']', color=colors[fi], verticalalignment='bottom', fontproperties=font)
                        ddd += 0.05
                        # txt.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=10))
                    ax.set_xlabel("Energy")
                    ax.set_ylabel("Intensity")
                    ax.legend(loc='upper right')
                    ind = (exp_e >= params['energyRange'][0]) & (exp_e <= params['energyRange'][1])
                    denom = utils.integral(exp_e[ind], exp_xanes[ind]**2)
                    if denom != 0: rFactor = utils.integral(exp_e[ind], (exp_xanes[ind] - smoothedPredictionNormalized[ind])**2)
                    else: rFactor = 0
                    ax.text(0.98, 0.02, 'norm = {:.4g} R-factor = {:.4f}'.format(norm1, rFactor), transform=ax.transAxes, horizontalalignment='right')
                    if len(warn)>0:
                        status = 'Warning: '
                        for w in warn: status += str(w.message)+'\n'
                        self.setStatus(status)
            except Exception as exc:
                self.setStatus(traceback.format_exc())
            except Warning:
                self.setStatus(traceback.format_exc())

        def startAutoFit(arg):
            project1 = copy.deepcopy(project)
            smoothParamNames = ['Gamma_hole', 'Ecent', 'Elarg', 'Efermi', 'Gamma_max', 'shift']
            commonParams0 = {}
            for pName in smoothParamNames:
                optimize.setValue(project1.defaultSmoothParams[smoothType], pName, fitBySlidersParams[pName])
                commonParams0[pName] = fitBySlidersParams[pName]
            optimParams = smoothLib.fitSmooth([project1], [spectrum], smoothType = smoothType, fixParamNames=[], commonParams0=commonParams0, targetFunc='l2(max)', plotTrace=False, minimizeMethodType='random', useGridSearch=False, useRefinement=False, optimizeWithoutPlot=True)
            for c in self.fitSmoothControls:
                if c.description in smoothParamNames:
                    c.value = optimize.value(optimParams, c.description)
                    fitBySlidersParams[c.description] = c.value
            plotXanes(**fitBySlidersParams)

        controls = []
        o = 'horizontal' # 'vertical'
        if 'shift' in defaultParams: shift = defaultParams['shift']
        else: shift = optimize.value(project.defaultSmoothParams[smoothType], 'shift')
        adjustFitIntervals(shift)
        message, exp_efermi, theory_efermi = smoothLib.checkShift(project.spectrum, spectrum, shift, smoothType)
        if message == '': minShift = shift-10.0; maxShift = shift+10.0
        else:
            self.setStatus(message)
            minShift = np.min(project.spectrum.energy) - np.max(spectrum.energy)
            maxShift = np.max(project.spectrum.energy) - np.min(spectrum.energy)
        controls.append(widgets.FloatSlider(description='shift', min=minShift, max=maxShift, step=1, value=shift, orientation=o, continuous_update=False))
        controls.append(widgets.FloatSlider(description='Gamma_hole', min=0.1, max=10, step=0.2, value=1, orientation=o, continuous_update=False))
        controls.append(widgets.FloatSlider(description='Ecent', min=1, max=100, step=1, value=50, orientation=o, continuous_update=False))
        controls.append(widgets.FloatSlider(description='Elarg', min=1, max=100, step=1, value=50, orientation=o, continuous_update=False))
        controls.append(widgets.FloatSlider(description='Gamma_max', min=1, max=100, step=1, value=15, orientation=o, continuous_update=False))
        if smoothType == 'adf':
            controls.append(widgets.FloatSlider(description='Efermi', min=-20, max=20, step=1, value=0, orientation=o, continuous_update=False))
        else:
            if fdmnes.useEpsiiShift:
                controls.append(widgets.FloatSlider(description='Efermi', min=np.min(project.spectrum.energy)-20, max=np.max(project.spectrum.energy), step=1, value=np.min(project.spectrum.energy), orientation=o, continuous_update=False))
            else:
                controls.append(widgets.FloatSlider(description='Efermi', min=-20, max=20, step=1, value=0, orientation=o, continuous_update=False))
        if 'norm' in defaultParams:
            nrm = defaultParams['norm']
            controls.append(widgets.FloatSlider(description='norm', min=nrm/2, max=nrm*2, step=nrm/20, value=nrm, orientation=o, continuous_update=False))
        if diffFrom is not None:
            controls.append(widgets.FloatSlider(description='purity', min=0, max=1, step=0.05, value=diffFrom['purity'], orientation=o, continuous_update=False))

        def changedCheckBoxSmoothWidth(p):
            if p['name'] == '_property_lock': return
            fitBySlidersParams['smooth width'] = p['new']
            plotXanes(**fitBySlidersParams)
        checkBoxSmoothWidth = widgets.Checkbox(description='smooth width', value=False)
        controls.append(checkBoxSmoothWidth)

        def changedCheckBoxNotConvoluted(p):
            if p['name'] == '_property_lock': return
            fitBySlidersParams['not convoluted'] = p['new']
            plotXanes(**fitBySlidersParams)
        checkBoxNotConvoluted = widgets.Checkbox(description='not convoluted', value=True)
        controls.append(checkBoxNotConvoluted)

        buttonAutoFit = widgets.Button(description="Start auto fit")
        buttonAutoFit.on_click(startAutoFit)
        controls.append(buttonAutoFit)

        e0 = project.spectrum.energy[0]-10; e1 = project.spectrum.energy[-1]+10
        v0 = project.intervals['plot'][0]
        v1 = project.intervals['plot'][1]
        # print(e0,e1,v0,v1,shift)
        if not(e0<=v0<=e1) or not(e0<=v1<=e1): # we have very bad shift
            e0 = min(e0,v0); v0 = e0
            e1 = max(e1,v1); v1 = e1
        controls.append(widgets.FloatRangeSlider(description='energyRange', min=e0, max=e1, step=(e1-e0)/30, value=[v0,v1], orientation='horizontal', continuous_update=False))
        if diffFrom is None: ylim0=0;  ylim1 = np.max(exp_xanes)*1.2
        else:
            d = np.max(exp_xanes)-np.min(exp_xanes)
            ylim0 = np.min(exp_xanes)-0.2*d; ylim1 = np.max(exp_xanes)+0.2*d
        xanes_tmp = spectrum.intensity / np.mean(spectrum.intensity[-3:]) * np.mean(exp_xanes[-3:])
        controls.append(widgets.FloatRangeSlider(description='ylim', min=min([ylim0,np.min(xanes_tmp)]), max=max([ylim1, np.max(xanes_tmp)]), step=(ylim1-ylim0)/10, value=[ylim0, ylim1], orientation='horizontal', continuous_update=False))

        defaultExpSmooth = project.defaultSmoothParams.getDict(smoothType)
        for c in controls:
            if c.description in defaultParams: c.value = defaultParams[c.description]
            elif c.description in defaultExpSmooth:
                if (c.description != 'Efermi') or (smoothType == 'adf'):
                    c.value = defaultExpSmooth[c.description]
            if hasattr(c, 'value'): fitBySlidersParams[c.description] = c.value

        # assign observers afterwards, because setting default parameters makes it run before all parameters are set
        checkBoxSmoothWidth.observe(changedCheckBoxSmoothWidth)
        checkBoxNotConvoluted.observe(changedCheckBoxNotConvoluted)

        ui = widgets.HBox(tuple(controls)+(self.statusHTML,))
        ui.layout.flex_flow = 'row wrap'
        ui.layout.justify_content = 'space-between'
        ui.layout.align_items = 'flex-start'
        ui.layout.align_content = 'flex-start'
        # outputWithFigure.layout.
        controlsDict = {}
        for c in controls:
            if hasattr(c, 'value'): controlsDict[c.description] = c
        out = widgets.interactive_output(plotXanes, controlsDict)
        # out.layout.min_height = '400px'
        display(ui, out)
        display(Javascript('$(this.element).addClass("fitBySlidersOutput");'))
        self.fitSmoothControls = controls

def plot_data(energy,data):
    _=plt.plot(energy,data)
    axes = plt.gca()
    axes.set_xlim([min(energy),max(energy)])
    plt.xlabel('Energy',fontweight='bold')
    plt.ylabel('Absorption',fontweight='bold')
    plt.title('Experimental Data')

def interpolation(energy,data,step):
    e_valor=np.arange(min(energy),max(energy),step)
    dat_valor=[]
    col=np.shape(data)[1]
    for i in range(col):
        d=np.interp(e_valor,energy,data[:,i])
        dat_valor.append(d)
    data=np.transpose(dat_valor)
    energy=e_valor
    return energy, data

def normalization(energy,data):
    scaled=np.zeros(np.shape(data)[1])
    for i in range(np.shape(data)[1]):
        scaled[i]=np.sqrt((1./((1./(np.max(energy)-np.min(energy)))*(np.trapz((data[:,i])**2,energy)))))
    for i in range(np.shape(data)[1]):
        data[:,i]=data[:,i]*scaled[i]
    return data

plotPCAcomponentsParams = {}
def plotPCAcomponents(energy, principal_components):
    global plotPCAcomponentsParams
    # Not Normalized Components
    n_col=np.shape(principal_components)[1]
    principal_components_v=copy.copy(principal_components)
    components=pd.DataFrame(principal_components)
    components.index=energy
    # Normalized Components
    val_N=pd.DataFrame(principal_components_v)
    val=np.transpose(val_N.values)
    for i in range(np.shape(val)[0]):
        val[i]=val[i]/(np.trapz(val[i],energy))
    dv=pd.DataFrame(np.transpose(val))
    dv.index=energy
    fig, ax = plt.subplots()
    plotPCAcomponentsParams["Switching"] = 'Not Normalized'
    plotPCAcomponentsParams["Component"] = 1

    def redraw():
        ax.clear()
        switching = plotPCAcomponentsParams["Switching"]
        valor=components if switching=='Not Normalized' else dv
        control = plotPCAcomponentsParams["Component"]
        for j in range(control):
            ax.plot(energy, valor.iloc[:,j],linewidth=2) ####
        plt.xlabel('Energy',fontweight='bold')
        plt.ylabel('Absorption',fontweight='bold')
        plt.title('Abstract Components')
        plt.xlim([min(energy),max(energy)])
        #plt.grid()
        filename = "N_abstract_components.dat"
        if switching=='Not Normalized': filename = 'Not_'+filename
        saveToFile('results'+os.sep+filename, valor.iloc[:,0:control])

    radio = widgets.RadioButtons( options=['Not Normalized', 'Normalized'], description='Switching:', disabled=False)
    def switch(change):
        global plotPCAcomponentsParams
        if change['type'] != 'change' or change['name'] != 'value': return
        plotPCAcomponentsParams["Switching"] = change['new']
        redraw()
    radio.observe(switch)

    componentWidget = widgets.BoundedIntText(value=1, min=1, max=n_col, step=1, description='Component:', disabled=False)
    def changeComp(change):
        global plotPCAcomponentsParams
        if change['type'] != 'change' or change['name'] != 'value': return
        plotPCAcomponentsParams["Component"] = change['new']
        redraw()
    componentWidget.observe(changeComp)

    uif = widgets.HBox((radio, componentWidget))
    uif.layout.flex_flow = 'row wrap'
    uif.layout.justify_content = 'space-between'
    uif.layout.align_items = 'flex-start'
    uif.layout.align_content = 'flex-start'
    display(uif)
    display(Javascript('$(this.element).addClass("pcaOutput");'))
    redraw()

PCAplot={}
def PCAcomparison(energy,data):
    global PCAplot
    n_col=np.shape(data)[1]
    u,s,vT=np.linalg.svd(data, full_matrices=False)
    PCAplot['column']=0
    PCAplot['number']=1
    fig, ax = plt.subplots()
    inset_ax = fig.add_axes([0.53, 0.25, 0.35, 0.2])
    def redrawf():
        ax.clear()
        inset_ax.clear()
        control1=PCAplot['column']
        control2=PCAplot['number']
        u_red = u[:,:control2]
        s_red = np.diag(s[:control2])
        v_red = vT[:control2,:]
        dataR=np.dot(np.dot(u_red,s_red),v_red)
        residuals=data-dataR
        ax.set_title('Experimental vs Reconstructed spectra')
        ax.set_xlabel('Energy',fontweight='bold')
        ax.set_ylabel('Absorption',fontweight='bold')
        ax.plot(energy,data[:,control1],linewidth=3,color='red',label='Exp.')
        ax.set_xlim([min(energy),max(energy)])
        ax.plot(energy,dataR[:,control1],linewidth=2,ls='--',color='black',label='Reconstructed')
        ax.legend()
        ax.legend()
        inset_ax.set_xlim([min(energy),max(energy)])
        inset_ax.axhline(0)
        inset_ax.plot(energy,residuals[:,control1])
        inset_ax.set_title('Residuals Plot')
    componentWidget1 = widgets.BoundedIntText(value=0, min=0, max=n_col-1, step=1, description='Spectrum:', disabled=False, continuous_update=True)
    def changeComp1(change1):
        global PCAplot
        if change1['type'] != 'change' or change1['name'] != 'value': return
        PCAplot["column"] = change1['new']
        redrawf()
    componentWidget1.observe(changeComp1)

    componentWidget2 = widgets.BoundedIntText(value=1, min=1, max=n_col, step=1, description='Component:', disabled=False, continuous_update=True)
    def changeComp2(change2):
        global PCAplot
        if change2['type'] != 'change' or change2['name'] != 'value': return
        PCAplot["number"] = change2['new']
        redrawf()
    componentWidget2.observe(changeComp2)
    uif = widgets.HBox((componentWidget1,componentWidget2))

    uif.layout.flex_flow = 'row wrap'
    uif.layout.justify_content = 'space-between'
    uif.layout.align_items = 'flex-start'
    uif.layout.align_content = 'flex-start'
    display(uif)
    display(Javascript('$(this.element).addClass("pcaOutput");'))
    redrawf()










def targetTransformationPCA_old(energy, data):
    n_row=np.shape(data)[0]
    n_col=np.shape(data)[1]
    def q(NumFactors):
        lisT=np.zeros(NumFactors)
        for i in range(NumFactors-1):
            lisT[i]=(0.5*2.**(1.-i))
        lisT=np.sort(lisT)
        for i in range(len(lisT)):
            lisT[i]=int(np.round(n_col*lisT[i]))
        lisT[len(lisT)-1]=lisT[len(lisT)-1]-1
        # Reference Matrix
        references= np.zeros((n_row,NumFactors))
        for i in range(len(lisT)):
            references[:,i]=data[:,np.int(lisT[i])]
        # Initialization
        mat_X = np.zeros((NumFactors, NumFactors))
        spectra = factorAnalysis.Spectra(data, references, NumFactors)
        n_sliders = NumFactors**2

        fig = plt.figure(figsize=(10, 7))
        axs = fig.add_subplot(121)
        axc = fig.add_subplot(122)

        def update_N(**xvalor):
            xvalor=[]
            for i in range(n_sliders):
                xvalor.append(controls[i].value)
            values=np.reshape(xvalor,(NumFactors,NumFactors))
            mat_X=np.transpose(values)
            sp,conc=spectra.get_spectrum(mat_X)
            new_xanes=pd.DataFrame(sp)
            new_xanes.index=energy
            new_concentrations=pd.DataFrame(conc)
            saveToFile("results"+os.sep+"Pure_XANES_Spectra.dat",new_xanes.values)
            saveToFile("results"+os.sep+"Pure_XANES_Concentrations.dat",new_concentrations.values)

            #plot figures
            axs.clear()
            axs.set_xlim([min(energy),max(energy)])
            axs.set_xlabel("Energy (eV)",fontweight='bold')
            axs.set_ylabel("Absorbance",fontweight='bold')
            axs.set(title="Pure XANES")
            axc.clear()
            axc.set_xlim([0,n_col-1])
            axc.set_ylim([-0.05,1.2])
            axc.set_xlabel("Scan Index",fontweight='bold')
            axc.set_ylabel("Fraction of Pure Components",fontweight='bold')
            axc.set(title="Pure Concentrations")
            new_xanes.plot(ax=axs, linewidth=3)
            new_concentrations.plot(ax=axc,linewidth=3)

        # Setup Widgets
        controls=[]
        o='vertical'
        for i in range(n_sliders):
            title="x%i%i" % (i%NumFactors+1, i//NumFactors+1)
            sl=widgets.FloatSlider(description=title,min=-5.0, max=5.0, step=0.1, orientation=o, continuous_update=False)
            controls.append(sl)
        controlsDict = {}
        for c in controls:
            controlsDict[c.description] = c
        uif = widgets.HBox(tuple(controls))

        outf = widgets.interactive_output(update_N,controlsDict)
        display(uif, outf)
        return NumFactors

    widgets.interact(q,NumFactors=widgets.BoundedIntText(
            value=2,
            min=2,
            max=n_col-1,
            step=1,
            description='PCs:',
            disabled=False))

class targetTransformationPCA:
    def __init__(self, energy, data, sign, min_val, max_val, step_val):
        self.params = {}
        self.fig = plt.figure()
        self.pureSpectra = None
        self.pureConcentrations = None

        def n_pc(components):
            NumFactors=components
            self.params['PCs'] = components
            u,s,vT=np.linalg.svd(data, full_matrices=False)
            u_red = u[:,:NumFactors]
            s_red = np.diag(s[:NumFactors])
            v_red = vT[:NumFactors,:]
            us=np.dot(u_red,s_red)
            n_col=np.shape(data)[1]
            # scaling
            scale=sign*(np.sqrt((1./((1./(np.max(energy)-np.min(energy)))*(np.trapz((us[:,0])**2,energy)))))) # Ho tolto -1 davanti a scale
            scale_M=[]
            for i in range(NumFactors):
                scale_M.append(scale)
        #mat_X_0 Not Normalized
            mat_X = np.zeros((NumFactors, NumFactors))
        ##############################################
        #mat_X_N Normalized
            mat_X_N = np.zeros((NumFactors, NumFactors))
            mat_X_N[0,:]=scale_M
            n_sliders = NumFactors**2
            n_sliders_N=(NumFactors**2)-NumFactors
        ##############################################
        #mat_X_1s first spectrum fixed
            col_1s = np.linalg.lstsq(us,data[:,0],rcond=None)[0]
            mat_X_1s=np.zeros((NumFactors, NumFactors))
            mat_X_1s[:,0]=col_1s
        #mat_X_L last spectrum fixed
            col_L=np.linalg.lstsq(us,data[:,np.shape(data)[1]-1],rcond=None)[0]
            mat_X_L=np.zeros((NumFactors, NumFactors))
            mat_X_L[:,NumFactors-1]=col_L
        #mat_X_1s_L first and last spectrum fixed
            mat_X_1s_L=np.zeros((NumFactors, NumFactors))
            mat_X_1s_L[:,0]=col_1s
            mat_X_1s_L[:,NumFactors-1]=col_L
            n_sliders_1s_L=(NumFactors**2)-2*NumFactors
        ###############################################
        #mat_X_N_1s Normalized and first spectrum fixed
            mat_X_N_1s = np.zeros((NumFactors, NumFactors))
            mat_X_N_1s[0,:]=scale_M
            mat_X_N_1s[:,0]=col_1s
            n_sliders_N_1s=(NumFactors-1)**2
        ###############################################
        #mat_X_N_L Normalized and Last spectrum fixed
            mat_X_N_L = np.zeros((NumFactors, NumFactors))
            mat_X_N_L[0,:]=scale_M
            mat_X_N_L[:,NumFactors-1]=col_L
            n_sliders_N_L=(NumFactors-1)**2
        #################################################
        #mat_X_N_L Normalized  1st and Last spectrum fixed
            mat_X_N_1st_L = np.zeros((NumFactors, NumFactors))
            mat_X_N_1st_L[0,:]=scale_M
            mat_X_N_1st_L[:,0]=col_1s
            mat_X_N_1st_L[:,NumFactors-1]=col_L
            n_sliders_N_1s_L=(NumFactors-1)*(NumFactors-2)

            def nature(control):
                ntype = control
                self.params['Switching'] = control
                if ntype=="Case: 1":
                    def assign(control_1):
                        s_spectrum=control_1
                        self.params['Constraints'] = control_1
                        if s_spectrum=='No Constraints':
                            matrix=mat_X
                            sliders=n_sliders
                        elif s_spectrum=="1st spectrum fixed":
                            matrix=mat_X_1s
                            sliders=n_sliders_N
                        elif s_spectrum=="Last spectrum fixed":
                            matrix=mat_X_L
                            sliders=n_sliders_N
                        elif s_spectrum=='1st and Last spectrum fixed':
                            matrix=mat_X_1s_L
                            sliders=n_sliders_1s_L
                        factorAnalysis.unNorm(s_spectrum,us,v_red,matrix,NumFactors,sliders,data,energy,min_val,max_val,step_val, self)

                    widgets.interact(assign, control_1 = widgets.RadioButtons(
                    options=['No Constraints','1st spectrum fixed', 'Last spectrum fixed','1st and Last spectrum fixed'],
                    description='Constraints:',
                    disabled=False))

                elif ntype=="Case: 2":
                    def assign2(control_2):
                        n_spectrum=control_2
                        self.params['Constraints'] = control_2
                        if n_spectrum=="Normalization":
                            matrix=mat_X_N
                            sliders=n_sliders_N
                        elif n_spectrum=="Norm. and 1st spectrum":
                            matrix=mat_X_N_1s
                            sliders=n_sliders_N_1s
                        elif n_spectrum=="Norm. and Last spectrum":
                            matrix=mat_X_N_L
                            sliders=n_sliders_N_L
                        elif n_spectrum=="Norm., 1s and last spectrum":
                            matrix=mat_X_N_1st_L
                            sliders=n_sliders_N_1s_L
                        factorAnalysis.Norm(n_spectrum,us,v_red,matrix,NumFactors,sliders,data,energy,min_val,max_val,step_val, self)

                    widgets.interact(assign2, control_2 = widgets.RadioButtons(
                    options=['Normalization','Norm. and 1st spectrum','Norm. and Last spectrum','Norm., 1s and last spectrum'],
                    description='Constraints:',
                    disabled=False))

            widgets.interact(nature, control = widgets.RadioButtons(
            options=['Case: 1', 'Case: 2'],
            description='Switching:',
            disabled=False))

        n_col=np.shape(data)[1]
        widgets.interact(n_pc,components=widgets.BoundedIntText(
            value=2,
            min=2,
            max=n_col-1,
            step=1,
            description='PCs:',
            disabled=False))

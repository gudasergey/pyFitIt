import copy, types, gc, os
from IPython.core.display import display, Javascript
from ipywidgets import widgets, HTML
from . import utils, smoothLib, inverseMethod, optimize, ML, plotting, funcModel, curveFitting
from .funcModel import FuncModel, ParamProperties
from scipy.interpolate import Rbf
import warnings, traceback, json
import numpy as np
from matplotlib.font_manager import FontProperties
from pyfitit.curveFitting import calculateRFactor

ENERGY_RANGE_PARAM = 'energyRange'
CONCENTRATION_PARAM = 'concentration'
METHOD_PARAM = 'method'

R_FACTOR_LABEL = 'R-factor'
THEORY_PLOT_LABEL = 'theory'
EXPERIMENT_PLOT_LABEL = 'experiment'


# ====================================================
#           Begin of abstract sliders engine
# ====================================================

class ControlsManager:
    styles = '''<div class='doNotShow'>
            <style>
            .doNotShow {width:0px; height:0px; margin:0; padding:0; overflow:hidden;}
            .prompt:empty {min-width:0px; width:0; padding-left:0; padding-right:0;}
            .container { width:100% !important; }
            .output_area {display:inline-block !important; }
            .cell > .output_wrapper > .output {margin-left: 14ex;}
            .out_prompt_overlay.prompt {min-width: 14ex;}
            .fitBySlidersOutput {flex-direction:row !important; flex-wrap: wrap;}
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
            </style></div>
            '''

    def __init__(self):
        self.context = types.SimpleNamespace()

        self.status = ''
        self.onControlChangedDelegate = None
        self.onClickDelegate = None
        self.onPlotDataPostProcess = None
        self.onExternalEval = None
        self.onInitDelegate = None
        self.predictionBackend = None
        self.plotter = None
        self.statusHTML = widgets.HTML(value='', placeholder='', description='')
        self.saver = None
        self.updatingModel = False

    def setup(self, controls, ui, debug=False, plotter=None, defaultParams=None, saver=None):
        self.plotter = DefaultPlotter() if plotter is None else plotter
        self.saver = saver
        self.context.plotData = {}
        self.context.controls = controls
        self.context.cache = utils.CacheInMemory()
        self.context.addToStatus = self.addToStatus
        self.context.addDebugToStatus = self.addDebugToStatus
        self.context.setStatus = self.setStatus
        self.context.debug = debug
        self.context.getFig = self.plotter.getFig
        self.context.getParam = lambda pname: FittingUtils.getParamFromContext(self.context, pname)
        self.bindControlsEvents()
        self.drawControls(ui)
        if defaultParams is not None:
            self.setControlValues(defaultParams)
        self.init()

    def setControlValues(self, values):
        self.updatingModel = True

        for key, value in values.items():
            FittingUtils.setParamForContext(self.context, key, value)

        self.updatingModel = False

    def init(self):
        self.updatingModel = True
        self.onInitDelegate(self.context)
        self.plotGraphs()
        self.updatingModel = False

    def updateLoop(self, name, oldValue, newValue, isButtonEvent):
        if self.updatingModel:
            return  # ignoring cascade updates, handling only the one initiated update
        self.updatingModel = True
        if isButtonEvent:
            self.onClickDelegate(self.context, name)
        else:
            self.loopWithoutRePlotting(name, oldValue, newValue)
        self.plotGraphs()  # re-plotting graph
        self.updatingModel = False

    def loopWithoutRePlottingNoParam(self):
        self.loopWithoutRePlotting(None, None, None)

    def loopWithoutRePlotting(self, name, oldValue, newValue):
        if self.onExternalEval is not None:
            self.onExternalEval(name, self.context, self.updateInternalNoParam)
        self.updateInternal(name, oldValue, newValue)

    def updateInternalNoParam(self):
        self.updateInternal(None, None, None)

    def updateInternal(self, name, oldValue, newValue):
        self.onControlChanged(name, oldValue, newValue)  # handling value update
        if self.onPlotDataPostProcess is not None:
            self.onPlotDataPostProcess(self.context)

    def safeUpdateLoop(self, name, oldValue, newValue, isButtonEvent):
        try:
            with warnings.catch_warnings(record=True) as warn:
                self.updateLoop(name, oldValue, newValue, isButtonEvent)
                if len(warn) > 0:
                    status = 'Warning: '
                    for w in warn: status += str(w.message) + '\n'
                    self.addToStatus(status)
        except Exception as exc:
            self.addToStatus(traceback.format_exc())
        except Warning:
            self.addToStatus(traceback.format_exc())

    def onControlChanged(self, name, oldValue, newValue):
        if self.onControlChangedDelegate is not None:
            return self.onControlChangedDelegate(self.context, name, oldValue, newValue)
        return None

    def drawControls(self, controls):
        self.initStyles()
        ui = widgets.VBox(tuple(controls) + (self.statusHTML,))
        # ui.layout.flex_flow = 'row wrap'
        # ui.layout.justify_content = 'space-between'
        # ui.layout.align_items = 'flex-start'
        # ui.layout.align_content = 'flex-start'

        display(ui)
        display(Javascript('$(this.element).addClass("fitBySlidersOutput");'))

    def setStatus(self, s):
        self.status = s
        self.statusHTML.value = '<div class="status">' + s + '</div>'

    def addToStatus(self, s):
        self.status += '<div>'+s+'</div>'
        self.setStatus(self.status)

    def addDebugToStatus(self, s):
        if self.context.debug:
            self.addToStatus(s)

    def bindControlsEvents(self):
        for name, c in self.context.controls.items():
            if isinstance(c, widgets.Button):
                c.on_click(lambda _, n=name: self.safeUpdateLoop(n, None, None, True))
            else:
                c.observe(lambda change, n=name: self.safeUpdateLoop(n, change['old'], change['new'], False), names='value')

    def initStyles(self):
        if utils.isJupyterNotebook():
            html = HTML(self.styles)
            html.add_class('doNotShow')
            display(html)

    def plotGraphs(self):
        self.plotter.clear()
        self.plotter.plot(self.context)

    def saveParams(self, path):
        if self.saver is None:
            with open(path, 'w') as outfile:
                json.dump(FittingUtils.getParamsFromContext(self.context), outfile)
        else: self.saver.saveParams(path)

    def saveAllData(self, folder):
        if self.saver is None:
            os.makedirs(folder, exist_ok=True)
            self.saveParams(folder+os.sep+'params.txt')
            plotting.savefig(folder+os.sep+'image.png', self.plotter.getFig())

            plotData = self.context.plotData
            metrics = {}
            with open(folder+os.sep+'graph_data.txt', 'w') as f:
                for key, graph in plotData.items():
                    if graph is None: continue
                    if graph['type'] == 'default':
                        f.write(graph['label']+' x: ')
                        np.savetxt(f, [graph['xData']], delimiter=',')
                        f.write(graph['label']+' y: ')
                        np.savetxt(f, [graph['yData']], delimiter=',')
                for key, graph in plotData.items():
                    if graph is None: continue
                    if graph['type'] == 'metric':
                        f.write(graph['label']+' = '+str(graph["value"])+'\n')
        else: self.saver.saveAllData(folder)


class DefaultPlotter:
    def __init__(self, **createfigParams):
        self.fig, self.ax = plotting.createfig(interactive=True, **createfigParams)
        if not isinstance(self.ax, np.ndarray): self.ax = np.array([self.ax])
        if len(self.ax.shape) == 2: self.ax = self.ax.flatten()

    def clear(self):
        for ax in self.ax: ax.clear()

    def plot(self, context):
        """
        Default plotter, outputs all gathered data to the end user (e.g. Jupyter Notebook)
        """
        plotData = context.plotData
        for ax in self.ax:
            ax.set_xlabel("Energy")
            ax.set_ylabel("Intensity")
        metrics = {ax:'' for ax in self.ax}
        for key, graph in plotData.items():
            if graph is None or not isinstance(graph, dict):
                continue
            axisInd = graph['axisInd'] if 'axisInd' in graph else 0
            ax = self.ax[axisInd]
            plotFuncStr = graph['plotFunc'] if 'plotFunc' in graph else None
            plotFunc = ax.scatter if plotFuncStr == 'scatter' else ax.plot

            if graph['type'] == 'default':
                plotFunc(
                    graph['xData'],
                    graph['yData'],
                    label=graph['label'] if 'label' in graph else 'unknown',
                    color=graph['color'] if 'color' in graph else None)
            elif graph['type'] == 'metric':
                metrics[ax] += f'{ graph["label"] }: { graph["text"] }\n'
            elif graph['type'] == 'xlim':
                ax.set_xlim(graph['xlim'])
                plotting.updateYLim(ax)
            elif graph['type'] == 'xlabel':
                ax.set_xlabel(graph['xlabel'])
        for ax in self.ax:
            if metrics[ax] != '':
                ax.text(0.98, 0.02, metrics[ax], transform=ax.transAxes, horizontalalignment='right')
            ax.legend(loc='upper right')

    def getFig(self):
        return self.fig


class ControlsBuilder:

    def __init__(self):
        self.controls = {}
        self.buildResult = []
        self.controlsStack = []

    def addSlider(self, name, min, max, step, value, type='f'):
        if type == 'f':
            s = widgets.FloatSlider(description=name, min=min, max=max, step=step, value=value,
                                    orientation='horizontal', continuous_update=False, readout_format='.3g')
        elif type == 'i':
            s = widgets.IntSlider(description=name, min=min, max=max, step=step, value=value,
                                  orientation='horizontal', continuous_update=False)
        else:
            raise Exception('Unknown type')
        self.pushControl(name, s)

    def addSelectionSlider(self, name, options, value=None):
        s = widgets.SelectionSlider(
            options=options,
            value=value,
            description=name,
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True
        )
        self.pushControl(name, s)

    def addFloatText(self, name, value, disabled=False):
        self.pushControl(name,
                         widgets.FloatText(
                             value=value,
                             description=name,
                             disabled=disabled
                         ))

    def addTextArea(self, name, value, disabled=False):
        self.pushControl(name,
                         widgets.Textarea(
                             value=value,
                             description=name,
                             disabled=disabled,
                         ))

    def addValidMark(self, name, value):
        self.pushControl(name,
                         widgets.Valid(
                             value=value,
                             description=name,
                         ))

    def addCheckbox(self, name, value):
        self.pushControl(name,
                         widgets.Checkbox(
                             value=value,
                             description=name,
                         ))

    def addButton(self, name):
        self.pushControl(
            name,
            widgets.Button(
                description=name,
                disabled=False,
                button_style=''  # 'success', 'info', 'warning', 'danger' or '',
            ))

    def addDropdown(self, name, values, default):
        self.pushControl(
            name,
            widgets.Dropdown(options=values, value=default, description=name, disabled=False))

    def addRangeSlider(self, name, value, min, max, step):
        self.pushControl(
            name,
            widgets.FloatRangeSlider(
                value=value,
                min=min,
                max=max,
                step=step,
                description=name,
                disabled=False,
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='.1f',
            )
        )

    def beginBox(self):
        self.controlsStack.append([])

    def endBox(self, type):
        controls = self.controlsStack.pop()
        if type == 'h':
            self.pushControl('', widgets.HBox(controls))
        else:
            self.pushControl('', widgets.VBox(controls))

    def pushControl(self, name, control):
        self.controls[name] = control
        if len(self.controlsStack) == 0:
            self.buildResult.append(control)
        else:
            self.controlsStack[-1].append(control)


class CombinedListeners:

    def __init__(self, manager, uiListeners):
        self.uiListeners = uiListeners
        manager.onControlChangedDelegate = self.onControlChanged  # for when user changes slider, dropdown list etc.
        manager.onClickDelegate = self.onClick  # for button clicks
        manager.onInitDelegate = self.onInit  # for the first call

    def onControlChanged(self, context, name, oldValue, newValue):
        """
        This function is called every time user changes value of slider or any other control in the Jupyter Notebook

        Note: this function won't be called when control is changed during processing of user input,
        e.g. when slider values are changed from the code, while handling user input
        This was made with a purpose of eliminating cascaded and looped calculations

        :param context: context, that contains all the data needed for calculations
        :param name: name of the slider or any other control, that changed its value
        :param oldValue: previous control value
        :param newValue: new control value
        """
        for listener in self.uiListeners:
            self.callIfExists(listener, 'onControlChanged', [context, name, oldValue, newValue])

    def onInit(self, context):
        for listener in self.uiListeners:
            self.callIfExists(listener, 'onInit', [context])

    def onClick(self, context, name):
        for listener in self.uiListeners:
            self.callIfExists(listener, 'onClick', [context, name])

    @staticmethod
    def callIfExists(target, methodName, params):
        method = getattr(target, methodName, None)
        if callable(method):
            method(*params)


class FittingUtils:

    @staticmethod
    def getParamsFromContext(context):
        dict = {}
        for name, control in context.controls.items():
            if isinstance(control, (widgets.Button, widgets.HBox, widgets.VBox)):
                continue
            dict[name] = control.value

        return dict

    @staticmethod
    def setParamForContext(context, name, value):
        if name in context.controls:
            context.controls[name].value = value
        else:
            context.addDebugToStatus(f"Error: could not find parameter {name}")

    @staticmethod
    def getParamFromContext(context, name):
        if name in context.controls:
            return context.controls[name].value
        else:
            context.addDebugToStatus(f"Error: could not find parameter {name}")
            return None

    @staticmethod
    def getPlotDataEntry(context, label, type='default'):
        for key, data in context.plotData.items():
            if data is None:
                continue

            if data['label'] == label and data['type'] == type:
                return data

        return None

    @staticmethod
    def getMetricValue(context, label):
        for key, data in context.plotData.items():
            if data is None:
                continue

            if data['label'] == label and data['type'] == 'metric':
                return data['value']

        return None

    @staticmethod
    def checkDefaultParams(values):
        if values is None:
            return {}

        assert isinstance(values, (dict, str)), 'Expected filename or dictionary'

        if isinstance(values, str):
            with open(values) as json_file:
                values = json.load(json_file)

        return values


# ====================================================
#           End of abstract sliders engine
# ====================================================


# ====================================================
#           Sliders for FuncModel class
# ====================================================

class FuncModelPlotter:
    def __init__(self, funcModel):
        self.funcModel = funcModel

    def clear(self):
        if self.funcModel.ax is not None:
            for ax in self.funcModel.ax: ax.clear()

    def plot(self, context):
        self.funcModel.plot()

    def getFig(self):
        return self.funcModel.fig


class FuncModelSliders:
    def __init__(self, funcModel:FuncModel, fitType='min', debug=False, defaultParams=None):
        """
        :param funcModel:
        :param paramProperties: dict{paramName -> {type:{'real','int','range','bool','list'}, domain:([a,b],list)}, default:val}
        :param fitType: 'min' or 'max'
        """
        self.funcModel = funcModel.copy()
        self.funcModel.createfigParams['interactive'] = True
        self.floatParams = [pn for pn in self.funcModel.paramProperties if self.funcModel.paramProperties[pn]['type']=='float']
        self.fittedParams = list(self.floatParams)
        self.fitType = fitType
        self.defaultParams = defaultParams if defaultParams is not None else {}
        self.setup(debug)
        self.manager = None

    def setup(self, debug):
        # building interface
        builder = ControlsBuilder()
        for pn, pp in self.funcModel.paramProperties.items():
            val = None
            if 'default' in pp: val = pp['default']
            if pn in self.defaultParams: val = self.defaultParams[pn]
            if pp['type'] == 'float':
                a, b = pp['domain']
                if val is None: val = (a+b)/2
                builder.addSlider(name=pn, min=a, max=b, step=(b-a)/50, value=val)
                fit = self.defaultParams['fit '+pn] if 'fit '+pn in self.defaultParams else True
                builder.addCheckbox(name='fit '+pn, value=fit)
            elif pp['type'] == 'int':
                a, b = pp['domain']
                if val is None: val = (a + b) // 2
                builder.addSlider(name=pn, min=a, max=b, step=1, value=val, type='i')
            elif pp['type'] == 'range':
                a, b = pp['domain']
                if val is None: val = [a,b]
                builder.addRangeSlider(name=pn, min=a, max=b, step=(b-a)/50, value=val)
            elif pp['type'] == 'bool':
                if val is None: val = True
                builder.addCheckbox(name=pn, value=val)
            elif pp['type'] == 'list':
                if val is None: val = pp['domain'][0]
                builder.addDropdown(name=pn, values=pp['domain'], default=val)
            else: assert False, 'Unknown type '+pp['type']
        if len(self.floatParams) > 0:
            builder.addButton(name='optimize')
            builder.addButton(name='find better')

        # binding callbacks
        manager = ControlsManager()
        manager.onControlChangedDelegate = self.onControlChanged  # for when user changes slider, dropdown list etc.
        manager.onClickDelegate = self.onClick  # for button clicks
        manager.onInitDelegate = self.update  # for the first call
        manager.setup(builder.controls, builder.buildResult, debug=debug, plotter=FuncModelPlotter(self.funcModel))
        self.manager = manager

    def onClick(self, context, name):
        params = FittingUtils.getParamsFromContext(context)

        def fun(x, *args):
            p = copy.deepcopy(params)
            for i, pn in enumerate(self.fittedParams): p[pn] = x[i]
            val = self.funcModel.evaluate(p)
            if self.fitType == 'max': val = -val
            return val

        def optim(startPoint):
            bounds = [self.funcModel.paramProperties[pn]['domain'] for pn in self.fittedParams]
            # COBYLA - 0.725  Powell - 0.227
            fmin, xmin = optimize.minimize(fun, startPoint, bounds, method='Powell')
            return fmin, xmin

        x0 = [params[pn] for pn in self.fittedParams]
        if name == 'optimize':
            assert len(self.funcModel.paramProperties.constrains) == 0, 'Not implemented yet!'
            fmin, xmin = optim(x0)
            for i,pn in enumerate(self.fittedParams):
                FittingUtils.setParamForContext(context, pn, xmin[i])
            self.update(context)
        elif name == 'find better':
            fminPrev = fun(x0)
            fmin = fminPrev
            i = 0
            while i < 10 and fmin >= fminPrev:
                x0 = self.funcModel.paramProperties.getRandom(self.fittedParams)
                x0 = [x0[pn] for pn in self.fittedParams]
                fmin, xmin = optim(x0)
            if fmin < fminPrev:
                for i, pn in enumerate(self.fittedParams):
                    FittingUtils.setParamForContext(context, pn, xmin[i])
                self.update(context)
        # context.addDebugToStatus('Button ' + name)

    def update(self, context):
        params = FittingUtils.getParamsFromContext(context)
        context.addDebugToStatus(str(params))
        val = self.funcModel.evaluate(params)
        self.funcModel.plot()
        context.addDebugToStatus(str(val))
        self.fittedParams = [pn for pn in self.floatParams if params['fit '+pn]]

    def onControlChanged(self, context, name, oldValue, newValue):
        self.update(context)

    def saveAllData(self, fileName):
        self.funcModel.save(fileName)


# ====================================================
#           Fitting spectrum by a single component
# ====================================================


class ProjectParametersBuilder:
    def __init__(self, controlsBuilder, project, theoryProcessingPipeline, defaultParams, prefix=None):
        self.project = project
        self.prefix = prefix
        self.controlsBuilder = controlsBuilder
        self.theoryProcessingPipeline = theoryProcessingPipeline
        self.defaultParams = defaultParams
        
    def withPrefix(self, name):
        return self.prefix + '_' + name if self.prefix is not None else name

    def getParamFromProject(self, pName, transType):
        if transType == 'fdmnes smooth':
            return self.project.FDMNES_smooth[pName]
        elif transType == 'adf smooth':
            return self.project.ADF_smooth[pName]
        else:
            return None

    def addPipelineSliders(self):
        for transformation in self.theoryProcessingPipeline:
            for p in transformation['params']:
                v = transformation['params'][p]
                if isinstance(v, list):
                    assert len(v) == 2
                    a, b = v
                    val = self.getParamFromProject(p, transformation['type'])
                    assert (a is not None) or (val is not None)
                    if a is None: a, b = val-b/2, val+b/2
                    if val is None: val = (a+b)/2
                    name = self.withPrefix(p)
                    if name in self.defaultParams: val = self.defaultParams[name]
                    self.controlsBuilder.addSlider(name=name, min=a, max=b, step=(b-a)/50, value=val)

    def addEnergyRange(self, minEnergy, maxEnergy, energyValue, step):
        name = self.withPrefix(ENERGY_RANGE_PARAM)
        if name in self.defaultParams: energyValue = self.defaultParams[name]
        self.controlsBuilder.addRangeSlider(name=name, min=minEnergy, max=maxEnergy, value=energyValue, step=step)

    def addGeometryParams(self, params):
        for pName in params.columns:
            min = params[pName].min()
            max = params[pName].max()
            name = self.withPrefix(pName)
            val = (max - min) / 2 + min
            if name in self.defaultParams: val = self.defaultParams[name]
            self.controlsBuilder.addSlider(name, min=min, max=max, step=(max-min)/50, value=val)

    def addConcentration(self):
        self.controlsBuilder.addSlider(name=self.prefix, min=0.01, max=1, step=0.05, value=1)

    def addEstimatorDropdown(self):
        default = inverseMethod.allowedMethods[0]
        if METHOD_PARAM in self.defaultParams: default = self.defaultParams[METHOD_PARAM]
        self.controlsBuilder.addDropdown(METHOD_PARAM, inverseMethod.allowedMethods, default)


class SpectrumSliders:
    
    def __init__(self, sample, project, debug=False, defaultParams=None, theoryProcessingPipeline=None):
        """

        Parameters
        ----------
        sample : ML.Sample
        project : Project.
        debug : Show debug info. The default is False.
        defaultParams : Default parameters
        theoryProcessingPipeline : list(str, lists)
                [['fdmnes smooth', {param1:... value or interval}], 'approximation', ['L1 norm', {norm:value or interval}], [userDefinedFunction, {param1:... value or interval}]]
                  If user sets value then param is fixed, interval means creating a slider for variation of the param.
                  userDefinedFunction - is a function(energy, intensity, params, context) -> energy, intensity
                  In case of missed smooth params the default sliders are generated.
                  In case of string instead of list (e.g. ['approximation', 'fdmnes smooth', 'fit L2 norm']) for smoothing default sliders are generated, for norm it is automatically fitted to experiment.
        Returns
        -------
        None.

        """
        defaultParams = FittingUtils.checkDefaultParams(defaultParams)

        self.project = project
        self.sample = sample
        self.manager = None
        self.spectrumFitter = None
        self.theoryProcessingPipeline = self.canonizePipeline(theoryProcessingPipeline)
        self.setup(debug, defaultParams, project)

    def setup(self, debug, defaultParams, project):
        # building interface
        builder = ControlsBuilder()
        parameters = ProjectParametersBuilder(builder, project, self.theoryProcessingPipeline, defaultParams)
        parameters.addGeometryParams(self.sample.params)
        parameters.addPipelineSliders()
        # energy range slider
        e0 = project.spectrum.energy[0] - 10
        e1 = project.spectrum.energy[-1] + 10
        v0 = project.intervals['fit_geometry'][0]
        v1 = project.intervals['fit_geometry'][1]
        parameters.addEnergyRange(e0, e1, [v0, v1], (e1-e0)/30)

        parameters.addEstimatorDropdown()

        # binding callbacks
        manager = ControlsManager()
        self.spectrumFitter = SpectrumFittingBackend(self.sample, self.project, self.theoryProcessingPipeline, manager)
        CombinedListeners(manager,
                          [
                              self.spectrumFitter,  # first we run fitting
                              RFactor()  # then calculate r-factor
                          ])

        # default params are values for parameters, can be dict or string - filepath
        # debug - whether or not show debug messages
        manager.setup(builder.controls, builder.buildResult, debug=debug, defaultParams=defaultParams)

        self.manager = manager

    @staticmethod
    def canonizePipeline(theoryProcessingPipeline):
        if theoryProcessingPipeline is None: theoryProcessingPipeline = ['approximation']
        pipeline = []
        names = []
        for i in range(len(theoryProcessingPipeline)):
            tr = theoryProcessingPipeline[i]
            if isinstance(tr, str) or callable(tr): tr = [tr, {}]
            if len(tr) == 1: tr = [tr[0], {}]
            tr_name = tr[0]
            if callable(tr_name):
                func = tr_name
                tr_name = 'func'
            params = tr[1]
            str_name = tr_name.split(' ')
            if len(str_name) == 2 and str_name[1] == 'smooth':
                smoothType = str_name[0]
                dsp = smoothLib.DefaultSmoothParams(0, 0)
                allParams = [p['paramName'] for p in dsp.params[smoothType]]
                for p in params:
                    assert p in allParams, 'Unknown '+smoothType+' smooth param: '+p
                for j in range(len(dsp.params[smoothType])):
                    p = dsp.params[smoothType][j]
                    paramName = p['paramName']
                    if paramName not in params:
                        borders = [p['leftBorder'], p['rightBorder']]
                        if paramName in ['Efermi', 'shift']: borders = [None, borders[1] - borders[0]]
                        params[paramName] = borders
            tr_name1 = tr_name
            if tr_name1 in names:
                assert tr_name1 != 'approximation'
                j = 2
                while tr_name1 in names:
                    tr_name1 = tr_name + '_' + str(j)
                    j += 1
            ctr = {'type': tr_name, 'params': params, 'name': tr_name1}
            if tr_name == 'func':
                ctr['func'] = func
            names.append(tr_name1)
            pipeline.append(ctr)
        assert 'approximation' in names, 'Pipeline must contain approximation by machine learning'
        return pipeline

    def saveAllData(self, folder):
        self.manager.saveAllData(folder)


class SpectrumFittingBackend:

    def __init__(self, sample, project, theoryProcessingPipeline, manager):
        self.paramsHash = None
        self.manager = manager
        self.cache = utils.CacheInMemory()

        self.exp_energy = project.spectrum.energy
        self.theory_energy = sample.energy

        self.geometryParams = sample.params
        self.geometryParamsColumns = sample.params.columns
        self.geometryParamsCount = sample.params.shape[1]

        self.sample = sample
        self.project = project
        self.theoryProcessingPipeline = theoryProcessingPipeline

    def getEstimator(self, methodName):
        method = inverseMethod.getMethod(methodName)
        return ML.Normalize(method, xOnly=False)

    def applyTransformation(self, transformation, tr_params, energy, intensity, context):
        # possible transforamtions: 'adf smooth' 'fdmnes smooth' , 'L1 norm' (also L2,3,4,5,...)
        tr_type = transformation['type']
        tr_name = transformation['name']
        if tr_type[:len('plot current')] == 'plot current':
            assert tr_type in ['plot current', 'plot current normalized'], 'Unknown transform: '+tr_type
            assert len(intensity) == 1, 'Can plot only one spectrum. Approximation transform should be applied before'
            e = energy + context.getParam('shift')
            intensity1 = intensity.reshape(-1)
            if tr_type == 'plot current normalized':
                intensity1 = intensity1 / intensity1[-1] * context.expSpectrum.intensity[-1]
            context.plotData['not smoothed'] = {'type': 'default', 'xData': e, 'yData': intensity1, 'label': 'not smoothed', 'color': 'orange'}
            assert len(intensity.shape) == 2, tr_type
            return energy, intensity
        if tr_type == 'approximation':
            def getFittedEstimator():
                # context.addDebugToStatus('Fitting non-smoothed estimator')
                estimator = self.getEstimator(context.getParam(METHOD_PARAM))
                estimator.fit(self.sample.params.to_numpy(), intensity)
                return estimator

            estimator = self.cache.getFromCacheOrEval(dataName='get estimator', evalFunc=getFittedEstimator, dependData=[self.sample.params.to_numpy(), intensity, context.getParam(METHOD_PARAM)])
            geomArg = np.array([context.params[pName] for pName in self.geometryParamsColumns]).reshape([1, self.geometryParamsCount])
            intensity = estimator.predict(geomArg)
            assert len(intensity.shape) == 2, tr_type
            return energy, intensity
        if tr_type == 'func':
            func = tr_params['func']

            def applyUserDefinedFunc():
                return func(energy, intensity, tr_params, self.manager.context)

            allParams = [context.params[p] for p in sorted(list(context.params.keys()))]
            res = self.cache.getFromCacheOrEval(dataName='applyUserDefinedFunc', evalFunc=applyUserDefinedFunc, dependData=[energy, intensity, tr_params, allParams])
            energy, intensity = res
            assert len(intensity.shape) == 2, tr_type
            return energy, intensity
        words = tr_type.split(' ')
        assert len(words) == 2, 'Unknown transform: '+tr_type
        if words[1] == 'norm':
            p = int(words[0][1:])
            if len(tr_params) == 0:
                a, b = self.project.intervals['fit_norm']
                ind = (a<=energy) & (energy<=b)
                norms = utils.norm_lp(energy[ind], intensity[:, ind], p).reshape(-1, 1)
                ind_exp = (a<=self.project.spectrum.energy) & (self.project.spectrum.energy<=b)
                exp_norm = utils.norm_lp(self.project.spectrum.energy[ind_exp], self.project.spectrum.intensity[ind_exp], p)
                intensity = (intensity/norms)*exp_norm
                if len(intensity) == 1:
                    self.manager.context.plotData[tr_name] = {'type': 'metric', 'label': tr_name, 'text': '%.4f' % (norms[0][0]/exp_norm), 'value': norms[0][0]/exp_norm}
            else:
                assert 'norm' in tr_params, 'Param name for norm should be "norm"'
                intensity = intensity / tr_params['norm']
        else:
            assert words[1] == 'smooth', 'Unknown transform: '+tr_type

            def smoothSample():
                res_int, res_energy = smoothLib.smoothDataFrame(tr_params, intensity, smoothType=words[0], exp_spectrum=self.project.spectrum, fit_norm_interval=[0,0], norm=1, energy=energy)
                return res_int, res_energy
            intensity, energy = self.cache.getFromCacheOrEval(dataName=tr_name+' of dataset', evalFunc=smoothSample, dependData=[tr_params, intensity])
        assert len(intensity.shape) == 2, tr_type
        return energy, intensity

    def makeTransformationParams(self, transformation, params):
        tr_params = copy.deepcopy(transformation['params'])
        for p in tr_params:
            if isinstance(tr_params[p], list):
                assert p in params
                tr_params[p] = params[p]
        if transformation['type'] == 'func':
            tr_params['func'] = transformation['func']
        return tr_params

    def calculatePrediction(self, params, context):
        # context.addDebugToStatus('params = ' + str(params))
        context.expSpectrum = self.project.spectrum
        context.params = params
        context.project = self.project

        def getTransformParams(transName):
            for transformation in self.theoryProcessingPipeline:
                if transformation['name'] == transName:
                    return self.makeTransformationParams(transformation, params)
            assert False, f'Transformation {transName} not found in pipeline'
        context.getTransformParams = getTransformParams
        spectra = self.sample.spectra.to_numpy()
        energy = self.sample.energy
        for transformation in self.theoryProcessingPipeline:
            # context.addDebugToStatus('old tr_params = '+str(transformation['params']))
            tr_params = self.makeTransformationParams(transformation, params)
            # context.addDebugToStatus('new tr_params = '+str(tr_params))
            energy, spectra = self.applyTransformation(transformation, tr_params, energy, spectra, context)
        assert spectra.shape[0] == 1, 'Only one spectrum must remain after all transformations. Do you forget approximation?'
        # plotting
        context.spectrumTheory = utils.Spectrum(energy, spectra.reshape(-1))
        context.plotData['theory'] = \
            {
                'type': 'default',
                'xData': context.spectrumTheory.energy,
                'yData': context.spectrumTheory.intensity,
                'label': THEORY_PLOT_LABEL,
                'color': 'blue'
            }
        context.plotData['experiment'] = \
            {
                'type': 'default',
                'xData': context.expSpectrum.energy,
                'yData': context.expSpectrum.intensity,
                'label': EXPERIMENT_PLOT_LABEL,
                'color': 'k'
            }
        context.plotData['plot xlim'] = {'type':'xlim', 'xlim':params['energyRange']}

    def update(self, context):
        self.calculatePrediction(FittingUtils.getParamsFromContext(context), context)

    def onControlChanged(self, context, name, oldValue, newValue):
        """
        This function is called every time user changes value of slider or any other control in the Jupyter Notebook

        Note: this function won't be called when control is changed during processing of user input,
        e.g. when slider values are changed from the code, while handling user input
        This was made with a purpose of eliminating cascaded and looped calculations

        :param context: context, that contains all the data needed for calculations
        :param name: name of the slider or any other control, that changed its value
        :param oldValue: previous control value
        :param newValue: new control value
        """
        self.update(context)

    def onInit(self, context):
        self.update(context)


class RFactor:

    def getRFactor(self, context):
        energyRange = FittingUtils.getParamFromContext(context, ENERGY_RANGE_PARAM)
        exp_e = np.array(FittingUtils.getPlotDataEntry(context, EXPERIMENT_PLOT_LABEL)['xData'])
        exp_xanes = np.array(FittingUtils.getPlotDataEntry(context, EXPERIMENT_PLOT_LABEL)['yData'])
        predictionXanes = np.interp(exp_e, FittingUtils.getPlotDataEntry(context, THEORY_PLOT_LABEL)['xData'],        FittingUtils.getPlotDataEntry(context, THEORY_PLOT_LABEL)['yData'])
        return calculateRFactor(exp_e, exp_xanes, predictionXanes, energyRange)

    def onControlChanged(self, context, name, oldValue, newValue):
        self.refreshRFactor(context)

    def onInit(self, context):
        self.refreshRFactor(context)

    def refreshRFactor(self, context):
        r_factor = self.getRFactor(context)
        context.plotData[R_FACTOR_LABEL] = \
            {
                'type': 'metric',
                'label': R_FACTOR_LABEL,
                'text': '%.4f' % r_factor,
                'value': r_factor
            }


# ====================================================
#           Fitting XANES by a mixture of components
# ====================================================

class RFactorMean:

    def __init__(self, sliderName, values, triggerName):
        self.values = values
        self.sliderName = sliderName
        self.triggerName = triggerName
        self.updateRunner = None
        self.context = None

    def getMeanRFactor(self, context):
        rFactors = []
        for c in self.values:
            self.context.controls[self.sliderName].value = c
            self.updateRunner()
            print(f'Temp: {c}, {[context.controls["HT"].value, context.controls["LT"].value]}')
            rFactors.append(FittingUtils.getMetricValue(self.context, R_FACTOR_LABEL))

        return np.mean(rFactors)

    def onClicked(self, context, updateRunner, name):
        self.updateRunner = updateRunner
        self.context = context
        mean_r_factor = self.getMeanRFactor(context)
        # addMetric(context.plotData, 'Mean R-Factor', '%.4f' % mean_r_factor, mean_r_factor)
        print('Mean R-Factor: %.4f' % mean_r_factor)


class MixtureConcentrationOptimizer:
    def __init__(self, concentrationParamNames, changeTrigger, bounds):
        self.bounds = bounds
        self.changeTrigger = changeTrigger
        self.paramNames = concentrationParamNames
        self.updateContextDelegate = None
        self.context = None

    def onExternalEval(self, name, context, updateInternal):
        if name != self.changeTrigger and name is not None:
            return

        self.updateContextDelegate = updateInternal
        self.context = context

        rFactor, params = optimize.minimize(
            self.minimizationFunc,
            [1. / len(self.bounds) for a in self.bounds],
            self.bounds,
            method='scipy')

        params = self.normalizedConcentrations(params)
        for i in range(len(self.paramNames)):
            context.controls[self.paramNames[i]].value = params[i]

    def normalizedConcentrations(self, params):
        return params / sum(params)

    def minimizationFunc(self, values, stub):
        for i in range(len(self.paramNames)):
            self.context.controls[self.paramNames[i]].value = values[i]

        self.updateContextDelegate()
        r = FittingUtils.getMetricValue(self.context, R_FACTOR_LABEL)
        return r


class XanesMixtureFittingBackend:

    def __init__(self, sampleList, projectList, defaultParams=None, debug=False):
        self.projectList = projectList
        self.sampleList = sampleList
        self.manager = None
        self.componentFitters = None
        self.rFactor = RFactor()
        self.setup(debug, defaultParams)

    def setup(self, debug, defaultParams):
        defaultParams = FittingUtils.checkDefaultParams(defaultParams)
        builder = ControlsBuilder()

        componentFitters = []
        # builder.addSelectionSlider("Tmp", spectra.labels)
        # builder.addButton('Eval Mean')
        parameters = ProjectParametersBuilder(builder)
        parameters.addEstimatorDropdown()

        # energy range slider
        project = self.projectList[0]
        e0 = project.spectrum.energy[0] - 10
        e1 = project.spectrum.energy[-1] + 10
        v0 = project.intervals['fit_geometry'][0]
        v1 = project.intervals['fit_geometry'][1]
        parameters.addEnergyRange(e0, e1, [v0, v1], (e1-e0)/30)

        for i in range(len(self.projectList)):
            parameters = ProjectParametersBuilder(builder, prefix=self.projectList[i].name)
            parameters.addConcentration()
            parameters.addSmoothing(-161, 20, 7713, 7730, 7700)
            parameters.addGeometryParams(self.sampleList[i].params)
            componentFitters.append(SpectrumFittingBackend(self.sampleList[i], self.projectList[i]))
        self.componentFitters = componentFitters
        manager = ControlsManager()
        # rFactorEvaluator = RFactor()
        # changeObserver = fitter  # PlotDataAggregator(fitter, OnValueChangedExperimentPlotter(spectra, "Tmp"))
        # manager.onExternalEval = MixtureConcentrationOptimizer(['HT', 'LT'], 'Tmp', [(0.01, 1), (0.01, 1)]).onExternalEval
        # manager.onPlotDataPostProcess = rFactorEvaluator.onPlotDataPostProcess
        # manager.onClickDelegate = RFactorMean('Tmp', spectra.labels, 'Eval Mean').onClicked

        # binding callbacks
        manager.onControlChangedDelegate = self.onControlChanged  # for when user changes slider, dropdown list etc.
        manager.onClickDelegate = self.onClick  # for button clicks
        manager.onInitDelegate = self.update  # for the first call

        # default params are values for parameters, can be dict or string - filepath
        # debug - whether or not show debug messages
        # plotter's class should have 3 functions: clear, getFig, plot
        manager.setup(builder.controls, builder.buildResult, debug=debug, defaultParams=defaultParams)

        self.manager = manager

    def onClick(self, context, name):
        pass

    def onControlChanged(self, context, name, oldValue, newValue):
        self.update(context)

    def saveAllData(self, folder):
        self.manager.saveAllData(folder)

    def update(self, context):
        predictedComponents = []
        concentrations = []
        for c in self.componentFitters:
            cParams = self.getParamsForComponent(context, c)
            c.calculatePrediction(cParams, context)
            predictedComponents.append(context.spectrumSmoothed)
            concentrations.append(cParams[CONCENTRATION_PARAM])
        concentrations = self.normalizeConcentration(concentrations)
        refEnergy = self.componentFitters[0].project.spectrum.energy
        aggregated = self.aggregateComponents(predictedComponents, concentrations, refEnergy)

        context.plotData[THEORY_PLOT_LABEL] = {
            'type': 'default',
            'xData': refEnergy,
            'yData': aggregated,
            'label': THEORY_PLOT_LABEL,
            'color': 'blue'
        }

        context.plotData[EXPERIMENT_PLOT_LABEL] = {
            'type': 'default',
            'xData': refEnergy,
            'yData': self.componentFitters[0].project.spectrum.intensity,
            'color': 'k',
            'label': EXPERIMENT_PLOT_LABEL
        }

        self.rFactor.refreshRFactor(context)

    def getParamsForComponent(self, context, component):
        params = FittingUtils.getParamsFromContext(context)
        filteredParams = {}
        for name, value in params.items():
            cName = component.project.name
            if name.startswith(cName):
                newName = self.renameComponentParam(name, cName)
                filteredParams[newName] = value

        filteredParams[METHOD_PARAM] = params[METHOD_PARAM]
        return filteredParams

    def getConvolution(self, prediction):
        for graph in prediction:
            if graph['label'] == THEORY_PLOT_LABEL:
                return graph

    def aggregateComponents(self, predictedComponents, concentrations, refEnergy):
        mixIntensities = [
            np.interp(refEnergy, predictedComponents[i].energy, predictedComponents[i].intensity) * concentrations[i]
            for i in range(len(predictedComponents))]
        mixIntensity = [sum(i) for i in zip(*mixIntensities)]
        return mixIntensity

    def renameComponentParam(self, name, componentName):
        if len(componentName) == len(name):
            return CONCENTRATION_PARAM
        else:
            return name[len(componentName) + 1:]

    def normalizeConcentration(self, concentrations):
        return np.array(concentrations) / sum(concentrations)


class PlotDataAggregator:
    def __init__(self, *args):
        self.sources = args

    def onControlChanged(self, context, name, oldValue, newValue):
        plotData = []
        for source in self.sources:
            plotData += source.onControlChanged(context, name, oldValue, newValue)

        return plotData


class OnValueChangedExperimentPlotter:
    def __init__(self, spectra, expSliderName):
        self.spectra = copy.deepcopy(spectra)
        self.expSliderName = expSliderName

    def onControlChanged(self, context, name, oldValue, newValue):
        spectrum = self.spectra.getSpectrumByLabel(context.controls[self.expSliderName].value)
        return [{
            'type': 'default',
            'xData': spectrum.energy,
            'yData': spectrum.intensity,
            'color': 'k',
            'label': EXPERIMENT_PLOT_LABEL
        }]


class OnValueChangedXanesFitter:
    def __init__(self, predictionBackend):
        self.predictionBackend = predictionBackend

    def onControlChanged(self, context, name, oldValue, newValue):
        return self.predictionBackend.getPlotData(FittingUtils.getParamsFromContext(context))


# TODO: get rid of it in favour of Cache
class CachedEstimator:
    def __init__(self, estimator):
        self.fitHash = None
        self.predictHash = None
        self.predictionCache = None
        self.estimator = estimator

    def fit(self, x, y):
        newHash = hash(x.data.tobytes()) ^ hash(y.data.tobytes())
        if newHash != self.fitHash:
            self.fitHash = newHash
            self.estimator.fit(x, y)

    def predict(self, x):
        newHash = hash(x.data.tobytes())
        if newHash != self.predictHash:
            self.predictHash = newHash
            self.predictionCache = self.estimator.predict(x)
        return self.predictionCache


# ====================================================
#                       FitSmooth
# ====================================================

class FitSmoothPlotter:
    def __init__(self):
        self.fig, self.ax = plotting.createfig(interactive=True)
        self.ax2 = None

    def clear(self):
        self.ax.clear()
        if self.ax2 is not None: self.ax2.clear()

    def plot(self, context):
        ax = self.ax
        plotData = context.plotData
        project = context.project
        params = FittingUtils.getParamsFromContext(context)
        metrics = ''
        for key, graph in plotData.items():
            if graph is None:
                continue
            if graph['type'] == 'default':
                if key == 'Smooth width':
                    if params['smooth width']:
                        if self.ax2 is None: self.ax2 = ax.twinx()
                        self.plotOn(self.ax2, graph)
                    else:
                        if self.ax2 is not None: self.ax2.remove(); self.ax2 = None
                else:
                    if key == 'initial':
                        if params['not convoluted']: self.plotOn(ax, graph)
                    else: self.plotOn(ax, graph)
            elif graph['type'] == 'metric':
                metrics += f'{ graph["label"] }: { graph["text"] }\n'
            else: assert False, 'Unknown graph type'
        
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
        if self.ax2 is not None: self.ax2.legend('upper left')
        self.ax.text(0.95, -0.05, metrics, transform=self.ax.transAxes, horizontalalignment='right', verticalalignment='bottom')

    def plotOn(self, ax, graph):
        ax.plot(
            graph['xData'],
            graph['yData'],
            label=graph['label'] if 'label' in graph else 'unknown',
            color=graph['color'] if 'color' in graph else None)

    def getFig(self):
        return self.fig


class fitSmooth:
    """Helps to choose smooth params. 
       extraSpectra - extra spectra to smooth and plot {'label':spectrum1, ...}
       extraGraphs - extra graphs to plot [{'x':.., 'y':.., 'label':..}, ...]
       bounds - new bounds for params {'Gamma_hole':[0.1,2], ...}
    """
    def __init__(self, spectrum, project, defaultParams=None, bounds=None, diffFrom=None, norm=None, smoothType='fdmnes', normType='multOnly', extraSpectra=None, extraGraphs=None, debug=True):
        defaultParams = FittingUtils.checkDefaultParams(defaultParams)
        self.spectrum = spectrum
        self.project = copy.deepcopy(project)
        self.diffFrom = diffFrom
        self.norm = norm
        self.smoothType = smoothType
        self.normType = normType
        self.extraSpectra = extraSpectra
        self.extraGraphs = extraGraphs
        self.manager = None
        self.bounds = {} if bounds is None else bounds
        self.setup(debug, defaultParams)

    def adjustFitIntervals(self, shift):
        e = np.sort(np.hstack((self.spectrum.energy+shift, self.project.spectrum.energy)).flatten())
        for fiName in ['plot', 'fit_norm', 'fit_smooth']:
            fi0 = self.project.intervals[fiName]
            fi = self.project.intervals[fiName]
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

    def setup(self, debug, defaultParams):
        builder = ControlsBuilder()
        manager = ControlsManager()
        project = self.project
        manager.context.project = project
        spectrum = self.spectrum
        exp_xanes = project.spectrum.intensity
        diffFrom = self.diffFrom
        if 'shift' in defaultParams: shift = defaultParams['shift']
        else: shift = project.defaultSmoothParams[self.smoothType]['shift']
        assert shift is not None
        self.adjustFitIntervals(shift)
        message, exp_efermi, theory_efermi = smoothLib.checkShift(project.spectrum, spectrum, shift, self.smoothType)
        if message == '': minShift = shift-10.0; maxShift = shift+10.0
        else:
            manager.addToStatus(message)
            minShift = np.min(project.spectrum.energy) - np.max(spectrum.energy)
            maxShift = np.max(project.spectrum.energy) - np.min(spectrum.energy)

        def addSlider(name, min, max, step, value):
            if name in self.bounds:
                min, max = self.bounds[name]
                value = max(value, min)
                value = min(value, max)
            builder.addSlider(name=name, min=min, max=max, step=step, value=value)
        addSlider(name='shift', min=minShift, max=maxShift, step=1, value=shift)
        addSlider(name='Gamma_hole', min=0.1, max=10, step=0.2, value=1)
        addSlider(name='Ecent', min=1, max=100, step=1, value=50)
        addSlider(name='Elarg', min=1, max=100, step=1, value=50)
        addSlider(name='Gamma_max', min=0, max=30, step=0.5, value=15)
        if self.smoothType == 'adf':
            # in adf smooth Efermi usually = eig energy[0], so parameter Efermi - is a shift from this value
            addSlider(name='Efermi', min=-20, max=20, step=1, value=0)
        else:
            addSlider(name='Efermi', min=np.min(project.spectrum.energy)-20, max=np.max(project.spectrum.energy), step=1, value=np.min(project.spectrum.energy))

        if 'norm' in defaultParams:
            nrm = defaultParams['norm']
            addSlider(name='norm', min=nrm/2, max=nrm*2, step=nrm/20, value=nrm)
        if diffFrom is not None:
            addSlider(name='purity', min=0, max=1, step=0.05, value=diffFrom['purity'])

        builder.addCheckbox(name='smooth width', value=False)
        builder.addCheckbox(name='not convoluted', value=True)
        builder.addButton(name='Start auto fit')
        e0 = project.spectrum.energy[0]-10; e1 = project.spectrum.energy[-1]+10
        v0 = project.intervals['plot'][0]
        v1 = project.intervals['plot'][1]
        # print(e0,e1,v0,v1,shift)
        if not(e0<=v0<=e1) or not(e0<=v1<=e1): # we have very bad shift
            e0 = min(e0,v0); v0 = e0
            e1 = max(e1,v1); v1 = e1
        builder.addRangeSlider(name='energyRange', min=e0, max=e1, value=[v0,v1], step=(e1-e0)/30)
        
        if diffFrom is None: ylim0=0;  ylim1 = np.max(exp_xanes)*1.2
        else:
            d = np.max(exp_xanes)-np.min(exp_xanes)
            ylim0 = np.min(exp_xanes)-0.2*d; ylim1 = np.max(exp_xanes)+0.2*d
        xanes_tmp = spectrum.intensity / np.mean(spectrum.intensity[-3:]) * np.mean(exp_xanes[-3:])
        builder.addRangeSlider(name='ylim', min=min([ylim0,np.min(xanes_tmp)]), max=max([ylim1, np.max(xanes_tmp)]), value=[ylim0, ylim1], step=(ylim1-ylim0)/10)

        # binding callbacks
        manager.onControlChangedDelegate = self.onControlChanged  # for when user changes slider, dropdown list etc.
        manager.onClickDelegate = self.onClick  # for button clicks
        manager.onInitDelegate = self.update  # for the first call

        # default params are values for parameters, can be dict or string - filepath
        # debug - whether or not show debug messages
        # plotter's class should have 3 functions: clear, getFig, plot
        manager.setup(builder.controls, builder.buildResult, debug=debug, defaultParams=defaultParams, plotter=FitSmoothPlotter())

        self.manager = manager

    def onClick(self, context, name):
        # here we handle all button clicks depending on the button name

        # resetting parameters to default
        if name == 'Start auto fit':
            params = FittingUtils.getParamsFromContext(context)
            project1 = copy.deepcopy(self.project)
            smoothParamNames = ['Gamma_hole', 'Ecent', 'Elarg', 'Efermi', 'Gamma_max', 'shift']
            commonParams0 = {}
            for pName in smoothParamNames:
                project1.defaultSmoothParams[self.smoothType].setAndExpandInterval(pName, params[pName])
                commonParams0[pName] = params[pName]
            optimParams = smoothLib.fitSmooth([project1], [self.spectrum], smoothType=self.smoothType, normType=self.normType, fixParamNames=[], commonParams0=commonParams0, targetFunc='l2(max)', optimizeWithoutPlot=True)
            for pName in smoothParamNames:
                FittingUtils.setParamForContext(context, pName, optimParams[pName])
            self.update(context)

    def smooth(self, extraSpectra, **params):
        project = self.project
        spectrum = self.spectrum
        smoothType = self.smoothType
        normType = self.normType
        exp_xanes = project.spectrum.intensity
        if 'norm' in params: norm = params['norm']
        else: norm = None
        xanes_sm, norm1 = smoothLib.smoothInterpNorm(params, spectrum, smoothType, project.spectrum, project.intervals['fit_norm'], norm, normType=normType)
        xanes_sm = xanes_sm.intensity
        e_last = project.intervals['fit_norm'][1]
        shift = params['shift']
        absorbPredictionNormalized = spectrum.intensity / spectrum.val(e_last-shift) * project.spectrum.val(e_last)
        extraSpectraSmoothed = {}
        if extraSpectra is not None:
            for sp in extraSpectra:
                extraSpectraSmoothed[sp] = smoothLib.smoothInterpNorm(params, extraSpectra[sp], smoothType, project.spectrum, project.intervals['fit_norm'], norm, normType=normType)[0]
        diffFrom = self.diffFrom
        if diffFrom is not None:
            smoothedXanesBase, _ = smoothLib.smoothInterpNorm(params, diffFrom['xanesBase'], smoothType, diffFrom['projectBase'].spectrum, project.intervals['fit_norm'], norm, normType=normType)
            smoothedXanesBase = smoothedXanesBase.intensity
            xanes_sm = (xanes_sm - smoothedXanesBase) * params['purity']
            absorbBaseNormalized,_ = diffFrom['xanesBase'].intensity  / np.mean(diffFrom['xanesBase'].intensity) * np.mean(exp_xanes[-3:])
            absorbPredictionNormalized = (absorbPredictionNormalized - absorbBaseNormalized)*params['purity']
        return absorbPredictionNormalized, xanes_sm, extraSpectraSmoothed, norm1

    def update(self, context):
        # using this method we can extract all parameters' values from context
        params = FittingUtils.getParamsFromContext(context)
        project = self.project
        for fi in ['plot', 'fit_norm', 'fit_smooth']:
            project.intervals[fi] = list(params['energyRange'])
        shift = params['shift']
        self.adjustFitIntervals(shift)
        absorbPredictionNormalized, smoothedPredictionNormalized, extraSpectraSmoothed, norm1 = self.smooth(self.extraSpectra, **params)
        if 'norm' not in params:
            if isinstance(norm1, dict):
                for norm_param in norm1:
                    context.plotData['norm_'+norm_param] = {'type': 'metric', 'label': 'norm_'+norm_param, 'text': '%.4g' % norm1[norm_param], 'value': norm1[norm_param]}
            else:
                context.plotData['norm'] = {'type': 'metric', 'label': 'norm', 'text': '%.4f' % norm1, 'value': norm1}
        spectrum = utils.Spectrum(self.spectrum.energy+shift, absorbPredictionNormalized, copy=True)
        #plotting
        exp_e = project.spectrum.energy
        exp_xanes = project.spectrum.intensity
        diffFrom = self.diffFrom
        if diffFrom is not None:
            expXanesBase = np.interp(exp_e, diffFrom['projectBase'].spectrum.energy, diffFrom['projectBase'].spectrum.intensity)
            exp_xanes = exp_xanes - expXanesBase
        spectrumSmoothed = utils.Spectrum(exp_e, smoothedPredictionNormalized, copy=True)
        expSpectrum = utils.Spectrum(exp_e, exp_xanes, copy=True)
        context.plotData['initial'] = {'type':'default', 'xData':spectrum.energy, 'yData':spectrum.intensity, 'label':'initial', 'color':'orange'}
        context.plotData['convolution'] = {'type':'default', 'xData':spectrumSmoothed.energy, 'yData':spectrumSmoothed.intensity, 'label':'convolution', 'color':'blue'}
        context.plotData['Experiment'] = {'type':'default', 'xData':expSpectrum.energy, 'yData':expSpectrum.intensity, 'label':'Experiment', 'color':'black'}
        for sp in extraSpectraSmoothed:
            context.plotData[sp] = {'type':'default', 'xData':exp_e, 'yData':extraSpectraSmoothed[sp].intensity, 'label':sp}
        extraGraphs = self.extraGraphs
        if extraGraphs is not None:
            for extraGraph in extraGraphs:
                context.plotData[extraGraph['label']] = {'type':'default', 'xData':extraGraph['x'], 'yData':extraGraph['y'], 'label':extraGraph['label']}
        
        if self.smoothType == 'fdmnes':
            smoothWidth = smoothLib.YvesWidth(exp_e, params['Gamma_hole'], params['Ecent'], params['Elarg'], params['Gamma_max'], params['Efermi'])
        elif self.smoothType == 'adf':
            smoothWidth = smoothLib.YvesWidth(exp_e-exp_e[0], params['Gamma_hole'], params['Ecent'], params['Elarg'], params['Gamma_max'], params['Efermi'])
        else: assert False, 'Unknown width'
        e_smoothWidth = exp_e # -exp_e[0]+spectrum.energy[0]+shift if self.smoothType == 'adf' else exp_e
        context.plotData['Smooth width'] = {'type':'default', 'xData':e_smoothWidth, 'yData':smoothWidth, 'label':'Smooth width', 'color':'red'}

        ind = (exp_e >= params['energyRange'][0]) & (exp_e <= params['energyRange'][1])
        denom = utils.integral(exp_e[ind], exp_xanes[ind]**2)
        if denom != 0: rFactor = utils.integral(exp_e[ind], (exp_xanes[ind] - smoothedPredictionNormalized[ind])**2) /  utils.integral(exp_e[ind], exp_xanes[ind]**2)
        else: rFactor = 0
        context.plotData['R-factor'] = {'type': 'metric', 'label': 'R-factor', 'text': '%.4f' % rFactor, 'value': rFactor }
        gc.collect()

    def onControlChanged(self, context, name, oldValue, newValue):
        self.update(context)

    def saveAllData(self, folder):
        self.manager.saveAllData(folder)


# ====================================================
#                ExtremaDescriptor
# ====================================================


class ExtremaDescriptor:
    def __init__(self, graphName, energyIntervalName):
        self.energyIntervalName = energyIntervalName
        self.graphName = graphName
        self.name_Degree = 'Poly Degree'

    def setup(self, builder):
        """

        :type builder: ControlsBuilder
        """
        builder.addSlider(self.name_Degree, 2, 10, 1, 2, type='i')

    def update(self, context):
        from pyfitit.descriptor import findExtremumByFit
        from pyfitit import Spectrum

        graph = context.plotData[self.graphName]
        expSpectrum = Spectrum(graph['xData'], graph['yData'])
        interval = FittingUtils.getParamFromContext(context, self.energyIntervalName)
        degree = FittingUtils.getParamFromContext(context, self.name_Degree)
        ds, poly = findExtremumByFit(expSpectrum, interval, degree, True)

        # queueing extrema and polynom for plotting
        x = ds[0]
        y = ds[1]

        context.plotData['Extrema'] = {
            'type': 'default',
            'xData': x,
            'yData': y,
            'label': 'Extrema',
            'color': 'red',
            'plotFunc': 'scatter'
        }

        x = np.linspace(interval[0], interval[1], num=500)
        y = poly(x)
        context.plotData['Poly'] = {
            'type': 'default',
            'xData': x,
            'yData': y,
            'label': 'Poly',
            'color': 'orange',
        }


# ====================================================
#                SampleInspector
# ====================================================

class SampleInspectorPlotter(DefaultPlotter):

    def __init__(self, customPlotter, name_CustomData, extraPlotter):
        super().__init__()
        self.customPlotter = customPlotter
        self.name_CustomData = name_CustomData
        self.extraPlotter = extraPlotter

    def plot(self, context):
        super().plot(context)
        for plot in context.plotData[self.name_CustomData]:
            self.customPlotter(self.ax[0], plot)
        if self.extraPlotter is not None:
            self.extraPlotter(self.ax[0])
        xlim = FittingUtils.getParamFromContext(context, 'xlim')
        self.ax[0].set_xlim(*xlim)
        self.ax[0].legend()


class SampleInspector:
    def __init__(self, spectra, debug=True, eachGraphProcessor=None, extraPlotter=None, defaultParams=None):
        """
        Show sample spectra by batches.

        :param spectra: DataFrame with each row - one spectrum. Energy values are contained in the column names 'e_value'. It can be constructed from numpy arrays by utils.makeDataFrame(energy, spectraMatrix)
        :param debug:
        :param eachGraphProcessor: dict with fields: customProcessor(spectrum, spectrumIndex), customPlotter(ax, customProcessorResult). These functions are called for each spectrum
        :param extraPlotter: function extraPlotter(ax)
        :param xlim: list of min and max x values on graphs
        """
        self.manager = None
        self.spectra = spectra
        self.spectraCount = len(self.spectra.index)
        self.energy = utils.getEnergy(spectra)

        self.name_Prev = 'Prev'
        self.name_Next = 'Next'
        self.name_SampleInfo = 'Sample Info'
        self.name_BatchNum = 'Batch Number'
        self.name_BatchSize = 'Batch Size'
        self.name_EnergyRange = 'xlim'
        self.name_Spectrum = 'Sample'
        self.name_CustomData = 'CustomData'
        if eachGraphProcessor is not None:
            self.customProcessor = eachGraphProcessor['customProcessor']
            self.customPlotter = eachGraphProcessor['customPlotter']
        else:
            self.customProcessor = None
            self.customPlotter = None
        self.extraPlotter = extraPlotter
        self.defaultParams = defaultParams if defaultParams is not None else {}
        self.setup(debug)

    def setup(self, debug):
        # building interface
        builder = ControlsBuilder()
        builder.addTextArea(name=self.name_SampleInfo, value='', disabled=True)
        builder.beginBox()
        builder.addButton(name=self.name_Prev)
        builder.addButton(name=self.name_Next)
        builder.endBox(type='h')
        v = self.defaultParams.get(self.name_BatchSize, 1)
        builder.addSlider(name=self.name_BatchSize, min=1, max=self.spectraCount, step=1, value=v, type='i')
        v = self.defaultParams.get(self.name_BatchNum, 1)
        builder.addSlider(name=self.name_BatchNum, min=1, max=self.spectraCount, step=1, value=v, type='i')

        min = np.min(self.energy)
        max = np.max(self.energy)
        v = self.defaultParams.get(self.name_EnergyRange, (min, max))
        builder.addRangeSlider(self.name_EnergyRange, v, min, max, 1)

        # binding callbacks
        manager = ControlsManager()
        manager.onControlChangedDelegate = self.onControlChanged  # for when user changes slider, dropdown list etc.
        manager.onClickDelegate = self.onClick  # for button clicks
        manager.onInitDelegate = self.update  # for the first call

        # default params are values for parameters, can be dict or string - filepath
        # debug - whether or not show debug messages
        # plotter's class should have 3 functions: clear, getFig, plot
        plotter = SampleInspectorPlotter(self.customPlotter, self.name_CustomData, self.extraPlotter)
        manager.setup(builder.controls, builder.buildResult, debug=debug, plotter=plotter)

        self.manager = manager

    def onClick(self, context, name):
        # here we handle all button clicks depending on the button name

        currentIndex = FittingUtils.getParamFromContext(context, self.name_BatchNum) - 1

        if name == self.name_Prev and currentIndex > 0:
            currentIndex -= 1

        if name == self.name_Next and currentIndex < self.spectraCount - 1:
            currentIndex += 1

        FittingUtils.setParamForContext(context, self.name_BatchNum, currentIndex + 1)
        self.update(context)

    def update(self, context):
        import math
        from pyfitit import Spectrum

        customData = []
        # recalculate batch index
        batchSize = FittingUtils.getParamFromContext(context, self.name_BatchSize)
        batchCount = math.ceil(self.spectraCount / batchSize)
        context.controls[self.name_BatchNum].max = batchCount
        batchIndex = FittingUtils.getParamFromContext(context, self.name_BatchNum) - 1

        energyRange = FittingUtils.getParamFromContext(context, self.name_EnergyRange)
        batchSpectraRange = self.getBatchIndices(batchIndex, batchSize)

        # clearing previous batch from cache
        for key in [x for x in context.plotData.keys() if x.startswith(self.name_Spectrum)]:
            del context.plotData[key]

        for i in batchSpectraRange:
            # taking spectrum at given index, limiting its energy
            x = self.energy
            ind = (x >= energyRange[0]) & (x <= energyRange[1])
            x = x[ind]
            y = self.spectra.iloc[i][ind]

            # here we save output data for later use
            # in this case it's plotting, which happens after all calculations are complete
            context.plotData[self.name_Spectrum + '_' + str(i)] = {
                'type': 'default',  # default type indicates this is an ordinary graph
                'xData': x,
                'yData': y,
                'label': str(i)
            }

            # print(FittingUtils.getParamsFromContext(context))
            if self.customProcessor is not None:
                data = self.customProcessor(Spectrum(x, y), i)
                customData.append(data)

        context.plotData[self.name_CustomData] = customData

        FittingUtils.setParamForContext(context, self.name_SampleInfo,
                                        f'Sample info:\n'
                                        f'Batch = {batchIndex + 1} / {batchCount}\n'
                                        f'Samples = {batchSpectraRange}')

        # in the end call plugin update, it uses the data we've set above

    def getBatchIndices(self, batchIndex, batchSize):
        left = batchIndex * batchSize
        right = left + batchSize
        return range(max(0, left), min(right, self.spectraCount))

    def onControlChanged(self, context, name, oldValue, newValue):
        """
        This function is called every time user changes value of slider or any other control in the Jupyter Notebook

        Note: this function won't be called when control is changed during processing of user input,
        e.g. when slider values are changed from the code, while handling user input
        This was made with a purpose of eliminating cascaded and looped calculations

        :param context: context, that contains all the data needed for calculations
        :param name: name of the slider or any other control, that changed its value
        :param oldValue: previous control value
        :param newValue: new control value
        """
        self.update(context)

    def saveAllData(self, folder):
        """
        Saves picture, parameters and plot data to files in user defined folder
        """
        self.manager.saveAllData(folder)


# ====================================================
#                        Example
# ====================================================

class SinCosPlotter:
    def __init__(self):
        self.fig, [self.axMain, self.axSnapshot] = plotting.createfig(interactive=True, nrows=2)

    def clear(self):
        self.axMain.clear()
        self.axSnapshot.clear()

    def plot(self, context):
        """
        Default plotter, outputs all gathered data to the end user (e.g. Jupyter Notebook)
        """
        plotData = context.plotData
        for key, graph in plotData.items():
            if graph is None:
                continue

            if key == 'SinCos-Snapshot':
                self.plotOn(self.axSnapshot, graph)

            if key == 'SinCos':
                self.plotOn(self.axMain, graph)

    def plotOn(self, ax, graph):
        ax.plot(
            graph['xData'],
            graph['yData'],
            label=graph['label'] if 'label' in graph else 'unknown',
            color=graph['color'] if 'color' in graph else None)
        ax.legend(loc='upper right')

    def getFig(self):
        return self.fig


class SinCosExample:
    def __init__(self, debug=True, defaultParams=None):
        defaultParams = FittingUtils.checkDefaultParams(defaultParams)

        self.boxName = 'Sin'
        self.bName = 'B'
        self.aName = 'A'
        self.bMaxName = 'B Max'

        self.manager = None
        self.setup(debug, defaultParams)

    def setup(self, debug, defaultParams):
        # building interface
        builder = ControlsBuilder()
        builder.addSlider(name='A', min=1, max=2, step=0.1, value=1.5)
        builder.addSlider(name='B', min=0, max=20, step=1, value=5)
        builder.addSlider(name='B Max', min=0, max=20, step=1, value=5)
        builder.addCheckbox(name='Sin', value=False)
        builder.addButton(name='Reset')
        builder.addButton(name='Clear')
        builder.addButton(name='Snapshot')

        # binding callbacks
        manager = ControlsManager()
        manager.onControlChangedDelegate = self.onControlChanged  # for when user changes slider, dropdown list etc.
        manager.onClickDelegate = self.onClick  # for button clicks
        manager.onInitDelegate = self.update  # for the first call

        # default params are values for parameters, can be dict or string - filepath
        # debug - whether or not show debug messages
        # plotter's class should have 3 functions: clear, getFig, plot
        manager.setup(builder.controls, builder.buildResult, debug=debug, defaultParams=defaultParams, plotter=SinCosPlotter())

        self.manager = manager

    def onClick(self, context, name):
        # here we handle all button clicks depending on the button name

        # resetting parameters to default
        if name == 'Reset':
            FittingUtils.setParamForContext(context, self.aName, 1.5)
            FittingUtils.setParamForContext(context, self.bName, 5)
            FittingUtils.setParamForContext(context, self.boxName, True)

        # clear all debug messages
        if name == 'Clear':
            context.setStatus('')

        # copying existing graph and changing color/label, effectively doing a snapshot
        if name == 'Snapshot':
            data = context.plotData['SinCos']
            context.plotData['SinCos-Snapshot'] = {
                'type': 'default',  # default type indicates this is an ordinary graph
                'xData': data['xData'],
                'yData': data['yData'],
                'label': 'snapshot',
                'color': 'black'
            }

    def update(self, context):
        # we can add any debug info using this method ("debug" should be True in order for this to work)
        # this message will tell us that our custom function was called
        context.addDebugToStatus('Update called')

        # using this method we can extract all parameters' values from context
        params = FittingUtils.getParamsFromContext(context)
        a = params[self.aName]
        b = params[self.bName]
        bMax = params[self.bMaxName]
        isSin = params[self.boxName]

        # note that since we're changing control's value inside update function
        # it won't trigger another update
        if b > bMax:
            b = bMax
            FittingUtils.setParamForContext(context, self.bName, bMax)

        x = np.linspace(0, 2 * np.pi, 500)

        # some heavy calculation
        def eval():
            # this message will tell us that our function is being evaluated
            context.addDebugToStatus('eval')
            return np.sin(a * x + b) if isSin else np.cos(a * x + b)

        # using cache to in order to save time
        # and not re-evaluate function when it doesn't depend on parameters
        # that have changed (since this method is called on any change of slider/checkbox etc.

        # it's very important to list each and every parameter that the function depends on (dependData array)
        # dataName should be unique for each function that uses cache
        y1 = context.cache.getFromCacheOrEval(dataName='y1', evalFunc=eval, dependData=np.array([a, b, isSin]))

        # here we save output data for later use
        # in this case it's plotting, which happens after all calculations are complete
        context.plotData['SinCos'] = {
            'type': 'default',  # default type indicates this is an ordinary graph
            'xData': x,
            'yData': y1,
            'label': 'Func',
            'color': 'orange'
        }

    def onControlChanged(self, context, name, oldValue, newValue):
        """
        This function is called every time user changes value of slider or any other control in the Jupyter Notebook

        Note: this function won't be called when control is changed during processing of user input,
        e.g. when slider values are changed from the code, while handling user input
        This was made with a purpose of eliminating cascaded and looped calculations

        :param context: context, that contains all the data needed for calculations
        :param name: name of the slider or any other control, that changed its value
        :param oldValue: previous control value
        :param newValue: new control value
        """
        self.update(context)

    def saveAllData(self, folder):
        """
        Saves picture, parameters and plot data to files in user defined folder
        """
        self.manager.saveAllData(folder)


# ====================================================
#      FDMNES Smooth based on FuncModel
# ====================================================

def smoothSliders(expSpectrum, theorySpectrum, shift='auto', smoothType='fdmnes', ylim='auto', defaultParams=None):
    """
    :param expSpectrum:
    :param theorySpectrum:
    :param smoothType:  'fdmnes' or 'adf'
    :param ylim: 'auto' or 'manual'
    """
    assert ylim in ['auto', 'manual']
    assert smoothType in ['fdmnes', 'adf']
    expEfermi = curveFitting.findEfermiFast(expSpectrum.energy, expSpectrum.intensity)
    if shift == 'auto':
        x = smoothLib.simpleSmooth(theorySpectrum.energy, theorySpectrum.intensity, sigma=5, assumeZeroInGaps=smoothType=='adf')
        theoryEfermi = curveFitting.findEfermiFast(theorySpectrum.energy, x)
        shift0 = expEfermi-theoryEfermi
    else:
        shift0 = shift

    def smoothFunction(slf, params):
        slf.data['exp'] = FuncModel.createDataItem('plot', x=expSpectrum.energy, y=expSpectrum.intensity, order=-1, color='black', lw=2)
        fitNormInterval = params['fitNormInterval']
        smoothed, norm = smoothLib.smoothInterpNorm(params, theorySpectrum, smoothType, expSpectrum, fitNormInterval, norm=None)
        shift = params['shift']
        slf.data['smoothed'] = FuncModel.createDataItem('plot', x=smoothed.energy, y=smoothed.intensity)
        slf.data['not smoothed'] = FuncModel.createDataItem('plot', x=theorySpectrum.energy + shift, y=theorySpectrum.intensity / norm, plot=params['not smoothed'])

        if smoothType == 'fdmnes':
            smoothWidth = smoothLib.YvesWidth(expSpectrum.energy, params['Gamma_hole'], params['Ecent'], params['Elarg'], params['Gamma_max'], params['Efermi'])
        elif smoothType == 'adf':
            smoothWidth = smoothLib.YvesWidth(expSpectrum.energy-expSpectrum.energy[0], params['Gamma_hole'], params['Ecent'], params['Elarg'], params['Gamma_max'], params['Efermi'])
        else: assert False, 'Unknown width'
        if hasattr(slf, 'ax2'):
            if not params['smooth width']:
                slf.ax2.remove()
                del slf.ax2
            else: slf.ax2.clear()
        def plotWidth(ax):
            if not params['smooth width']: return
            if not hasattr(slf, 'ax2'): slf.ax2 = ax.twinx()
            slf.ax2.clear()
            slf.ax2.plot(expSpectrum.energy, smoothWidth, color='red')
        slf.data['smooth width for plotting'] = FuncModel.createDataItem('custom', plotter=plotWidth, save=False)
        slf.data['smooth width'] = FuncModel.createDataItem('plot', x=expSpectrum.energy, y=smoothWidth, plot=False)

        slf.data['fitNormInterval1'] = FuncModel.createDataItem('text', transform=None, x=fitNormInterval[0], y=0, str='[', save=False)
        slf.data['fitNormInterval2'] = FuncModel.createDataItem('text', transform=None, x=fitNormInterval[1], y=0, str=']', save=False)

        energyRange = params['energyRange']
        def xlim(ax):
            ax.set_xlim(energyRange)
            if ylim == 'auto':
                plotting.updateYLim(ax)
        slf.data['xlim'] = FuncModel.createDataItem('custom', plotter=xlim, save=False, order=1000)
        if ylim == 'manual':
            def setylim(ax):
                ax.set_ylim(params['ylim'])
            slf.data['ylim'] = FuncModel.createDataItem('custom', plotter=setylim, save=False, order=1001)
        error = utils.rFactorSp(smoothed, expSpectrum, sub1=True, interval=energyRange)
        slf.data['info'] = FuncModel.createDataItem('text', str='norm=%.3g' % norm + '  err=%.3g' % error, order=1002)
        return error

    paramProperties = ParamProperties()
    dsp = smoothLib.DefaultSmoothParams(expEfermi, shift0)
    for p in dsp.params['fdmnes']:
        paramProperties[p['paramName']] = {'type': 'float', 'domain': [p['leftBorder'], p['rightBorder']], 'default': p['value']}
    paramProperties['energyRange'] = {'type':'range', 'domain':[expSpectrum.energy[0],expSpectrum.energy[-1]]}
    paramProperties['fitNormInterval'] = {'type': 'range', 'domain': [expSpectrum.energy[0], expSpectrum.energy[-1]]}
    paramProperties['not smoothed'] = {'type': 'bool', 'default': True}
    paramProperties['smooth width'] = {'type': 'bool', 'default': True}
    if ylim != 'auto':
        tmp_norm = theorySpectrum.intensity[-1]/expSpectrum.intensity[-1]
        paramProperties['ylim'] = {'type': 'range', 'domain': [min(np.min(theorySpectrum.intensity),np.min(expSpectrum.intensity)), max(np.max(theorySpectrum.intensity/tmp_norm),np.max(expSpectrum.intensity))]}
        print(paramProperties['ylim']['domain'])
    funcModel = FuncModel(function=smoothFunction, paramProperties=paramProperties)
    return FuncModelSliders(funcModel, defaultParams=defaultParams)


def addNoFit(paramName, paramProperties, defaultParams):
    if paramName in paramProperties: names = [paramName]
    else:
        names = []
        for p in paramProperties:
            if p[:len(paramName) + 1] == paramName+ '_': names.append(p)
    for n in names:
        fn = f'fit {n}'
        if fn not in defaultParams: defaultParams[fn] = False


def xanesSliders(project, xanesSampleFolder, defaultParams=None):
    xanesModel = FuncModel.createXanesFittingModel(project, ML.readSample(xanesSampleFolder), name=project.name)
    if defaultParams is None: defaultParams = {}
    for p in ['Gamma_hole', 'Gamma_max', 'Ecent', 'Efermi', 'Elarg', 'shift', 'norm']:
        addNoFit(p, xanesModel.paramProperties, defaultParams)
    return FuncModelSliders(funcModel=xanesModel, debug=False, defaultParams=defaultParams)

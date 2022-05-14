from . import utils
utils.fixDisplayError()
import numpy as np
import copy, os, json, io, matplotlib
from . import plotting, ML, smoothLib, inverseMethod, optimize
from types import MethodType


class ParamProperties(dict):
    def __init__(self):
        """
        dict {paramName -> {type:{'float','int','range','bool','list'}, domain:([a,b],list)}, default:val}. Default is optional
        self.constrains - list of constrains. Example: {'type':'linear', 'lowerBound':0, 'upperBound':1, 'paramMultipliers':{paramName:mult,...}. Bounds and multipliers can be columns.
        """
        super().__init__()
        self.constrains = []

    def getMiddle(self):
        assert len(self.constrains) == 0, 'TODO - check constrains for middle and throw exception'
        res = {}
        for pn, pp in self.items():
            if pp['type'] == 'float': res[pn] = np.mean(pp['domain'])
            elif pp['type'] == 'int': res[pn] = int(np.mean(pp['domain']))
            elif pp['type'] == 'range': res[pn] = pp['domain']
            elif pp['type'] == 'bool': res[pn] = True
            elif pp['type'] == 'list': res[pn] = pp['domain'][len(pp['domain'])//2]
            else: assert False, 'Unknown param type'
        return res

    def getRandom(self, fittedParams=None, rng=None):
        assert len(self.constrains) == 0, 'TODO: generate random in a loop until constrains are met'
        if fittedParams is None: fittedParams = list(self.keys())
        if rng is None: rng = np.random.default_rng(seed=None)
        res = {}

        def getFloat(rng, domain):
            f = rng.random()
            a, b = domain
            return a + f * (b - a)

        for pn, pp in self.items():
            if pn not in fittedParams: continue
            if pp['type'] == 'float': res[pn] = getFloat(rng, pp['domain'])
            elif pp['type'] == 'int':
                a, b = pp['domain']
                res[pn] = rng.integers(low=a, high=b)
            elif pp['type'] == 'range':
                x1 = getFloat(rng, pp['domain'])
                x2 = getFloat(rng, pp['domain'])
                res[pn] = [min(x1,x2), max(x1,x2)]
            elif pp['type'] == 'bool': res[pn] = rng.random()<0.5
            elif pp['type'] == 'list':
                i = rng.integers(low=0, high=len(pp['domain']))
                res[pn] = pp['domain'][i]
            else: assert False, 'Unknown param type'
        return res

    def getFloatParamNames(self):
        return sorted([pn for pn in self if self[pn]['type'] == 'float'])

    def unionWith(self, other, conflictResolver='throw exception'):
        """
        Add param properties from other.
        :param other:
        :param conflictResolver: 'throw exception', 'intersect domains'
        """
        for name, pp_other in other.items():
            if name in self:
                if conflictResolver == 'intersect domains':
                    pp_self = self[name]
                    assert pp_self['type'] == pp_other['type']
                    if 'default' not in pp_self and 'default' in pp_other:
                        pp_self['default'] = pp_other['default']
                    if pp_self['domain'] != pp_other['domain']:
                        if pp_self['type'] in ['int', 'float', 'range']:
                            assert isinstance(pp_self['domain'], list) and len(pp_self['domain']) == 2
                            assert isinstance(pp_other['domain'], list) and len(pp_other['domain']) == 2
                            a1, b1 = pp_self['domain']
                            a2, b2 = pp_other['domain']
                            pp_self['domain'] = [max(a1, a2), min(b1, b2)]
                        else:
                            assert pp_self['type'] == 'list'
                            pp_self['domain'] = list(set(pp_self['domain']) & set(pp_other['domain']))
                else:
                    assert False, f"Duplicate param name {name}"
            else: self[name] = pp_other
        self.constrains = self.constrains + other.constrains


class FuncModel:
    def __init__(self, name='', function=None, paramProperties=None, createfigParams=None, figureSettings=None, userPlotFunc=None, userSaveFunc=None):
        """
        Class with function, used for fitting. Function while evaluating add data to be plotted and saved. self.data - dictionary with key - dataItem name, possible values (for default plotter and saver):
            curve {'type':plot', 'save':True, 'plot':True, 'axisInd':0, 'x':None, 'y':None, 'kwargs':{}}
            points {'type':'scatter', 'save':True, 'plot':True, 'axisInd':0, 'x':None, 'y':None, 'kwargs':{}}
            text {'type':typ, 'save':True, 'plot':True, 'axisInd':0, 'x':0.98, 'y':0.02, 's':'text', 'transform':'ax.transAxes', 'horizontalalignment':'right', 'verticalalignment':'bottom', 'kwargs':{}}
        :param name: useful for those, who works with many Models
        :param function: function(self, paramDict). Has access to the current class. Better not to change any global variables, because in the future the model will be copied and copies will change the common variable - it is not good. Use cache from self.cache
        :param paramProperties: ParamProperties instance
        :param figureSettings: array of dict with setFigureSettings arguments for each axis
        :param userPlotFunc: plotMoreFunction(self), plot instead of default plotter. Has access to the current class. It is recommended to use custom data items with user defined plotter instead of userPlotFunc to enable combining of FuncModels
        :param userSaveFunc: saveMoreFunction(self), save instead of default saver. Has access to the current class. It is recommended to use custom data items with user defined str to save instead of userSaveFunc to enable combining of FuncModels
        """
        self.name = name
        self.function = MethodType(function, self) if function is not None else None
        assert isinstance(paramProperties, ParamProperties)
        self.paramProperties = copy.deepcopy(paramProperties)
        self.cache = utils.CacheInMemory()
        self.data = {}  # for saving to file and plotting
        self.params = None  # current params values to evaluate function
        self.userPlotFunc = MethodType(userPlotFunc, self) if userPlotFunc is not None else None
        self.userSaveFunc = MethodType(userSaveFunc, self) if userSaveFunc is not None else None
        self.fig, self.ax = None, None
        self.createfigParams = copy.deepcopy(createfigParams) if createfigParams is not None else {}
        if 'subplotpars' not in self.createfigParams:
            self.createfigParams['subplotpars'] = matplotlib.figure.SubplotParams(top=0.85)
        self.figureSettings = figureSettings
        self.value = None

    def copy(self):
        func = self.function.__func__ if self.function is not None else None
        userPlotFunc = self.userPlotFunc.__func__ if self.userPlotFunc is not None else None
        userSaveFunc = self.userSaveFunc.__func__ if self.userSaveFunc is not None else None
        res = FuncModel(name=self.name, function=func, paramProperties=self.paramProperties, createfigParams=self.createfigParams, figureSettings=self.figureSettings, userPlotFunc=userPlotFunc, userSaveFunc=userSaveFunc)
        res.params = copy.deepcopy(self.params)
        res.value = copy.deepcopy(self.value)
        return res

    def createFig(self):
        assert self.fig is None
        self.fig, self.ax = plotting.createfig(**self.createfigParams)
        # make ax one dimensional array for uniform treatment
        if not isinstance(self.ax, np.ndarray): self.ax = np.array([self.ax])
        if len(self.ax.shape) == 2: self.ax = self.ax.flatten()
        if not isinstance(self.figureSettings, list): self.figureSettings = [self.figureSettings] * len(self.ax)

    # def __del__(self):
        # thows exception AttributeError: 'FuncModel' object has no attribute 'fig'
        # if self.fig is not None: plotting.closefig(self.fig)

    def evaluate(self, params):
        """
        Evaluate model for given dict params
        """
        if self.params is not None:
            # assert len(self.params) == len(params), str(self.params)+' != '+str(params)  - breaks combination of models
            # assert set(self.params.keys()) == set(params.keys()), str(self.params)+' != '+str(params)
            for pn in self.params:
                self.params[pn] = copy.deepcopy(params[pn])
        else: self.params = copy.deepcopy(params)
        for pn in set(self.params.keys())-set(self.paramProperties.keys()):
            del self.params[pn]
        self.value = self.function(self.params) if self.function is not None else None
        return self.value

    @staticmethod
    def defaultDataItem(typ):
        # text "[label]" in filePostfix is replaced by label
        res = {'type':typ, 'save':True, 'plot':True, 'axisInd':0, 'order':0, 'filePostfix':''}
        if typ == 'plot': res = {**res, 'x':None, 'y':None, 'kwargs':{}}
        elif typ == 'scatter': res = {**res, 'x':None, 'y':None, 'kwargs':{}}
        elif typ == 'text': res = {**res, 'x':0.98, 'y':0.02, 'str':'', 'transform':'ax.transAxes', 'horizontalalignment':'right', 'verticalalignment':'bottom', 'kwargs':{}}
        elif typ == 'lazy': res = {**res, 'generator':None}  # generator() -> dataItem
        elif typ == 'custom': res = {**res, 'str':'', 'plotter':None}
        return res

    @staticmethod
    def createDataItem(typ, **p):
        """
        All parameters that are not in default data item for this type are put in 'kwargs' automatically
        """
        res0 = FuncModel.defaultDataItem(typ)
        if 'kwargs' in res0:
            res = copy.deepcopy(res0)
            for pn in res0:
                if pn in p: res[pn] = p[pn]
            # fill kwargs by items in p, that are not in res0
            p_kwargs = set(p.keys()) - set(res0.keys())
            for pn in p_kwargs: res['kwargs'][pn] = p[pn]
        else:
            res = res0
            for pn in p:
                res[pn] = p[pn]
        return res

    @staticmethod
    def evaluateLazyDataItems(data):
        res = {}
        for name in data:
            dataItem = data[name]
            if dataItem['type'] == 'lazy': dataItem = dataItem['generator']()
            res[name] = dataItem
        return res

    def plotData(self):
        data = FuncModel.evaluateLazyDataItems(self.data)
        plotCounts = {ax:0 for ax in self.ax}
        items = sorted(data.items(), key=lambda o: o[1]['order'])
        for name, dataItem in items:
            if not dataItem['plot']: continue
            kwargs = copy.deepcopy(dataItem['kwargs']) if 'kwargs' in dataItem else {}
            if dataItem['type'] in ['plot', 'scatter']:
                if 'label' not in kwargs: kwargs['label'] = name
            axisInd = dataItem['axisInd']
            ax = self.ax[axisInd]
            if dataItem['type'] == 'plot':
                ax.plot(dataItem['x'], dataItem['y'], **kwargs)
                plotCounts[ax] += 1
            elif dataItem['type'] == 'scatter':
                ax.scatter(dataItem['x'], dataItem['y'], **kwargs)
                plotCounts[ax] += 1
            elif dataItem['type'] == 'text':
                assert dataItem['transform'] in ['ax.transAxes', None]
                # transform=None is not the default ax.text argument, we have to use if and copy command
                if dataItem['transform'] == 'ax.transAxes':
                    ax.text(dataItem['x'], dataItem['y'], dataItem['str'], transform=ax.transAxes, ha=dataItem['horizontalalignment'], va=dataItem['verticalalignment'], **kwargs)
                else:
                    ax.text(dataItem['x'], dataItem['y'], dataItem['str'], ha=dataItem['horizontalalignment'], va=dataItem['verticalalignment'], **kwargs)
            elif dataItem['type'] == 'custom':
                dataItem['plotter'](ax)
        for i, ax in enumerate(self.ax):
            if self.figureSettings[i] is not None:
                plotting.setFigureSettings(ax, **self.figureSettings[i])
            if plotCounts[ax] > 0: ax.legend(loc='upper right')

    def plotTitle(self):
        title = self.name
        if self.function is not None and self.value is not None:
            if title == '': title = 'value'
            if isinstance(self.value, float): title += " = %.3g" % self.value
            else: title += " = " + str(self.value)
        if self.params is not None:
            title += ' ' + FuncModel.params2str(self.params)
        self.fig.suptitle(utils.wrap(title, 80, maxLineCount=3))

    def defaultPlotter(self):
        if self.fig is None: self.createFig()
        self.plotData()
        self.plotTitle()

    def plot(self):
        if self.fig is not None:
            for ax in self.ax: ax.clear()
        if self.userPlotFunc is None:
            self.defaultPlotter()
        else:
            self.userPlotFunc()

    def saveFigure(self, fileName):
        if self.fig is None: return
        plotting.savefig(fileName=fileName, fig=self.fig)
        plotting.closefig(self.fig)  # we have to close figure here, because __del__ doesn't work

    @staticmethod
    def params2str(params):
        return utils.dictToStr(params)

    @staticmethod
    def data2str(data):
        f = io.StringIO()
        items = sorted(data.items(), key=lambda o: o[1]['order'])
        for name, dataItem in items:
            if not dataItem['save']: continue
            if dataItem['type'] in ['plot', 'scatter']:
                f.write(f'{name} x: ')
                np.savetxt(f, [dataItem['x']], delimiter=',')
                f.write(f'{name} y: ')
                np.savetxt(f, [dataItem['y']], delimiter=',')
            elif dataItem['type'] in ['text', 'custom']:
                if len(data) > 1: f.write(f'{name}: ')
                f.write(dataItem['str'])
                if len(data) > 1: f.write('\n')
            else: f.write(json.dumps(dataItem, cls=utils.NumpyEncoder))
        f.seek(0)
        return f.read()

    @staticmethod
    def aggregateSavedDataByFile(data):
        res = {}
        for name, dataItem in data.items():
            if not dataItem['save']: continue
            fn = dataItem['filePostfix'].replace('[label]', name)
            if fn not in res: res[fn] = {}
            res[fn][name] = dataItem
        return res

    def saveData(self, fileName):
        folder = os.path.split(fileName)[0]
        if folder == '': folder = '.'
        os.makedirs(folder, exist_ok=True)
        ext = os.path.splitext(fileName)[-1]
        fn = os.path.splitext(fileName)[0]
        assert ext in ['.txt', '.json', '.pkl']
        data = FuncModel.evaluateLazyDataItems(self.data)
        aggData = FuncModel.aggregateSavedDataByFile(data)
        for filePostfix in aggData:
            fileName1 = f'{fn}_{filePostfix}' if filePostfix!='' else fileName
            if ext == '.txt':
                with open(fileName1, 'w') as f:
                    if filePostfix == '':
                        if self.params is not None:
                            f.write(FuncModel.params2str(self.params)+'\n')
                        if self.value is not None:
                            f.write(f'value={self.value}\n')
                    f.write(FuncModel.data2str(aggData[filePostfix]))
            else: utils.saveData(aggData[filePostfix], fileName1)

    def defaultSaver(self, fileName):
        if self.userSaveFunc is None:
            self.saveData(fileName)
        else:
            self.userSaveFunc(fileName)
        self.saveFigure(os.path.splitext(fileName)[0]+'.png')

    def save(self, fileName): self.defaultSaver(fileName)

    def setFunction(self, func):
        self.function = MethodType(func, self)

    def push_front(self, func, valueChooser=None):
        assert False, "TODO: implement pushing of FuncModel with its own paramProperties"
        def new_func(slf, params):
            val1 = func(slf, params)
            val2 = self.function.__func__(slf, params)
            if valueChooser is None:
                if val2 is None: return val1
                return val2
            else: return valueChooser(val1, val2)
        if self.function is not None:
            self.setFunction(new_func)
        else:
            self.setFunction(func)

    def push_back(self, func, valueChooser=None):
        assert False, "TODO: implement pushing of FuncModel with its own paramProperties"
        def new_func(slf, params):
            val1 = self.function.__func__(slf, params)
            val2 = func(slf, params)
            if valueChooser is None:
                if val2 is None: return val1
                return val2
            else: return valueChooser(val1, val2)
        if self.function is not None:
            self.setFunction(new_func)
        else:
            self.setFunction(func)

    def addParamsGenerator(self, transform, newParamProperties):
        assert self.function is not None
        self.setFunction(lambda slf, params: self.function.__func__(slf, transform(slf, params)))
        self.paramProperties = newParamProperties

    def fixParams(self, toFixParamDict):
        oldFunc = self.function.__func__

        def newFunc(slf, params):
            params = copy.deepcopy(params)
            for pn in toFixParamDict: params[pn] = toFixParamDict[pn]
            return oldFunc(slf, params)
        for pn in toFixParamDict:
            del self.paramProperties[pn]
        self.setFunction(newFunc)

    def check(self, fileName):
        p = self.paramProperties.getMiddle()
        print('params =',p)
        val = self.evaluate(p)
        print('Value =', val)
        self.plot()
        self.save(fileName)

    def combineWith(self, other, newName=None, valueMergeRule=None, dataMergeRule=None, merger=None, createfigParams=None):
        """
        Constructs combination of self and other models. You should specify valueMergeRule and dataMergeRule or merger. Parameters with common names become common. Data with common keys is renamed

        :param other: other FuncModel
        :param newName: name of new model
        :param valueMergeRule: list with 2 weights or 'make tuple'
        :param dataMergeRule: str - 'add axes', 'plot on 0', 'plot on 1', ...
        :param merger: new function, merger(self, other) -> new value (self and other are already evaluated!)
        :param createfigParams: new createfigParams
        :returns: new combined FuncModel
        """
        assert dataMergeRule == 'add axes' or dataMergeRule[:8] == 'plot on '
        assert (merger is None and valueMergeRule is not None and dataMergeRule is not None) or (merger is not None and valueMergeRule is None and dataMergeRule is None)
        old = self.copy()
        other = other.copy()

        def newFunc(slf, params):
            oldValue = old.evaluate(params)
            otherValue = other.evaluate(params)
            if merger is None:
                slf.data = copy.deepcopy(old.data)
                oldAxesCount = np.max([di['axisInd'] for _, di in old.data.items()]) + 1
                for name, dataItem in other.data.items():
                    if dataMergeRule == 'add axes':
                        dataItem['axisInd'] += oldAxesCount
                    else:
                        ind = int(dataMergeRule[8:])
                        dataItem['axisInd'] = ind
                for name in other.data:
                    if name in slf.data:
                        assert old.name != other.name, 'Use different names for combined FuncModels because you have same data keys'
                        oldDataItem = slf.data[name]
                        del slf.data[name]
                        slf.data[old.name +' '+ name] = oldDataItem
                        slf.data[other.name +' '+ name] = other.data[name]
                    else: slf.data[name] = other.data[name]
            else: merger(old, other)
            if valueMergeRule == 'make tuple': return oldValue, otherValue
            else: return oldValue*valueMergeRule[0] + otherValue*valueMergeRule[1]

        newUserPlotFunc = None if old.userPlotFunc is None else old.userPlotFunc.__func__
        newUserSaveFunc = None if old.userSaveFunc is None else old.userSaveFunc.__func__
        createfigParams = self.createfigParams if createfigParams is None else createfigParams
        paramProperties = copy.deepcopy(old.paramProperties)
        paramProperties.unionWith(other.paramProperties, conflictResolver='intersect domains')
        if newName is None: newName = f'Combination of {old.name} and {other.name}'
        res = FuncModel(newName, function=newFunc, paramProperties=paramProperties, createfigParams=createfigParams, figureSettings=old.figureSettings, userPlotFunc=newUserPlotFunc, userSaveFunc=newUserSaveFunc)
        return res

    @staticmethod
    def createXanesFittingModel(project, sample, distToExperiment=None, smooth=True, additionalFunc=None, **kwargs):
        """
        Returns FuncModel for xanes fitting, paramProperties, fixedParams(sample, expSpectrum), defaultParams(smooth and ML method) to use in FuncModelSliders

        :param project:
        :param sample:
        :param distToExperiment: func(theorySp, expSp, params) -> distance to experiment
        :param smooth:
        :param additionalFunc: function(FuncModel, params) to call in the end of FuncModel function. If returns not None, FuncModel returned value replaced by additionalFunc value
        :param kwargs: arguments for FuncModel constructor (except function)
        """
        expSpectrum = project.spectrum

        def xanesPredictorFunc(slf, params):
            energyRange = params['energyRange']
            slf.data['exp'] = FuncModel.createDataItem('plot', x=expSpectrum.energy, y=expSpectrum.intensity, order=-1, color='black', lw=2)
            methodName = params['method']

            def getFittedEstimator():
                method = inverseMethod.getMethod(methodName)
                estimator = ML.Normalize(method, xOnly=False)
                estimator.fit(sample.params.to_numpy(), sample.spectra.to_numpy())
                return estimator
            estimator = slf.cache.getFromCacheOrEval(dataName='estimator', evalFunc=getFittedEstimator, dependData=[methodName])
            geomParams = [params[pn] for pn in sample.paramNames]
            sp = slf.cache.getFromCacheOrEval(dataName='prediction', evalFunc=lambda: utils.Spectrum(sample.energy, estimator.predict(np.array(geomParams).reshape(1, -1)).reshape(-1)), dependData=[methodName, *geomParams])
            if smooth:
                shift = params['shift']
                smoothed, norm = smoothLib.smoothInterpNorm(smoothParams=params, spectrum=sp, smoothType='fdmnes', expSpectrum=expSpectrum, fitNormInterval=energyRange)
                slf.data['not smoothed'] = FuncModel.createDataItem('plot', x=sp.energy + shift, y=sp.intensity / norm, plot=params['not smoothed'])
            else: smoothed = utils.Spectrum(expSpectrum.energy, np.interp(expSpectrum.energy, sp.energy, sp.intensity))
            slf.data['theory'] = FuncModel.createDataItem('plot', x=smoothed.energy, y=smoothed.intensity, order=1)

            geomParamsDict = {pn:params[pn] for pn in sample.paramNames}
            if hasattr(project, 'moleculeConstructor') and project.moleculeConstructor is not None:
                slf.data['molecule'] = FuncModel.createDataItem('lazy', generator=lambda: FuncModel.createDataItem('text', str=project.moleculeConstructor(geomParamsDict).export_xyz_string(), filePostfix='[label].xyz', plot=False))

            def xlim(ax):
                ax.set_xlim(energyRange)
                plotting.updateYLim(ax)
            slf.data['xlim'] = FuncModel.createDataItem('custom', plotter=xlim, save=False, order=1000)
            if distToExperiment is None:
                error = utils.rFactorSp(smoothed, expSpectrum, p=1, sub1=True, interval=energyRange)
            else:
                error = distToExperiment(smoothed, expSpectrum, params)
            slf.data['error'] = FuncModel.createDataItem('text', str='err = %.3g' % error, order=1001)
            if additionalFunc is not None:
                val = additionalFunc(slf, params)
                if val is not None: return val
            return error

        paramProperties = ParamProperties()
        for pn, interval in project.geometryParamRanges.items():
            paramProperties[pn] = {'type':'float', 'domain':interval}
        # add smooth params
        if smooth:
            dsp = smoothLib.DefaultSmoothParams(project.FDMNES_smooth['Efermi'], project.FDMNES_smooth['shift'])
            for p in dsp.params['fdmnes']:
                paramProperties[p['paramName']] = {'type':'float', 'domain':[p['leftBorder'], p['rightBorder']], 'default':project.FDMNES_smooth[p['paramName']]}
            if 'norm' in project.FDMNES_smooth:
                norm = project.FDMNES_smooth['norm']
                paramProperties['norm'] = {'type':'float', 'domain':[norm*0.5, norm*2], 'default':norm}
        paramProperties['energyRange'] = {'type':'range', 'domain':[expSpectrum.energy[0], expSpectrum.energy[-1]]}
        paramProperties['method'] = {'type': 'list', 'domain': inverseMethod.allowedMethods, 'default':'RBF'}
        paramProperties['not smoothed'] = {'type': 'bool', 'default': True}
        funcModel = FuncModel(function=xanesPredictorFunc, paramProperties=paramProperties, **kwargs)
        return funcModel

    @staticmethod
    def createExafsFittingModel(sampleFolder, expSpectrum, moleculeConstructor=None, multipleS02=False, kPower=2, RSpaceParams={'kmin':2, 'kmax':10}, exafsParamsFuncArgBounds={'enot':[-10,10], 'S02':[0, 2], 'sigma2':[0.001,0.04]}):
        from . import msPathGroupManager
        exafsPredictor = msPathGroupManager.ExafsPredictor(sampleFolder)
        exafsPredictor.fit()
        exafsParamsFuncArgBounds = copy.deepcopy(exafsParamsFuncArgBounds)
        if multipleS02:
            for g in exafsPredictor.pathGroups: exafsParamsFuncArgBounds[f'S02_{g}'] = exafsParamsFuncArgBounds['S02']
            del exafsParamsFuncArgBounds['S02']
        for g in exafsPredictor.pathGroups:
            exafsParamsFuncArgBounds[f'sigma2_{g}'] = exafsParamsFuncArgBounds['sigma2']
        del exafsParamsFuncArgBounds['sigma2']
        geomParamBounds = utils.loadData(sampleFolder + os.sep + 'dataset' + os.sep + 'paramRanges.json')
        exafsModel = exafsPredictor.getFuncModel(expSpectrum=expSpectrum, geomParamBounds=geomParamBounds, kPower=kPower, exafsParamsFuncArgBounds=exafsParamsFuncArgBounds, multipleS02=multipleS02, RSpaceParams=RSpaceParams, name='EXAFS', moleculeConstructor=moleculeConstructor)
        return exafsModel

    @staticmethod
    def makeMixture(funcModelList, expDataItemNames='exp', distToExperiment=None, mixDataItemNames=None, errorDataItemNames=None, plotIndividual=False, commonParamNames=None):
        """
        Makes mixture of uniform func models. Parameters of models are prepended with model names. Concentration parameters C1,C2,... are appended.

        :param funcModelList:
        :param expDataItemNames: exp data item name. Str or list corresponding to mixDataItemNames
        :param distToExperiment: func(theorySp, expSp, params) -> distance to experiment or list of such funcs corresponding to mixDataItemNames
        :param mixDataItemNames: str or list of dataItem names to make mixtures
        :param errorDataItemNames: str or list of dataItem names containing function values (rFactor or relative error)
        :param plotIndividual: plot individual components or not
        :param commonParamNames: list of common params of all the models in mixture
        :returns: new mixture func model
        """
        assert mixDataItemNames is not None
        if distToExperiment is None: distToExperiment = lambda theorySp, expSp, params: utils.rFactorSp(theorySp, expSp)
        if commonParamNames is None: commonParamNames = []
        funcModelList = [m.copy() for m in funcModelList]
        componentNames = np.array([m.name for m in funcModelList], dtype=object)
        for i in range(len(componentNames)):
            cn = componentNames[i]
            if np.sum(componentNames == cn) > 1:
                for k,ind in enumerate(np.where(componentNames == cn)[0]):
                    componentNames[ind] = componentNames[ind] + f'_{k+1}'
        assert len(set(componentNames)) == len(funcModelList), f'Duplicate func model names: '+str(componentNames)
        concNames = ['Conc_'+componentName for componentName in componentNames[:-1]]
        if isinstance(mixDataItemNames, str): mixDataItemNames = [mixDataItemNames]
        if isinstance(expDataItemNames, str): expDataItemNames = [expDataItemNames]*len(mixDataItemNames)
        if not isinstance(distToExperiment, list): distToExperiment = [distToExperiment]*len(mixDataItemNames)
        if errorDataItemNames is not None and isinstance(errorDataItemNames, str):
            errorDataItemNames = [errorDataItemNames]*len(mixDataItemNames)

        def getAllConcentrations(fullParams):
            res = [fullParams[concName] for concName in concNames]
            res.append(1-np.sum(res))
            return res

        def getComponentParamVector(fullParams, componentName):
            p = {}
            for pn in fullParams:
                pref = componentName+'_'
                if pn.startswith(pref):
                    p[pn[len(pref):]] = fullParams[pn]
            for pn in commonParamNames:
                p[pn] = fullParams[pn]
            return p

        def getAllComponentsParams(fullParams):
            return {cn:getComponentParamVector(fullParams, cn) for cn in componentNames}

        def mixtureFunc(slf, params):
            paramsByComponent = getAllComponentsParams(params)
            concentrations = getAllConcentrations(params)
            for i,m in enumerate(funcModelList):
                m.evaluate(paramsByComponent[componentNames[i]])
            slf.data = {}
            x = {name:None for name in mixDataItemNames}
            y = {name:None for name in mixDataItemNames}
            for i,m in enumerate(funcModelList):
                # mix spectra
                for name in mixDataItemNames:
                    di = m.data[name]
                    if x[name] is None:
                        assert i == 0, str(i)
                        x[name] = di['x']
                        y[name] = concentrations[i]*di['y']
                    else:
                        assert np.all(x[name] == di['x'])
                        y[name] += concentrations[i]*di['y']
                for name, di in m.data.items():
                    if name not in expDataItemNames:
                        slf.data[componentNames[i]+'_'+name] = di
            for name in mixDataItemNames:
                di = copy.deepcopy(funcModelList[0].data[name])
                di['x'], di['y'] = x[name], y[name]
                slf.data[name] = di
            if not plotIndividual:
                for i, m in enumerate(funcModelList):
                    for name in mixDataItemNames:
                        slf.data[componentNames[i] + '_' + name]['plot'] = False
            dat = funcModelList[0].data
            exps = {name: utils.Spectrum(dat[name]['x'], dat[name]['y']) for name in expDataItemNames}
            for name in expDataItemNames: slf.data[name] = dat[name]
            for i in range(1, len(funcModelList)):
                for name in expDataItemNames:
                    di = funcModelList[i].data[name]
                    assert np.all(di['x']==exps[name].energy) and np.all(di['y']==exps[name].intensity), 'Different exp detected!'
            errors = []
            for j,name in enumerate(expDataItemNames):
                theory = utils.Spectrum(slf.data[mixDataItemNames[j]]['x'], slf.data[mixDataItemNames[j]]['y'])
                errors.append(distToExperiment[j](theory, exps[name], params))
            if errorDataItemNames is not None:
                for j,name in enumerate(errorDataItemNames):
                    di = slf.data[componentNames[0] + '_' + name]
                    di['str'] = "%.3g" % errors[j]
                    slf.data[name] = di
                    for i, m in enumerate(funcModelList):
                        del slf.data[componentNames[i] + '_' + name]
            return np.sum(errors)

        # make mixture param properties
        paramProperties = ParamProperties()
        for concName in concNames: paramProperties[concName] = {'type':'float', 'domain':[0,1]}
        if len(concNames) > 1: # i.e. number of components > 2
            paramProperties.constrains.append({'type':'linear', 'lowerBound':0, 'upperBound':1, 'paramMultipliers':{concName:1 for concName in concNames}})
        for i, m in enumerate(funcModelList):
            for name, pp in m.paramProperties.items():
                if name in commonParamNames:
                    if name in paramProperties:
                        assert paramProperties[name] == pp
                    else:
                        paramProperties[name] = copy.deepcopy(pp)
                else:
                    newName = componentNames[i] + '_' + name
                    assert newName not in paramProperties, f'{newName} in '+str(list(paramProperties.keys()))
                    paramProperties[newName] = copy.deepcopy(pp)

        name = '+'.join(componentNames)
        funcModel = FuncModel(name=name, function=mixtureFunc, paramProperties=paramProperties, createfigParams=funcModelList[0].createfigParams, figureSettings=funcModelList[0].figureSettings, userPlotFunc=funcModelList[0].userPlotFunc, userSaveFunc=funcModelList[0].userSaveFunc)
        funcModel.concNames = concNames
        # make initial params
        if np.all(np.array([m.params is not None for m in funcModelList])):
            params = {concName:1/len(funcModelList) for concName in concNames}
            for i, m in enumerate(funcModelList):
                for name in m.params:
                    newName = componentNames[i] + '_' + name
                    if name in commonParamNames:
                        if name in params:
                            assert params[name] == m.params[name], f'Common param {name} is not common'
                        else: params[name] = m.params[name]
                    else:
                        params[newName] = m.params[name]
            funcModel.params = params
        return funcModel

    def optimize(self, optType='min', trysCount=1, fittedParams=None, method=None, folderToSaveResult=None, contourMapCalcMethod='fast', plotContourMaps='all', extraPlotFuncContourMaps=None):
        """
        Find best params

        :param optType: min or max
        :param trysCount: number of attempts to find minimum
        :param fittedParams: list of float fitted param names. If None - take all float params
        :param method: str - scipy.optimize.minimize method. If None, use 'Powell' for unconstrained optimization and 'trust-constr' for constrained
        :param folderToSaveResult: if None, save nothing
        :param contourMapCalcMethod: 'fast' - plot contours of the target function; 'thorough' - plot contours of the min of target function by all arguments except axes
        :param plotContourMaps: 'all' or list of 1-element or 2-elements lists of axes names to plot contours of target function
        :param extraPlotFuncContourMaps: user defined function to plot something on result contours: func(ax, axisNamesList, xminDict)
        """
        assert len(self.paramProperties.constrains) == 0, "TODO"
        assert optType in ['min', 'max']
        if fittedParams is None: fittedParams = self.paramProperties.getFloatParamNames()
        assert self.params is not None, f"Initialize params of model {self.name}"
        var = copy.deepcopy(self.params)

        def targetFunction(argsList):
            for i,pn in enumerate(fittedParams): var[pn] = argsList[i]
            sign = +1 if optType=='min' else -1
            return self.evaluate(var)*sign
        bounds = [self.paramProperties[pn]['domain'] for pn in fittedParams]
        constraints = None  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        optRes = optimize.findGlobalMinimum(targetFunction, trysCount, bounds, constraints=constraints, fun_args=None, paramNames=fittedParams, folderToSaveResult=folderToSaveResult, fixParams=None, contourMapCalcMethod=contourMapCalcMethod, plotContourMaps=plotContourMaps, extraPlotFunc=extraPlotFuncContourMaps, printOnline=True, method=method)
        if folderToSaveResult is not None:
            for i,res in enumerate(optRes):
                val = targetFunction(res['x'])
                assert val == res['value']
                self.plot()
                self.save(folderToSaveResult + os.sep + f'{utils.zfill(i,trysCount)}_{val:.4f}.txt')
        val = targetFunction(optRes[0]['x'])
        return val, {pn:self.params[pn] for pn in fittedParams}

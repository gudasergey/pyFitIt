# this file is needed for convenient import of functions from library before pyFitIt deploy
import numpy as np
import os, sys, tempfile, copy
import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import RidgeCV
from IPython.display import display

sys.path.insert(0, '../lib')
import smoothLib, experiment, sampling, utils, optimize
from ML import makeQuadric

LoadExperiment = experiment.LoadExperiment
sample = sampling.sample

def importSampledXanes(folder):
    geometryParams = pd.read_csv(folder+'/params.txt', sep=' ')
    xanes = pd.read_csv(folder+'/xanes.txt', sep=' ')
    return geometryParams, xanes

def fitBySliders(geometryParams, xanes, exp):
    # normalize params
    geometryParamsMin = np.min(geometryParams.values, axis=0)
    geometryParamsMax = np.max(geometryParams.values, axis=0)
    geometryParams = 2*(geometryParams-geometryParamsMin)/(geometryParamsMax-geometryParamsMin) - 1
    # machine learning estimator training
    estimator = makeQuadric(RidgeCV(alphas=[0.01,0.1,1,10,100]))
    estimator.fit(geometryParams.values, xanes.values)
    # need for smoothing by fdmnes
    xanesFolder = tempfile.mkdtemp(prefix='smooth_')
    e_names = xanes.columns
    xanes_energy = np.array([float(e_names[i][2:]) for i in range(e_names.size)])

    def plotXanes(**params):
        geomArg = np.array([params[pName] for pName in geometryParams.columns]).reshape([1,geometryParams.shape[1]])
        geomArg = 2*(geomArg-geometryParamsMin)/(geometryParamsMax-geometryParamsMin) - 1
        # prediction
        absorbPrediction = estimator.predict(geomArg)[0]
        # smoothing
        xanesPrediction = utils.Xanes(xanes_energy, absorbPrediction, None, None)
        xanesPrediction.save(xanesFolder+'/out.txt', exp.defaultSmoothParams.fdmnesSmoothHeader)
        _, smoothedXanes = smoothLib.smooth_fdmnes(None, None, params['Gamma_hole'], params['Ecent'], params['Elarg'], params['Gamma_max'], params['Efermi'], xanesFolder)
        #plotting
        shift = params['shift']
        exp_e = exp.xanes.energy
        exp_xanes = exp.xanes.absorb
        e_fdmnes = exp_e-shift
        absorbPredictionNormalized = utils.fit_arg_to_experiment(xanesPrediction.energy, exp_e, xanesPrediction.absorb, shift, lastValueNorm=True)
        smoothedPredictionNormalized = utils.fit_arg_to_experiment(xanesPrediction.energy, exp_e, smoothedXanes, shift, lastValueNorm=True)
        fig, ax = plt.subplots()
        if params['notConvoluted']: ax.plot(e_fdmnes, absorbPredictionNormalized, label='initial')
        ax.plot(e_fdmnes, smoothedPredictionNormalized, label='convolution')
        ax.plot(e_fdmnes, exp_xanes, c='k', label="Experiment")
        if params['smoothWidth']:
            ax2 = ax.twinx()
            smoothWidth = smoothLib.YvesWidth(e_fdmnes, params['Gamma_hole'], params['Ecent'], params['Elarg'], params['Gamma_max'], params['Efermi'])
            ax2.plot(e_fdmnes, smoothWidth, c='r', label='Smooth width')
            ax2.legend()
        ax.set_xlim([params['energyRange'][0], params['energyRange'][1]])
        ax.set_ylim([0, np.max(exp_xanes)*1.2])
        ax.set_xlabel("Energy")
        ax.set_ylabel("Absorb")
        ax.legend()
        fig.set_size_inches((16/3*2, 9/3*2))
        plt.show()

    controls = []
    o = 'vertical'
    for pName in exp.geometryParamRanges:
        p0 = exp.geometryParamRanges[pName][0]; p1 = exp.geometryParamRanges[pName][1]
        controls.append(widgets.FloatSlider(description=pName, min=p0, max=p1, step=(p1-p0)/30, value=(p0+p1)/2, orientation=o))
    shift = optimize.value(exp.defaultSmoothParams['fdmnes'], 'shift')
    controls.append(widgets.FloatSlider(description='shift', min=shift-10.0, max=shift+10.0, step=0.3, value=shift, orientation=o))
    controls.append(widgets.FloatSlider(description='Gamma_hole', min=0.1, max=10, step=0.2, value=2, orientation=o))
    controls.append(widgets.FloatSlider(description='Ecent', min=0, max=100, step=1, value=50, orientation=o))
    controls.append(widgets.FloatSlider(description='Elarg', min=0, max=100, step=1, value=50, orientation=o))
    controls.append(widgets.FloatSlider(description='Gamma_max', min=1, max=100, step=1, value=15, orientation=o))
    controls.append(widgets.FloatSlider(description='Efermi', min=-30, max=30, step=1, value=0, orientation=o))
    controls.append(widgets.Checkbox(description='smoothWidth', value=True))
    controls.append(widgets.Checkbox(description='notConvoluted', value=True))
    e0 = xanes_energy[0]; e1 = xanes_energy[-1]
    controls.append(widgets.FloatRangeSlider(description='energyRange', min=e0,max=e1,step=(e1-e0)/30,value=[e0,e1], orientation='horizontal'))
    ui = widgets.HBox(tuple(controls))
    ui.layout.flex_flow = 'row wrap'
    ui.layout.justify_content = 'space-between'
    ui.layout.align_items = 'flex-start'
    ui.layout.align_content = 'flex-start'
    controlsDict = {}
    for c in controls: controlsDict[c.description] = c
    out = widgets.interactive_output(plotXanes, controlsDict)
    out.layout.min_height = '400px'
    display(ui, out)

from . import utils
utils.fixDisplayError()
import os, warnings, json, matplotlib, copy
from . import optimize, ML
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import cycler


figsize = (16/3*2, 9/3*2)
dpi = 300
matplotlib.rcParams['figure.dpi'] = dpi


def closefig(fig):
    #if not utils.isJupyterNotebook(): plt.close(fig)  - notebooks also have limit - 20 figures
    if fig.number in plt.get_fignums():
        plt.close(fig)


# append = {'label':..., 'data':..., 'twinx':True} or  [{'label':..., 'data':...}, {'label':..., 'data':...}, ...]
def plotToFolder(folder, exp, xanes0, smoothed_xanes, append=None, fileName='', title='', shift=None):
    if fileName=='': fileName='xanes'
    os.makedirs(folder, exist_ok=True)
    exp_e = exp.spectrum.energy
    exp_xanes = exp.spectrum.intensity
    shiftIsAbsolute = exp.defaultSmoothParams.shiftIsAbsolute
    search_shift_level = exp.defaultSmoothParams.search_shift_level
    fit_interval = exp.intervals['plot']
    fig, ax = plt.subplots(figsize=figsize)
    fdmnes_en = smoothed_xanes.energy
    fdmnes_xan = smoothed_xanes.intensity
    if shift is None:
        if os.path.isfile(folder+'/args_smooth.txt'):
            with open(folder+'/args_smooth.txt', 'r') as f: smooth_params = json.load(f)
            if type(smooth_params) is list: shift = optimize.value(smooth_params,'shift')
            else: shift = smooth_params['shift']
        else:
            if xanes0 is not None:
                warnings.warn("Can't find file args_smooth.txt in folder "+folder+" use default shift from experiment fdmnes")
            shift = exp.defaultSmoothParams['fdmnes']['shift']
        if not shiftIsAbsolute: shift += utils.getInitialShift(exp_e, exp_xanes, fdmnes_en, fdmnes_xan, search_shift_level)
    if xanes0 is not None:
        fdmnes_xan0 = xanes0.intensity / np.mean(xanes0.intensity[-3:]) * np.mean(exp_xanes[-3:])
        ax.plot(xanes0.energy+shift, fdmnes_xan0, label='initial')
    ax.plot(fdmnes_en, fdmnes_xan, label='convolution')
    ax.plot(exp_e, exp_xanes, c='k', label="Experiment")
    if append is not None:
        if type(append) is dict: append = [append]
        assert type(append) is list
        for graph in append: 
            assert type(graph) is dict
            if ('twinx' in graph) and graph['twinx']:
                ax2 = ax.twinx()
                ax2.plot(exp_e, graph['data'], c='r', label='Smooth width')
                ax2.legend()
            else:
                if 'data' in graph:
                    ax.plot(exp_e, graph['data'], label=graph['label'])
                else:
                    ax.plot(graph['x'], graph['y'], label=graph['label'])
    ax.set_xlim([fit_interval[0], fit_interval[1]])
    ymin = 0 if np.min(exp_xanes)>=0 else np.min(exp_xanes)*1.2
    ymax = np.max(exp_xanes)*1.2
    ax.set_ylim([ymin, ymax])
    i = 1
    font = FontProperties(); font.set_weight('black'); font.set_size(20)
    for fi in exp.intervals:
        if exp.intervals[fi] == '': continue
        txt = ax.text(exp.intervals[fi][0], ax.get_ylim()[0], '[', color='green', verticalalignment='bottom', fontproperties=font)
        txt = ax.text(exp.intervals[fi][1], ax.get_ylim()[0], ']', color='green', verticalalignment='bottom', fontproperties=font)
        # ax.plot(np.array(exp.intervals[fi])-shift, [0.03*i*(ymax-ymin)+ymin]*2, 'r*', ms=10); i+=1
    ax.set_xlabel("Energy")
    ax.set_ylabel("XANES")
    ax.legend()
    if title!='':
        title = wrap(title, 100)
        ax.set_title(title)
    fig.set_size_inches((16/3*2, 9/3*2))
    fig.savefig(folder+'/'+fileName+'.png', dpi=dpi)
    closefig(fig)
    # print(exp_e.size, exp_xanes.size, fdmnes_xan.size)
    with open(folder+'/'+fileName+'.csv', 'a') as f:
        np.savetxt(f, [exp_e, exp_xanes], delimiter=',')
        np.savetxt(f, [fdmnes_en, fdmnes_xan], delimiter=',')


# region = {paramName1:[a,b], paramName2:[c,d]}, N = {paramName1:N1, paramName2:N2}
# estimatorInverse must be fitted to predict smoothed xanes on the same energy grid as experiment
def plot3DL2Norm(region, otherParamValues, estimatorInverse, paramNames, exp, N=None, folder='.', density=False):
    axisNames = list(region.keys())
    axisNames.sort()
    axisInds = [np.where(paramNames==axisNames[0])[0][0], np.where(paramNames==axisNames[1])[0][0]]
    if N is None: N = {axisNames[0]:50, axisNames[1]:50}
    N0 = N[axisNames[0]]; N1 = N[axisNames[1]]
    axisValues = [np.linspace(region[name][0], region[name][1], N[name]) for name in axisNames]
    param1mesh, param2mesh = np.meshgrid(axisValues[0], axisValues[1])
    forPrediction = np.zeros([N0*N1, paramNames.size])
    k = 0
    centerPoint = np.zeros(paramNames.size)
    for p in otherParamValues: centerPoint[paramNames==p] = otherParamValues[p]
    for i0 in range(N0):
        for i1 in range(N1):
            forPrediction[k] = np.copy(centerPoint)
            forPrediction[k,axisInds[0]] = param1mesh[i0,i1]
            forPrediction[k,axisInds[1]] = param2mesh[i0,i1]
            k += 1
    
    ind = (exp.intervals['fit_smooth'][0]<=exp.spectrum.energy) & (exp.spectrum.energy<=exp.intervals['fit_smooth'][1])
    if density:
        dens = np.zeros(forPrediction.shape[0])
        for i in range(forPrediction.shape[0]):
            dens[i] = estimatorInverse.predict(exp.spectrum.intensity[ind].reshape(1,-1), forPrediction[i].reshape(1,-1), k=10, bandwidth=0.2)
    else:
        xanesPredicted = estimatorInverse.predict(forPrediction)
    normValues = np.zeros(param1mesh.shape)
    k = 0
    for i0 in range(N0):
        for i1 in range(N1):
            if density:
                normValues[i0,i1] = dens[k]
            else:
                normValues[i0,i1] = np.sqrt(utils.integral(exp.spectrum.energy[ind], (xanesPredicted[k][ind]-exp.spectrum.intensity[ind])**2))
            k += 1

    fig = plt.figure()
    CS = plt.contourf(param1mesh, param2mesh, normValues, cmap='plasma')
    plt.clabel(CS, fmt='%2.2f', colors='k', fontsize=30, inline=False)
    # plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel(axisNames[0])
    plt.ylabel(axisNames[1])
    plotTitle = 'Probability density for project '+exp.name if density else 'L2-distance from project '+exp.name
    plt.title(plotTitle)
    fig.set_size_inches((16/3*2, 9/3*2))
    postfix = '_density' if density else '_l2_norm'
    fig.savefig(folder+'/'+axisNames[0]+'_'+axisNames[1]+postfix+'_contour.png', dpi=dpi)
    closefig(fig)
    #for cmap in ['inferno', 'spectral', 'terrain', 'summer']:
    cmap = 'inferno'
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # cmap = 'summer'
    ax.plot_trisurf(param1mesh.flatten(), param2mesh.flatten(), normValues.flatten(), linewidth=0.2, antialiased=True, cmap=cmap)
    ax.view_init(azim=310, elev=40)
    plt.xlabel(axisNames[0])
    plt.ylabel(axisNames[1])
    plt.title(plotTitle)
    fig.set_size_inches((16/3*2, 9/3*2))
    # plt.legend()
    fig.savefig(folder+'/'+axisNames[0]+'_'+axisNames[1]+postfix+'.png', dpi=dpi)
    closefig(fig)
    data2csv = np.vstack((param1mesh.flatten(), param2mesh.flatten(), normValues.flatten())).T
    np.savetxt(folder+'/'+axisNames[0]+'_'+axisNames[1]+postfix+'.csv', data2csv, delimiter=',')


def wrap(s, n):
    if len(s) <= n: return s
    words = s.split(' ')
    s = '';
    line = ''
    for w in words:
        if line != '': line += ' '
        line += w
        if len(line) > 80:
            if s != '': s += "\n"
            s += line;
            line = ''
    if line != '':
        if s != '': s += "\n"
        s += line;
    return s


# x1,y1,label1,x2,y2,label2,....,filename
def plotToFile(*p, fileName=None, save_csv=True, title='', xlim=None):
    assert len(p)%3 == 0
    fig, ax = plt.subplots(figsize=figsize)
    n = len(p)//3
    for i in range(n):
        # print('plotting '+p[i*3+2]+': ', p[i*3], p[i*3+1])
        ax.plot(p[i*3], p[i*3+1], label=p[i*3+2])
    if title != '':
        # print(title)
        title = wrap(title, 100)
        ax.set_title(title)
    ax.legend()
    if xlim is not None: ax.set_xlim(xlim[0], xlim[1])
    fig.set_size_inches((16/3*2, 9/3*2))
    if fileName is None: fileName = 'graph.png'
    folder = os.path.split(os.path.expanduser(fileName))[0]
    if not os.path.exists(folder): os.makedirs(folder)
    # print('saving to file: '+fileName)
    fig.savefig(fileName, dpi=dpi)
    closefig(fig)

    if save_csv:
        with open(os.path.splitext(fileName)[0]+'.csv', 'w') as f:
            for i in range(n):
                f.write(p[i*3+2]+' x: ')
                np.savetxt(f, p[i*3], delimiter=',')
                f.write(p[i*3+2]+' y: ')
                np.savetxt(f, p[i*3+1], delimiter=',')


def xanesEvolution(centerPoint, axisName, axisRange, outputFileName, geometryParams, xanes, N=20, estimator=ML.Normalize(ML.makeQuadric(ML.RidgeCV(alphas=[0.01,0.1,1,10,100])), xOnly=False) ):
    paramNames = geometryParams.columns
    centerPoint = np.array([centerPoint[p] for p in paramNames])
    axisInd = np.where(paramNames==axisName)[0][0]
    axisMin = axisRange[0]; axisMax = axisRange[1]
    axisValues = np.linspace(axisMin, axisMax, N)
    forPrediction = np.zeros([N, centerPoint.size])
    for i in range(N):
        forPrediction[i,:] = np.copy(centerPoint)
        forPrediction[i,axisInd] = axisValues[i]
    estimator.fit(geometryParams.values, xanes.values)
    xanesPredicted = estimator.predict(forPrediction)
    e_names = xanes.columns
    xanes_energy = np.array([float(e_names[i][2:]) for i in range(e_names.size)])
    evo = np.hstack((np.expand_dims(xanes_energy, axis=1), xanesPredicted.T))
    header = 'energy'
    for v in axisValues: header += ' %6.3f' % v
    np.savetxt(outputFileName, evo, header=header)
    # нарисовать график!!!!!!

def plotDirectMethodResult(predRegr, predProba, paramName, paramRange, folder):
    fig, ax = plt.subplots(figsize=figsize)
    classNum = predProba.size
    a = paramRange[0]; b = paramRange[1]
    h = (b-a)/classNum
    barPos = np.linspace(a+h/2,b-h/2,classNum)
    ax.bar(barPos, predProba.reshape((classNum,)), width=h*0.9, label='probability')
    ax.plot([predRegr,predRegr],[0,1], color='red', linewidth=4, label='regression')
    ax.set_title('Prediction of '+paramName+' = {:.2g}'.format(predRegr))
    ax.legend()
    plt.ylabel('Probability')
    fig.set_size_inches((16/3*2, 9/3*2))
    if not os.path.exists(folder): os.makedirs(folder)
    fig.savefig(folder+'/'+paramName+'.png', dpi=dpi)
    closefig(fig)
    np.savetxt(folder+'/'+paramName+'.csv', [barPos-h/2, barPos+h/2, predProba], delimiter=',')


def truncate_colormap(cmapName, minval=0.0, maxval=1.0, n=100):
    import matplotlib.colors as colors
    cmap = plt.get_cmap(cmapName)
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmapName, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plotSample(energy, spectra, color_param=None, sortByColors=False, fileName=None):
    """Plot all spectra on the same graph colored by some parameter. Order of plotting is controled by sortByColors parameter

    :param energy: energy values
    :param spectra: 2d numpy matrix, each row is a spectrum
    :param color_param: Values of parameter to use for color, defaults to None
    :param sortByColors: Order of plotting: random or sorted by color parameter, defaults to False
    :param fileName: File to save graph. If none - figure is returned 
    """
    assert len(energy) == spectra.shape[1]
    if isinstance(spectra, pd.DataFrame): spectra = spectra.to_numpy()
    nanExists = False
    if color_param is not None:
        indNan = np.isnan(color_param)
        nanExists = np.any(indNan)
        if nanExists:
            color_param0 = color_param
            spectra0 = spectra
            color_param = color_param[~indNan]
            spectra = spectra[~indNan]
    if sortByColors:
        assert color_param is not None
        ind = np.argsort(color_param)
    else:
        ind = np.random.permutation(spectra.shape[0])
    spectra = spectra[ind]

    fig, ax = plt.subplots(figsize=(16*0.5, 9*0.5), dpi=100)
    if color_param is not None:
        assert len(color_param) == spectra.shape[0]
        colors = (color_param-np.min(color_param))/(np.max(color_param)-np.min(color_param)) * 0.9
        colors = plt.cm.hsv(colors)
        colors = colors[ind]
        ax.set_prop_cycle(cycler.cycler('color', colors))
    else: colors = ['k']*spectra.shape[0]

    for i in range(spectra.shape[0]):
        ax.plot(energy, spectra[i], lw=0.5, alpha=0.8)
    if nanExists:
        for i in np.where(indNan)[0]:
            ax.plot(energy, spectra0[i], color='k', lw=0.5, alpha=0.8)

    if fileName is not None:
        fig.savefig(fileName, dpi=dpi)
        closefig(fig)
    else: return fig


def getPlotLim(z, gap=0.1):
    m = np.min(z)
    M = np.max(z)
    d = M-m
    return [m-d*gap, M+d*gap]
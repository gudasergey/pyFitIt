import scipy.spatial.distance

from . import utils
utils.fixDisplayError()
import os, warnings, json, matplotlib, sklearn, seaborn, cycler, logging
from . import optimize, ML
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.contour, matplotlib.colors
from matplotlib.font_manager import FontProperties


def createfig(interactive=False, figsize=None, figdpi=None, **kwargs):
    if not interactive: plt.ioff()
    else:
        if utils.isJupyterNotebook(): plt.ion()
    if interactive:
        if figsize is None: figsize = (16 * 0.5, 9 * 0.5)
        if figdpi is None: figdpi = 100
    else:
        if figsize is None: figsize = (16 * 0.6, 9 * 0.6)
        if figdpi is None: figdpi = 300
    fig, ax = plt.subplots(figsize=figsize, dpi=figdpi, **kwargs)
    return fig, ax


def savefig(fileName, fig, figdpi=None):
    if figdpi is None:
        figdpi = 300
    folder = os.path.split(fileName)[0]
    if folder!= '' and not os.path.exists(folder): os.makedirs(folder, exist_ok=True)
    fig.savefig(fileName, dpi=figdpi)


def closefig(fig, interactive=False):
    if matplotlib.get_backend() == 'nbAgg': return
    #if not utils.isJupyterNotebook(): plt.close(fig)  - notebooks also have limit - 20 figures
    if fig.number in plt.get_fignums():
        if utils.isJupyterNotebook() and interactive: plt.show(block=False)  # Even if plt.isinteractive() == True jupyter notebook doesn't show graph if in past plt.ioff/ion was called
        plt.close(fig)
    else:
        print('Warning: can\'t close not existent figure')
    if utils.isJupyterNotebook() and interactive: plt.ion()


# append = {'label':..., 'data':..., 'twinx':True} or  [{'label':..., 'data':...}, {'label':..., 'data':...}, ...]
def plotToFolder(folder, exp, xanes0, smoothed_xanes, append=None, fileName='', title='', shift=None):
    if fileName=='': fileName='xanes'
    os.makedirs(folder, exist_ok=True)
    exp_e = exp.spectrum.energy
    exp_xanes = exp.spectrum.intensity
    shiftIsAbsolute = exp.defaultSmoothParams.shiftIsAbsolute
    search_shift_level = exp.defaultSmoothParams.search_shift_level
    fit_interval = exp.intervals['plot']
    fig, ax = createfig()
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
    savefig(folder+'/'+fileName+'.png', fig)
    closefig(fig)
    # print(exp_e.size, exp_xanes.size, fdmnes_xan.size)
    if os.path.exists(folder+'/'+fileName+'.csv'): os.remove(folder+'/'+fileName+'.csv')
    with open(folder+'/'+fileName+'.csv', 'a') as f:
        np.savetxt(f, [exp_e, exp_xanes], delimiter=',')
        np.savetxt(f, [fdmnes_en, fdmnes_xan], delimiter=',')
        if xanes0 is not None:
            np.savetxt(f, [xanes0.energy+shift, fdmnes_xan0], delimiter=',')


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

    fig, _ = createfig()
    CS = plt.contourf(param1mesh, param2mesh, normValues, cmap='plasma')
    plt.clabel(CS, fmt='%2.2f', colors='k', fontsize=30, inline=False)
    # plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel(axisNames[0])
    plt.ylabel(axisNames[1])
    plotTitle = 'Probability density for project '+exp.name if density else 'L2-distance from project '+exp.name
    plt.title(plotTitle)
    postfix = '_density' if density else '_l2_norm'
    savefig(folder+'/'+axisNames[0]+'_'+axisNames[1]+postfix+'_contour.png', fig)
    closefig(fig)
    #for cmap in ['inferno', 'spectral', 'terrain', 'summer']:
    cmap = 'inferno'
    fig, _ = createfig()
    ax = fig.gca(projection='3d')
    # cmap = 'summer'
    ax.plot_trisurf(param1mesh.flatten(), param2mesh.flatten(), normValues.flatten(), linewidth=0.2, antialiased=True, cmap=cmap)
    ax.view_init(azim=310, elev=40)
    plt.xlabel(axisNames[0])
    plt.ylabel(axisNames[1])
    plt.title(plotTitle)
    # plt.legend()
    savefig(folder+'/'+axisNames[0]+'_'+axisNames[1]+postfix+'.png', fig)
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


def setFigureSettings(ax, title='', xlabel='', ylabel='', xlim=None, ylim=None, plotMoreFunction=None, yscale=None, tight_layout=True, grid=False, legend=True):
    if title != '':
        title = wrap(title, 100)
        ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
        updateYLim(ax)
    if ylim is not None: ax.set_ylim(ylim[0], ylim[1])
    if xlabel != '': ax.set_xlabel(xlabel)
    if ylabel != '': ax.set_ylabel(ylabel)
    if plotMoreFunction is not None:
        plotMoreFunction(ax)
    if yscale is not None:
        ax.set_yscale(yscale)

    l = logging.getLogger()
    level = l.getEffectiveLevel()
    l.setLevel(logging.CRITICAL)
    if legend: ax.legend()
    l.setLevel(level)
    if grid: ax.grid()
    if tight_layout: ax.figure.tight_layout()


def plotToFile(*p, axisMatrix=None, axisMatrixKw=None, fileName=None, save_csv=True, showInNotebook=False, **kw):
    """
    Simple plot multiple graphs to file
    :param p: sequence of triads x1,y1,label1,x2,y2,label2,.... You can use dict with plot params instead of label (for example 'fmt' - line format)
    :param axisMatrix: list(list(*p))
    :param fileName:
    :param save_csv: True/False
    :param kw: title='', xlabel='', ylabel='', xlim=None, ylim=None, plotMoreFunction=None (function(ax)), yscale=None, tight_layout=True, grid=False
    """
    assert len(p)%3 == 0, f'Number of parameters {len(p)} is not multiple of 3'
    assert len(p) == 0 or axisMatrix is None
    assert len(p) > 0 or axisMatrix is not None
    if axisMatrix is not None: assert isinstance(axisMatrix, list) and isinstance(axisMatrix[0], list) and isinstance(axisMatrix[0][0], tuple)
    if axisMatrix is None:
        fig, ax = createfig(interactive=showInNotebook)
        ax = [[ax]]
        axisMatrixKw = [[kw]]
        axisMatrix = [[p]]
    else:
        fig, ax = createfig(interactive=showInNotebook, nrows=len(axisMatrix), ncols=len(axisMatrix[0]), squeeze=False)
    toSave = []

    def plot(ax, p, kw):
        n = len(p)//3
        for i in range(n):
            if isinstance(p[i*3+2], str):
                label = p[i*3+2]
                ax.plot(p[i*3], p[i*3+1], label=label)
            else:
                params = p[i*3+2]
                assert isinstance(params, dict)
                ax.plot(p[i*3], p[i*3+1], **params)
                label = params['label'] if 'label' in params else str(i)
            toSave.append({'label':label, 'x':p[i*3], 'y':p[i*3+1]})
        setFigureSettings(ax, **kw)
    for i in range(len(axisMatrix)):
        for j in range(len(axisMatrix[0])):
            plot(ax[i][j], axisMatrix[i][j], axisMatrixKw[i][j])
    if fileName is None: fileName = 'graph.png'
    folder = os.path.split(os.path.expanduser(fileName))[0]
    if folder != '' and not os.path.exists(folder): os.makedirs(folder, exist_ok=True)
    savefig(fileName, fig)
    closefig(fig, interactive=showInNotebook)

    if save_csv:
        def save(file, obj):
            if not isinstance(obj, np.ndarray): obj = np.array(obj)
            obj = obj.reshape(1,-1)
            np.savetxt(file, obj, delimiter=',')
        with open(os.path.splitext(fileName)[0]+'.txt', 'w') as f:
            for item in toSave:
                label = item['label']
                f.write(label+' x: ')
                save(f,item['x'])
                f.write(label+' y: ')
                save(f, item['y'])


def readPlottingFile(fileName):
    with open(fileName) as f: s = f.read().strip()
    d = {}
    for line in s.split('\n'):
        name0 = line.split(':')[0]
        name = name0
        i = 1
        while name in d:
            name = f'{name}_{i}'
            i+=1
        d[name] = np.array([float(w) for w in line.split(':')[1].split(',')])
    return d


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
    fig, ax = createfig()
    classNum = predProba.size
    a = paramRange[0]; b = paramRange[1]
    h = (b-a)/classNum
    barPos = np.linspace(a+h/2,b-h/2,classNum)
    ax.bar(barPos, predProba.reshape((classNum,)), width=h*0.9, label='probability')
    ax.plot([predRegr,predRegr],[0,1], color='red', linewidth=4, label='regression')
    ax.set_title('Prediction of '+paramName+' = {:.2g}'.format(predRegr))
    ax.legend()
    plt.ylabel('Probability')
    if not os.path.exists(folder): os.makedirs(folder)
    savefig(folder+'/'+paramName+'.png', fig)
    closefig(fig)
    np.savetxt(folder+'/'+paramName+'.csv', [barPos-h/2, barPos+h/2, predProba], delimiter=',')


def truncate_colormap(cmapName, minval=0.0, maxval=1.0, n=100):
    import matplotlib.colors as colors
    if isinstance(cmapName, str): cmap = plt.get_cmap(cmapName)
    else: cmap = cmapName
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmapName, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plotSample(energy, spectra, colorParam=None, sortByColors=False, fileName=None, alpha=None, colorBarFormat='%.2g', colorBarLabel=None, cmap='gist_rainbow', **kw):
    """Plot all spectra on the same graph colored by some parameter. Order of plotting is controled by sortByColors parameter

    :param energy: energy values
    :param spectra: 2d numpy matrix, each row is a spectrum
    :param colorParam: Values of parameter to use for color, defaults to None
    :param sortByColors: Order of plotting: random or sorted by color parameter, defaults to False
    :param fileName: File to save graph. If none - figure is returned
    :param alpha: alpha
    :param cmap: color map name, examples: 'seaborn husl' 'gist_rainbow' 'jet' 'nipy_spectral'
    """
    assert len(energy) == spectra.shape[1]
    colorMap = parseColorMap(cmap)
    if alpha is None: alpha = getDefaultSpectraAlpha(len(spectra))
    if isinstance(spectra, pd.DataFrame): spectra = spectra.to_numpy()
    nanExists = False
    if colorParam is not None:
        indNan = np.isnan(colorParam)
        nanExists = np.any(indNan)
        if nanExists:
            color_param0 = colorParam
            spectra0 = spectra
            colorParam = colorParam[~indNan]
            spectra = spectra[~indNan]
    if sortByColors:
        assert colorParam is not None
        ind = np.argsort(colorParam)
    else:
        ind = np.random.permutation(spectra.shape[0])
    spectra = spectra[ind]

    fig, ax = createfig(interactive=True)
    if colorParam is not None:
        assert len(colorParam) == spectra.shape[0]
        c_min, c_max = np.min(colorParam), np.max(colorParam)
        transform = lambda r: (r - c_min) / (c_max - c_min)
        colors = transform(colorParam)
        colors = colorMap(colors)
        colors = colors[ind]
        ax.set_prop_cycle(cycler.cycler('color', colors))
        if ML.isClassification(colorParam):
            ticks = np.unique(colorParam)
        else:
            ticks = np.linspace(np.min(colorParam), np.max(colorParam), 10)
        ticksPos = transform(ticks)
        addColorBar(mappable=plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=1), cmap=colorMap), fig=fig, ax=ax, labelMaps={}, label='', ticksPos=ticksPos, ticks=ticks, format=colorBarFormat, colorBarLabel=colorBarLabel)
    else:
        colors = ['k']*spectra.shape[0]
        ax.set_prop_cycle(cycler.cycler('color', colors))

    for i in range(spectra.shape[0]):
        ax.plot(energy, spectra[i], lw=0.5, alpha=alpha)
    if nanExists:
        for i in np.where(indNan)[0]:
            ax.plot(energy, spectra0[i], color='k', lw=0.5, alpha=alpha)
    setFigureSettings(ax, **kw)
    if fileName is not None:
        savefig(fileName, fig)
        closefig(fig, interactive=True)
    else: return fig


def getPlotLim(z, gap=0.1):
    z1 = z[np.isfinite(z)]
    if len(z1) == 0: return None
    m = np.min(z1)
    M = np.max(z1)
    d = M-m
    return [m-d*gap, M+d*gap]


def updateYLim(ax):
    xlim = ax.get_xlim()
    ybounds = []
    for line in ax.lines:
        x = line.get_xdata()
        y = line.get_ydata()
        lim = getPlotLim(y[(xlim[0] <= x) & (x <= xlim[1])])
        if lim is not None: ybounds.append(lim)
    ybounds = np.array(ybounds)
    ylim = [np.min(ybounds[:, 0]), np.max(ybounds[:, 1])]
    ax.set_ylim(ylim)


def addColorBar(mappable, fig, ax, labelMaps, label, ticksPos, ticks, format='%.2g', colorBarLabel=None):
    if isinstance(mappable, matplotlib.contour.QuadContourSet): ex = {}
    else: ex = {'extend':'max'}
    cbar = fig.colorbar(mappable, ax=ax, **ex, orientation='vertical', ticks=ticksPos, format='%.1g', label=colorBarLabel)
    # old code (not work properly too)
    # cbar = fig.colorbar(mappable, ax=ax, extend='max', orientation='vertical', ticks=ticksPos, format='%.1g')
    if label in labelMaps:
        cbarTicks = [None]*len(labelMaps[label])
        for name in labelMaps[label]:
            cbarTicks[labelMaps[label][name]] = name
        cbar.ax.set_yticklabels(cbarTicks)
    else:
        if isinstance(ticks[0], float):
            ticks = [format % t for t in ticks]
        cbar.ax.set_yticklabels(ticks)


def parseColorMap(cmap):
    """
    :param cmap: pyplot color map name, or 'seaborn ...' - seaborn (if it is map object - returns itself). Examples: 'seaborn husl' 'gist_rainbow' 'jet' 'nipy_spectral'
    """
    if 'seaborn' in cmap:
        colorMap = seaborn.color_palette(cmap.split(' ')[1], as_cmap=True)
        if np.linalg.norm(np.array(colorMap(0))-colorMap(1)) < 0.2:
            colorMap = truncate_colormap(colorMap, minval=0, maxval=0.8)
    else:
        if isinstance(cmap, str):
            colorMap = plt.cm.get_cmap(name=cmap)
        else:
            colorMap = cmap
    return colorMap


def getScatterDefaultParams(x, y, dpi):
    if not isinstance(x, np.ndarray): x = np.array(x)
    if not isinstance(y, np.ndarray): y = np.array(y)
    scale = 1 #dpi / 300  # 1 - for 300dpi
    alpha = 0.8
    markersize = 22
    minSize = 6
    maxSize = 30
    z = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
    z = z/np.std(z,axis=0)
    d = scipy.spatial.distance.cdist(z,z)
    mind = np.array([np.min(d[i][d[i]>0]) for i in range(len(z))])
    maxIntersectPercent = 0.5
    r = np.quantile(mind, q=maxIntersectPercent)

    max = np.max(d)
    if max > 0:
        # markersize=1 means 500 markers in diagonal
        diagonalCount = max/r
        markersize = np.max([500/diagonalCount,minSize])
        markersize = np.min([markersize, maxSize])
        if markersize == minSize:
            overlapMarkerCount = minSize / (500/diagonalCount)
            alpha = 1/overlapMarkerCount
    return markersize*scale, alpha


def getDefaultSpectraAlpha(count):
    return min(1/count*100,1)


def scatter(x, y, color=None, colorMap='plasma', marker_text=None, text_size=None, markersize=None, marker='o', alpha=None, edgecolor=None, fileName='scatter.png', **kw):
    """

    :param x:
    :param y:
    :param color:
    :param colorMap: 'plasma' (default) or 'gist_rainbow' or any other
    :param marker_text:
    :param text_size: if None - auto evaluation
    :param markersize:
    :param marker:
    :param alpha:
    :param title:
    :param xlabel:
    :param ylabel:
    :param fileName:
    :param plotMoreFunction: function(ax)
    :return:
    """
    if isinstance(x, pd.Series): x = x.to_numpy()
    if isinstance(y, pd.Series): y = y.to_numpy()
    if isinstance(color, pd.Series): color = color.to_numpy()
    if isinstance(color, np.ndarray):
        assert color.dtype in ['float64', 'int32'], color.dtype
    assert len(x) == len(y)
    if color is not None: assert len(color) == len(x)
    fig,ax = createfig()
    defaultMarkersize, defaulAlpha = getScatterDefaultParams(x, y, fig.dpi)
    if markersize is None: markersize = defaultMarkersize
    if alpha is None: alpha = defaulAlpha
    if edgecolor is None: edgecolor = '#555'
    colorMap = parseColorMap(colorMap)
    if color is not None:
        c = color
        c_min = np.min(c)
        c_max = np.max(c)
        transform = lambda r: (r - c_min) / (c_max - c_min)
        if ML.isClassification(color):
            ticks = np.unique(color)
        else:
            ticks = np.linspace(c_min, c_max, 10)
        ticksPos = transform(ticks)
        sc = ax.scatter(x, y, s=markersize**2, marker=marker, c=transform(c), cmap=colorMap, vmin=0, vmax=1, alpha=alpha, edgecolor=edgecolor)
        addColorBar(sc, fig, ax, {}, '', ticksPos, ticks)
    else:
        ax.scatter(x, y, s=markersize**2, color='green', alpha=alpha, edgecolor=edgecolor)
    if marker_text is not None and (text_size is None or text_size > 0):
        assert len(marker_text) == len(x), f'{len(marker_text)} != {len(x)}'
        if text_size is None: text_size = markersize*0.4
        for i in range(len(marker_text)):
            ax.text(x[i], y[i], str(marker_text[i]), ha='center', va='center', size=text_size)
    ax.set_xlim(getPlotLim(x))
    ax.set_ylim(getPlotLim(y))
    setFigureSettings(ax, **kw)
    savefig(fileName, fig)
    closefig(fig)
    csv_data = pd.DataFrame()
    csv_data['x'] = x
    csv_data['y'] = y
    if color is not None: csv_data['color'] = color
    csv_data.to_csv(os.path.splitext(fileName)[0] + '.csv', index=False)


def plotMatrix(mat, cmap=None, ticklabels=None, annot=False, fmt='.2g', fileName=None, **kw):
    fig, ax = createfig()
    if ticklabels is not None:
        assert mat.shape[0] == len(ticklabels) and mat.shape[1] == len(ticklabels)
        df = pd.DataFrame(data=mat[::-1,:], columns=ticklabels, index=ticklabels[::-1])
        ax = seaborn.heatmap(df, annot=annot, fmt=fmt, cmap=cmap, ax=ax)
    else:
        ax = seaborn.heatmap(mat, annot=annot, fmt=fmt, cmap=cmap)
    setFigureSettings(ax, **kw)
    if fileName is not None:
        savefig(fileName, fig)
        closefig(fig)
    else: return fig, ax


def plotConfusionMatrixHelper(conf_mat, accuracy, labelName, uniqueLabelValues, fileName, **kw):
    fig, ax = createfig()
    pos = ax.matshow(conf_mat.T, cmap='plasma')
    fig.colorbar(pos, ax=ax)
    title = 'Confusion matrix for label ' + labelName + f'. Accuracy = {accuracy:.2f}'
    ax.set_title(title)
    ax.set_xlabel('predicted ' + labelName)
    ax.set_ylabel('true ' + labelName)
    ticks = ['%g' % v for v in uniqueLabelValues]
    m = len(uniqueLabelValues)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.arange(m)))
    ax.set_xticklabels(ticks)
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.arange(m)))
    ax.set_yticklabels(ticks)
    # print(ticks)
    for i in range(m):
        for j in range(m):
            ax.text(i, j, '%.2g' % conf_mat[i, j], ha='center', va='center', size=10)
    setFigureSettings(ax, **kw)
    savefig(fileName, fig)
    closefig(fig)


def plotConfusionMatrix(trueLabels, predictedLabels, labelName, labelMap=None, cmap=None, fileName='', **kw):
    """
    :param labelMap: dict{userFriendlyLabel:index}
    """
    n = len(trueLabels)
    assert n == len(predictedLabels)
    conf_mat = sklearn.metrics.confusion_matrix(trueLabels, predictedLabels)/n
    if labelMap is not None:
        lm = {labelMap[l]:l for l in labelMap}
        trueLabels = np.array([lm[int(l)] for l in trueLabels])
        predictedLabels = np.array([lm[int(l)] for l in predictedLabels])
    # print(conf_mat)
    if fileName == '': fileN = f'conf_matr_{labelName}.png'
    else: fileN = fileName
    acc = np.sum(trueLabels == predictedLabels) / len(trueLabels)
    if 'title' not in kw:
        kw['title'] = 'Confusion matrix for label ' + labelName + f'. Accuracy = {acc:.2f}'
    if 'xlabel' not in kw:
        kw['xlabel'] = 'predicted ' + labelName
    if 'ylabel' not in kw:
        kw['ylabel'] = 'true ' + labelName
    plotMatrix(conf_mat, cmap=cmap, ticklabels=np.unique(trueLabels), annot=True, fmt='.2g', fileName=fileN, **kw)
    # plotConfusionMatrixHelper(conf_mat, acc, labelName, np.unique(trueLabels), fileN)
    for j in range(len(np.unique(trueLabels))):
        sum = np.sum(conf_mat[:,j])
        if sum != 0: conf_mat[:,j] /= sum
    fileN = os.path.splitext(fileN)[0] + '_normed' + os.path.splitext(fileN)[1]
    plotMatrix(conf_mat, cmap=cmap, ticklabels=np.unique(trueLabels), annot=True, fmt='.2g', fileName=fileN, **kw)
    # plotConfusionMatrixHelper(conf_mat, acc, labelName, np.unique(trueLabels), fileN)


def plotHeatMap(func, xlim, ylim, N1=50, N2=50, cmap='plasma', fileName='heatmap.png', **kw):
    """
    Plot heatmap for func(z), z == [x,y]
    """
    axisValues = [np.linspace(xlim[0], xlim[1], N1), np.linspace(ylim[0], ylim[1], N2)]
    param1mesh, param2mesh = np.meshgrid(axisValues[0], axisValues[1])
    funcValues = np.zeros(param1mesh.shape)
    k = 0
    for i0 in range(N1):
        for i1 in range(N2):
            funcValues[i0, i1] = func([param1mesh[i0, i1], param2mesh[i0, i1]])
            k += 1
    fig, ax = createfig()
    CS = plt.contourf(param1mesh, param2mesh, funcValues, cmap=cmap)
    plt.clabel(CS, fmt='%2.2f', colors='k', fontsize=15, inline=False)
    fig.colorbar(CS)
    setFigureSettings(ax, **kw)
    savefig(fileName, fig)
    closefig(fig)


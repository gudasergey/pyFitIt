import os, sys, copy, sklearn, shutil, random, itertools
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pyfitit import *
from scipy.optimize import minimize
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import scipy.signal
import scipy.interpolate
from scipy.spatial import distance
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
if utils.isLibExists("lightgbm"):
    import lightgbm as lgb


def plotting_spectra(sample, to_delete, centering, normalize):
    energy, spectra, parameters = sample.energy, sample.spectra.values.T, sample.params.values
    if centering:
        spectra = spectra - np.mean(spectra, axis=1, keepdims=True)
        if normalize:
            spectra = spectra / np.std(spectra, axis=1, keepdims=True)
    if to_delete is not None:
        spectra = np.delete(spectra,np.sort(to_delete),1)
        parameters = np.delete(parameters,np.sort(to_delete),0)

    def Range(s):
        v1 = s[0]
        v2 = s[1]
        fig,ax = plt.subplots(figsize=(13,5))
        ax.plot(energy[v1:v2], spectra[v1:v2], linewidth=0.1, color='blue')
        ax.set_xlabel('Energy', fontweight='bold')
        ax.set_ylabel('Absorption', fontweight='bold')
        ax.axvline(x=energy[v1], color='black', linestyle='--')
        ax.axvline(x=energy[v2], color='black', linestyle='--')
        ax.grid()

    _ = interact(Range,s=widgets.IntRangeSlider(
            value=[0,len(energy)-1],
            min=0,
            max=len(energy)-1,
            step=1,
            description='Energy:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
        ))
    p = pd.DataFrame(data=parameters, columns=sample.params.columns)
    s = pd.DataFrame(data=spectra.T, columns=sample.spectra.columns)
    res = ML.Sample(p, s)
    return res


def error(spectra,nPCs):
    u,s,vT=np.linalg.svd(spectra,full_matrices=False)
    ur=u[:,:nPCs]
    sr=np.diag(s[:nPCs])
    vr=vT[:nPCs,:]
    #print('Ur:',ur.shape,'Sr:',sr.shape,'Vr:',vr.shape)
    snew=np.dot(np.dot(ur,sr),vr)
    return np.mean(np.sqrt(np.mean((spectra-snew)**2, axis=1))/np.sqrt(np.mean((spectra)**2, axis=1)))


def plot_error(s,error_plot):
    fig,ax=plt.subplots(nrows=1, ncols=2,figsize=(13,5))
    ax[0].plot(np.arange(1,21),s[0:20],'-o',label='Scree Plot',color='blue')
    ax[0].set_xlabel('PCs',fontweight='bold')
    ax[0].set_ylabel('Singular Values',fontweight='bold')
    ax[0].grid()
    ax[1].plot(np.arange(1,21),error_plot,'-o',label='Scree Plot',color='red')
    ax[1].set_xlabel('PCs',fontweight='bold')
    ax[1].set_ylabel('Error',fontweight='bold')
    ax[1].grid()

def statistic(spectra,log_scale,number=20):
    u,s,v=np.linalg.svd(spectra)
    v=v.T
    error_plot=[]
    for i in range(0,number):
        error_plot.append(error(spectra,i))

    if log_scale==True:
        s=np.log10(s)
        error_plot=np.log10(error_plot)
        plot_error(s,error_plot)
    elif log_scale==False:
        s=s
        error_plot=error_plot
        plot_error(s,error_plot)
    elif log_scale !=False and log_scale !=True:
        print('Error in log_scale name')

def cross_val_predict(method, X, y):
    cv = sklearn.model_selection.KFold(n_splits=7, shuffle=True, random_state=0)
    res = np.zeros(y.shape)
    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        method.fit(X_train, y_train)
        res[test_index] = method.predict(X_test).reshape(-1)
    return res


def buildDescriptor(spectra, descriptor_type='PCA'):
    if descriptor_type == 'PCA':
        u,s,vT=np.linalg.svd(spectra,full_matrices=False)
        s=np.diag(s)
        C = np.dot(s,vT).T
    elif descriptor_type == 'intervals average':
        m = 10
        spectraT = spectra.T
        C = np.zeros((spectraT.shape[0], m))
        for i in range(m):
            ne = spectraT.shape[1]
            C[:,i] = np.mean(spectraT[:, ne*i//m:ne*(i+1)//m], axis=1)
    else: raise Error('Error in typing regressor name')
    return C


def training(descriptor, parameters, regressor_name, number=10):
    if number>descriptor.shape[1]: number = descriptor.shape[1]
    clf = [None]*number
    for i in range(number):
        y = descriptor[:,i]
        if regressor_name=='Ridge':
            clf[i] = RidgeCV()
        elif regressor_name=='ExtraTrees':
            clf[i]=ExtraTreesRegressor(n_estimators=100,random_state=0,min_samples_leaf=10)
        elif regressor_name=='RBF':
            clf[i] = ML.RBF(function='linear', baseRegression='linear')
        else:
            print('Error in typing regressor name')
        clf[i].fit(parameters, y)
    return clf

def plot_training_error(descriptor, parameters, clf):
    graf=[]
    number = len(clf)
    for i in range(number):
        y = descriptor[:,i]
        pred = cross_val_predict(clf[i], parameters, y)
        graf.append(sklearn.metrics.r2_score(y, pred)*100)
    fig,ax=plt.subplots(figsize=(13,5))
    ax.plot(np.arange(1,number+1),graf,'-o')
    ax.set_ylabel('Quality %',fontweight='bold')
    ax.set_xlabel('PCs',fontweight='bold')
    ax.grid()

def create_C_plot(val1,val2,dim_param,npt,PC,min_val,max_val):
    if val1 == val2:
        print('Error: The two values are the same')
    elif np.remainder(val1,1) !=0 or np.remainder(val2,1) !=0:
        print('Chosed not integer values')
    else:
        matrix_values=np.zeros((npt**2,dim_param))
        var_vector=np.array(np.meshgrid(np.linspace(min_val, max_val,npt), np.linspace(min_val, max_val,npt))).T.reshape(-1,2)
        matrix_values[:,val1]=var_vector[:,0]
        matrix_values[:,val2]=var_vector[:,1]
        matrix_grid=np.zeros(matrix_values.shape[0])
        for i in range(matrix_values.shape[0]):
            matrix_grid[i]=clf[PC].predict(matrix_values[i].reshape(1,-1))
        return matrix_grid.reshape(npt,npt)

def twoDmap(clf,param,npt,min_val,max_val):
    dim_param=np.shape(param.values)[1]
    def selectPC(number):
        control=number
        def selectP1(s1):
            val1=s1
            def selectP2(s2):
                val2=s2
                if val1 == val2:
                    return(print('The two values are the same'))
                else:
                    matrix_values=np.zeros((npt**2,dim_param))
                    var_vector=np.array(np.meshgrid(np.linspace(min_val, max_val,npt), np.linspace(min_val, max_val,npt))).T.reshape(-1,2)
                    matrix_values[:,val1]=var_vector[:,0]
                    matrix_values[:,val2]=var_vector[:,1]
                    matrix_grid=np.zeros(matrix_values.shape[0])
                    for i in range(matrix_values.shape[0]):
                        matrix_grid[i]=clf[number].predict(matrix_values[i].reshape(1,-1))
                    x= np.linspace(min_val, max_val,npt)
                    y= np.linspace(min_val, max_val,npt)
                    X,Y=np.meshgrid(x,y)
                    Z=matrix_grid.reshape(npt,npt)
                    dgy,dgx=np.gradient(Z)
                    fig = plt.figure(figsize=(13,5))
                    ax = fig.gca()
                    surf=ax.contourf(X, Y, Z, 30,cmap='Spectral')
                    ax.tick_params(direction='in',labelsize=15,width=2)
                    ax.set_xlabel(param.keys()[val1],fontweight='bold')
                    ax.set_ylabel(param.keys()[val2],fontweight='bold')
                    fig.colorbar(surf)
                    ax.streamplot(X, Y, dgx,dgy,color='black')
            _=interact(selectP2, s2 = widgets.BoundedIntText(value=1, min=0, max=dim_param-1, step=1, description='p2:', disabled=False))
        _=interact(selectP1, s1 = widgets.BoundedIntText(value=0, min=0, max=dim_param-1, step=1, description='p1:', disabled=False))
    _=interact(selectPC, number = widgets.BoundedIntText(value=0, min=0, max=20, step=1, description='PC:', disabled=False))

def iso(parameters,PC,C,clf):
    return (C-clf[PC].predict(parameters.reshape(1,-1))[0])**2

def find_isosurface(param,min_val,max_val,clf,iso_param):
    PC=iso_param[0]
    C=iso_param[1]
    p1=iso_param[2]
    p2=iso_param[3]
    #C = C_vector[PC]
    #C = C_values
    dim=np.shape(param.values)[1]
    p_values=np.zeros((1000,2))
    bnds=np.zeros((dim,2))
    for i in range(len(bnds)):
        bnds[p1]=[min_val,max_val]
        bnds[p2]=[min_val,max_val]
    for i in range(1000):
        p=np.zeros(dim)
        p[p1]=random.uniform(-0.2, 0.2)
        p[p2]=random.uniform(-0.2, 0.2)
        #res=minimize(iso, p,args=(PC,C),bounds=bnds)#bounds=bnds)
        res=minimize(iso, p,args=(PC,C,clf),bounds=bnds)
        p_values[i][0]=res.x[p1]
        p_values[i][1]=res.x[p2]
    return p_values

def plot_isosurfaces(g1,g2,iso_param_1,iso_param_2,param):
    fig,ax=plt.subplots(nrows=1, ncols=3,figsize=(13,5))
    if iso_param_1[2] != iso_param_2[2] or iso_param_1[3] != iso_param_2[3]: print('Error Not common parameters chosen for the plot')
    else:
        ax[0].set_title('PC:'+str(iso_param_1[0]))
        ax[0].plot(g1[:,0],g1[:,1],'o')
        ax[0].set_xlim(-0.22,0.22)
        ax[0].set_ylim(-0.22,0.22)
        ax[0].set_xlabel(param.keys()[iso_param_1[2]],fontweight='bold')
        ax[0].set_ylabel(param.keys()[iso_param_1[3]],fontweight='bold')
        ax[0].grid()
        ax[1].set_title('PC:'+str(iso_param_2[0]))
        ax[1].plot(g2[:,0],g2[:,1],'o',color='orange')
        ax[1].set_xlim(-0.22,0.22)
        ax[1].set_ylim(-0.22,0.22)
        ax[1].set_xlabel(param.keys()[iso_param_1[2]],fontweight='bold')
        ax[1].set_ylabel(param.keys()[iso_param_1[3]],fontweight='bold')
        ax[1].grid()
        ax[2].set_title('Merging Plot')
        ax[2].plot(g1[:,0],g1[:,1],'o')
        ax[2].plot(g2[:,0],g2[:,1],'o',color='orange')
        ax[2].set_xlabel(param.keys()[iso_param_1[2]],fontweight='bold')
        ax[2].set_ylabel(param.keys()[iso_param_1[3]],fontweight='bold')
        ax[2].grid()


def findExtrema(sample, extremaType, energyInterval, maxRf=.1, allowExternal=True, maxExtremumPointsDist=5, intensityNormForMaxExtremumPointsDist=1, maxAdditionIter=-1, refineExtremum=True, extremumInterpolationRadius=10, returnIndices=False, plotToFile=None):
    """
        finds descriptors for XANES extrema

    :param extremumInterpolationRadius:
        Radius of interpolation, only used if refineExtremum=True
    :param refineExtremum:
        Interpolate extremum's neighbourhood in order to find more accurate position of extremum
    :param maxAdditionIter:
        Amount af bad to good transfer iterations
    :param maxExtremumPointsDist:
        Max euclidean distance between good and bad extrema points (y is normed to intensityNormForMaxExtremumPointsDist)
    :param allowExternal:
        Allow searching of bad extrema outside of energyInterval
    :param maxRf:
        Max R-Factor between good and bad spectra
    :param energyInterval: Tuple
        Energy range inside which to search extrema
    :param sample: ML.Sample or tuple (intensities, energy). Each row is a spectrum
    :param extremaType: str
        'min' for pits, 'max' for peaks
    :return:
        newSample (only good spectra), descriptors (extrema_x,y, diff2), and goodSpectrumIndices if returnIndices=True
    """

    assert extremaType in ['min', 'max']

    def calculateRFactor(energy, xanes1, xanes2):
        return utils.integral(energy, (xanes1 - xanes2) ** 2) / \
               utils.integral(energy, xanes1 ** 2)

    def divideSpectra(energy, extremaIndices, energyInterval):
        # allowedIndices = np.logical_and(energy >= energyInterval[0], energy <= energyInterval[1])
        good = []
        bad = []
        spectrumIndex = 0
        for extremumIndices in extremaIndices:
            extremumCount = 0
            validExtremumIndex = -1
            for extremumIndex in extremumIndices:
                if isIndexInRange(energy, extremumIndex, energyInterval):
                    extremumCount += 1
                    validExtremumIndex = extremumIndex
            if extremumCount == 1:
                good.append((spectrumIndex, validExtremumIndex))
            else:
                bad.append(spectrumIndex)
            spectrumIndex += 1

        return good, bad

    def closestExtremumIndex(extremumEnergy, extrema, energy):
        if len(extrema) == 0:
            return -1

        extremaEnergies = list(map(lambda x: abs(extremumEnergy - energy[x]), extrema))
        return np.argmin(extremaEnergies)

    def isIndexInRange(energy, energyIndex, energyInterval):
        return (energy[energyIndex] >= energyInterval[0]) and (energy[energyIndex] <= energyInterval[1])

    def moveBadSpectraToGood(good, bad, energy, intensities, extrema, energyInterval):
        if len(good) == 0 or len(bad) == 0:
            return good, bad

        goodSpectra = list(map(lambda x: intensities[x[0]], good))
        badSpectra = list(map(lambda x: intensities[x], bad))
        d = distance.cdist(badSpectra, goodSpectra, lambda x, y: calculateRFactor(energy, x, y))

        newGood = copy.deepcopy(good)
        newBad = copy.deepcopy(bad)
        sortedIndices = np.argsort(d)
        goodExtrIntensities = list(map(lambda x: intensities[x[0]][x[1]], good))
        for badIndex in range(d.shape[0]):
            for goodIndex in sortedIndices[badIndex]:
                # filtering those which exceed max rfactor
                if d[badIndex][goodIndex] > maxRf:
                    break

                globalBadIndex = bad[badIndex]
                globalGoodIndex = good[goodIndex][0]
                goodExtremumEnergy = energy[good[goodIndex][1]]

                closestIndex = closestExtremumIndex(goodExtremumEnergy, extrema[globalBadIndex], energy)
                if closestIndex<0: continue
                badExtrEnergyIndex = extrema[globalBadIndex][closestIndex]
                badExtrPoint = np.array((energy[badExtrEnergyIndex], intensities[globalBadIndex][badExtrEnergyIndex] / intensityNormForMaxExtremumPointsDist))
                goodExtrPoint = np.array((goodExtremumEnergy, intensities[globalGoodIndex][good[goodIndex][1]] / intensityNormForMaxExtremumPointsDist))
                dist = np.linalg.norm(badExtrPoint - goodExtrPoint)

                # filtering by distance
                if dist > maxExtremumPointsDist or \
                   not allowExternal and not isIndexInRange(energy, closestIndex, energyInterval):
                    continue

                # move filtered bad spectrum from 'bad' to 'good' list
                newGood.append((globalBadIndex, badExtrEnergyIndex))
                newBad.remove(globalBadIndex)
                break

        return newGood, newBad

    def getFinalExtremaAndDerivatives(good, energy, intensities, refineExtremum, extremumInterpolationRadius, comparator):
        if not refineExtremum:
            xExtrema = np.array(list(map(lambda pair: energy[pair[1]], good)))
            yExtrema = np.array(list(map(lambda pair: intensities[pair[0]][pair[1]], good)))

            def f(x):
                (sIndex, extremumIndex) = x
                intensity = intensities[sIndex]
                d1Left = (intensity[extremumIndex] - intensity[extremumIndex - 1]) / (energy[extremumIndex] - energy[extremumIndex - 1])
                d1Right = (intensity[extremumIndex + 1] - intensity[extremumIndex]) / (energy[extremumIndex + 1] - energy[extremumIndex])
                leftMeanEnergy = (energy[extremumIndex] - energy[extremumIndex - 1]) / 2
                rightMeanEnergy = (energy[extremumIndex + 1] - energy[extremumIndex]) / 2
                return (d1Right - d1Left) / (rightMeanEnergy - leftMeanEnergy)

            derivatives = np.apply_along_axis(f, 1, np.array(good))
            return np.array([xExtrema, yExtrema]), derivatives

        xStep = 0.01
        xExtrema = []
        yExtrema = []

        derivative = []
        for (spectrumIndex, extremumIndex) in good:
            xnew = np.arange(energy[extremumIndex] - extremumInterpolationRadius, energy[extremumIndex] + extremumInterpolationRadius, xStep)
            spline = scipy.interpolate.CubicSpline(energy, intensities[spectrumIndex])
            ynew = spline(xnew)
            extrema = np.array(scipy.signal.argrelextrema(ynew, comparator)[0])
            index = np.argsort(abs(xnew[extrema] - energy[extremumIndex]))[0]
            newExtremumIndex = extrema[index]
            xExtrema.append(xnew[newExtremumIndex])
            yExtrema.append(ynew[newExtremumIndex])
            # assert abs(yExtrema[-1] - np.interp(xExtrema[-1], energy, intensities[spectrumIndex])) < 0.1

            derivative.append((ynew[newExtremumIndex - 1] - 2*ynew[newExtremumIndex] + ynew[newExtremumIndex+1]) / xStep ** 2)

        return np.array([np.array(xExtrema), np.array(yExtrema)]), np.array(derivative)

    def plot(good, bad, energy, intensities, extremaPoints):
        if plotToFile is None:
            return

        goodBadIndices = []
        for (i, e) in good:
            goodBadIndices.append((i, e, 'good'))
        for i in bad:
            goodBadIndices.append((i, -1, 'bad'))
        # random.shuffle(goodBadIndices)

        fig, ax = plt.subplots(figsize=plotting.figsize)
        alpha = min(1/len(goodBadIndices)*100,1)
        for (spectrum, extremum, label) in goodBadIndices:
            if label == 'good':
                ax.plot(energy, intensities[spectrum], color='blue', lw=.3, alpha=alpha)
        for x, y in zip(extremaPoints[0], extremaPoints[1]):
            ax.plot(x, y, marker='o', markersize=3, color="red", alpha=alpha)
        for (spectrum, extremum, label) in goodBadIndices:
            if label == 'bad':
                ax.plot(energy, intensities[spectrum], color='black', lw=.3)

        fig.set_size_inches((16/3*2, 9/3*2))
        plt.show()
        fig.savefig(plotToFile, dpi=plotting.dpi)
        plotting.closefig(fig)

    def ensureCorrectResults(intensities, good, bad, extremaPoints, derivatives):
        goodIndices = list(map(lambda x: x[0], good))
        assert len(goodIndices) == len(set(goodIndices)), 'indistinct good values'
        assert len(bad) == len(set(bad)), 'indistinct bad values'
        assert len(bad) + len(good) == len(intensities), 'spectra inconsistency'
        assert extremaPoints[0].shape[0] == extremaPoints[1].shape[0] and \
               extremaPoints[0].shape[0] == derivatives.shape[0], 'bad descriptor integrity'

    def findExtremaIndices(intensities, comparator):
        extrema = map(lambda x: scipy.signal.argrelextrema(x, comparator)[0], intensities)
        return list(extrema)

    # main function
    if isinstance(sample, ML.Sample):
        energy = sample.energy
        intensities = sample.spectra.values
    else: #tuple 
        (intensities, energy) = sample
    comparator = np.less if extremaType == 'min' else np.greater
    # getting indices of energies in which extrema are present
    extrema = findExtremaIndices(intensities, comparator)

    good, bad = divideSpectra(energy, extrema, energyInterval)
    print('First split: ', len(good), len(bad))

    # try to identify good spectra in bad list and move them
    iterations = 0
    while iterations < maxAdditionIter or maxAdditionIter == -1:
        newGood, newBad = moveBadSpectraToGood(good, bad, energy, intensities, extrema, energyInterval)
        added = len(newGood) - len(good)
        print('Iteration = %i, added %i spectra' % (iterations + 1, added))
        shouldContinue = added > 0
        good = newGood
        bad = newBad
        iterations += 1

        if not shouldContinue:
            break
    good = sorted(good, key=lambda pair: pair[0])
    extremaPoints, derivatives = getFinalExtremaAndDerivatives(good, energy, intensities, refineExtremum, extremumInterpolationRadius, comparator)
    goodSpectrumIndices = list(map(lambda x: x[0], good))
    descr = np.vstack((extremaPoints, derivatives)).T

    # plot results
    plot(good, bad, energy, intensities, extremaPoints)

    ensureCorrectResults(intensities, good, bad, extremaPoints, derivatives)
    if isinstance(sample, ML.Sample):
        newSpectra = sample.spectra.loc[goodSpectrumIndices].reset_index(drop=True)
        newParams = sample.params.loc[goodSpectrumIndices].reset_index(drop=True)
        newSample = ML.Sample(newParams, newSpectra)
    else: newSample = intensities[goodSpectrumIndices]

    if returnIndices:
        return newSample, descr, goodSpectrumIndices
    else:
        return newSample, descr


def findExtremumByFit(spectrum, energyInterval, fitByPolynomDegree = 2, returnPolynom = False):
    """
    :param spectrum:
    :param energyInterval: 
    :param fitByPolynomDegree: 
    :param returnPolynom: 
    :param extremaType: 'min' or 'max'
    :returns: descriptors [(extremum energy, extremum value, 2d derivative at extremum)] sorted by 2d derivative
    """
    x = spectrum.energy
    ind = (x >= energyInterval[0]) & (x <= energyInterval[1])
    x = x[ind]
    y = spectrum.intensity[ind]
    plt.plot(x, y)

    poly = np.polynomial.polynomial.Polynomial.fit(x, y, fitByPolynomDegree)
    dpoly = poly.deriv(1)
    extrema = dpoly.roots()
    extrema_val = poly(extrema)
    d2poly = poly.deriv(2)
    d2val = d2poly(extrema)
    ind = np.argsort(d2val)
    descriptors = (extrema[ind], extrema_val[ind], d2val[ind])

    if returnPolynom:
        return descriptors, poly
    else:
        return descriptors


def stableExtrema(spectra, energy, extremaType, energyInterval, plotResultToFolder=None, maxRf=.1, allowExternal=True, maxExtremumPointsDist=5, intensityNormForMaxExtremumPointsDist=1, maxAdditionIter=-1, refineExtremum=True, extremumInterpolationRadius=10, smoothRad=5):
    assert extremaType in ['min', 'max'], 'invalid extremaType'
    spectra1 = np.copy(spectra)
    for i in range(len(spectra)):
        spectra1[i] = smoothLib.simpleSmooth(energy, spectra1[i], smoothRad, kernel='Gauss')
    newSpectra, descr = findExtrema((spectra1, energy), extremaType, energyInterval, maxRf=maxRf, allowExternal=allowExternal, maxExtremumPointsDist=maxExtremumPointsDist, intensityNormForMaxExtremumPointsDist=intensityNormForMaxExtremumPointsDist, maxAdditionIter=maxAdditionIter, refineExtremum=refineExtremum, extremumInterpolationRadius=extremumInterpolationRadius, returnIndices=False, plotToFile='stableExtrema-'+extremaType+'.png')
    assert len(newSpectra) == len(spectra), f'Can\'t find {extremaType} for {len(spectra)-len(newSpectra)} spectra. Try changing search interval or expand energy interval for all spectra. See plot for details.'
    if plotResultToFolder is not None:
        plt.ioff()
        extrema_x = descr[:,0]
        extrema_y = descr[:,1]
        for i1 in range(min(100, len(spectra))):
            if i1 == 0:
                if os.path.exists(plotResultToFolder): shutil.rmtree(plotResultToFolder)
                os.makedirs(plotResultToFolder, exist_ok=True)
            # i = np.random.randint(0,len(spectra))
            i = i1
            fig, ax = plt.subplots(figsize=plotting.figsize)
            ax.plot(energy, spectra1[i], label='stable smooth')
            ax.plot(energy, spectra[i], label='spectrum')
            d = np.max(spectra[i])-np.min(spectra[i])
            ax.set_ylim(np.min(spectra[i])-d*0.1, np.max(spectra[i])+d*0.1)
            ax.scatter([extrema_x[i]], [extrema_y[i]], 10)
            ax.legend()
            fig.set_size_inches((16/3*2, 9/3*2))
            fig.savefig(plotResultToFolder+os.sep+str(i)+'.png', dpi=plotting.dpi)
            plotting.closefig(fig)
        plt.ion()
    return descr


def stableExtremaOld(spectra, energy, extremaType, energyInterval, fitByPolynomDegree=2, plotResultToFolder=None, plotBadOnly=True):
    assert extremaType in ['min', 'max'], 'invalid extremaType'
    assert fitByPolynomDegree in [2,3]
    extrema_x = np.zeros(len(spectra))
    extrema_y = np.zeros(len(spectra))
    extrema_sharpness = np.zeros(len(spectra))
    for i in range(len(spectra)):
        spectrum = utils.Spectrum(energy, spectra[i])
        if plotResultToFolder is None:
            extrema, extrema_val, d2val = findExtremumByFit(spectrum, energyInterval, fitByPolynomDegree)
        else:
            ds, poly = findExtremumByFit(spectrum, energyInterval, fitByPolynomDegree, returnPolynomGraph=True)
            extrema, extrema_val, d2val = ds
        if fitByPolynomDegree==2:
            if extremaType == 'min':
                assert d2val[0]>=0, 'There is no min for spectrum '+str(i)+' on interval '+str(energyInterval)
            else:
                assert d2val[0]<=0, 'There is no max for spectrum '+str(i)+' on interval '+str(energyInterval)
        j = -1 if extremaType == 'min' else 0
        extrema_x[i] = extrema[j]
        extrema_y[i] = extrema_val[j]
        extrema_sharpness[i] = d2val[j]
        if plotResultToFolder is not None:
            if i == 0:
                if os.path.exists(plotResultToFolder): shutil.rmtree(plotResultToFolder)
                os.makedirs(plotResultToFolder, exist_ok=True)
            if plotBadOnly and (extrema_x[i]<energy[0] or extrema_x[i]>energy[-1]):
                fig, ax = plt.subplots(figsize=plotting.figsize)
                ax.plot(energy, poly(energy), label='polynom')
                ax.plot(energy, spectra[i], label='spectrum')
                d = np.max(spectra[i])-np.min(spectra[i])
                ax.set_ylim(np.min(spectra[i])-d*0.1, np.max(spectra[i])+d*0.1)
                ax.scatter([extrema_x[i]], [extrema_y[i]], 10)
                ax.legend()
                fig.set_size_inches((16/3*2, 9/3*2))
                fig.savefig(plotResultToFolder+os.sep+str(i)+'.png', dpi=plotting.dpi)
                plotting.closefig(fig)
    return np.array([extrema_x,extrema_y,extrema_sharpness]).T


def pcaDescriptor(spectra, count=None):
    """Build pca descriptor.
    
    :param spectra: 2-d matrix of spectra (each row is one spectrum)
    :param count: count of pca descriptors to build (default: all)
    :returns: 2-d matrix (each column is one descriptor values for all spectra)
    """
    u,s,vT = np.linalg.svd(spectra.T,full_matrices=False)
    s = np.diag(s)
    C = np.dot(s,vT).T
    if count is not None: C = C[:,:min([C.shape[1],count])]
    return C


def relPcaDescriptor(spectra, energy0, Efermi, count=None):
    """Build relative pca descriptor. It is a pca descriptor for aligned spectra
    
    :param spectra: 2-d matrix of spectra (each row is one spectrum)
    :param energy: energy values (common for all spectra)
    :param Efermi: Efermi values for all spectra
    :param count: count of pca descriptors to build (default: all)
    :returns: 2-d matrix (each column is one descriptor values for all spectra)
    """
    spectra_rel = copy.deepcopy(spectra)
    energy = energy0 - Efermi[0]
    for i in range(len(spectra_rel)):
        spectra_rel[i] = np.interp(energy, energy0-Efermi[i], spectra_rel[i])
    return pcaDescriptor(spectra_rel, count)


def efermiDescriptor(spectra, energy):
    """Find Efermi and arctan grow rate for all spectra and returns as 2 column matrix
    
    :param spectra: 2-d matrix of spectra (each row is one spectrum)
    """
    d = np.zeros((spectra.shape[0],2))
    for i in range(spectra.shape[0]):
        arcTanParams, _ = curveFitting.findEfermiByArcTan(energy, spectra[i])
        d[i] = [arcTanParams['x0'], arcTanParams['a']]
    return d

    
def plot_descriptors_1d(data, spectra, energy, label_names, desc_points_names, folder):
    """Plot 1d graphs of descriptors vs labels
    
    Args:
        data (pandas dataframe):  data with desctriptors and labels
        spectra (TYPE): numpy 2d array of spectra (one row - one spectrum)
        energy (TYPE): energy values for spectra
        label_names (list): label names
        desc_points_names (list of pairs): points to plot on spectra graphs (for example: [[max energies, max intensities], [pit energies, pit intensities]])
        folder (TYPE): Description
    """
    if os.path.exists(folder): shutil.rmtree(folder)
    data = data.sample(frac=1).reset_index(drop=True)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(folder+os.sep+'by_descriptor', exist_ok=True)
    os.makedirs(folder+os.sep+'by_label', exist_ok=True)
    descriptors = set(data.columns) - set(label_names)
    for label in label_names:
        os.makedirs(folder+os.sep+'by_label'+os.sep+label, exist_ok=True)
        if label not in data.columns: continue

        fig = plotting.plotSample(energy, spectra, color_param=data[label].values, sortByColors=False, fileName=None)
        ax = fig.axes[0]
        for desc_points in desc_points_names:
            ax.scatter(data[desc_points[0]], data[desc_points[1]], 10)
        ax.set_title('Spectra colored by '+label)
        fig.savefig(folder+os.sep+'by_label'+os.sep+'xanes_spectra_'+label+'.png', dpi=plotting.dpi)
        plotting.closefig(fig)

        for d in descriptors:
            os.makedirs(folder+os.sep+'by_descriptor'+os.sep+d, exist_ok=True)
            fig, ax = plt.subplots(figsize=plotting.figsize)
            # known
            ind = pd.notnull(data[label])
            ma = np.max(data.loc[ind,label]); mi = np.min(data.loc[ind,label])
            delta = (np.random.rand(data.shape[0])*2-1)*(ma-mi)/30
            ax.scatter(data.loc[ind,d], data.loc[ind,label]+delta[ind], 600, c='black')
            ax.scatter(data.loc[ind,d], data.loc[ind,label]+delta[ind], 500, c='white')
            # text
            for i in range(data.shape[0]):
                if not np.isnan(data.loc[i,label]):
                    ax.text(data.loc[i,d], data.loc[i,label]+delta[i], str(i), ha='center', va='center', size=8)
            ax.set_xlabel(d)
            ax.set_ylabel(label)
            ax.set_xlim(plotting.getPlotLim(data.loc[ind,d]))
            ax.set_ylim(plotting.getPlotLim(data.loc[ind,label]+delta[ind]))
            fig.set_size_inches((16/3*2, 9/3*2))
            fig.savefig(folder+os.sep+'by_descriptor'+os.sep+d+os.sep+label+'.png', dpi=plotting.dpi)
            fig.savefig(folder+os.sep+'by_label'+os.sep+label+os.sep+d+'.png', dpi=plotting.dpi)
            plotting.closefig(fig)


def isClassification(data, column):
    if data[column].dtype != 'float64': return True
    ind = pd.notnull(data[column])
    return np.all(np.round(data.loc[ind,column]) == data.loc[ind,column].to_numpy())


def getQuality(data, columns, label_names, model0=None, m=1, cv_count=10, returnModels=False, printDebug=True):
    """Get cross validation quality
    
    Args:
        data (pandas dataframe):  data with desctriptors and labels
        columns (list of strings): features to use (x)
        label_names (list of strings): labels (y)
        model0 (None, optional): model to use for prediction
        m (int, optional): number of cross validation attempts (each cv composed of cv_count parts)
        cv_count (int, optional): number of parts to divide dataset in cv
        returnModels (bool, optional): whether to return models
        printDebug (boolean): show debug output
    Returns:
        quality dict{label:quality}, predictions dict{label:array of predictions}, models dict{label:model}
    """
    quality = {}; quality_std = {}
    predictions = {}
    models = {}
    X = data.loc[:, columns].to_numpy()
    if printDebug:
        print('Try predict by columns:', columns)

    def getScore(model, X, y):
        try:
            with warnings.catch_warnings(record=True) as warn:
                pred = sklearn.model_selection.cross_val_predict(model, X, y, cv=sklearn.model_selection.KFold(cv_count, shuffle=True))
        except Warning:
            pass
        # print(classification, model, y, pred)
        res = sklearn.metrics.accuracy_score(y, pred) if classification else sklearn.metrics.r2_score(y, pred)
        return res, pred
    tryParams = [{'n_estimators':40, 'min_samples_leaf':4}]
    # if len(columns) == 1:
    tryParams.append({'n_estimators':40, 'min_samples_leaf':20})
    # model = ML.Normalize(sklearn.linear_model.LogisticRegression(multi_class='ovr'), xOnly=True)
    # model = ML.Normalize(sklearn.svm.SVC(C=1), xOnly=True)
    for name in label_names:
        assert name not in columns
        y = np.ravel(data[name].to_numpy())
        classification = isClassification(data, name)
        acc = np.zeros(m)
        # print(model0)
        if model0 is None:
            try_acc = np.zeros(len(tryParams))
            try_model = [None]*len(tryParams)
            for modelParams, j in zip(tryParams, range(len(tryParams))):
                if classification:
                    try_model[j] = sklearn.ensemble.ExtraTreesClassifier(**modelParams)
                else:
                    try_model[j] = sklearn.ensemble.ExtraTreesRegressor(**modelParams)
                # print(classification, try_model[j])
                try_acc[j], pred = getScore(try_model[j], X, y)
            bestj = np.argmax(try_acc)
            acc[0] = try_acc[bestj]
            model = try_model[bestj]
            if printDebug and len(tryParams)>1:
                print('Best model params: ', tryParams[bestj], f'Delta ={try_acc[bestj]-np.min(try_acc)}')
            for i in range(1,m):
                acc[i], pred = getScore(model, X, y)
        else:
            for i in range(m):
                model = copy.deepcopy(model0)
                acc[i], pred = getScore(model, X, y)
        quality[name] = np.mean(acc)
        quality_std[name] = np.std(acc)
        predictions[name] = pred
        if returnModels:
            try:
                with warnings.catch_warnings(record=True) as warn:
                    models[name] = model.fit(X,y)
            except Warning:
                pass
        cl = 'classification' if classification else 'regression'
        if printDebug:
            print('{} - {} score: {:.2f}-{:.2f}'.format(name, cl, np.min(acc), np.max(acc)))
    if printDebug: print('')
    if returnModels:
        return quality, predictions, models, quality_std
    else:
        return quality, predictions, quality_std


def check_done(allTrys, columns):
    import itertools
    for cs in list(itertools.permutations(columns)):
        s = '|'.join(cs)
        if s in allTrys: return True
    allTrys.append('|'.join(columns))
    return False


def descriptor_quality(data, label_names, all_features, feature_subset_size=2, cv_parts_count=10, cv_repeat=5, unknown_data=None, model=None, folder='quality_by_label', printDebug=False):
    """Calculate cross-validation result for all feature subsets of the given size for all labels
    
    Args:
        data (pandas dataframe):  data with desctriptors and labels
        label_names: list of label names to predict
        all_features (list of strings): features from which subsets are taken
        feature_subset_size (int, optional): size of subsets
        cv_parts_count (int, optional): cross validation count (divide all data into cv_parts_count parts)
        cv_repeat: repeat cross validation times
        unknown_data: features with unknown labels to make prediction
        model: ML model to use
        folder (str, optional): output folder to save results
        printDebug (boolean): show debug output
    """
    qualities = {}
    for label in label_names: qualities[label] = []
    allTrys = []
    for fs in itertools.product(*([all_features]*feature_subset_size)):
        if (len(set(fs)) != feature_subset_size) or check_done(allTrys, fs):
            continue
        # if len(qualities[label])>=2: continue
        fs = list(fs)
        res = getQuality(data, fs, label_names, model0=model, cv_count=cv_parts_count, m=cv_repeat, returnModels=unknown_data is not None, printDebug=printDebug)
        quality = res[0]
        if unknown_data is None: quality_std = res[2]
        else:
            models = res[2]
            quality_std = res[3]

        for label in quality:
            res_d = {'features':' + '.join(fs), 'quality':quality[label], 'quality_std':quality_std[label]}
            if unknown_data is not None:
                tmp_model = models[label]
                res_d['predictions'] = tmp_model.predict(unknown_data.loc[:,fs])
            qualities[label].append(res_d)
    os.makedirs(folder, exist_ok=True)
    for label in qualities:
        results = sorted(qualities[label], key=lambda res_list: res_list['quality'], reverse=True)
        with open(folder+os.sep+label+'.txt', 'w') as f:
            for r in results:
                s = f"{r['quality']:.3f}±{r['quality_std']:.3f}  -  {r['features']}\n"
                f.write(s)
                print(s[:-1])
                if unknown_data is not None:
                    pred = " ".join([f"{i}:{r['predictions'][i]:.2f}" for i in range(unknown_data.shape[0])])
                    pred = f"predictions: {pred}\n"
                    f.write(pred)
                    print(pred[:-1])


def plot_descriptors_2d(data, descriptor_names, label_names, labelMaps=None, folder_prefix='', unknown=None, markersize=500, textsize=8, alpha=1, cv_count=2, plot_data_only=False, doNotPlotRemoteCount=0):
    """Plot 2d prediction map.
    
    Args:
        data (pandas dataframe):  data with desctriptors and labels
        descriptor_names (list - pair): 2 names of descriptors to use for prediction
        label_names (list): all label names to predict
        labelMaps (dict): {label: {'valueString':number, ...}, ...} - maps of label vaules to numbers
        folder_prefix (string): output folders prefix
        plot_data_only (boolean): do not predict
        doNotPlotRemoteCount (integer): calculate mean and do not plot the most remote doNotPlotRemoteCount points
        returns: saves all graphs to two folders: folder_prefix_by_label, folder_prefix_by descriptors
    """
    assert len(descriptor_names) == 2
    if labelMaps is None: labelMaps = {}
    data = data.sample(frac=1).reset_index(drop=True)
    folder = folder_prefix + '_by_descriptors'+os.sep+descriptor_names[0]+'_'+descriptor_names[1]
    os.makedirs(folder, exist_ok=True)
    folder2 = folder_prefix+'_by_label'
    os.makedirs(folder2, exist_ok=True)
    if not plot_data_only:
        quality, predictions, models, _ = getQuality(data, descriptor_names, label_names, m=1, cv_count=cv_count, returnModels=True)

    def get_color(c):
        c = (c-np.min(c)) / (np.max(c) - np.min(c))
        c = c*0.9
        return c
    colorMap = plotting.truncate_colormap('hsv', minval=0, maxval=0.9)
    x = data[descriptor_names[0]].to_numpy()
    y = data[descriptor_names[1]].to_numpy()
    if doNotPlotRemoteCount > 0:
        xm = np.median(x); ym = np.median(y)
        x_normed = (x-xm)/np.sqrt(np.median((x-xm)**2))
        y_normed = (y - ym) / np.sqrt(np.median((y - ym) ** 2))
        ind = np.argsort(-x_normed**2-y_normed**2)
        bad_ind = ind[:doNotPlotRemoteCount]
        good_ind = np.setdiff1d(np.arange(len(x)), bad_ind)
        x = x[good_ind]
        y = y[good_ind]
    for label in label_names:
        if label not in data.columns: continue
        labelData = data[label].to_numpy()
        if doNotPlotRemoteCount > 0:
            labelData = labelData[good_ind]
        os.makedirs(folder2+os.sep+label, exist_ok=True)
        fig, ax = plt.subplots(figsize=plotting.figsize)

        c = labelData
        assert np.all(pd.notnull(c))
        c_min = np.min(c); c_max = np.max(c)
        transform = lambda r: (r-c_min) / (c_max - c_min) * 0.9
        if not plot_data_only:
            # contours
            x_min, x_max = np.min(x), np.max(x)
            x_min, x_max = x_min-0.2*(x_max-x_min), x_max+0.2*(x_max-x_min)
            y_min, y_max = np.min(y), np.max(y)
            y_min, y_max = y_min-0.2*(y_max-y_min), y_max+0.2*(y_max-y_min)
            xx, yy =  np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
            preds0 = models[label].predict(np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1))))
            preds = transform(preds0.reshape(xx.shape))
        # print(label,'- classification =', ML.isClassification(data, label))
        if ML.isClassification(data, label):
            ticks = np.unique(data[label])
            # levels = transform( np.append( np.unique(data[label]), data[label].max()+1 ) )
            levels = transform( np.append( ticks-0.5, np.max(ticks)+0.5 ) )
        else:
            ticks = np.linspace(data[label].min(), data[label].max(), 10)
            delta = ticks[1]-ticks[0]
            levels = np.append(ticks-delta/2, ticks[-1]+delta/2)
            levels = transform(levels)
        if not plot_data_only:
            CF = ax.contourf(xx, yy, preds, cmap=colorMap, vmin=0, vmax=1, levels=levels, extend='both')
            # save to file
            cont_data = pd.DataFrame()
            cont_data[descriptor_names[0]] = xx.reshape(-1)
            cont_data[descriptor_names[1]] = yy.reshape(-1)
            cont_data[label] = preds0.reshape(-1)
            cont_data.to_csv(folder+'/'+label+'.csv', index=False)

        # known
        c = transform(c)
        linewidth = 1 if not plot_data_only else 0
        sc = ax.scatter(x, y, markersize, c=c, cmap=colorMap, vmin=0, vmax=1, alpha=alpha, linewidth=linewidth, edgecolor='black')
        if not plot_data_only:
            c = transform(predictions[label])
            ax.scatter(x, y, markersize/10, c=c, cmap=colorMap, vmin=0, vmax=1)
        mappable = sc if plot_data_only else CF
        cbar = fig.colorbar(mappable, ax=ax, extend='max', orientation='vertical', ticks=levels[:-1], format='%.1g')
        if label in labelMaps:
            cbarTicks = [None]*len(labelMaps[label])
            for name in labelMaps[label]:
                cbarTicks[labelMaps[label][name]] = name
            cbar.ax.set_yticklabels(cbarTicks)
        else: cbar.ax.set_yticklabels(ticks)

        # unknown
        if unknown is not None:
            if not plot_data_only:
                pred_unk = models[label].predict(unknown.loc[:,descriptor_names])
                c_params = {'c':transform(pred_unk), 'cmap':colorMap}
            else: c_params = {'c':'white'}
            ax.scatter(unknown[descriptor_names[0]], unknown[descriptor_names[1]], markersize, **c_params, vmin=0, vmax=1, edgecolor='black')
            for i in range(len(unknown)):
                ax.text(unknown.loc[i, descriptor_names[0]], unknown.loc[i, descriptor_names[1]], f'u{i}', ha='center', va='center', size=np.sqrt(markersize)/2)

        # text
        if textsize>0:
            for i in range(data.shape[0]):
                if i not in good_ind: continue
                ind = i  # data.loc[i,'Ind']
                ax.text(data.loc[i,descriptor_names[0]], data.loc[i,descriptor_names[1]], str(ind), ha='center', va='center', size=textsize)

        ax.set_xlabel(descriptor_names[0])
        ax.set_ylabel(descriptor_names[1])
        if not plot_data_only:
            qs = "{:.2f}".format(quality[label])
            ax.set_title(label+' prediction. Acc = '+qs)
            qs += '_'
        else: 
            ax.set_title(label)
            qs = ''
        ax.set_xlim(plotting.getPlotLim(x))
        ax.set_ylim(plotting.getPlotLim(y))
        plt.show(block=False)  # Even if plt.isinteractive() == True jupyter notebook doesn't show graph if in past plt.ioff/ion was called
        fig.savefig(folder+'/'+label+'.png', dpi=plotting.dpi)
        fig.savefig(folder2+os.sep+label+os.sep+qs+descriptor_names[0]+'_'+descriptor_names[1]+'.png', dpi=plotting.dpi)
        plotting.closefig(fig)


@ignore_warnings(category=ConvergenceWarning)
def getLinearAnalyticModel(data, features, label, l1_ratio, try_alpha=None, cv_count=10):
    if try_alpha is None: try_alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    data = data.sample(frac=1).reset_index(drop=True)
    label_data = data[label]
    scores = np.zeros(len(try_alpha))
    models = [None] * len(try_alpha)
    for i in range(len(try_alpha)):
        models[i] = sklearn.linear_model.ElasticNet(alpha=try_alpha[i], l1_ratio=l1_ratio, fit_intercept=True, normalize=False)
        scores[i] = np.mean(sklearn.model_selection.cross_val_score(models[i], data.loc[:, features], label_data, cv=cv_count))
    i_best = np.argmax(scores)
    model = models[i_best]
    score = scores[i_best]
    if score > 0.5:
        model.fit(data.loc[:, features], label_data)
        model_str = ''
        for i in range(len(features)):
            if abs(model.coef_[i]) >= 0.005:
                model_str += f'{model.coef_[i]:+.2f}*' + features[i]
        if abs(model.intercept_) >= 0.005:
            model_str += f'{model.intercept_:+.2f}'
        model_string = f'Score={scores[i_best]:.2f}. Model: {label} = {model_str}'
    else: model_string = f'No linear formula for {label}'
    return score, model, model_string


@ignore_warnings(category=ConvergenceWarning)
def getAnalyticFormulasForGivenFeatures(data0, features0, label_names, l1_ratio=1, try_alpha=None, cv_count=10, normalize=True, output_file='formulas.txt'):
    if try_alpha is None: try_alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    data0 = data0.sample(frac=1).reset_index(drop=True)
    if isinstance(features0, list):
        features0 = np.array(features0)
    if normalize:
        mean = data0.mean()
        std = data0.std()
        data = (data0-mean)/std
    data2 = ML.transformFeatures2Quadric(data.loc[:, features0], addConst=False)
    data2_features = np.array(data2.columns)
    for label in label_names: data2[label] = data[label]
    dataSets = [data, data2]
    featureSets = [features0, data2_features]
    result_file = open(output_file, 'w')
    for label in label_names:
        # label_data = (data[label]+data[label].min())**2
        label_data = data[label]
        for di in range(len(dataSets)):
            d = dataSets[di]
            features = featureSets[di]
            # check possibility
            model = ExtraTreesRegressor(n_estimators=100, random_state=0, min_samples_leaf=10)
            # model = lgb.LGBMRegressor(num_leaves=31, learning_rate=0.02, n_estimators=100)
            score = np.mean(sklearn.model_selection.cross_val_score(model, d, label_data, cv=cv_count))
            if score <= 0.5:
                model_string = f'{label} can\'t be expressed in terms of features: '+str(features)
            else:
                score, model, model_string = getLinearAnalyticModel(d, features, label, l1_ratio=l1_ratio, try_alpha=try_alpha, cv_count=cv_count)
            print(model_string)
            result_file.write(model_string+'\n')

            if score > 0.5:
                # get simple models
                ind = np.argsort(model.coef_)
                max_print_num = 5
                print_num = 0
                for i in range(len(ind)-1):
                    fs = features[ind[-i-1:]]
                    score, model, model_string = getLinearAnalyticModel(d, fs, label, l1_ratio=l1_ratio, try_alpha=try_alpha, cv_count=cv_count)
                    if score > 0.5:
                        print(' '*8 + model_string)
                        result_file.write(' '*8 + model_string + '\n')
                        print_num += 1
                        if print_num > max_print_num: break
    result_file.close()

import os, copy, sklearn, shutil, itertools, statsmodels, warnings, scipy.signal, scipy.interpolate
from . import ML, mixture, utils, plotting, smoothLib, curveFitting
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt


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

        fig, ax = plotting.createfig()
        alpha = min(1/len(goodBadIndices)*100,1)
        for (spectrum, extremum, label) in goodBadIndices:
            if label == 'good':
                ax.plot(energy, intensities[spectrum], color='blue', lw=.3, alpha=alpha)
        for x, y in zip(extremaPoints[0], extremaPoints[1]):
            ax.plot(x, y, marker='o', markersize=3, color="red", alpha=alpha)
        for (spectrum, extremum, label) in goodBadIndices:
            if label == 'bad':
                ax.plot(energy, intensities[spectrum], color='black', lw=.3)
        plotting.savefig(plotToFile, fig)
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
    # print('First split: ', len(good), len(bad))

    # try to identify good spectra in bad list and move them
    iterations = 0
    while iterations < maxAdditionIter or maxAdditionIter == -1:
        newGood, newBad = moveBadSpectraToGood(good, bad, energy, intensities, extrema, energyInterval)
        added = len(newGood) - len(good)
        # print('Iteration = %i, added %i spectra' % (iterations + 1, added))
        shouldContinue = added > 0
        good = newGood
        bad = newBad
        iterations += 1

        if not shouldContinue:
            break
    assert len(good) > 0, f'No good spectra'
    good = sorted(good, key=lambda pair: pair[0])
    extremaPoints, derivatives = getFinalExtremaAndDerivatives(good, energy, intensities, refineExtremum, extremumInterpolationRadius, comparator)
    goodSpectrumIndices = np.array(list(map(lambda x: x[0], good)))
    descr = np.vstack((extremaPoints, derivatives)).T

    # plot results
    plot(good, bad, energy, intensities, extremaPoints)

    ensureCorrectResults(intensities, good, bad, extremaPoints, derivatives)
    if isinstance(sample, ML.Sample):
        newSpectra = sample.spectra.loc[goodSpectrumIndices].reset_index(drop=True)
        newParams = sample.params.loc[goodSpectrumIndices].reset_index(drop=True)
        newSample = ML.Sample(newParams, newSpectra)
    else:
        newSample = intensities[goodSpectrumIndices]

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


def stableExtrema(spectra, energy, extremaType, energyInterval, plotFolderPrefix=None, plotIndividualSpectra=False, extraPlotInd=None, maxRf=.1, allowExternal=True, maxExtremumPointsDist=5, intensityNormForMaxExtremumPointsDist=1, maxAdditionIter=-1, refineExtremum=True, extremumInterpolationRadius=10, smoothRad=5):
    assert extremaType in ['min', 'max'], 'invalid extremaType'
    if plotFolderPrefix is None: plotFolderPrefix = 'stable_'+extremaType
    spectra1 = np.copy(spectra)
    for i in range(len(spectra)):
        spectra1[i] = smoothLib.simpleSmooth(energy, spectra1[i], smoothRad, kernel='Gauss')
    newSpectra, descr, goodSpectrumIndices = findExtrema((spectra1, energy), extremaType, energyInterval, maxRf=maxRf, allowExternal=allowExternal, maxExtremumPointsDist=maxExtremumPointsDist, intensityNormForMaxExtremumPointsDist=intensityNormForMaxExtremumPointsDist, maxAdditionIter=maxAdditionIter, refineExtremum=refineExtremum, extremumInterpolationRadius=extremumInterpolationRadius, returnIndices=True, plotToFile=plotFolderPrefix + os.sep + 'stableExtrema-' + extremaType + '.png')
    if len(newSpectra) != len(spectra):
        warnings.warn(f'Can\'t find {extremaType} for {len(spectra)-len(newSpectra)} spectra. Try changing search interval or expand energy interval for all spectra. See plot for details.')
    if plotIndividualSpectra:
        indSpectraFolder = plotFolderPrefix + os.sep + 'individual_spectra_'+extremaType
        extrema_x = descr[:,0]
        extrema_y = descr[:,1]
        maxNumToPlot = 100
        if len(spectra) > maxNumToPlot:
            rng = np.random.default_rng(0)
            plot_inds = rng.choice(len(spectra), maxNumToPlot)
        else:
            plot_inds = np.arange(len(spectra))
        if extraPlotInd is not None:
            plot_inds = np.array(list(set(plot_inds).union(set(extraPlotInd))))
        if os.path.exists(indSpectraFolder): shutil.rmtree(indSpectraFolder)
        for i in plot_inds:
            def plotMoreFunction(ax):
                ax.scatter([extrema_x[i]], [extrema_y[i]], 50)
            plotting.plotToFile(energy, spectra1[i], 'stable smooth', energy, spectra[i], 'spectrum', plotMoreFunction=plotMoreFunction, fileName=indSpectraFolder + os.sep + str(i) + '.png', title=f'{extremaType}_x = {descr[i,0]:.1f} {extremaType}_y = {"%.3g" % descr[i,1]} d2 = {"%.3g" % descr[i,2]}')
    return descr, goodSpectrumIndices


def pcaDescriptor(spectra, count=None, returnU=False, U=None):
    """Build pca descriptor.
    
    :param spectra: 2-d matrix of spectra (each row is one spectrum)
    :param count: count of pca descriptors to build (default: all)
    :param returnU: return also pca building matrix u
    :returns: 2-d matrix (each column is one descriptor values for all spectra)
    """
    if U is None:
        U,s,vT = np.linalg.svd(spectra.T,full_matrices=False)
    C = np.dot(spectra, U)
    if count is not None: C = C[:,:min([C.shape[1],count])]
    if returnU: return C, U
    else: return C


def relPcaDescriptor(spectra, energy0, Efermi, prebuildData=None, count=None, returnU=False):
    """Build relative pca descriptor. It is a pca descriptor for aligned spectra
    
    :param spectra: 2-d matrix of spectra (each row is one spectrum)
    :param energy0: energy values (common for all spectra)
    :param Efermi: Efermi values for all spectra
    :param prebuildData: tuple(U, relEnergy) from other built descriptor
    :param count: count of pca descriptors to build (default: all)
    :param returnU: return also pca building matrix u
    :returns: 2-d matrix (each column is one descriptor values for all spectra)
    """
    spectra_rel = copy.deepcopy(spectra)
    relEnergy = energy0 - Efermi[0] if prebuildData is None else prebuildData[1]
    for i in range(len(spectra_rel)):
        spectra_rel[i] = np.interp(relEnergy, energy0-Efermi[i], spectra_rel[i])
    if prebuildData is not None: U = prebuildData[0]
    else: U = None
    if returnU:
        return pcaDescriptor(spectra_rel, count, returnU, U) + (relEnergy,)
    else:
        return pcaDescriptor(spectra_rel, count, returnU, U)


def efermiDescriptor(spectra, energy):
    """Find Efermi and arctan grow rate for all spectra and returns as 2 column matrix
    
    :param spectra: 2-d matrix of spectra (each row is one spectrum)
    :param energy:
    :param returnArctan: whether to return arctan y
    """
    d = np.zeros((spectra.shape[0],2))
    arctan_y = np.zeros(spectra.shape)
    for i in range(spectra.shape[0]):
        arcTanParams, arctan_y[i] = curveFitting.findEfermiByArcTan(energy, spectra[i])
        d[i] = [arcTanParams['x0'], arcTanParams['a']]
    return d, arctan_y


def addDescriptors(sample, descriptors):
    """
    :param sample: Sample instance
    :param descriptors: list of str (descriptor type) or dict{'type':.., 'columnName':.., 'arg1':.., 'arg2':.., ...}. Possible descriptor types: 'stableExtrema' (params - see. stableExtrema function), 'efermi', 'pca', 'rel_pca', 'max', 'min', 'variation', 'polynom'
    :return: new sample with descriptors, goodSpectrumIndices
    """
    # canonizations
    newD = []
    for d in descriptors:
        if isinstance(d, str): newD.append({'type':d})
        else: newD.append(d)
    descriptors = newD
    goodSpectrumIndices_all = None
    unique_names = []
    for d in descriptors:
        typ = d['type']
        params = copy.deepcopy(d)
        del params['type']
        name = typ
        if 'columnName' in params:
            name = params['columnName']
            del params['columnName']
        name1 = name
        i = 1
        while name1 in unique_names:
            name1 = f'{name}{i}'
            i += 2
        name = name1
        unique_names.append(name)

        if typ == 'stableExtrema':
            assert 'extremaType' in params
            if name == typ:
                name = params['extremaType']
            ext_e = name+'_e'
            ext_int = name+'_i'
            ext_d2 = name+'_d2'
            assert (ext_e not in sample.paramNames) and (ext_int not in sample.paramNames) and (ext_d2 not in sample.paramNames), f'Duplicate descriptor names while adding {ext_e}, {ext_int}, {ext_d2} to {sample.paramNames}. Use columnName argument in descriptor parameters'
            ext, goodSpectrumIndices = stableExtrema(sample.spectra, sample.energy, **params)
            assert np.all(np.diff(goodSpectrumIndices) >= 0)
            if goodSpectrumIndices_all is None: goodSpectrumIndices_all = goodSpectrumIndices
            else: goodSpectrumIndices_all = np.intersect1d(goodSpectrumIndices_all, goodSpectrumIndices)
            if len(goodSpectrumIndices) != sample.getLength():
                known, unknown, indKnown, indUnknown = sample.splitUnknown(returnInd=True)
                common = np.intersect1d(indUnknown, goodSpectrumIndices)
                assert len(common) == len(indUnknown), f"Can\'t find {params['extremaType']} for unknown spectra. Try changing search energy interval or expand energy interval for all spectra. See plot for details."

            def expandByZeros(col):
                r = np.zeros(sample.getLength())
                r[goodSpectrumIndices] = col
                return r
            sample.addParam(paramName=ext_e, paramData=expandByZeros(ext[:, 0]))
            sample.addParam(paramName=ext_int, paramData=expandByZeros(ext[:, 1]))
            sample.addParam(paramName=ext_d2, paramData=expandByZeros(ext[:, 2]))
        elif typ in ['max', 'min']:
            extr_es = np.zeros(sample.getLength())
            extr_is = np.zeros(sample.getLength())
            extr_d2 = np.zeros(sample.getLength())
            sign = +1 if typ == 'max' else -1
            smoothRad = params['smoothRad'] if 'smoothRad' in params else 5
            for i in range(sample.getLength()):
                sp = sample.getSpectrum(i=i)
                sp.intensity = smoothLib.simpleSmooth(sp.energy, sp.intensity, sigma=smoothRad, kernel='Gauss')
                ps = sample.params.loc[i]
                _, all_extr_ind = utils.argrelmax((sign*sp).intensity, returnAll=True)
                if 'constrain' in params:
                    all_extr_ind1 = []
                    for extr_ind in all_extr_ind:
                        if params['constrain'](sp.energy[extr_ind], sp.intensity[extr_ind], sp, ps):
                            all_extr_ind1.append(extr_ind)
                    all_extr_ind = all_extr_ind1

                def defaultSelector(extr_energies, extr_intensities, spectrum, params):
                    if len(extr_energies) > 0:
                        best_ind = np.argmax(sign * extr_intensities)
                        return extr_energies[best_ind], extr_intensities[best_ind]
                    else:
                        best_ind = np.argmax(sign * spectrum.intensity)
                        return spectrum.energy[best_ind], spectrum.intensity[best_ind]
                selector = params['selector'] if 'selector' in params else defaultSelector
                e,inte = (sp.energy[all_extr_ind], sp.intensity[all_extr_ind]) if len(all_extr_ind)>0 else ([],[])
                extr_e, _ = selector(e, inte, sp, ps)
                extr_i = np.where(sp.energy >= extr_e)[0][0]
                ai = max(0, extr_i-2)
                bi = min(ai+5, len(sp.energy))
                p = np.polyfit(sp.energy[ai:bi], sp.intensity[ai:bi], 2)
                assert len(p) == 3
                extr_es[i] = -p[1]/(2*p[0])
                extr_is[i] = np.polyval(p, extr_es[i])
                extr_d2[i] = 2*p[0]
            sample.addParam(paramName=name+'_e', paramData=extr_es)
            sample.addParam(paramName=name+'_i', paramData=extr_is)
            sample.addParam(paramName=name+'_d2', paramData=extr_d2)
        elif typ == 'variation':
            var = np.zeros(sample.getLength())
            smoothRad = params['smoothRad'] if 'smoothRad' in params else 5
            for i in range(sample.getLength()):
                sp = sample.getSpectrum(i=i)
                sp.intensity = smoothLib.simpleSmooth(sp.energy, sp.intensity, sigma=smoothRad, kernel='Gauss')
                energyInterval = params['energyInterval'] if 'energyInterval' in params else [sp.energy[0], sp.energy[-1]]
                if not isinstance(energyInterval, list):
                    assert callable(energyInterval), 'energyInterval should be list or function(spectrum, params), which returns list'
                    energyInterval = energyInterval(sp, sample.params.loc[i])
                sp = sp.limit(energyInterval)
                e, y = sp.energy, sp.intensity
                diff = (y[1:] - y[:-1]) / (e[1:] - e[:-1])
                var[i] = utils.integral((e[1:] + e[:-1])/2, np.abs(diff))
            sample.addParam(paramName=name, paramData=var)
        elif typ == 'efermi':
            efermi, arctan_y = efermiDescriptor(sample.spectra.to_numpy(), sample.energy)
            if 'plotFolder' in params:
                plotFolder = params['plotFolder']
                if 'maxNumToPlot' not in params: maxNumToPlot = 100
                else: maxNumToPlot = params['maxNumToPlot']
                if sample.getLength() > maxNumToPlot:
                    rng = np.random.default_rng(0)
                    plot_inds = rng.choice(sample.getLength(), maxNumToPlot)
                else:
                    plot_inds = np.arange(sample.getLength())
                if 'extraPlotInd' in params:
                    plot_inds = np.array(list(set(plot_inds).union(set(params['extraPlotInd']))))
                if os.path.exists(plotFolder): shutil.rmtree(plotFolder)
                for i in plot_inds:
                    plotting.plotToFile(sample.energy, sample.spectra.loc[i].to_nupy(), 'spectrum', sample.energy, arctan_y[i], 'arctan', fileName=f'{plotFolder}/{i}.png', title=f'efermi={efermi[i, 0]:.1f} efermiRate={efermi[i, 1]}')
            sample.addParam(paramName=f'{name}_e', paramData=efermi[:, 0])
            sample.addParam(paramName=f'{name}_slope', paramData=efermi[:, 1])
        elif typ == 'pca':
            assert set(params.keys()) <= {'usePcaPrebuildData', 'count', 'fileName'}, 'Wrong param names: '+str(set(params.keys()) - {'usePcaPrebuildData', 'count', 'fileName'})
            assert 'usePcaPrebuildData' in params, "Use pca in the following way: {'type':'pca', 'count':3, 'usePcaPrebuildData':True/False, 'fileName':'?????.pkl'}"
            count = params['count'] if 'count' in params else 3
            if 'energyInterval' in params:
                spectra = sample.limit(energyRange=params['energyInterval'], inplace=False).spectra.to_numpy()
            else:
                spectra = sample.spectra.to_numpy()
            if params['usePcaPrebuildData']:
                assert 'fileName' in params
                pca_u = utils.load_pkl(params['fileName'])
                pca = pcaDescriptor(spectra, count=count, U=pca_u)
            else:
                pca, pca_u = pcaDescriptor(spectra, count=count, returnU=True)
                if 'fileName' in params:
                    utils.save_pkl(pca_u, params['fileName'])
            for j in range(3):
                sample.addParam(paramName=f'{name}{j+1}', paramData=pca[:, j])
        elif typ == 'rel_pca':
            assert set(params.keys()) <= {'usePcaPrebuildData', 'count', 'fileName'}
            assert 'usePcaPrebuildData' in params, "Use rel_pca in the following way: {'type':'rel_pca', 'count':3, 'usePcaPrebuildData':True/False, 'fileName':'?????.pkl'}"
            count = params['count'] if 'count' in params else 3
            if 'energyInterval' in params:
                sample1 = sample.limit(energyRange=params['energyInterval'], inplace=False)
                spectra = sample1.spectra.to_numpy()
                energy = sample1.energy
            else:
                spectra = sample.spectra.to_numpy()
                energy = sample.energy
            if 'efermi' not in sample.paramNames:
                efermi, _ = efermiDescriptor(spectra, energy)
                efermi = efermi[:, 0]
            else: efermi = sample.params['efermi']
            if params['usePcaPrebuildData']:
                assert 'fileName' in params
                relpca_u, relEnergy = utils.load_pkl(params['fileName'])
                relpca = relPcaDescriptor(spectra, energy, efermi, count=count, prebuildData=(relpca_u, relEnergy))
            else:
                relpca, relpca_u, relEnergy = relPcaDescriptor(spectra, energy, efermi, count=count, returnU=True)
                if 'fileName' in params:
                    utils.save_pkl((relpca_u, relEnergy), params['fileName'])
            for j in range(count):
                sample.addParam(paramName=f'{name}{j+1}', paramData=relpca[:, j])
        elif typ == 'polynom':
            deg = params['deg'] if 'deg' in params else 3
            descr = np.zeros((sample.getLength(), deg+1))
            for i in range(sample.getLength()):
                sp = sample.getSpectrum(i=i)
                energyInterval = params['energyInterval'] if 'energyInterval' in params else [sp.energy[0], sp.energy[-1]]
                if not isinstance(energyInterval, list):
                    assert callable(energyInterval), 'energyInterval should be list or function(spectrum, params), which returns list'
                    energyInterval = energyInterval(sp, sample.params.loc[i])
                sp = sp.limit(energyInterval)
                descr[i] = np.polyfit((sp.energy-sp.energy[0])/(sp.energy[-1]-sp.energy[0]), sp.intensity, deg)
            for j in range(deg+1):
                sample.addParam(paramName=f'{name}_{j}', paramData=descr[:, j])
        else:
            assert False, f"Unknown descriptor type {typ}"

    if goodSpectrumIndices_all is not None and len(goodSpectrumIndices_all) != sample.getLength():
        sample = sample.takeRows(goodSpectrumIndices_all)
    if goodSpectrumIndices_all is None: goodSpectrumIndices_all = np.arange(sample.getLength())
    return sample, goodSpectrumIndices_all


def plotDescriptors1d(data, spectra, energy, label_names, desc_points_names=None, folder='.'):
    """Plot 1d graphs of descriptors vs labels
    
    Args:
        data (pandas dataframe):  data with desctriptors and labels
        spectra (TYPE): numpy 2d array of spectra (one row - one spectrum)
        energy (TYPE): energy values for spectra
        label_names (list): label names
        desc_points_names (list of pairs): points to plot on spectra sample graph (for example: [[max energies, max intensities], [pit energies, pit intensities]])
        folder (TYPE): Description
    """
    # if os.path.exists(folder): shutil.rmtree(folder)
    if desc_points_names is None: desc_points_names = []
    data = data.sample(frac=1).reset_index(drop=True)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(folder+os.sep+'by_descriptor', exist_ok=True)
    os.makedirs(folder+os.sep+'by_label', exist_ok=True)
    descriptors = set(data.columns) - set(label_names)
    for label in label_names:
        os.makedirs(folder+os.sep+'by_label'+os.sep+label, exist_ok=True)
        if label not in data.columns: continue

        fig = plotting.plotSample(energy, spectra, colorParam=data[label].values, sortByColors=False, fileName=None)
        ax = fig.axes[0]
        for desc_points in desc_points_names:
            ax.scatter(data[desc_points[0]], data[desc_points[1]], 10)
        ax.set_title('Spectra colored by '+label)
        plotting.savefig(folder+os.sep+'by_label'+os.sep+'xanes_spectra_'+label+'.png', fig)
        plotting.closefig(fig)

        for d in descriptors:
            os.makedirs(folder+os.sep+'by_descriptor'+os.sep+d, exist_ok=True)
            fig, ax = plotting.createfig()
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
            plotting.savefig(folder+os.sep+'by_descriptor'+os.sep+d+os.sep+label+'.png', fig)
            plotting.savefig(folder+os.sep+'by_label'+os.sep+label+os.sep+d+'.png', fig)
            plotting.closefig(fig)


def getXYFromSample(sample, features, label_names, textColumn=None):
    """Construct X, y from sample

    Args:
        sample:  sample or DataFrame with descriptors and labels
        features (list of strings or string): features to use, or 'spectra' or 'spectra_d_i1,i2,i3,...' (spectra derivatives together), or ['spType1 spectra_d_i1,i2', 'spType2 spectra_d_i1,i2', ...]
        label_names (list)
        textColumn (str): title of column to use as names
    """
    if isinstance(sample, pd.DataFrame):
        data = sample
        assert not isinstance(features, str)
        assert set(features) <= set(data.columns), str(features) + " is not subset of " + str(data.columns)
        assert not (set(label_names) & set(features))
        X = data.loc[:, features].to_numpy()
    else:
        assert isinstance(sample, ML.Sample)
        data = sample.params
        if isinstance(features, str): features = [features]
        X = None
        for fs in features:
            if 'spectra' in fs:
                # determine spType
                if fs[:len('spectra')] != 'spectra':
                    i = fs.rfind(' ')
                    assert i>0
                    spType = fs[:i]
                    fs = fs[i+1:]
                else: spType = sample.getDefaultSpType()
                if fs == 'spectra': fs = 'spectra_d_0'
                assert fs[:10] == 'spectra_d_', features
                diffs = [int(s) for s in fs[10:].split(',')]
                X1 = getSpectraFeatures(sample, diff=diffs, spType=spType)
            else:
                X1 = data.loc[:, fs].to_numpy()
                if len(X1.shape) == 1: X1 = X1.reshape(-1,1)
                assert not (set(label_names) & set(fs))
            if X is None: X = X1
            else: X = np.hstack((X,X1))
    y = data.loc[:, label_names].to_numpy()
    if textColumn is None:
        return X, y
    else:
        return X, y, data.loc[:, textColumn].to_numpy()


def getQuality(sample, features, label_names, makeMixtureParams=None, model_class=None, model_regr=None, m=1, cv_count=10, returnModels=False, printDebug=True):
    """Get cross validation quality
    
    Args:
        sample:  sample (for mixture) or DataFrame with descriptors and labels
        features (list of strings or string): features to use (x), or 'spectra' or 'spectra_d_i1,i2,i3,...' (spectra derivatives together)
        label_names (list of strings): labels (y)
        makeMixtureParams: arguments for mixture.generateMixtureOfSample excluding sample, size, label_names, randomSeed
        model_class: model for classification
        model_regr: model for regression
        m (int, optional): number of cross validation attempts (each cv composed of cv_count parts)
        cv_count (int, optional): number of parts to divide dataset in cv
        returnModels (bool, optional): whether to return models
        printDebug (boolean): show debug output
    Returns:
        dict {label: dict{'quality':..., 'predictedLabels':..., 'trueLabels':...(for mixture), 'model':..., 'quality_std':...}}
    """
    mix = makeMixtureParams is not None
    quality = {}; quality_std = {}
    predictedLabels = {}; trueLabels = {}
    models = {}
    X,Y = getXYFromSample(sample, features, label_names)
    if printDebug:
        print('Try predict by:', features)
    n_estimators = 40
    if X.shape[1] > 5: n_estimators = 100
    if X.shape[1] > 10: n_estimators = 200
    tryParams = [{'n_estimators': n_estimators, 'min_samples_leaf': 4}, {'n_estimators': n_estimators, 'min_samples_leaf': 1}]
    if X.shape[0] > 500: tryParams = [{'n_estimators': n_estimators}]
    for il, label in enumerate(label_names):
        y = Y[:,il]
        classification = ML.isClassification(y)
        if mix: classification = False
        model0 = model_class if classification else model_regr
        acc = [None]*m
        if model0 is None:
            try_acc = np.zeros(len(tryParams))
            try_model = [None]*len(tryParams)
            for modelParams, j in zip(tryParams, range(len(tryParams))):
                if classification:
                    try_model[j] = sklearn.ensemble.ExtraTreesClassifier(**modelParams)
                else:
                    try_model[j] = sklearn.ensemble.ExtraTreesRegressor(**modelParams)
                if mix:
                    try_acc_mix, trueVals, pred, mod = mixture.score_cv(try_model[j], sample, features, label, label_names, makeMixtureParams, testRatio=0.5, repetitions=1, model_class=sklearn.ensemble.ExtraTreesClassifier(**modelParams))
                    try_acc[j] = try_acc_mix['avgLabels']
                else:
                    try_acc[j], pred = ML.score_cv(try_model[j], X, y, cv_count)
                    trueVals = y
            bestj = np.argmax(try_acc)
            acc[0] = try_acc[bestj]
            model = try_model[bestj]
            model_class = sklearn.ensemble.ExtraTreesClassifier(**tryParams[bestj])
            if printDebug and len(tryParams)>1:
                print('Best model params: ', tryParams[bestj], f'Delta ={try_acc[bestj]-np.min(try_acc)}')

        start_i = 0 if mix or model0 is not None else 1
        for i in range(start_i, m):
            if model0 is not None: model = copy.deepcopy(model0)
            if mix:
                acc[i], trueVals, pred, mod = mixture.score_cv(model, sample, features, label, label_names, makeMixtureParams, testRatio=1 / cv_count, repetitions=cv_count, model_class=model_class)
            else:
                acc[i], pred = ML.score_cv(model, X, y, cv_count)
                trueVals = y
        if mix:
            quality[label] = {problemType: np.mean([acc[i][problemType] for i in range(m)], axis=0) for problemType in acc[0]}
            quality_std[label] = {problemType: np.std([acc[i][problemType] for i in range(m)], axis=0) for problemType in acc[0]}
        else:
            quality[label] = np.mean(acc)
            quality_std[label] = np.std(acc)
        predictedLabels[label] = pred
        trueLabels[label] = trueVals
        if returnModels:
            if mix:
                models[label] = mod
            else:
                try:
                    with warnings.catch_warnings(record=True) as warn:
                        models[label] = model.fit(X,y)
                except Warning:
                    pass
        cl = 'classification' if classification else 'regression'
        if printDebug:
            if mix:
                print(f'{label} - {cl} score: {quality[label]["avgLabels"]:.2f}+-{quality_std[label]["avgLabels"]:.2f}')
            else:
                print(f'{label} - {cl} score: {np.min(acc):.2f}-{np.max(acc):.2f}')
    if printDebug: print('')
    result = {label:{'quality':quality[label], 'predictedLabels':predictedLabels[label], 'trueLabels':trueLabels[label], 'quality_std':quality_std[label]} for label in label_names}
    if returnModels:
        for label in label_names:
            result[label]['model'] = models[label]
    return result


def plot_quality_2d(features, label_data, markersize=None, alpha=None, title='', fileName=None, plotMoreFunction=None, plot_axes=None):
    """

    :param features: DataFrame to use as features
    :param label_data:
    :param markersize:
    :param alpha:
    :param title:
    :param fileName:
    :param plotMoreFunction: functions(ax) called after plotting before saving file
    :param plot_axes: dict{'x_name', 'x_data', 'y_name', 'y_data'} - if you want to plot quality on another axes than features[0,1]
    :return:
    """
    features = copy.deepcopy(features)
    if fileName is None:
        if plot_axes is None:
            fileName = f'quality_{features.columns[0]}_{features.columns[1]}.png'
    feature_columns = features.columns
    features['label'] = label_data
    quality, predictions, quality_std = getQuality(features, feature_columns, ['label'], returnModels=False, printDebug=False)
    title += f' {quality["label"]:.2f}'
    if plot_axes is None:
        plotting.scatter(features[features.columns[0]], features[features.columns[1]], color=np.abs(predictions['label']-label_data), markersize=markersize, alpha=alpha, title=title, xlabel=features.columns[0], ylabel=features.columns[1], fileName=fileName, plotMoreFunction=plotMoreFunction)
    else:
        plotting.scatter(plot_axes['x_data'], plot_axes['y_data'], color=np.abs(predictions['label'] - label_data), markersize=markersize, alpha=alpha, title=title, xlabel=plot_axes['x_name'], ylabel=plot_axes['y_name'], fileName=fileName, plotMoreFunction=plotMoreFunction)


def check_done(allTrys, columns):
    import itertools
    for cs in list(itertools.permutations(columns)):
        s = '|'.join(cs)
        if s in allTrys: return True
    allTrys.append('|'.join(columns))
    return False


def descriptorQuality(data, label_names, all_features, feature_subset_size=2, cv_parts_count=10, cv_repeat=5, unknown_data=None, textColumn=None, model_class=None, model_regr=None, folder='quality_by_label', printDebug=False):
    """Calculate cross-validation result for all feature subsets of the given size for all labels
    
    Args:
        data (pandas dataframe):  data with descriptors and labels
        label_names: list of label names to predict
        all_features (list of strings): features from which subsets are taken
        feature_subset_size (int, optional): size of subsets
        cv_parts_count (int, optional): cross validation count (divide all data into cv_parts_count parts)
        cv_repeat: repeat cross validation times
        unknown_data: features with unknown labels to make prediction
        textColumn: exp names in unknown_data
        model_class: model for classification
        model_regr: model for regression
        folder (str, optional): output folder to save results
        printDebug (boolean): show debug output
    """
    assert set(label_names) < set(data.columns)
    assert set(all_features) < set(data.columns)
    qualities = {}
    for label in label_names: qualities[label] = []
    allTrys = []
    for fs in itertools.product(*([all_features]*feature_subset_size)):
        if (len(set(fs)) != feature_subset_size) or check_done(allTrys, fs):
            continue
        # if len(qualities[label])>=2: continue
        fs = list(fs)
        # returns dict {label: dict{'quality':..., 'predictions':..., 'trueLabels':...(for mixture), 'model':..., 'quality_std':...}}
        getQualityResult = getQuality(data, fs, label_names, model_class=model_class, model_regr=model_regr, cv_count=cv_parts_count, m=cv_repeat, returnModels=unknown_data is not None, printDebug=printDebug)
        for label in getQualityResult:
            quality = getQualityResult[label]['quality']
            quality_std = getQualityResult[label]['quality_std']
            res_d = {'features':','.join(fs), 'quality':quality, 'quality_std':quality_std}
            if unknown_data is not None:
                model = getQualityResult[label]['model']
                res_d['predictions'] = model.predict(unknown_data.loc[:,fs].to_numpy())
            qualities[label].append(res_d)
    os.makedirs(folder, exist_ok=True)
    for label in qualities:
        results = sorted(qualities[label], key=lambda res_list: res_list['quality'], reverse=True)
        with open(folder+os.sep+label+'.csv', 'w') as f:
            if unknown_data is None:
                for r in results:
                    f.write(f"{r['quality']:.3f}±{r['quality_std']:.3f}  -  {r['features']}\n")
            else:
                def getRow(col1, col2, other):
                    return f'{col1};{col2};' + ';'.join(other) + '\n'
                s = getRow('unknown', 'true value', [r['features'] for r in results])
                s += getRow('quality', '', [f"{r['quality']:.3f}±{r['quality_std']:.3f}" for r in results])
                trueValues = unknown_data.loc[:, label]
                predicted = np.zeros((unknown_data.shape[0], len(results)))
                for i in range(unknown_data.shape[0]):
                    if textColumn is None:  name = f'{i}'
                    else: name = unknown_data.loc[i, textColumn]
                    true = str(unknown_data.loc[i, label])
                    s += getRow(name, true, ["%.3g" % r['predictions'][i] for r in results])
                    for j in range(len(results)):
                        predicted[i,j] = results[j]['predictions'][i]
                goodInd = np.where(~np.isnan(trueValues))[0]
                true_count = len(goodInd)
                if true_count > 0:
                    trueValues = trueValues[goodInd]
                    predicted = predicted[goodInd,:]
                    if ML.isClassification(data[label]):
                        accuracy = ['%.3g' % (np.sum(trueValues == predicted[:,j])/predicted.shape[0]) for j in range(predicted.shape[1])]
                        s += getRow('accuracy by exp', '1', accuracy)
                    else:
                        mean = np.mean(trueValues)
                        meanErr = np.sum((trueValues-mean)**2)
                        quality = ['%.3g' % (1-np.sum((trueValues-predicted[:,j])**2)/meanErr) for j in range(predicted.shape[1])]
                        s += getRow('quality by exp', '1', quality)
                f.write(s)
        # sort by 'quality by exp'
        if unknown_data is not None:
            resData = pd.read_csv(folder + os.sep + label + '.csv', sep=';')
            resData.to_excel(folder + os.sep + label + '.xlsx', index=False)
            n = resData.shape[0]
            if resData.loc[n - 1, 'unknown'] in ['accuracy by exp', 'quality by exp']:
                expQualities = [-float(resData.loc[n - 1, resData.columns[j]]) for j in range(2, resData.shape[1])]
                ind = np.argsort(expQualities)
                data1 = pd.DataFrame()
                data1['unknown'] = resData['unknown']
                data1['true value'] = resData['true value']
                for jj in range(len(expQualities)):
                    j = ind[jj] + 2
                    col = resData.columns[j]
                    data1[col] = resData[col]
                data1.to_excel(folder + os.sep + label + '_sort_by_exp.xlsx', index=False)


def plotDescriptors2d(data, descriptor_names, label_names, labelMaps=None, folder_prefix='', unknown=None, markersize=None, textsize=None, alpha=None, cv_count=2, plot_only='', doNotPlotRemoteCount=0, textColumn=None, additionalMapPlotFunc=None, cmap='seaborn husl', edgecolor=None, textcolor=None, linewidth=None, dpi=None, plotPadding=0.1):
    """Plot 2d prediction map.
    
        :param data: (pandas dataframe)  data with descriptors and labels
        :param descriptor_names: (list - pair) 2 names of descriptors to use for prediction
        :param label_names: (list) all label names to predict
        :param labelMaps: (dict) {label: {'valueString':number, ...}, ...} - maps of label vaules to numbers
        :param folder_prefix: (string) output folders prefix
        :param plot_only: 'data', 'data and quality', default='' - all including prediction
        :param doNotPlotRemoteCount: (integer) calculate mean and do not plot the most remote doNotPlotRemoteCount points
        :param textColumn: if given, use to put text inside markers
        :param additionalMapPlotFunc: function(ax) to plot some additional info
        :param cmap: pyplot color map name, or 'seaborn ...' - seaborn
        returns: saves all graphs to two folders: folder_prefix_by_label, folder_prefix_by descriptors
    """
    assert len(descriptor_names) == 2
    assert plot_only in ['', 'data', 'data and quality']
    if edgecolor is None:
        edgecolor = '#DDD' if plot_only == '' else '#555'
    if linewidth is None:
        linewidth = 1
    if textcolor is None:
        textcolor = '#FFF' if plot_only == '' else '#000'
    if labelMaps is None: labelMaps = {}
    folder = folder_prefix + '_by_descriptors'+os.sep+descriptor_names[0]+'_'+descriptor_names[1]
    # if os.path.exists(folder): shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)
    folder2 = folder_prefix+'_by_label'
    # if os.path.exists(folder2): shutil.rmtree(folder2)
    os.makedirs(folder2, exist_ok=True)
    if plot_only != 'data':
        qualityRes = getQuality(data, descriptor_names, label_names, m=1, cv_count=cv_count, returnModels=True)
    colorMap = plotting.parseColorMap(cmap)
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
    x1 = x if unknown is None else np.concatenate([x, unknown[descriptor_names[0]]])
    y1 = y if unknown is None else np.concatenate([y, unknown[descriptor_names[1]]])

    def getScatterParams(fig):
        defaultMarkersize, defaulAlpha = plotting.getScatterDefaultParams(x1, y1, fig.dpi)
        markersize1 = defaultMarkersize if markersize is None else markersize
        alpha1 = defaulAlpha if alpha is None else alpha
        if plot_only == '': alpha1 = 1
        textsize1 = markersize1/2 if textsize is None else textsize
        return markersize1, alpha1, textsize1

    for label in label_names:
        if label not in data.columns: continue
        if plot_only != 'data':
            quality = qualityRes[label]['quality']
            predictions = qualityRes[label]['predictedLabels']
            model = qualityRes[label]['model']
        labelData = data[label].to_numpy()
        if doNotPlotRemoteCount > 0:
            labelData = labelData[good_ind]
        os.makedirs(folder2+os.sep+label, exist_ok=True)
        fileName1 = folder + '/' + label
        if plot_only == 'data':
            fileName2 = folder2 + os.sep + label + os.sep + f'{descriptor_names[0]}  {descriptor_names[1]}'
        else:
            fileName2 = folder2 + os.sep + label + os.sep + f'{quality:.2f} {descriptor_names[0]}  {descriptor_names[1]}'
        fig, ax = plotting.createfig(figdpi=dpi, interactive=True)
        markersize, alpha, textsize = getScatterParams(fig)
        assert np.all(pd.notnull(labelData))
        c_min = np.min(labelData); c_max = np.max(labelData)
        transform = lambda r: (r-c_min) / (c_max - c_min)
        if plot_only == '':
            # contours
            x_min, x_max = np.min(x1), np.max(x1)
            x_min, x_max = x_min-0.2*(x_max-x_min), x_max+0.2*(x_max-x_min)
            y_min, y_max = np.min(y1), np.max(y1)
            y_min, y_max = y_min-0.2*(y_max-y_min), y_max+0.2*(y_max-y_min)
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
            preds0 = model.predict(np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1))))
            preds = transform(preds0.reshape(xx.shape))
        # print(label,'- classification =', ML.isClassification(data, label))
        isClassification = ML.isClassification(data, label)
        if isClassification:
            ticks = np.unique(labelData)
            h_tick = np.unique(ticks[1:] - ticks[:-1])[0]
            assert np.all(ticks[1:] - ticks[:-1] == h_tick), f'contourf use middles between levels to set colors of areas - so if labels are not equally spaced, we have to use labelMaps and equally spaces label ids.\nLabel = {label}'
            levels = transform(np.append(ticks - h_tick/2, np.max(ticks) + h_tick/2))
            ticksPos = transform(ticks)
        else:
            ticks = np.linspace(data[label].min(), data[label].max(), 10)
            delta = ticks[1]-ticks[0]
            levels = transform( np.append(ticks-delta/2, ticks[-1]+delta/2) )
            ticksPos = transform(ticks)
        if plot_only == '':
            CF = ax.contourf(xx, yy, preds, cmap=colorMap, vmin=0, vmax=1, levels=levels, extend='both')
            # save to file
            cont_data = pd.DataFrame()
            cont_data[descriptor_names[0]] = xx.reshape(-1)
            cont_data[descriptor_names[1]] = yy.reshape(-1)
            cont_data[label] = preds0.reshape(-1)
            cont_data.to_csv(fileName1+'.csv', index=False)
            cont_data.to_csv(fileName2+'.csv', index=False)

        # known
        c = labelData
        c = transform(c)
        sc = ax.scatter(x, y, s=markersize**2, c=c, cmap=colorMap, vmin=0, vmax=1, alpha=alpha, linewidth=linewidth, edgecolor=edgecolor)
        if plot_only == '':
            c = transform(predictions)
            if doNotPlotRemoteCount > 0: c = c[good_ind]
            ax.scatter(x, y, s=(markersize/3)**2, c=c, cmap=colorMap, vmin=0, vmax=1)

        if plot_only != '': plotting.addColorBar(sc, fig, ax, labelMaps, label, ticksPos, ticks)
        else: plotting.addColorBar(CF, fig, ax, labelMaps, label, ticksPos, ticks)

        # unknown
        if unknown is not None:
            umarkersize = markersize * 1.2
            if plot_only == '':
                pred_unk = model.predict(unknown.loc[:,descriptor_names])
                c_params = {'c':transform(pred_unk), 'cmap':colorMap}
            else: c_params = {'c':'white'}
            ax.scatter(unknown[descriptor_names[0]], unknown[descriptor_names[1]], s=umarkersize**2, **c_params, vmin=0, vmax=1, edgecolor='black')
            for i in range(len(unknown)):
                if textColumn is None:
                    name = str(i)
                else:
                    name = unknown.loc[i,textColumn]
                if textsize == 0: umarkerTextSize = umarkersize/2
                else: umarkerTextSize = textsize
                ax.text(unknown.loc[i, descriptor_names[0]], unknown.loc[i, descriptor_names[1]], name, ha='center', va='center', size=umarkerTextSize, color=textcolor)

        # text
        if textsize>0:
            for i in range(data.shape[0]):
                if doNotPlotRemoteCount > 0 and i not in good_ind: continue
                if textColumn is None:
                    name = i
                else:
                    name = data.loc[i,textColumn]
                ax.text(data.loc[i,descriptor_names[0]], data.loc[i,descriptor_names[1]], str(name), ha='center', va='center', size=textsize, color=textcolor)

        ax.set_xlabel(descriptor_names[0])
        ax.set_ylabel(descriptor_names[1])
        if plot_only != 'data':
            qt = 'Accuracy' if isClassification else 'R2-score'
            ax.set_title(f'{label} prediction. {qt} = {quality:.2f}')
        else:
            ax.set_title(label)
        ax.set_xlim(plotting.getPlotLim(x, gap=plotPadding))
        ax.set_ylim(plotting.getPlotLim(y, gap=plotPadding))
        if unknown is not None:
            ax.set_xlim(plotting.getPlotLim(np.concatenate([x, unknown[descriptor_names[0]]]), gap=plotPadding))
            ax.set_ylim(plotting.getPlotLim(np.concatenate([y, unknown[descriptor_names[1]]]), gap=plotPadding))
        if additionalMapPlotFunc is not None:
            additionalMapPlotFunc(ax)
        plotting.savefig(fileName1+'.png', fig)
        plotting.savefig(fileName2+'.png', fig)
        plotting.closefig(fig, interactive=True)

        # plot CV result
        if plot_only == '':
            xx = labelData
            yy = predictions
            if doNotPlotRemoteCount > 0: yy = yy[good_ind]
            if isClassification:
                labelMap = labelMaps[label] if label in labelMaps else None
                plotting.plotConfusionMatrix(xx, yy, label, labelMap=labelMap, fileName=fileName1 + '_cv.png')
                plotting.plotConfusionMatrix(xx, yy, label, labelMap=labelMap, fileName=fileName2 + '_cv.png')
            else:
                fig, ax = plotting.createfig()
                cx = transform(xx)
                cy = transform(yy)
                sc = ax.scatter(xx, yy, s=markersize**2, c=cx, cmap=colorMap, vmin=0, vmax=1, alpha=alpha, linewidth=1, edgecolor=edgecolor)
                ax.scatter(xx, yy, s=(markersize/3)**2, c=cy, cmap=colorMap, vmin=0, vmax=1)
                ax.plot([xx.min(), xx.max()], [xx.min(), xx.max()], 'r', lw=2)
                plotting.addColorBar(sc, fig, ax, labelMaps, label, ticksPos, ticks)
                if textsize > 0:
                    k = 0
                    for i in range(data.shape[0]):
                        if doNotPlotRemoteCount > 0 and i not in good_ind: continue
                        if textColumn is None:
                            name = i
                        else:
                            name = data.loc[i, textColumn]
                        ax.text(xx[k], yy[k], str(name), ha='center', va='center', size=textsize)
                        k += 1
                ax.set_xlim(plotting.getPlotLim(xx))
                ax.set_ylim(plotting.getPlotLim(yy))
                ax.set_title('CV result for label '+label+f'. R2-score = {quality:.2f}')
                ax.set_xlabel('true '+label)
                ax.set_ylabel('predicted '+label)
                plotting.savefig(fileName1 + '_cv.png', fig)
                plotting.savefig(fileName2 + '_cv.png', fig)
                plotting.closefig(fig)

            cv_data = pd.DataFrame()
            cv_data['true '+label] = xx
            cv_data['predicted '+label] = yy
            cv_data.to_csv(fileName1 + '_cv.csv', index=False)
            cv_data.to_csv(fileName2 + '_cv.csv', index=False)


def plot_cv_result(sample, features, label_names, makeMixtureParams=None, model_class=None, model_regr=None, labelMaps=None, folder='', markersize=None, textsize=None, alpha=None, cv_count=2, repForStdCalc=3, unknown_sample=None, textColumn=None, unknown_data_names=None, fileName=None, plot_diff=True):
    """Plot cv result graph.

        :param sample:  sample (for mixture) or DataFrame with descriptors and labels
        :param features: (list of strings or string) features to use (x), or 'spectra' or 'spectra_d_i1,i2,i3,...' (spectra derivatives together), or ['spType1 spectra_d_i1,i2', 'spType2 spectra_d_i1,i2', ...]
        :param label_names: all label names to predict
        :param makeMixtureParams: arguments for mixture.generateMixtureOfSample excluding sample. Sample is divided into train and test and then make mixtures separately. All labels becomes real (concentration1*labelValue1 + concentration2*labelValue2).
        :param model_class: model for classification tasks
        :param model_regr: model for regression tasks
        :param labelMaps: dict {label: {'valueString':number, ...}, ...} - maps of label vaules to numbers
        :param folder: output folder
        :param markersize: markersize for scatter plot
        :param textsize: textsize for textColumn labels
        :param alpha: alpha for scatter plot
        :param cv_count: cv_count - for cross validation, inf means - LOO
        :param repForStdCalc: - number of cv repetitions to calculate std of quality
        :param unknown_sample: sample with unknown labels to make prediction
        :param textColumn: if given, use to put text inside markers
        :param unknown_data_names: if given it is used to print unk names
        :param fileName: file to save result
        :param plot_diff: True/False - whether to plot error (smoothed difference between true and predicted labels)
    """
    assert repForStdCalc>=1
    if labelMaps is None: labelMaps = {}
    mix = makeMixtureParams is not None
    os.makedirs(folder, exist_ok=True)
    qualityResult = getQuality(sample, features, label_names, makeMixtureParams=makeMixtureParams, model_class=model_class, model_regr=model_regr, m=repForStdCalc, cv_count=cv_count, returnModels=True)
    cv_result = {}
    res = getXYFromSample(sample, features, label_names, textColumn if textColumn in sample.paramNames else None)
    text = None if textColumn is None or textColumn not in sample.paramNames else res[2]
    if unknown_sample is not None:
        res = getXYFromSample(unknown_sample, features, label_names, textColumn)
        unkX, unky = res[0], res[1]
    for il, label in enumerate(label_names):
        # unknown prediction
        dns = ' + '.join(features) if isinstance(features, list) else features
        if len(dns) > 100: dns = dns[:100] + f' {len(dns)}'
        cv_result[label] = qualityResult[label]
        cv_result[label]['features'] = dns
        trueLabels = cv_result[label]['trueLabels']
        predictedLabels = cv_result[label]['predictedLabels']
        if mix:
            if unknown_sample is not None:
                cv_result[label]['probPredictionsForUnknown'] = {}
                cv_result[label]['predictionsForUnknown'] = {}
                for problemType in trueLabels:
                    if problemType=='componentLabels':
                        # print(label, ML.isClassification(trueLabels[problemType]))
                        # print(trueLabels[problemType])
                        if ML.isClassification(trueLabels['componentLabels']):
                            cv_result[label]['probPredictionsForUnknown']['componentLabels'] = cv_result[label]['model']['componentLabels'].predict_proba(unkX)
                    cv_result[label]['predictionsForUnknown'][problemType] = cv_result[label]['model'][problemType].predict(unkX)
        else:
            isClassification = ML.isClassification(trueLabels)
            if unknown_sample is not None:
                if isClassification:
                    prob = cv_result[label]['model'].predict_proba(unkX)
                    cv_result[label]['probPredictionsForUnknown'] = prob
                cv_result[label]['predictionsForUnknown'] = cv_result[label]['model'].predict(unkX)
        if fileName is None:
            dns = ' + '.join(features) if isinstance(features, list) else features
            if len(dns)>30: dns = dns[:30]+f'_{len(dns)}'
            fileNam = folder + os.sep + label + ' ' + dns
        else:
            fileNam = folder + os.sep + fileName
        cv_result[label]['fileNam'] = fileNam

        def plot(true, pred, quality, filePostfix=''):
            plotFileName = fileNam+'_'+filePostfix
            pred = pred.reshape(-1)
            if ML.isClassification(true):
                labelMap = labelMaps[label] if label in labelMaps else None
                plotting.plotConfusionMatrix(true, pred, label, labelMap=labelMap, fileName=plotFileName + '.png')
            else:
                err = np.abs(true - pred)
                def plotIdentity(ax):
                    ax.plot([pred.min(), pred.max()], [pred.min(), pred.max()], 'r', lw=2)
                plotting.scatter(pred, true, color=err, alpha=alpha, markersize=markersize, text_size=textsize, marker_text=text, xlabel='predicted ' + label, ylabel='true ' + label, title='CV result for label ' + label + f'. Quality = {quality:.2f}', fileName=plotFileName + '.png', plotMoreFunction=plotIdentity)
                if plot_diff:
                    kr = statsmodels.nonparametric.kernel_regression.KernelReg(endog=err, exog=pred, var_type='c', bw=[(np.max(pred)-np.min(pred))/20])
                    gr_pred = np.linspace(np.min(pred), np.max(pred), 200)
                    smoothed_err, smoothed_std = kr.fit(gr_pred)
                    plotting.plotToFile(gr_pred, smoothed_err, 'mean abs diff', xlabel='predicted '+label, ylabel='abs error', title='abs(predicted-true) for label '+label, fileName=plotFileName+'_diff.png')
        if mix:
            for problemType in trueLabels:
                true = trueLabels[problemType]; pred = predictedLabels[problemType]
                if len(true.shape) > 1 and true.shape[1] > 1:
                    for j in range(true.shape[1]):
                        plot(true[:,j], pred[:,j], cv_result[label]["quality"][problemType][j], filePostfix=f'{problemType}_{j}')
                else:
                    plot(true, pred, cv_result[label]["quality"][problemType], filePostfix=problemType)
        else:
            plot(trueLabels, predictedLabels, cv_result[label]["quality"])

    unk_text = None
    if textColumn is not None: unk_text = res[2]
    if unknown_data_names is not None: unk_text = unknown_data_names
    if unk_text is None and unknown_sample is not None: unk_text = [str(i) for i in range(unknown_sample.getLength())]

    # apply labelMaps
    if unknown_sample is not None:
        cv_result_for_print = copy.deepcopy(cv_result)
        for label in cv_result_for_print:
            if label not in labelMaps: continue
            lm = {labelMaps[label][l]: l for l in labelMaps[label]}
            def convert(vec):
                return np.array([lm[int(x)] for x in vec])
            r = cv_result_for_print[label]
            r['trueLabels'] = convert(r['trueLabels'])
            if mix:
                for problemType in r['predictionsForUnknown']:
                    r['predictionsForUnknown'][problemType] = convert(r['predictionsForUnknown'][problemType])
            else: r['predictionsForUnknown'] = convert(r['predictionsForUnknown'])
    for label in cv_result_for_print:
        r = cv_result_for_print[label]
        with open(r['fileNam'] + '_unkn.txt', 'w') as f:
            s = f"Prediction of {label} by " + r['features'] + '\n'
            s += "quality = " + str(r['quality']) + '\n'
            s += "quality_std = " + str(r['quality_std']) + '\n'
            f.write(s)
            if unknown_sample is not None:
                def printPredictions(ps):
                    pred = ""
                    for i in range(len(unk_text)):
                        p = ps[i]
                        pred += f"{unk_text[i]}: {p}"
                        true = unknown_sample.params.loc[i, label]
                        # print(true)
                        if not np.isnan(true):
                            pred += f" true = {true}  err = {np.abs(true - p)}"
                        pred += "\n"
                    return pred
                if mix:
                    pred = ''
                    for problemType in r['predictionsForUnknown']:
                        pred += f'\n{problemType} prediction:\n'
                        pred += printPredictions(r['predictionsForUnknown'][problemType])
                        pred += "\n"
                else: pred = printPredictions(r['predictionsForUnknown'])
                pred = f"predictions:\n{pred}\n"
                f.write(pred)
                if 'probPredictionsForUnknown' in r:
                    probPredictionsForUnknown = r['probPredictionsForUnknown']
                    if mix:
                        if len(probPredictionsForUnknown) > 0:
                            assert len(probPredictionsForUnknown) == 1, probPredictionsForUnknown
                            probPredictionsForUnknown = probPredictionsForUnknown['componentLabels']
                            assert isinstance(probPredictionsForUnknown, list), str(probPredictionsForUnknown)
                            for component in range(len(probPredictionsForUnknown)):
                                probPredictionsForUnknown1 = probPredictionsForUnknown[component]
                                s = f'probabilities for component {component}:\nunknownInd'
                                ticks = np.unique(r['trueLabels']['componentLabels'][:,component])
                                for tick in ticks: s += f' prob_{tick}'
                                s += '\n'
                                for i in range(len(probPredictionsForUnknown1)):
                                    s += unk_text[i]
                                    for j in range(len(probPredictionsForUnknown1[i])):
                                        s += ' %.2g' % probPredictionsForUnknown1[i, j]
                                    s += '\n'
                                s += '\n'
                                f.write(s)
                    else:
                        s = 'probabilities:\nunknownInd'
                        ticks = np.unique(r['trueLabels'])
                        for tick in ticks: s += f' prob_{tick}'
                        s += '\n'
                        for i in range(len(probPredictionsForUnknown)):
                            s += unk_text[i]
                            for j in range(len(probPredictionsForUnknown[i])):
                                s += ' %.2g' % probPredictionsForUnknown[i,j]
                            s += '\n'
                        f.write(s)
                # print(pred[:-1])


def getLinearAnalyticModel(data, features, label, l1_ratio, try_alpha=None, cv_count=10):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        if try_alpha is None: try_alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        if isinstance(features, str): features = [features]
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


def getAnalyticFormulasForGivenFeatures(data, features, label_names, l1_ratio=1, try_alpha=None, cv_count=10, normalize=True, output_file='formulas.txt'):
    """
    Finds analytical formulas for labels in terms of linear and quadratic functions of features. Data is normilized to zero mean and std=1.
    Algorithm. For each label we check whether the label can be expressed in terms of features using general nonlinear model (ExtraTreesRegressor). If score>0.5 we try to build linear model using feature selecting algoritm Elastic Net. If its score>0.5 we print it. The linear formula returned by Elastic Net is also heavy, so we sort coefficients by absolute value and try to build linear model based on subsets of features with the largest absolute coeffitients. The attempts are done for subsets of each size: 1, 2, 3, ...
    :param data: data frame
    :param features: feature names
    :param label_names:
    :param l1_ratio:
    :param try_alpha:
    :param cv_count:
    :param normalize:
    :param output_file:
    :return:
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        if try_alpha is None: try_alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        data = data.loc[:,list(features)+list(label_names)]
        data = data.sample(frac=1).reset_index(drop=True)
        if isinstance(features, list):
            features = np.array(features)
        for fname in features:
            assert np.all(~np.isnan(data[fname])), f'{fname} contains NaNs:\n' + str(data[fname])
        if normalize:
            mean = data.mean()
            std = data.std()
            data1 = (data - mean) / std
        else:
            data1 = data
        data2 = ML.transformFeatures2Quadric(data1.loc[:, features], addConst=False)
        data2_features = np.array(data2.columns)
        for label in label_names: data2[label] = data1[label]
        dataSets = [data1, data2]
        featureSets = [features, data2_features]
        result_file = open(output_file, 'w')
        for label in label_names:
            # label_data = (data[label]+data[label].min())**2
            label_data = data1[label]
            for di in range(len(dataSets)):
                d = dataSets[di]
                features1 = featureSets[di]
                # check possibility
                model = sklearn.ensemble.ExtraTreesRegressor(n_estimators=100, random_state=0, min_samples_leaf=10)
                # model = lgb.LGBMRegressor(num_leaves=31, learning_rate=0.02, n_estimators=100)
                score = np.mean(sklearn.model_selection.cross_val_score(model, d, label_data, cv=cv_count))
                if score <= 0.5:
                    model_string = f'{label} can\'t be expressed in terms of features: '+str(features1)
                else:
                    score, model, model_string = getLinearAnalyticModel(d, features1, label, l1_ratio=l1_ratio, try_alpha=try_alpha, cv_count=cv_count)
                print(model_string)
                result_file.write(model_string+'\n')

                if score <= 0.5: continue
                # get simple models

                def searchBestSimple(subsets, max_print_num):
                    simpleModels = []
                    for f in subsets:
                        score, model, model_string = getLinearAnalyticModel(d, f, label, l1_ratio=l1_ratio, try_alpha=try_alpha, cv_count=cv_count)
                        simpleModels.append({'score':score, 'model':model, 'model_string':model_string, 'features':f})
                    simpleModels = sorted(simpleModels, key=lambda r:r['score'], reverse=True)
                    print_num = 0
                    for sm in simpleModels:
                        if sm['score'] > 0.5:
                            print(' ' * 8 + sm['model_string'])
                            result_file.write(' ' * 8 + sm['model_string'] + '\n')
                            print_num += 1
                            if print_num >= max_print_num: break
                        else: break
                    return simpleModels

                searchBestSimple(features1, 1)
                simpleModels2 = searchBestSimple(itertools.combinations(features1, 2), 2)
                best2features = simpleModels2[0]['features']
                subsets = [[*best2features,f] for f in features1 if f not in best2features]
                searchBestSimple(subsets, 3)
        result_file.close()


def calcCalibrationData(expData, theoryData, componentNameColumn=None, folder=None, excludeColumnNames=None, multiplierOnlyColumnNames=None, shiftOnlyColumnNames=None, stableLinearRegression=True):
    """
    Calculate calibration coefficients
    :param expData: params DataFrame of experimental sample
    :param theoryData: params DataFrame of thoretical sample
    :param componentNameColumn: to find coinside exp and theory. if None use index
    :param folder: to save calibration graphs (if None - do not plot)
    :param excludeColumnNames: list of columns to exclude from calibration
    :param multiplierOnlyColumnNames: list of columns to calibrate only by multiplier
    :param shiftOnlyColumnNames: list of columns to calibrate only by shift
    :param stableLinearRegression: throw away 10% of points with max error and rebuild regression
    :return: dict {descriptorName: [toDiv, toSub]} theory = toDiv*exp + toSub, then calibration:  newTheoryDescr = (theoryDescr - toSub) / toDiv
    """
    if excludeColumnNames is None: excludeColumnNames = []
    if multiplierOnlyColumnNames is None: multiplierOnlyColumnNames = []
    if shiftOnlyColumnNames is None: shiftOnlyColumnNames = []
    calibration = {}
    if componentNameColumn is None:
        expComponentNames = np.arange(len(expData))
        theoryComponentNames = np.arange(len(theoryData))
    else:
        expComponentNames = expData[componentNameColumn].to_numpy()
        theoryComponentNames = theoryData[componentNameColumn].to_numpy()
    assert set(excludeColumnNames) <= set(expData.columns), 'Unknown names in excludeColumnNames'
    assert set(multiplierOnlyColumnNames) <= set(expData.columns), 'Unknown names in multiplierOnlyColumnNames'
    assert set(shiftOnlyColumnNames) <= set(expData.columns), 'Unknown names in shiftOnlyColumnNames'
    for descriptor_name in expData.columns:
        if descriptor_name in excludeColumnNames: continue
        x = []; y = []; names = []
        for i in range(theoryData.shape[0]):
            i_exp = np.where(expComponentNames == theoryComponentNames[i])[0]
            if len(i_exp) == 0: continue
            assert len(i_exp) == 1, f'Multiple experiments with the same name {theoryComponentNames[i]}'
            i_exp = i_exp[0]
            x.append(expData.loc[i_exp,descriptor_name])
            y.append(theoryData.loc[i, descriptor_name])
            names.append(theoryComponentNames[i])
        x = np.array(x)
        y = np.array(y)

        def buildRegression(desc, x, y):
            if desc in multiplierOnlyColumnNames:
                a = curveFitting.linearReg_mult_only(x,y,np.ones(len(x)))
                b = 0
            elif desc in shiftOnlyColumnNames:
                a = 1
                b = np.mean(y - x)
            else:
                b,a = curveFitting.linearReg(x,y,np.ones(len(x)))
            return a, b
        a, b = buildRegression(descriptor_name, x, y)
        y_regr = a*x+b
        if stableLinearRegression:
            error = np.abs(y-y_regr)
            q = np.quantile(error, q=0.9)
            a, b = buildRegression(descriptor_name, x[error<=q], y[error<=q])
            y_regr = a * x + b
        # print('Descriptor '+descriptor_name+' std error after calibration =', np.sqrt(np.mean((x - (y-b)/a)**2)))
        calibration[descriptor_name] = [a, b]
        if folder is not None:
            fig, ax = plotting.createfig()
            ax.scatter(x, y, 600, c='yellow')
            for i in range(len(names)):
                ax.text(x[i], y[i], names[i], ha='center', va='center', size=10)
            ax.plot(x, y_regr, color='k')
            ax.set_xlabel('exp')
            ax.set_ylabel('theory')
            ax.set_xlim(np.min(x), np.max(x))
            ax.set_ylim(np.min(y), np.max(y))
            ax.set_title(f"{descriptor_name}: theory = {a:.3f}*exp + {b:.2f}")
            plotting.savefig(folder+os.sep+descriptor_name+'.png', fig)
            plotting.closefig(fig)
    return calibration


def calibrateSample(sample, calibrationDataForDescriptors=None, calibrationDataForSpectra=None, inplace=False):
    """
    Linear calibration of sample
    :param sample: Sample
    :param calibrationDataForDescriptors: dict {descriptorName: [toDiv, toSub]}  descr = (descr - toSub) / toDiv
    :param calibrationDataForSpectra: dict{'spType':[spectraDivisor, energySub]}  (energySub, spectraDivisor - scalars or arrays)
    """
    if not inplace:
        sample = copy.deepcopy(sample)
    if calibrationDataForDescriptors is not None:
        data = sample.params
        for descriptor_name in sample.paramNames:
            if descriptor_name not in calibrationDataForDescriptors: continue
            toDiv, toSub = calibrationDataForDescriptors[descriptor_name]
            data[descriptor_name] = (data[descriptor_name] - toSub) / toDiv
    if calibrationDataForSpectra is not None:
        for spType in calibrationDataForSpectra:
            spectraDivisor, energySub = calibrationDataForSpectra[spType]
            n = sample.getLength()
            if not isinstance(energySub, np.ndarray):
                energySub = energySub + np.zeros(n)
            if not isinstance(spectraDivisor, np.ndarray):
                spectraDivisor = spectraDivisor + np.zeros(n)
            energy = sample.getEnergy(spType=spType)
            spectra = []
            commonSub = np.mean(energySub)
            newEnergy = energy-commonSub
            for i in range(n):
                sp = sample.getSpectrum(i, spType=spType, returnIntensityOnly=True)
                sp = np.interp(newEnergy, energy - energySub[i], sp, left=0, right=0)
                sp = sp / spectraDivisor[i]
                spectra.append(sp)
            sample.setSpectra(spectra=np.array(spectra), energy=newEnergy, spType=spType)
    return sample


def concatCalibratedDatasets(expSample, theorySample, componentNameColumn=None, debugInfo=None):
    """
    Concat exp and theory samples after calibration
    :param expSample: experimental sample
    :param theorySample: thoretical sample
    :param componentNameColumn: to find coinside exp and theory. if None use index
    :param debugInfo: dict{folder, uncalibratedSample}
    :return: combined sample
    """
    if componentNameColumn is None:
        expComponentNames = np.arange(len(expSample))
        theoryComponentNames = np.arange(len(theorySample))
    else:
        expComponentNames = expSample.params[componentNameColumn].to_numpy()
        theoryComponentNames = theorySample.params[componentNameColumn].to_numpy()
        assert len(np.unique(expComponentNames) == len(expComponentNames)), 'Duplicate exp names!'
        assert len(np.unique(theoryComponentNames) == len(theoryComponentNames)), 'Duplicate theory names!'
    _, commExp, commTheor = np.intersect1d(expComponentNames, theoryComponentNames, return_indices=True)

    if debugInfo is not None:
        for spType in expSample.spTypes():
            expEn = expSample.getEnergy(spType=spType)
            thEn = theorySample.getEnergy(spType=spType)
            rFactors = []
            for i in range(len(commExp)):
                expSp = expSample.getSpectra(spType=spType).loc[commExp[i]]
                thSp = theorySample.getSpectra(spType=spType).loc[commTheor[i]]
                rf = utils.rFactor(expEn, np.interp(expEn, thEn, thSp), expSp)
                rFactors.append(rf)
                graphs = (expEn, expSp, 'exp', thEn, thSp, 'calibrated theory')
                if 'uncalibratedSample' in debugInfo:
                    uncTheorySample = debugInfo['uncalibratedSample']
                    uncThSp = uncTheorySample.getSpectra(spType=spType).loc[commTheor[i]]
                    uncThEn = uncTheorySample.getEnergy(spType=spType)
                    graphs += (uncThEn, uncThSp, 'uncalibrated theory')
                plotting.plotToFile(*graphs, xlabel='energy', title=f'Comparison of exp spectrum {expComponentNames[commExp[i]]} with calibrated theory. rFactor = {rf}', fileName=f'{debugInfo["folder"]}/{spType}_{expComponentNames[commExp[i]]}.png')
            rFactors = np.array(rFactors)
            print(f'Std-mean rFactor between calibrated and true spectra[{spType}] =', np.sqrt(np.mean(rFactors**2)))


    combined = copy.deepcopy(expSample)
    theory = copy.deepcopy(theorySample)
    theory.delRow(commTheor)
    for spType in theory.spTypes():
        theory.changeEnergy(energy=combined.getEnergy(spType=spType), spType=spType)
    combined.unionWith(theory)
    return combined


def energyImportance(sample, label_names, folder, model=None, method='one model', spType=None):
    assert method in ['one model', 'model for each energy']
    n = sample.getLength()
    if n < 100: min_samples_leaf = 1
    else: min_samples_leaf = 4
    modelIsNone = model is None
    spectra = sample.getSpectra(spType=spType).to_numpy()
    energy = sample.getEnergy(spType=spType)
    denergy = energy[1:]-energy[:-1]
    dspectra = (spectra[:,1:]-spectra[:,:-1])/denergy
    d2spectra = (dspectra[:,1:]-dspectra[:,:-1])/denergy[:-1]
    spectras = [spectra, dspectra, d2spectra]
    energies = [energy, energy[:-1], energy[1:-1]]
    for label in label_names:
        classification = ML.isClassification(sample.params, label)
        if modelIsNone:
            n_estimators = 20 if method == 'model for each energy' else 1000
            if classification:
                model = sklearn.ensemble.ExtraTreesClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf)
            else:
                model = sklearn.ensemble.ExtraTreesRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf)
        for isp,sp in enumerate(spectras):
            overall_quality = ML.score_cv(model, sp, sample.params[label], cv_count=10, returnPrediction=False)
            m = len(energies[isp])
            quality = np.zeros(m)
            if method == 'model for each energy':
                for j in range(m):
                    quality[j] = ML.score_cv(model, sp[:,j], sample.params[label], cv_count=10, returnPrediction=False)
            else:
                model.fit(sp, sample.params[label])
                quality[:] = model.feature_importances_
            plotting.plotToFile(energies[isp], quality, 'enImportance', title=f'Energy importance for label {label}. Overall quality = {overall_quality:.2f}', fileName=f'{folder}/{label}_d{isp}.png')
        overall_quality = ML.score_cv(model, np.hstack(spectras), sample.params[label], cv_count=10, returnPrediction=False)
        m = len(energies[-1])
        quality = np.zeros(m)
        if method == 'model for each energy':
            for j in range(m):
                spj = np.hstack((spectras[0][:,j+1].reshape(-1,1), spectras[1][:,j].reshape(-1,1), spectras[2][:,j].reshape(-1,1)))
                quality[j] = ML.score_cv(model, spj, sample.params[label], cv_count=10, returnPrediction=False)
        else:
            sps = np.hstack((spectras[0][:, 1:-1], spectras[1][:, :-1], spectras[2]))
            assert sps.shape[1] == 3*m
            model.fit(sps, sample.params[label])
            f = model.feature_importances_
            quality[:] = 0
            quality[:] = quality[:] + f[:m]
            quality[:] = quality[:] + f[m:2*m]
            quality[:] = quality[:] + f[2*m:]
        plotting.plotToFile(energies[-1], quality, 'enImportance', title=f'Energy importance for label {label}. Overall quality = {overall_quality:.2f}', fileName=f'{folder}/{label}_d012.png')


def energyImportanceInverse(sample, features, filePrefix=None, model=None):
    n = sample.getLength()
    if n < 100: min_samples_leaf = 1
    else: min_samples_leaf = 4
    modelIsNone = model is None
    if modelIsNone:
        model = sklearn.ensemble.ExtraTreesRegressor(n_estimators=100, min_samples_leaf=min_samples_leaf)
    if filePrefix is None: filePrefix='invEnergyImp'
    for spType in sample.spTypes():
        spectra = sample.getSpectra(spType).to_numpy()
        energy = sample.getEnergy(spType)
        m = len(energy)
        quality = np.zeros(m)
        n = spectra.shape[0]
        for j in range(m):
            if n < 100: cv_count = n
            elif n < 500: cv_count = 10
            elif n < 2000: cv_count = 5
            else: cv_count = 2
            quality[j] = ML.score_cv(model, sample.params.loc[:,features], spectra[:,j], cv_count=cv_count, returnPrediction=False)
        postfix = '' if len(sample.spTypes()) == 1 else '_'+spType
        plotting.plotToFile(energy, quality, 'invEnImportance', title=f'Inverse energy importance', fileName=f'{filePrefix}{postfix}.png')


def diffSpectraDataFrame(spectra, energy, smoothWidth=3):
    assert len(energy) == spectra.shape[1]
    spectra = np.copy(spectra)
    if smoothWidth > 0:
        for i in range(len(spectra)):
            spectra[i] = smoothLib.simpleSmooth(energy, spectra[i], smoothWidth, kernel='Gauss')
    denergy = energy[1:] - energy[:-1]
    dspectra = (spectra[:, 1:] - spectra[:, :-1]) / denergy
    return dspectra, (energy[1:] + energy[:-1])/2


def getSpectraFeatures(sample, diff=None, spType=None):
    if diff is None: diff = [0]
    spectra = sample.getSpectra(spType=spType).values
    energy = sample.getEnergy(spType=spType)
    result = None
    for i in range(np.max(diff)+1):
        if i in diff:
            if result is None: result = spectra
            else: result = np.hstack((result, spectra))
        smoothWidth = 3 if i == 0 else 0
        if i < np.max(diff):
            spectra, energy = diffSpectraDataFrame(spectra, energy, smoothWidth)
    return result


def plotProbabilityMapsForLabelPair(sample, features, label_pair, maxPlotCount='all', textColumn=None, unknownSample=None, folder='probMaps', bandwidth=0.2):
    """

    :param features: (list of strings or string) features to use, or 'spectra' or 'spectra_d_i1,i2,i3,...' (spectra derivatives together), or ['spType1 spectra_d_i1,i2', 'spType2 spectra_d_i1,i2', ...]
    :param textColumn: if not None, can be absent in sample (for example - mixture sample), but not in unknownSample
    """
    assert len(label_pair) == 2
    xlim = [sample.params[label_pair[0]].min(), sample.params[label_pair[0]].max()]
    ylim = [sample.params[label_pair[1]].min(), sample.params[label_pair[1]].max()]

    res = getXYFromSample(sample, features, label_pair, textColumn if textColumn in sample.paramNames else None)
    X, y = res[0], res[1]
    text = list(map(str, range(len(X)))) if textColumn is None or textColumn not in sample.paramNames else res[2]
    if unknownSample is not None:
        res = getXYFromSample(unknownSample, features, [], textColumn)
        unkX, unky = res[0], res[1]
        unkText = map(str, range(len(unkX))) if textColumn is None else res[2]

    def plotForOneSpectrum(spectrum, name, trueLabels=None):
        def density(z):
            return densityEstimator.predict(spectrum.reshape(1, -1), np.array(z).reshape(1, -1), k=10, bandwidth=bandwidth)[0][0]
        fileName = f'{folder}/unknown/{name}.png' if trueLabels is None else f'{folder}/{name}.png'

        def plotMoreFunction(ax):
            ax.scatter([trueLabels[0]], [trueLabels[1]], color='red', s=500)
        plotting.plotHeatMap(density, xlim, ylim, title=f'Probability density for {name}', xlabel=label_pair[0], ylabel=label_pair[1], fileName=fileName, plotMoreFunction=plotMoreFunction if trueLabels is not None else None)

    if maxPlotCount == 'all': maxPlotCount = len(X)
    for i in range(min(maxPlotCount, len(X))):
        X1 = X[np.arange(len(X)) != i, :]
        y1 = y[np.arange(len(X)) != i, :]
        densityEstimator = ML.NNKCDE(X1, y1)
        plotForOneSpectrum(X[i], text[i], y[i])
    if unknownSample is not None:
        densityEstimator = ML.NNKCDE(X, y)
        for i in range(len(unkX)):
            plotForOneSpectrum(unkX[i], unkText[i])

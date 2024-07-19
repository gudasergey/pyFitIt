import os, copy, sklearn, sklearn.feature_selection, sklearn.cross_decomposition, shutil, itertools, statsmodels, warnings, scipy.signal, scipy.interpolate, types, logging, scipy.spatial, sklearn.manifold
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
    assert spectra.shape[1] == U.shape[0], f'Numbers of energies are not equal: {spectra.shape[1]} != {U.shape[0]}'
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


def efermiDescriptor(spectra, energy, maxlevel=None):
    """Find Efermi and arctan grow rate for all spectra and returns as 2 column matrix
    
    :param spectra: 2-d matrix of spectra (each row is one spectrum)
    :param energy:
    :param returnArctan: whether to return arctan y
    """
    d = np.zeros((spectra.shape[0],2))
    arctan_y = np.zeros(spectra.shape)
    for i in range(spectra.shape[0]):
        arcTanParams, arctan_y[i] = curveFitting.findEfermiByArcTan(energy, spectra[i], maxlevel)
        d[i] = [arcTanParams['x0'], arcTanParams['a']]
    return d, arctan_y


def addDescriptors(sample:ML.Sample, descriptors, inplace=True):
    """
    :param sample: Sample instance
    :param descriptors: list of str (descriptor type) or dict{'type':.., 'columnName':.., 'arg1':.., 'arg2':.., ...}.
        Params: common: spType, energyInterval, columnName
            type=stableExtrema - see. stableExtrema function
            type in ['max', 'min', 'maxGroup', 'minGroup'] - smoothRad, constrain=func(extr_e, extr_intensity, spectrum, params) -> True/False, selector=func(all_extr_e, all_extr_inten, spectrum, params) -> list_extr_e, list_extr_inten
            type=variation - smoothRad, energyInterval can be func(spectrum, params)
            type=efermi - plotFolder, maxNumToPlot (choose by rand), extraPlotInd (obligatory plots), maxlevel (fit from zero up to this level)
            type='1st_peak' - centroid of xanes_y > ylevel & xanes_e < elevel
            type in [pca, rel_pca, scaled_pca] - usePrebuiltData=True/False, count, fileName (for prebuilt data)
            type=tsne - features(list) or spType with energyInterval, usePrebuiltData=True/False, count, fileName (for prebuilt data)
            type=pls - features(list) or spType with energyInterval, usePrebuiltData=True/False, count, fileName (for prebuilt data)
            type=best_linear - features(list) or None (then spType with energyInterval used), usePrebuiltData=True/False, fileName (for prebuilt data), cv_parts, best_alpha (set it to None first), infoFile - file used to choose best_alpha (test score should be as much as possible, but approx. equal to train score!!!), baggingParams - if you need to use bagging, for example: dict(max_samples=0.1, max_features=1.0, n_estimators=10)
            type=polynom - deg
            type=moment - deg in [0,1,2,3,...]
            type=area
            type=center
            type=value - energies(list) - values of spectra in this energy points
            type=separable - 'labels':'all' or list, 'redefineLabelMap': labelMap to redefine (returned by findBestLabelMap function), params of separableDesctriptors: 'normalize':True, 'pairwiseTransformType':'binary', 'features':'exafs spectra', 'debugFolder':None

    :return: new sample with descriptors, goodSpectrumIndices
    """
    if not inplace: sample = sample.copy()
    # canonizations
    assert isinstance(descriptors, list)
    newD = []
    for d in descriptors:
        if isinstance(d, str): newD.append({'type':d})
        else: newD.append(d)
    descriptors = newD
    goodSpectrumIndices_all = None
    unique_names = []
    all_prebuilt_files = []

    def getCommonParams(params):
        params = copy.deepcopy(params)
        del params['type']
        name = typ
        if 'columnName' in params:
            name = params['columnName']
            del params['columnName']
        else:
            if typ == 'moment':
                assert 'deg' in params
                name += params['deg']
            elif typ == 'best_linear':
                name = f'BL_{params["label"]}'
            name1 = name
            i = 1
            while name1 in unique_names:
                name1 = f'{name}{i}'
                i += 2
            if name != 'pls': name = name1
            unique_names.append(name)
        if 'spType' in params:
            spType = params['spType']
            del params['spType']
        else: spType = sample.getDefaultSpType()
        if 'energyInterval' in params and not callable(params['energyInterval']):
            energyInterval = params['energyInterval']
            del params['energyInterval']
        else:
            energy = sample.getEnergy(spType)
            energyInterval = [energy[0], energy[-1]]
        return name, spType, energyInterval, params

    N = sample.getLength()
    common_efermi = None
    def get_common_efermi():
        if common_efermi is None:
            efermi, _ = efermiDescriptor(spectra1.to_numpy(), energy1)
            return efermi[:, 0]
        else: return common_efermi

    def checkParams(params, allParamNames, obligatoryParams, help):
        assert set(params.keys()) <= allParamNames, 'Wrong param names: ' + str(set(params.keys()) - allParamNames)
        for pn in obligatoryParams:
            assert pn in params, help

    for d in descriptors:
        typ = d['type']
        # print(typ)
        name, spType, energyInterval, params = getCommonParams(d)
        energy = sample.getEnergy(spType)
        spectra = sample.getSpectra(spType)
        sample1 = sample.limit(energyInterval, spType=spType, inplace=False)
        energy1 = sample1.getEnergy(spType)
        spectra1 = sample1.getSpectra(spType)
        paramNames = sample.paramNames
        # =======================================================================
        # =======================================================================
        # =======================================================================
        if typ == 'stableExtrema':
            assert 'extremaType' in params
            if name == typ:
                name = params['extremaType']
            ext_e = name+'_e'
            ext_int = name+'_i'
            ext_d2 = name+'_d2'
            assert (ext_e not in paramNames) and (ext_int not in paramNames) and (ext_d2 not in paramNames), f'Duplicate descriptor names while adding {ext_e}, {ext_int}, {ext_d2} to {paramNames}. Use columnName argument in descriptor parameters'
            ext, goodSpectrumIndices = stableExtrema(spectra, energy, energyInterval=energyInterval, **params)
            assert np.all(np.diff(goodSpectrumIndices) >= 0)
            if goodSpectrumIndices_all is None: goodSpectrumIndices_all = goodSpectrumIndices
            else: goodSpectrumIndices_all = np.intersect1d(goodSpectrumIndices_all, goodSpectrumIndices)
            if len(goodSpectrumIndices) != N:
                known, unknown, indKnown, indUnknown = sample.splitUnknown(returnInd=True)
                common = np.intersect1d(indUnknown, goodSpectrumIndices)
                assert len(common) == len(indUnknown), f"Can\'t find {params['extremaType']} for unknown spectra. Try changing search energy interval or expand energy interval for all spectra. See plot for details."

            def expandByZeros(col):
                r = np.zeros(N)
                r[goodSpectrumIndices] = col
                return r
            sample.addParam(paramName=ext_e, paramData=expandByZeros(ext[:, 0]))
            sample.addParam(paramName=ext_int, paramData=expandByZeros(ext[:, 1]))
            sample.addParam(paramName=ext_d2, paramData=expandByZeros(ext[:, 2]))
        # =======================================================================
        # =======================================================================
        # =======================================================================
        elif typ in ['max', 'min', 'maxGroup', 'minGroup']:
            if 'Group' in typ:
                count = np.zeros(N)
                centroid = np.zeros(N)
                centroid[:] = np.nan
                mean_e = np.zeros(N)
                mean_e[:] = np.nan
                mean_i = np.zeros(N)
                mean_i[:] = np.nan
            else:
                extr_es = np.zeros(N)
                extr_is = np.zeros(N)
                extr_d2 = np.zeros(N)
            sign = +1 if typ == 'max' else -1
            smoothRad = params['smoothRad'] if 'smoothRad' in params else 5
            for i in range(N):
                sp = sample.getSpectrum(ind=i, spType=spType)
                if smoothRad > 0:
                    sp.intensity = smoothLib.simpleSmooth(sp.energy, sp.intensity, sigma=smoothRad, kernel='Gauss')
                sp = sp.limit(interval=energyInterval)
                ps = sample.params.loc[i]
                _, all_extr_ind = utils.argrelmax((sign*sp).intensity, returnAll=True)
                if 'constrain' in params:
                    all_extr_ind1 = []
                    for extr_ind in all_extr_ind:
                        if params['constrain'](sp.energy[extr_ind], sp.intensity[extr_ind], sp, ps):
                            all_extr_ind1.append(extr_ind)
                    all_extr_ind = all_extr_ind1
                if 'Group' in typ:
                    count[i] = len(all_extr_ind)
                    if count[i] > 0:
                        centroid[i] = np.sum(sp.intensity[all_extr_ind]*sp.energy[all_extr_ind]) / np.sum(sp.intensity[all_extr_ind])
                        mean_e[i] = np.mean(sp.energy[all_extr_ind])
                        mean_i[i] = np.mean(sp.intensity[all_extr_ind])
                else:
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
            if 'Group' in typ:
                sample.addParam(paramName=name + '_count', paramData=count)
                sample.addParam(paramName=name + '_centroid', paramData=centroid)
                sample.addParam(paramName=name + '_mean_e', paramData=mean_e)
                sample.addParam(paramName=name + '_mean_i', paramData=mean_i)
            else:
                sample.addParam(paramName=name+'_e', paramData=extr_es)
                sample.addParam(paramName=name+'_i', paramData=extr_is)
                sample.addParam(paramName=name+'_d2', paramData=extr_d2)
        # =======================================================================
        # =======================================================================
        # =======================================================================
        elif typ == 'variation':
            var = np.zeros(N)
            smoothRad = params['smoothRad'] if 'smoothRad' in params else 5
            for i in range(N):
                sp = sample.getSpectrum(ind=i, spType=spType)
                if smoothRad > 0:
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
        # =======================================================================
        # =======================================================================
        # =======================================================================
        elif typ == 'efermi':
            maxlevel = params.get('maxlevel',None)
            efermi, arctan_y = efermiDescriptor(sample.getSpectra(spType).to_numpy(), energy, maxlevel)
            if common_efermi is None: common_efermi = efermi[:, 0]
            if 'plotFolder' in params:
                plotFolder = params['plotFolder']
                maxNumToPlot = params.get('maxNumToPlot', 100)
                if N > maxNumToPlot:
                    rng = np.random.default_rng(0)
                    plot_inds = rng.choice(N, maxNumToPlot)
                else:
                    plot_inds = np.arange(N)
                if 'extraPlotInd' in params:
                    plot_inds = np.array(list(set(plot_inds).union(set(params['extraPlotInd']))))
                if os.path.exists(plotFolder): shutil.rmtree(plotFolder)
                toPlot = ([energy[0], energy[-1]], [maxlevel, maxlevel], 'maxlevel') if maxlevel is not None else tuple()
                for i in plot_inds:
                    sp_name = str(i) if sample.nameColumn is None else sample.params.loc[i,sample.nameColumn]
                    y = sample.getSpectrum(ind=i, spType=spType).y
                    if maxlevel is not None:
                        dy = np.max(y)-np.min(y)
                        ylim = (np.min(y)-dy*0.1, np.max(y)+dy*0.1)
                    else: ylim=None
                    plotting.plotToFile(energy, y, 'spectrum', energy, arctan_y[i], 'arctan', *toPlot, fileName=f'{plotFolder}/{sp_name}.png', title=f'efermi={efermi[i, 0]:.1f} efermiRate={efermi[i, 1]}', ylim=ylim)
            sample.addParam(paramName=f'{name}_e', paramData=efermi[:, 0])
            sample.addParam(paramName=f'{name}_slope', paramData=efermi[:, 1])
        # =======================================================================
        # =======================================================================
        # =======================================================================
        elif typ in ['pca', 'scaled_pca']:
            checkParams(params, allParamNames={'usePrebuiltData', 'count', 'fileName', 'energyInterval'}, obligatoryParams=['usePrebuiltData'], help="Use pca in the following way: {'type':'pca/scaled_pca', 'count':3, 'usePrebuiltData':True/False, 'fileName':'?????.pkl'}")
            count = params.get('count',3)
            if 'fileName' in params:
                assert params['fileName'] not in all_prebuilt_files, f'Duplicate prebuilt fileName detected: '+params['fileName']
                all_prebuilt_files.append(params['fileName'])
            spectra1_1 = spectra1.to_numpy() if typ == 'pca' else sklearn.preprocessing.StandardScaler().fit_transform(spectra1.to_numpy())
            if params['usePrebuiltData']:
                pca_u, en1 = utils.load_pkl(params['fileName'])
                if len(en1) != len(energy1) or np.any(en1 != energy1):
                    assert set(en1) >= set(energy1), f'PCA components were calculated for energy \n{en1}\nBut now you try to calculate descriptors for another energy:\n{energy1}'
                    ind = np.isin(en1,energy1)
                    pca_u = pca_u[ind,:]
                pca = pcaDescriptor(spectra1_1, count=count, U=pca_u)
            else:
                pca, pca_u = pcaDescriptor(spectra1_1, count=count, returnU=True)
                if 'fileName' in params:
                    utils.save_pkl((pca_u, energy1), params['fileName'])
            for j in range(count):
                sample.addParam(paramName=f'{name}{j+1}', paramData=pca[:, j])
        # =======================================================================
        # =======================================================================
        # =======================================================================
        elif typ == 'rel_pca':
            checkParams(params, allParamNames={'usePrebuiltData', 'count', 'fileName', 'energyInterval'}, obligatoryParams=['usePrebuiltData'], help="Use rel_pca in the following way: {'type':'rel_pca', 'count':3, 'usePrebuiltData':True/False, 'fileName':'?????.pkl'}")
            count = params.get('count',3)
            common_efermi = get_common_efermi()
            if 'fileName' in params:
                assert params['fileName'] not in all_prebuilt_files, f'Duplicate prebuilt fileName detected: '+params['fileName']
                all_prebuilt_files.append(params['fileName'])
            if params['usePrebuiltData']:
                relpca_u, relEnergy = utils.load_pkl(params['fileName'])
                relpca = relPcaDescriptor(spectra1.to_numpy(), energy1, common_efermi, count=count, prebuildData=(relpca_u, relEnergy))
            else:
                relpca, relpca_u, relEnergy = relPcaDescriptor(spectra1.to_numpy(), energy1, common_efermi, count=count, returnU=True)
                if 'fileName' in params:
                    utils.save_pkl((relpca_u, relEnergy), params['fileName'])
            for j in range(count):
                sample.addParam(paramName=f'{name}{j+1}', paramData=relpca[:, j])
        # =======================================================================
        # =======================================================================
        # =======================================================================
        elif typ == 'pls':
            checkParams(params, allParamNames={'label', 'features', 'spType', 'energyInterval',  'usePrebuiltData', 'count', 'fileName'}, obligatoryParams=['label', 'usePrebuiltData', 'fileName'], help="Use pls in the following way: {'type':'pls', 'label':..., 'features':[...], 'count':2, 'usePrebuiltData':True/False, 'fileName':'?????.pkl'} or {'type':'pls', 'label':..., 'spType':[...], 'energyInterval':[..,..], 'count':2, 'usePrebuiltData':True/False, 'fileName':'?????.pkl'}")
            label = params['label']
            count = params.get('count', 2)
            fileName = params['fileName']
            assert fileName not in all_prebuilt_files, f'Duplicate prebuilt fileName detected: {fileName}'
            all_prebuilt_files.append(fileName)
            features = params.get('features', None)
            if features is not None: spType = None
            if params['usePrebuiltData']:
                pls_regr = utils.load_pkl(fileName)
            else:
                pls_regr = generatePLSData(sample1, label, fileName, features=features, spType=spType, n_components=count)
            fname = spType if features is None else 'features'
            pls = getPLS(sample1, pls_regr=pls_regr, features=features, spType=spType)
            for j in range(count):
                sample.addParam(paramName=f'{name}{j+1}_{fname}_{label}', paramData=pls[:, j])
        # =======================================================================
        # =======================================================================
        # =======================================================================
        elif typ == 'tsne':
            from openTSNE import TSNE
            logging.getLogger('openTSNE.tsne').setLevel(logging.ERROR)
            checkParams(params, allParamNames={'features', 'spType', 'usePrebuiltData', 'count', 'fileName', 'energyInterval', 'perplexity', 'preprocess'}, obligatoryParams=['usePrebuiltData', 'fileName'], help="Use "+typ+" in the following way: {'features':[...] or 'spType':'...', 'energyInterval':[..,..], 'type':'"+typ+"', 'count':2, 'perplexity':..., 'preprocess':None/'pca number'/'scaler and pca number' 'usePrebuiltData':True/False, 'fileName':'?????.pkl'}")
            count = params.get('count',2)
            features = params.get('features', None)
            if features is not None: f = sample.params.loc[:, features]
            else: f = spectra1
            preprocess = params.get('preprocess', None)
            assert params['fileName'] not in all_prebuilt_files, f'Duplicate prebuilt fileName detected: ' + params['fileName']
            all_prebuilt_files.append(params['fileName'])
            if params['usePrebuiltData']:
                tr, tsne = utils.load_pkl(params['fileName'])
                if tr is not None:
                    f = tr.transform(f)
                embedding = tsne.transform(f)
            else:
                perplexity = params.get('perplexity', min(30, len(sample)//4))
                tsne = TSNE(perplexity=perplexity, random_state=0, verbose=False)
                if preprocess is not None:
                    w1 = preprocess.split(' ')[0]
                    assert w1 in ['pca', 'scaler']
                    num = int(preprocess.split(' ')[-1])
                    tr = sklearn.decomposition.PCA(n_components=num, random_state=0)
                    if w1 == 'scaler': tr = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), tr)
                else:
                    tr = sklearn.preprocessing.StandardScaler()
                f = tr.fit_transform(f)
                tsne = tsne.fit(f)
                utils.saveData((tr, tsne), file_name=params['fileName'])
                embedding = tsne.transform(f)
            fname = spType if features is None else 'features'
            for j in range(count):
                sample.addParam(paramName=f'{name}{j + 1}_{fname}', paramData=embedding[:, j])
        # =======================================================================
        # =======================================================================
        # =======================================================================
        elif typ == 'best_linear':
            checkParams(params, allParamNames={'label', 'features', 'spType', 'energyInterval', 'usePrebuiltData', 'fileName', 'baggingParams', 'cv_parts', 'debug'}, obligatoryParams=['label', 'usePrebuiltData', 'fileName'], help="Use best_linear in the following way: {'type':'best_linear', 'label':..., 'features':[...], 'cv_parts':4, 'usePrebuiltData':True/False, 'fileName':'?????.pkl', 'baggingParams':dict(max_samples=0.1, max_features=1.0, n_estimators=10)} or {'type':'best_linear', 'label':..., 'spType':[...], 'energyInterval':[..,..], 'cv_parts':4, 'usePrebuiltData':True/False, 'fileName':'?????.pkl', 'baggingParams':dict(max_samples=0.1, max_features=1.0, n_estimators=10)}")
            label = params['label']
            assert isinstance(label, str), str(label)
            if not params['usePrebuiltData']: assert label in sample.paramNames
            fileName = params['fileName']
            if fileName[-1] == '?': fileName = fileName[:-1]+f'prebuilt_{name}.pkl'
            assert fileName not in all_prebuilt_files, f'Duplicate prebuilt fileName detected: {fileName}'
            all_prebuilt_files.append(fileName)
            features = params.get('features', None)
            cv_parts = params.get('cv_parts', 10)
            debug = params.get('debug', 'auto')
            infoFile = os.path.splitext(fileName)[0]+'_info.txt'
            best_alpha = params.get('best_alpha', None)
            baggingParams = params.get('baggingParams', None)
            if features is not None: spType = None
            if params['usePrebuiltData']:
                model = utils.loadData(fileName)
            else:
                model = generateBestLinearTransformation(sample1, label, best_alpha=best_alpha, features=features, spType=spType, cv_parts=cv_parts, infoFile=infoFile, baggingParams=baggingParams, debug=debug)
                utils.saveData(model, fileName)
            if features is None: features = f'{spType} spectra'
            X, _ = getXYFromSample(sample1, features, None)
            pred = model.predict(X)
            sample.addParam(paramName=name, paramData=pred)
        # =======================================================================
        # =======================================================================
        # =======================================================================
        elif typ == 'polynom':
            deg = params['deg'] if 'deg' in params else 3
            descr = np.zeros((N, deg+1))
            for i in range(N):
                sp = sample.getSpectrum(ind=i, spType=spType)
                energyInterval = params['energyInterval'] if 'energyInterval' in params else [sp.energy[0], sp.energy[-1]]
                if not isinstance(energyInterval, list):
                    assert callable(energyInterval), 'energyInterval should be list or function(spectrum, params), which returns list'
                    energyInterval = energyInterval(sp, sample.params.loc[i])
                sp = sp.limit(energyInterval)
                descr[i] = np.polyfit((sp.energy-sp.energy[0])/(sp.energy[-1]-sp.energy[0]), sp.intensity, deg)
            for j in range(deg+1):
                sample.addParam(paramName=f'{name}_{j}', paramData=descr[:, j])
        # =======================================================================
        # =======================================================================
        # =======================================================================
        elif typ == 'moment':
            area = utils.integral(energy1, spectra1.to_numpy())
            center = utils.integral(energy1, spectra1.to_numpy() * energy1) / area
            deg = params['deg']
            descr = utils.integral(energy1, spectra1.to_numpy()*(energy1-center)**deg) / area
            sample.addParam(paramName=name, paramData=descr)
        elif typ == 'area':
            area = utils.integral(energy1, spectra1.to_numpy())
            sample.addParam(paramName=name, paramData=area)
        elif typ == 'center':
            area = utils.integral(energy1, spectra1.to_numpy())
            center = utils.integral(energy1, spectra1.to_numpy()*energy1) / area
            sample.addParam(paramName=name, paramData=center)
        elif typ == 'value':
            for e in params['energies']:
                v = [np.interp(e, energy1, spectra1.loc[i]) for i in range(spectra1.shape[0])]
                sample.addParam(paramName=f'{name}_{e}', paramData=v)
        # type=value - energies(list) - values of spectra in this energy points
        # =======================================================================
        # =======================================================================
        # =======================================================================
        elif typ == '1st_peak':
            # '1st_peak' - centroid of xanes_y > ylevel & xanes_e < elevel
            common_efermi = get_common_efermi()
            elevel = params.get('elevel', np.mean(common_efermi) + 20)
            ylevel = params.get('ylevel', 0.6)
            peak_x, peak_y = np.zeros(N), np.zeros(N)
            smoothRad = params.get('smoothRad',3)
            for i in range(N):
                sp = sample.getSpectrum(ind=i, spType=spType)
                if smoothRad > 0:
                    sp.y = smoothLib.simpleSmooth(sp.x, sp.y, sigma=smoothRad, kernel='Gauss')
                ind = sp.x <= elevel
                j = np.where((sp.y<=ylevel) & ind)[0]
                assert len(j)>0, f'No points in spectrum < {ylevel}'
                j = j[-1]
                e0 = sp.x[j]
                edge_x, edge_y = sp.x[ind & (sp.x>=e0)], sp.y[ind & (sp.x>=e0)]
                w = np.sqrt(np.diff(edge_x)**2 + np.diff(edge_y)**2)
                assert np.sum(w) > 0
                w = w/np.sum(w)
                peak_x[i] = np.sum(edge_x[:-1] * w)
                peak_y[i] = np.sum(edge_y[:-1] * w)
            if 'plotFolder' in params:
                plotFolder = params['plotFolder']
                maxNumToPlot = params.get('maxNumToPlot', 100)
                if N > maxNumToPlot:
                    rng = np.random.default_rng(0)
                    plot_inds = rng.choice(N, maxNumToPlot)
                else:
                    plot_inds = np.arange(N)
                if 'extraPlotInd' in params:
                    plot_inds = np.array(list(set(plot_inds).union(set(params['extraPlotInd']))))
                if os.path.exists(plotFolder): shutil.rmtree(plotFolder)
                toPlot = ([energy[0], energy[-1]], [ylevel, ylevel], 'ylevel', [elevel,elevel], [0,1], 'elevel')
                for i in plot_inds:
                    sp_name = str(i) if sample.nameColumn is None else sample.params.loc[i,sample.nameColumn]
                    y = sample.getSpectrum(ind=i, spType=spType).y
                    y0 = copy.deepcopy(y)
                    if smoothRad > 0:
                        y = smoothLib.simpleSmooth(energy, y0, sigma=smoothRad, kernel='Gauss')
                    plotting.plotToFile(energy, y, 'smoothed', energy, y0, 'initial', [peak_x[i]], [peak_y[i]], {'label':'1st_peak', 'fmt':'o'}, *toPlot, fileName=f'{plotFolder}/{sp_name}.png', title=f'1st_peak=({peak_x[i]:.1f}, {peak_y[i]:.2f})')
            sample.addParam(paramName=name+'_e', paramData=peak_x)
            sample.addParam(paramName=name+'_i', paramData=peak_y)
        elif typ == 'separable':
            redefineLabelMap = params.get('redefineLabelMap', {})
            labels = params.get('labels', 'all')
            if isinstance(labels, str) and labels == 'all':
                labels = sample.labels
            normalize = params.get('normalize', True)
            pairwiseTransformType = params.get('pairwiseTransformType', True)
            features = params.get('features', sample.getDefaultSpType()+' spectra')
            debugFolder = params.get('debugFolder', None)
            for il,label in enumerate(labels):
                if label in redefineLabelMap:
                    sample_redefLM = sample1.copy()
                    if label in sample.labelMaps: sample_redefLM.decode(label)
                    sample_redefLM.encode(label, redefineLabelMap[label])
                else: sample_redefLM = sample1
                d = separableYDescriptor(sample_redefLM, features=features, label=label, normalize=normalize, pairwiseTransformType=pairwiseTransformType, debugFolder=debugFolder)
                sample.addParam(paramName=name+'_'+label, paramData=d)
        else:
            assert False, f"Unknown descriptor type {typ}"

    if goodSpectrumIndices_all is not None and len(goodSpectrumIndices_all) != N:
        sample = sample.takeRows(goodSpectrumIndices_all)
    if goodSpectrumIndices_all is None: goodSpectrumIndices_all = np.arange(N)
    return sample, goodSpectrumIndices_all


def generatePLSData(sample:ML.Sample, label, fileName, features=None, spType=None, n_components=2):
    assert features is None or spType is None
    known, unknown = sample.splitUnknown(columnNames=label)
    pls_regr = sklearn.cross_decomposition.PLSRegression(n_components=n_components)
    l = known.params[label]
    if features is None: f = known.getSpectra(spType=spType)
    else: f = known.params.loc[:,features]
    pls_regr.fit(f, l)
    pred = pls_regr.predict(f)
    isClassification = ML.isClassification(l)
    if isClassification: pred = np.round(pred).astype(int)
    q = ML.calcAllMetrics(l, pred, isClassification)
    name = spType if features is None else 'features'
    # print(f'PLS X={name} Y={label} quality =', q)
    utils.saveData(pls_regr, fileName)
    return pls_regr


def getPLS(sample:ML.Sample, pls_regr, features=None, spType=None):
    if features is None:
        f = sample.getSpectra(spType=spType)
    else:
        f = sample.params.loc[:,features]
    return pls_regr.transform(f)


def plotDescriptors1d(data, spectra, energy, label_names, desc_points_names=None, folder='.'):
    """Plot 1d graphs of descriptors vs labels
    
    Args:
        data (pandas dataframe):  data with descriptors and labels
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
        if spectra is not None:
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

        :param sample:  sample or DataFrame with descriptors and labels
        :param features: (list of strings or string) features to use, or 'spectra' or 'spectra_d_i1,i2,i3,...' (spectra derivatives together), or ['spType1 spectra_d_i1,i2', 'spType2 spectra_d_i1,i2', ...]
        :param label_names: (list)
        :param textColumn: (str) title of column to use as names
        :return: X,y or X,y,text if textColumn is not None
    """
    if label_names is None: label_names = []
    if isinstance(label_names, str): label_names = [label_names]
    def checkLabels(df):
        assert set(label_names) <= set(data.columns), str(label_names) + " is not subset of " + str(df.columns)
        assert not (set(label_names) & set(features)), f'There are common labels and features: {set(label_names) & set(features)}'
    if isinstance(sample, pd.DataFrame):
        data = sample
        assert not isinstance(features, str)
        assert set(features) <= set(data.columns), str(features) + " is not subset of " + str(data.columns)
        checkLabels(data)
        X = data.loc[:, features].to_numpy()
    else:
        assert isinstance(sample, ML.Sample)
        data = sample.params
        if isinstance(features, str): features = [features]
        checkLabels(data)
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
                assert fs not in label_names, f'Labels inside features are detected. Labels = {label_names}. Features = {features}'
            if X is None: X = X1
            else: X = np.hstack((X,X1))
    y = data.loc[:, label_names].to_numpy() if len(label_names)>0 else None
    if y is not None: assert len(X) == len(y)
    if textColumn is None:
        return X, y
    else:
        assert len(X) == data.shape[0]
        return X, y, data.loc[:, textColumn].to_numpy()


def getQuality(sample, features, label_names, makeMixtureParams=None, model_class=None, model_regr=None, m=1, cv_count=10, testSample=None, returnModels=False, random_state=0, printDebug=True):
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
        testSample: if not None calculate quality using the testSample, instead of CV
        returnModels (bool, optional): whether to return models
        random_state (int): seed of random generator
        printDebug (boolean): show debug output
    Returns:
        dict {label: dict{'quality':..., 'predictedLabels':... (dict for mixture), 'trueLabels':...(dict for mixture), 'model':..., 'quality_std':..., 'singlePred':... (for mix), 'singleTrue':... (for mix)}}
    """
    def applyToArrayOfDicts(func, arr):
        if len(arr) == 0: return arr
        assert isinstance(arr[0], dict), str(arr[0])
        res = {}
        for name in arr[0]:
            res[name] = func([el[name] for el in arr])
        return res

    def applyToArrayOfDictsOfDicts(func, arr):
        if len(arr) == 0: return arr
        assert isinstance(arr[0], dict), str(arr[0])
        res = {}
        for name1 in arr[0]:
            if isinstance(arr[0][name1], dict):
                res[name1] = {}
                for name2 in arr[0][name1]:
                    res[name1][name2] = func([el[name1][name2] for el in arr])
            elif isinstance(arr[0][name1], list):
                res[name1] = [{} for _ in range(len(arr[0][name1]))]
                for i in range(len(arr[0][name1])):
                    for name2 in arr[0][name1][i]:
                        res[name1][i][name2] = func([el[name1][i][name2] for el in arr])
            else:
                assert False, f'{name1} {arr[0][name1]}'
        return res
    rng = np.random.default_rng(seed=random_state)
    isSample = isinstance(sample, ML.Sample)
    data = sample.params if isSample else sample
    for l in label_names:
        assert np.all(np.isfinite(data[l])), f'Unknown labels detected for {l}: {data[l]}'
    mix = makeMixtureParams is not None
    quality, quality_std = {}, {}
    predictedLabels, trueLabels, singleCVAll = {}, {}, {}
    models = {}
    X,Y = getXYFromSample(sample, features, label_names)
    if testSample is not None:
        X_test, Y_test = getXYFromSample(testSample, features, label_names)
        assert m == 1
        assert cv_count == 1
        assert not mix, 'TODO: not implemented yet'
    if printDebug:
        print('Try predict by:', features)
    n_estimators = 40
    if X.shape[1] > 5: n_estimators = 100
    if X.shape[1] > 10: n_estimators = 200
    # do not provide min_samples_leaf > 1, because for small samples it causes too large smoothing of the target function
    tryParams = [{'n_estimators': n_estimators}]
    for il, label in enumerate(label_names):
        y = Y[:,il]
        classification = ML.isClassification(y)
        if mix: classification = False
        model0 = model_class if classification else model_regr
        acc, trueVals, pred, singleCV = [None]*m, [None]*m, [None]*m, [None]*m
        if model0 is None:
            try_acc = np.zeros(len(tryParams))
            try_model = [None]*len(tryParams)
            for j,modelParams in enumerate(tryParams):
                if classification:
                    try_model[j] = sklearn.ensemble.ExtraTreesClassifier(**modelParams, random_state=rng.integers(np.iinfo(np.int32).max, dtype=np.int32))
                else:
                    try_model[j] = sklearn.ensemble.ExtraTreesRegressor(**modelParams, random_state=rng.integers(np.iinfo(np.int32).max, dtype=np.int32))
                if mix:
                    try:
                        try_acc_mix, _, _, _ = mixture.score_cv(try_model[j], sample, features, label, label_names, makeMixtureParams, testRatio=0.5, repetitions=1, model_class=sklearn.ensemble.ExtraTreesClassifier(**modelParams))
                        try_acc[j] = try_acc_mix['avgLabels']['R2-score']
                    except mixture.LabelValuesLackError as e:
                        print(f'Error: label {label} has too few values.', e)
                        try_acc[j] = -1
                else:
                    r = ML.score_cv(try_model[j], X, y, cv_count, returnPrediction=False, random_state=rng.integers(np.iinfo(np.int32).max, dtype=np.int32))
                    try_acc[j] = r['accuracy'] if classification else r['R2-score']
            bestj = np.argmax(try_acc)
            model = try_model[bestj]
            model_class = sklearn.ensemble.ExtraTreesClassifier(**tryParams[bestj])
            if printDebug and len(tryParams)>1:
                print('Best model params: ', tryParams[bestj], f'Delta ={try_acc[bestj]-np.min(try_acc)}')

        # run CV multiple times to estimate standard deviation of quality
        # переиспользовать try выше нельзя т.к. testRatio разный!
        for i in range(m):
            if model0 is not None: model = copy.deepcopy(model0)
            if hasattr(model, 'random_state'): model.random_state = rng.integers(np.iinfo(np.int32).max, dtype=np.int32)
            if mix:
                try:
                    # repetitions=cv_count - means not m, but covering all the dataset, because for repetitions=1 mixture.score_cv only one train-test split
                    acc[i], trueVals[i], pred[i], singleCV[i] = mixture.score_cv(model, sample, features, label, label_names, makeMixtureParams, testRatio=1 / cv_count, repetitions=cv_count, model_class=model_class, random_state=rng.integers(np.iinfo(np.int32).max, dtype=np.int32))
                except mixture.LabelValuesLackError as e:
                    print(f'Error: label {label} has too few values for cv_count={cv_count} and sample size={len(sample)}.', e)
            else:
                if testSample is None:
                    acc[i], pred = ML.score_cv(model, X, y, cv_count)
                    trueVals = y
                else:
                    model.fit(X,y)
                    pred = model.predict(X_test)
                    trueVals = Y_test[:,il]
                    acc[i] = ML.calcAllMetrics(trueVals, pred, ML.isClassification(trueVals))
        # gather results
        if mix:
            acc = [a for a in acc if a is not None]
            pred = [a for a in pred if a is not None]
            trueVals = [a for a in trueVals if a is not None]
            singleCV = [a for a in singleCV if a is not None]
            if len(acc) > 0:
                singleCVAll[label] = singleCV[0]  # take first only (better to average, but we need reencode labels first)
                quality[label] = applyToArrayOfDictsOfDicts(lambda x: np.mean(x,axis=0), acc)
                quality_std[label] = applyToArrayOfDictsOfDicts(lambda x: np.std(x,axis=0), acc)
                sz = len(pred[0]['avgLabels'])
                predictedLabels[label] = {problemType: np.array([p[problemType] for p in pred]).reshape(sz*len(acc),-1) for problemType in acc[0]}
                trueLabels[label] = {problemType: np.array([t[problemType] for t in trueVals]).reshape(sz*len(acc),-1) for problemType in acc[0]}
                for problemType in acc[0]:
                    assert predictedLabels[label][problemType].shape[0] == sz*len(acc)
                    if isinstance(acc[0][problemType], dict):
                        assert predictedLabels[label][problemType].size == sz*len(acc), f'{predictedLabels[label][problemType].size} != {sz*len(acc)}, problemType={problemType} acc[0][problemType]={acc[0][problemType]}'
                    else:
                        assert isinstance(acc[0][problemType], list)
                        assert predictedLabels[label][problemType].size == sz*len(acc)*len(acc[0][problemType]), f'{predictedLabels[label][problemType].size} != {sz*len(acc)*len(acc[0][problemType])}, problemType={problemType} acc[0][problemType]={acc[0][problemType]}'
            else:
                print(f'No good CV attempts for label {label}. Decrease cv_count = {cv_count}')
                continue
        else:
            quality[label] = applyToArrayOfDicts(lambda x: np.mean(x,axis=0), acc)
            quality_std[label] = applyToArrayOfDicts(lambda x: np.std(x,axis=0), acc)
            predictedLabels[label] = pred
            trueLabels[label] = trueVals
        if returnModels:
            if mix:
                model_regr_avgLabel, model_regr_conc, model_comp_labels = mixture.fit_mixture_models(sample, features=features, mixSampleSize=500, label=label, label_names=label_names, randomSeed=rng.integers(np.iinfo(np.int32).max, dtype=np.int32), makeMixtureParams=makeMixtureParams, model_regr=model, isClassification=ML.isClassification(sample.params[label]), model_class=model_class)
                models[label] = {'avgLabels': model_regr_avgLabel, 'concentrations': model_regr_conc, 'componentLabels': model_comp_labels}
            else:
                try:
                    with warnings.catch_warnings(record=True) as warn:
                        model.fit(X, y)
                        models[label] = copy.deepcopy(model)
                except Warning:
                    pass
        cl = 'classification' if classification else 'regression'
        if printDebug:
            if mix:
                print(f'{label} - {cl} score: {quality[label]["avgLabels"]}+-{quality_std[label]["avgLabels"]}')
            else:
                print(f'{label} - {cl} score: {quality[label]}+-{quality_std[label]}')
    if printDebug: print('')
    result = {label: dict(quality=quality[label], predictedLabels=predictedLabels[label], trueLabels=trueLabels[label], quality_std=quality_std[label]) for label in quality}
    if mix:
        for label in quality: result[label] = {**result[label], **singleCVAll[label]}
    if returnModels:
        for label in quality:
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


def descriptorQuality(sample:ML.Sample, label_names=None, all_features=None, feature_subset_size=2, cv_parts_count=10, cv_repeat=5, labelMaps=None, unknown_sample=None, textColumn=None, makeMixtureParams=None, model_class=None, model_regr=None, folder='quality_by_label', printDebug=False):
    """Calculate cross-validation result for all feature subsets of the given size for all labels
    
    Args:
        sample:
        label_names: list of label names to predict (if None take from sample)
        all_features (list of strings): features from which subsets are taken (if None take from sample)
        feature_subset_size (int, optional): size of subsets
        cv_parts_count (int, optional): cross validation count (divide all data into cv_parts_count parts)
        cv_repeat: repeat cross validation times
        :param labelMaps: dict {label: {'valueString':number, ...}, ...} - maps of label vaules to numbers (if None take from sample)
        unknown_sample:
        textColumn: exp names in unknown_data
        model_class: model for classification
        model_regr: model for regression
        :param makeMixtureParams: arguments for mixture.generateMixtureOfSample excluding sample. Sample is divided into train and test and then make mixtures separately. All labels becomes real (concentration1*labelValue1 + concentration2*labelValue2)
        folder (str, optional): output folder to save results
        printDebug (boolean): show debug output
    """
    if all_features is None: all_features = sample.features
    assert len(all_features) >= 1
    sample = sample.copy()
    if unknown_sample is not None: unknown_sample = unknown_sample.copy()
    if label_names is None: label_names = sample.labels
    if all_features is None: all_features = sample.features
    assert set(label_names) < set(sample.params.columns)
    assert set(all_features) < set(sample.params.columns)
    for fn in all_features:
        assert np.all(pd.notna(sample.params[fn])), f'Feature {fn} contains NaN values'
    for fn in label_names:
        assert np.all(pd.notna(sample.params[fn])), f'Label {fn} contains NaN values'
    mix = makeMixtureParams is not None
    if mix:
        assert labelMaps is None, 'We decode all features to make mixture. Set labelMaps=None'
        sample.decodeAllLabels()
        unknown_sample.decodeAllLabels()
        for l in label_names:
            if not ML.isOrdinal(sample.params, l):
                sample.delParam(l)
                unknown_sample.delParam(l)
    qualities, cv_data = {}, {}
    for label in label_names:
        qualities[label] = []
        cv_data[label] = pd.DataFrame()
    allTrys = []
    use_common_model = feature_subset_size == 1 and model_class is None and model_regr is None
    if use_common_model:
        commonQualityResult = getQuality(sample, all_features, label_names, makeMixtureParams=makeMixtureParams, model_class=model_class, model_regr=model_regr, cv_count=cv_parts_count, m=1, returnModels=True, printDebug=printDebug)
        for label in commonQualityResult:
            model = commonQualityResult[label]['model']['avgLabels'] if mix else commonQualityResult[label]['model']
            assert len(model.feature_importances_) == len(all_features)
            for i in range(len(all_features)):
                commonQualityResult[label][all_features[i]] = model.feature_importances_[i]
    for fs in itertools.product(*([all_features]*feature_subset_size)):
        if (len(set(fs)) != feature_subset_size) or check_done(allTrys, fs):
            continue
        # if len(qualities[label])>=2: continue
        fs = list(fs)
        # returns dict {label: dict{'quality':..., 'predictions':...(dict for mixture), 'trueLabels':...(dict for mixture), 'model':..., 'quality_std':..., 'singlePred':... (for mix), 'singleTrue':... (for mix)}}
        getQualityResult = getQuality(sample, fs, label_names, makeMixtureParams=makeMixtureParams, model_class=model_class, model_regr=model_regr, cv_count=cv_parts_count, m=cv_repeat, returnModels=unknown_sample is not None, printDebug=printDebug)
        for label in getQualityResult:
            quality = getQualityResult[label]['quality']['avgLabels'] if mix else getQualityResult[label]['quality']
            quality_std = getQualityResult[label]['quality_std']['avgLabels'] if mix else getQualityResult[label]['quality_std']
            if use_common_model:
                quality['fi'] = commonQualityResult[label][fs[0]]
                quality_std['fi'] = 0
            if mix:
                y_pred = getQualityResult[label]['singlePred']
                y_true = getQualityResult[label]['singleTrue']
            else:
                y_pred = getQualityResult[label]['predictedLabels']
                y_true = getQualityResult[label]['trueLabels']
            assert np.all(y_true == sample.params[label]), f'{y_true.tolist()}\n{sample.params[label].tolist()}'
            isClassification = 'accuracy' in quality
            if label in sample.labelMaps:
                y_true = sample.decode(label=label, values=y_true)
                if not mix:
                    y_pred = sample.decode(label=label, values=y_pred)
            # add mutual information
            if feature_subset_size == 1:
                if isClassification:
                    quality['mi'] = sklearn.feature_selection.mutual_info_classif(sample.params[fs], y_true)[0]
                else:
                    quality['mi'] = sklearn.feature_selection.mutual_info_regression(sample.params[fs], y_true)[0]
                quality_std['mi'] = 0

            res_d = {'features':','.join(fs), 'feature list':fs, 'quality':quality, 'quality_std':quality_std}
            if 'true' not in cv_data[label].columns:
                if sample.nameColumn is not None:
                    cv_data[label][sample.nameColumn] = sample.params[sample.nameColumn]
                cv_data[label]['true'] = y_true
            cv_data[label][res_d['features']] = y_pred
            if unknown_sample is not None:
                model = getQualityResult[label]['model']['avgLabels'] if mix else getQualityResult[label]['model']
                res_d['unk_predictions'] = model.predict(unknown_sample.params.loc[:, fs].to_numpy())
                if not mix and label in unknown_sample.labelMaps:
                    res_d['unk_predictions'] = unknown_sample.decode(label, values=res_d['unk_predictions'])
            qualities[label].append(res_d)

    os.makedirs(folder, exist_ok=True)
    for label in qualities:
        cv_data[label].to_csv(f'{folder}{os.sep}{label} cv.csv', sep=';', index=False)
        for metric in qualities[label][0]['quality']:
            if metric.endswith(' interval'): continue
            results = sorted(qualities[label], key=lambda res_list: res_list['quality'][metric], reverse=True)

            # plot feature mutual information matrix
            if 'fi' in qualities[label][0]['quality']: plot_mi_matrix_metrics = ['fi']
            elif 'mi' in qualities[label][0]['quality']: plot_mi_matrix_metrics = ['mi']
            else: plot_mi_matrix_metrics = ['accuracy', 'R2-score']
            if feature_subset_size==1 and metric in plot_mi_matrix_metrics:
                results_mi = results[:30]
                mi_matrix = np.zeros([len(results_mi)]*2)
                mi_fs = [r['feature list'][0] for r in results_mi]
                for mi_i in range(len(results_mi)):
                    r = results_mi[mi_i]
                    mi_matrix[mi_i] = sklearn.feature_selection.mutual_info_regression(sample.params.loc[:,mi_fs], sample.params.loc[:,mi_fs[mi_i]], random_state=0)
                    mi_matrix[mi_i,mi_i] = 0
                plotting.plotMatrix(mi_matrix, ticklabelsX=mi_fs, ticklabelsY=mi_fs, fileName=f'{folder}{os.sep}{label} descriptor MI.png', title=f'Mutual information for best features for label {label}', wrapXTickLabelLength=20, figsize=(15,10), interactive=True)

            csv_filename = f'{folder}{os.sep}{label} {metric}.csv'
            with open(csv_filename, 'w', encoding='utf-8') as f:
                if unknown_sample is None:
                    for r in results:
                        f.write(f"{r['quality'][metric]:.3f}±{r['quality_std'][metric]:.3f}  -  {r['features']}\n")
                else:
                    def getRow(col1, col2, other):
                        return f'{col1};{col2};' + ';'.join(other) + '\n'
                    s = getRow('unknown', 'true value', [r['features'] for r in results])
                    s += getRow('quality', '', [f"{r['quality'][metric]:.3f}±{r['quality_std'][metric]:.3f}" for r in results])
                    trueValues = unknown_sample.params.loc[:, label]
                    if label in unknown_sample.labelMaps:
                        trueValues = unknown_sample.decode(label, values=trueValues)
                    predicted = np.zeros((len(unknown_sample), len(results)))
                    for i in range(len(unknown_sample)):
                        if textColumn is None:  name = f'{i}'
                        else: name = unknown_sample.params.loc[i, textColumn]
                        true = str(unknown_sample.params.loc[i, label])
                        s += getRow(name, true, ["%.3g" % r['unk_predictions'][i] for r in results])
                        for j in range(len(results)):
                            predicted[i,j] = results[j]['unk_predictions'][i]
                    goodInd = np.where(~np.isnan(trueValues))[0]
                    true_count = len(goodInd)
                    if true_count > 0:
                        trueValues = trueValues[goodInd]
                        predicted = predicted[goodInd,:]
                        unk_q = ['%.3g' % ML.calcAllMetrics(trueValues, predicted[:,j], ML.isClassification(sample.params[label]) and not mix)[metric] for j in range(predicted.shape[1])]
                        s += getRow(f'{metric} by exp', '1', unk_q)
                    f.write(s)
            # sort by 'quality by exp'
            if unknown_sample is not None:
                resData = pd.read_csv(csv_filename, sep=';')
                resData.to_excel(os.path.splitext(csv_filename)[0] + '.xlsx', index=False)
                n = resData.shape[0]
                if ' by exp' in resData.loc[n - 1, 'unknown']:
                    expQualities = [-float(resData.loc[n-1, resData.columns[j]]) for j in range(2, resData.shape[1])]
                    ind = np.argsort(expQualities)
                    data1 = pd.DataFrame()
                    data1['unknown'] = resData['unknown']
                    data1['true value'] = resData['true value']
                    for jj in range(len(expQualities)):
                        j = ind[jj] + 2
                        col = resData.columns[j]
                        data1[col] = resData[col]
                    data1.to_excel(folder + os.sep + label + '_sort_by_exp.xlsx', index=False)


def plotDescriptors2d(data, descriptorNames, labelNames, labelMaps=None, folder_prefix='', fileExtension='.png', unknown=None, markersize=None, textsize=None, alpha=None, cv_count=None, plot_only='', doNotPlotRemoteCount=0, textColumn=None, additionalMapPlotFunc=None, cmap='seaborn husl', edgecolor=None, textcolor=None, linewidth=None, dpi=None, plotPadding=0.1, minQualityForPlot=None, model_class=None, model_regr=None, debug=True):
    """Plot 2d prediction map.
    
        :param data: (pandas dataframe)  data with descriptors and labels
        :param descriptorNames: (list - pair) 2 names of descriptors to use for prediction
        :param labelNames: (list) all label names to predict
        :param labelMaps: (dict) {label: {'valueString':number, ...}, ...} - maps of label vaules to numbers
        :param folder_prefix: (string) output folders prefix
        :param fileExtension: file extension
        :param plot_only: 'data', 'data and quality', default='' - all including prediction
        :param doNotPlotRemoteCount: (integer) calculate mean and do not plot the most remote doNotPlotRemoteCount points
        :param textColumn: if given, use to put text inside markers
        :param additionalMapPlotFunc: function(ax) to plot some additional info
        :param cmap: pyplot color map name, or 'seaborn ...' - seaborn
        :param minQualityForPlot: dict{label: {metrics: float})} - plot only if quality greater this number
        returns: saves all graphs to two folders: folder_prefix_by_label, folder_prefix_by descriptors
    """
    assert len(descriptorNames) == 2
    assert plot_only in ['', 'data', 'data and quality']
    if cv_count is None:
        if len(data) < 50: cv_count = len(data)
        elif len(data) < 500: cv_count = 10
        else: cv_count = 2
    if linewidth is None:
        if len(data) < 10: linewidth = 1
        elif len(data) < 200: linewidth = 0.5
        else: linewidth = 0
    if labelMaps is None: labelMaps = {}
    folder = folder_prefix + '_by_descriptors' + os.sep + descriptorNames[0] + '_' + descriptorNames[1]
    # if os.path.exists(folder): shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)
    folder2 = folder_prefix+'_by_label'
    # if os.path.exists(folder2): shutil.rmtree(folder2)
    os.makedirs(folder2, exist_ok=True)
    if plot_only != 'data':
        qualityRes = getQuality(data, descriptorNames, labelNames, m=1, cv_count=cv_count, model_class=model_class, model_regr=model_regr, returnModels=True, printDebug=debug)
        for label in qualityRes:
            assert qualityRes[label]['model'] is not None
    colorMap = plotting.parseColorMap(cmap)
    x = data[descriptorNames[0]].to_numpy()
    y = data[descriptorNames[1]].to_numpy()
    if doNotPlotRemoteCount > 0:
        xm = np.median(x); ym = np.median(y)
        x_normed = (x-xm)/np.sqrt(np.median((x-xm)**2))
        y_normed = (y - ym) / np.sqrt(np.median((y - ym) ** 2))
        ind = np.argsort(-x_normed**2-y_normed**2)
        bad_ind = ind[:doNotPlotRemoteCount]
        good_ind = np.setdiff1d(np.arange(len(x)), bad_ind)
        x = x[good_ind]
        y = y[good_ind]
    x1 = x if unknown is None else np.concatenate([x, unknown[descriptorNames[0]]])
    y1 = y if unknown is None else np.concatenate([y, unknown[descriptorNames[1]]])

    markersize0, textsize0, alpha0 = markersize, textsize, alpha
    def getScatterParams(fig):
        defaultMarkersize, defaulAlpha = plotting.getScatterDefaultParams(x1, y1, fig.dpi, common=False)
        markersize1 = defaultMarkersize if markersize0 is None else markersize0
        alpha1 = defaulAlpha if alpha0 is None else alpha0
        if plot_only == '': alpha1 = 1
        textsize1 = markersize1/2 if textsize0 is None else textsize0
        if not utils.isArray(markersize1): markersize1 = np.zeros(len(x1))+markersize1
        if not utils.isArray(alpha1): alpha1 = np.zeros(len(x1)) + alpha1
        if not utils.isArray(textsize1): textsize1 = np.zeros(len(x1)) + textsize1
        assert len(x1) == len(markersize1), f'{len(x1)} != {len(markersize1)}'
        if unknown is None: return markersize1, alpha1, textsize1
        else: return markersize1[:len(x)], alpha1[:len(x)], textsize1[:len(x)], markersize1[len(x):], alpha1[len(x):], textsize1[len(x):]

    for label in labelNames:
        if label not in data.columns: continue
        if plot_only != 'data':
            quality = qualityRes[label]['quality']
            predictions = qualityRes[label]['predictedLabels']
            model = qualityRes[label]['model']
            if minQualityForPlot is not None:
                skip_plot = False
                for metric, threshold in minQualityForPlot[label].items():
                    skip_plot = skip_plot or (quality[metric] < threshold)
                if skip_plot:
                    continue
        labelData = data[label].to_numpy()
        if doNotPlotRemoteCount > 0:
            labelData = labelData[good_ind]
        os.makedirs(folder2+os.sep+label, exist_ok=True)
        fileName1 = folder + '/' + label
        if plot_only == 'data':
            fileName2 = folder2 + os.sep + label + os.sep + f'{descriptorNames[0]}  {descriptorNames[1]}'
        else:
            q1 = quality['R2-score'] if 'R2-score' in quality else quality['accuracy']
            fileName2 = folder2 + os.sep + label + os.sep + f'{q1:.2f} {descriptorNames[0]}  {descriptorNames[1]}'
        fig, ax = plotting.createfig(figdpi=dpi, interactive=True)
        scatterParams = getScatterParams(fig)
        markersize, alpha, textsize = scatterParams[:3]
        assert np.all(pd.notnull(labelData))
        c_min = np.min(labelData); c_max = np.max(labelData)
        if c_min == c_max:
            c_min, c_max = 0, 1
            const = True
        else: const = False
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
        isClassification = ML.isClassification(data, label)
        if isClassification:
            if label in labelMaps: ticks = np.sort(np.array(list(labelMaps[label].values())))
            else: ticks = np.unique(labelData)
            if const:
                levels = ticks
            else:
                h_tick = np.unique(ticks[1:] - ticks[:-1])[0]
                assert np.all(ticks[1:] - ticks[:-1] == h_tick), f'contourf use middles between levels to set colors of areas - so if labels are not equally spaced, we have to use labelMaps and equally spaces label ids.\nLabel = {label}'
                levels = transform(np.append(ticks - h_tick/2, np.max(ticks) + h_tick/2))
                ticksPos = transform(ticks)
        else:
            ticks = np.linspace(data[label].min(), data[label].max(), 10)
            delta = ticks[1]-ticks[0]
            levels = transform( np.append(ticks-delta/2, ticks[-1]+delta/2) )
            ticksPos = transform(ticks)
        if plot_only == '' and not const:
            CF = ax.contourf(xx, yy, preds, cmap=colorMap, vmin=0, vmax=1, levels=levels, extend='both')
            # save to file
            cont_data = pd.DataFrame()
            cont_data[descriptorNames[0]] = xx.reshape(-1)
            cont_data[descriptorNames[1]] = yy.reshape(-1)
            cont_data[label] = preds0.reshape(-1)
            cont_data.to_csv(fileName1+'.csv', index=False)
            cont_data.to_csv(fileName2+'.csv', index=False)

        # known
        c = labelData
        assert np.all(np.isfinite(c)), str(list(c))
        c = transform(c)
        assert np.all(np.isfinite(c)), str(list(c))
        if edgecolor is None:
            edgecolor = ['#000' if np.mean(colorMap(ci)[:3])>0.5 else '#FFF' for ci in c]
        sc = ax.scatter(x, y, s=markersize**2, c=c, cmap=colorMap, vmin=0, vmax=1, alpha=alpha, linewidth=linewidth, edgecolor=edgecolor)
        if plot_only == '':
            c = transform(predictions)
            if doNotPlotRemoteCount > 0: c = c[good_ind]
            ax.scatter(x, y, s=(markersize/3)**2, c=c, cmap=colorMap, vmin=0, vmax=1)

        if not const:
            if plot_only != '': plotting.addColorBar(sc, fig, ax, labelMaps, label, ticksPos, ticks)
            else: plotting.addColorBar(CF, fig, ax, labelMaps, label, ticksPos, ticks)

        # unknown
        if unknown is not None:
            umarkersize = scatterParams[3]
            if plot_only == '':
                pred_unk = model.predict(unknown.loc[:, descriptorNames])
                c_params = {'c':transform(pred_unk), 'cmap':colorMap}
            else: c_params = {'c':'white'}
            ax.scatter(unknown[descriptorNames[0]], unknown[descriptorNames[1]], s=umarkersize ** 2, **c_params, vmin=0, vmax=1, edgecolor='black', linestyle=':')
            if textcolor is None:
                if utils.isArray(c_params['c']):
                    textcolor1 = ['#000' if np.mean(colorMap(ci)[:3])>0.5 else '#FFF' for ci in c_params['c']]
                else:
                    textcolor1 = '#FFF' if plot_only == '' else '#000'
            else: textcolor1 = textcolor
            if not isinstance(textcolor1, list): textcolor1 = [textcolor1]*len(unknown)
            umarkerTextSize = scatterParams[5]
            if np.all(umarkerTextSize == 0): umarkerTextSize = umarkersize / 2
            for i in range(len(unknown)):
                if textColumn is None:
                    name = str(i)
                else:
                    name = unknown.loc[i,textColumn]
                ax.text(unknown.loc[i, descriptorNames[0]], unknown.loc[i, descriptorNames[1]], name, ha='center', va='center', size=umarkerTextSize[i], color=textcolor1[i])

        # text
        if np.any(textsize > 0):
            c = transform(labelData)
            if textcolor is None:
                if plot_only == '':
                    textcolor1 = ['#000' if np.mean(colorMap(ci)[:3]) > 0.5 else '#FFF' for ci in c]
                else:textcolor1 = ['#000']*len(c)
            else:
                textcolor1 = textcolor
                if not isinstance(textcolor1, list): textcolor1 = [textcolor1]*data.shape[0]
            for i in range(data.shape[0]):
                if doNotPlotRemoteCount > 0 and i not in good_ind: continue
                if textColumn is None or textColumn not in data.columns:
                    name = i
                else:
                    name = data.loc[i,textColumn]
                ax.text(data.loc[i, descriptorNames[0]], data.loc[i, descriptorNames[1]], str(name), ha='center', va='center', size=textsize[i], color=textcolor1[i])

        ax.set_xlabel(descriptorNames[0])
        ax.set_ylabel(descriptorNames[1])
        if plot_only != 'data':
            if isClassification:
                ax.set_title(f'{label} prediction. Accuracy = {quality["accuracy"]:.2f}')
            else:
                ax.set_title(f'{label} prediction. R2-score = {quality["R2-score"]:.2f}')
        else:
            ax.set_title(label)
        ax.set_xlim(plotting.getPlotLim(x, gap=plotPadding))
        ax.set_ylim(plotting.getPlotLim(y, gap=plotPadding))
        if unknown is not None:
            ax.set_xlim(plotting.getPlotLim(np.concatenate([x, unknown[descriptorNames[0]]]), gap=plotPadding))
            ax.set_ylim(plotting.getPlotLim(np.concatenate([y, unknown[descriptorNames[1]]]), gap=plotPadding))
        if additionalMapPlotFunc is not None:
            additionalMapPlotFunc(ax)
        plotting.savefig(fileName1+fileExtension, fig)
        plotting.savefig(fileName2+fileExtension, fig)
        plotting.closefig(fig, interactive=True)

        # plot CV result
        if plot_only == '':
            xx = labelData
            yy = predictions
            if doNotPlotRemoteCount > 0: yy = yy[good_ind]
            if isClassification:
                labelMap = labelMaps[label] if label in labelMaps else None
                plotting.plotConfusionMatrix(xx, yy, label, labelMap=labelMap, fileName=fileName1 + '_cv'+fileExtension)
                plotting.plotConfusionMatrix(xx, yy, label, labelMap=labelMap, fileName=fileName2 + '_cv'+fileExtension)
            else:
                fig, ax = plotting.createfig()
                cx = transform(xx)
                cy = transform(yy)
                sc = ax.scatter(xx, yy, s=markersize**2, c=cx, cmap=colorMap, vmin=0, vmax=1, alpha=alpha, linewidth=1, edgecolor=edgecolor)
                ax.scatter(xx, yy, s=(markersize/3)**2, c=cy, cmap=colorMap, vmin=0, vmax=1)
                ax.plot([xx.min(), xx.max()], [xx.min(), xx.max()], 'r', lw=2)
                plotting.addColorBar(sc, fig, ax, labelMaps, label, ticksPos, ticks)
                if np.any(textsize > 0):
                    k = 0
                    for i in range(data.shape[0]):
                        if doNotPlotRemoteCount > 0 and i not in good_ind: continue
                        if textColumn is None:
                            name = i
                        else:
                            name = data.loc[i, textColumn]
                        ax.text(xx[k], yy[k], str(name), ha='center', va='center', size=textsize[i])
                        k += 1
                ax.set_xlim(plotting.getPlotLim(xx))
                ax.set_ylim(plotting.getPlotLim(yy))
                ax.set_title('CV result for label '+label+f'. R2-score = {quality["R2-score"]:.2f}')
                ax.set_xlabel('true '+label)
                ax.set_ylabel('predicted '+label)
                plotting.savefig(fileName1 + '_cv'+fileExtension, fig)
                plotting.savefig(fileName2 + '_cv'+fileExtension, fig)
                plotting.closefig(fig)

            cv_data = pd.DataFrame()
            cv_data['true '+label] = xx
            cv_data['predicted '+label] = yy
            cv_data.to_csv(fileName1 + '_cv.csv', index=False)
            cv_data.to_csv(fileName2 + '_cv.csv', index=False)


def generateBestLinearTransformation(sample:ML.Sample, label, best_alpha=None, features=None, spType=None, cv_parts=10, infoFile=None, baggingParams=None, debug='auto'):
    """
    Find coefficients of the best linear descriptor. For classification problems it calculates an array of linear models (one model per class)

    :param features: (list of strings or string) features to use (x), or 'spectra' or 'spectra_d_i1,i2,i3,...' (spectra derivatives together), or ['spType1 spectra_d_i1,i2', 'spType2 spectra_d_i1,i2', ...]
    :param baggingParams: parameters of MyBaggingClassifier, for example: dict(max_samples=0.1, max_features=1.0, n_estimators=10). If None - bagging is not used
    returns array or list of arrays of coeffs (first coeff is w0)
    """
    if debug == 'auto': debug = best_alpha is None
    if debug:
        os.makedirs(os.path.split(infoFile)[0], exist_ok=True)
        # df = open(infoFile, 'w', encoding='utf-8')
        df = types.SimpleNamespace(write=lambda s: print(s, end=""))
    def getRegressor(**params):
        # est = sklearn.linear_model.Ridge(**params)
        # est = sklearn.svm.SVR(C=params['alpha'])
        est = sklearn.svm.LinearSVR(C=params['alpha'], random_state=0)
        if baggingParams is not None:
            est = sklearn.ensemble.BaggingRegressor(est, **baggingParams)
        return sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), est)

    def getClassifier(**params):
        # return sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.linear_model.LogisticRegression(**params))
        # return sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.svm.LinearSVC(**params))
        # return sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.ensemble.BaggingClassifier(sklearn.svm.LinearSVC(**params), max_samples=0.3, max_features=0.3, bootstrap=False, n_estimators=50))
        # return sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.ensemble.BaggingClassifier(sklearn.linear_model.LogisticRegression(**params), max_samples=0.1, max_features=1.0, bootstrap=False, n_estimators=10))
        est = ML.FixedClassesClassifier(sklearn.svm.LinearSVC(**params), classes=np.unique(Y))
        if baggingParams is not None:
            est = ML.MyBaggingClassifier(est, **baggingParams)
        return sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), est)
        # tol=10

    def getBestEstimator(X, Y, typ):
        if best_alpha is None:
            if debug: df.write(f'X shape: {X.shape}\n')
            if len(X) < cv_parts*2:
                cv = sklearn.model_selection.LeaveOneOut()
                if debug: df.write('Using LOO\n')
            else:
                cv = sklearn.model_selection.KFold(cv_parts, shuffle=True, random_state=0)
                if debug: df.write(f'Using {cv_parts}-fold CV\n')
            alphas = [1e-20, 1e-14, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]
            scores = []
            for alpha in alphas:
                # if typ == 'regression':
                model = getRegressor(alpha=alpha)
                pred = sklearn.model_selection.cross_val_predict(model, X, Y, cv=cv)
                test_score = sklearn.metrics.r2_score(Y, pred)
                model.fit(X, Y)
                pred = model.predict(X)
                train_score = sklearn.metrics.r2_score(Y, pred)
                # else:
                #     в качестве итогового признака я все равно использую значения регрессора
                #     удаление малопопулярных классов при тренировке, портит значения регрессора
                #     и делает задачу восстановления итоговых значений по labelMaps очень запутанной
                #     assert typ == 'classification', typ
                #     model = getClassifier(C=alpha)
                #     Yd = pd.get_dummies(Y.reshape(-1))
                #     pred = ML.cross_val_predict(model, X, Y, cv=cv, predictFuncName='decision_function')
                #     # throw strange error
                #     # pred = sklearn.model_selection.cross_val_predict(model, X, Y, cv=cv, method='predict_proba')
                #     # pred = sklearn.model_selection.cross_val_predict(model, X, Y, cv=cv)
                #     test_score = np.mean(sklearn.metrics.roc_auc_score(Yd, pred, average=None))
                #     # test_score = sklearn.metrics.accuracy_score(Y, pred)
                #     model.fit(X, Y)
                #     pred = model.decision_function(X)
                #     # pred = model.predict(X)
                #     # train_score = sklearn.metrics.accuracy_score(Y, pred)
                #     train_score = np.mean(sklearn.metrics.roc_auc_score(Yd, pred, average=None))
                if debug: df.write(f'alpha ={alpha} train_score ={train_score:.3f} test_score ={test_score:.3f}\n')
                scores.append([train_score, test_score])
            scores = np.array(scores)
            ind = np.where((scores[:,0] <= scores[:,1]*1.1) & (scores[:,0]>0) & (scores[:,1] > 0))[0]
            if len(ind) == 0:
                print('Best linear feature search was failed!!!!!!!!!!!!!!!!!!!!!!!')
                diff = np.abs(scores[:,0]-scores[:,1])
                diff[scores[:,0]<0] = np.max(diff)
                i = np.argmin(diff)
            else:
                i = ind[np.argmax(scores[ind,1])]
            best_alpha_auto = alphas[i]
            if debug: df.write(f'We take best_alpha = {best_alpha_auto}\n')
        else:
            best_alpha_auto = best_alpha
        # if typ == 'regression':
        return getRegressor(alpha=best_alpha_auto)
        # else:
        #     cl = getClassifier(C=best_alpha_auto)
        #     def predict(X):
        #         proba = cl.predict_proba(X)
        #         if labelMap is not None:
        #             if len(invLabelMap) != proba.shape[1]:
        #                 print('label =', label)
        #                 print(Y)
        #             assert len(invLabelMap) == proba.shape[1], f'{len(invLabelMap)} != {proba.shape[1]}\nlabelMap = {labelMap}\nclasses = {cl.classes_}'
        #             assert len(invLabelMap) == len(cl.classes_), f'{len(invLabelMap)} != {len(cl.classes_)}'
        #             pred = np.zeros(len(proba))
        #             for ic,c in enumerate(cl.classes_):
        #                 pred += invLabelMap[c]*proba[:,ic]
        #         return pred
        #     return types.SimpleNamespace(predict=predict, fit=lambda X,y: cl.fit(X,y))

    assert features is None or spType is None
    if features is None: features = f'{spType} spectra'
    if debug: df.write(f'features: {features}\n')
    # labelMap = sample.labelMaps.get(label, None)
    # if labelMap is not None:
    #     invLabelMap = {labelMap[v]:v for v in labelMap}
    assert ML.isOrdinal(sample.params, label)
    if label in sample.labelMaps:
        sample = sample.copy()
        sample.decode(label)
    sample0 = sample
    if debug: df.write(f'label: {label}\n')
    sample,_ = sample0.splitUnknown(columnNames=label)
    # if ML.isClassification(sample.params, label):
    #     lv, lc = np.unique(sample.params[label], return_counts=True)
    #     toDelete = lv[lc <= len(sample)*0.1]
    #     for l in toDelete:
    #         sample.delRow(sample.params[label]==l, inplace=True)
    #     if debug and len(toDelete)>0:
    #         left = np.unique(sample.params[label])
    #         df.write(f'Delete label {label} values: {toDelete}. Values left: {left}\n')
    X, Y = getXYFromSample(sample, features, [label])
    # if ML.isClassification(sample.params, label):
    #     model_class = getBestEstimator(X,Y, 'classification')
    #     model_class.fit(X, Y)
    #     pred = model_class.predict(X)
    #     df.write(f'adj.bal.acc = {sklearn.metrics.balanced_accuracy_score(Y, pred)}\n')
    #     # res = {'model':model_class, 'type':'classification'}
    #     res = model_class
    # else:
    model_regr = getBestEstimator(X,Y, 'regression')
    model_regr.fit(X,Y)
    pred = model_regr.predict(X)
    if debug: df.write(f'R2 score = {sklearn.metrics.r2_score(Y, pred)}\n')
    # res = {'model':model_regr, 'type':'regression'}
    res = model_regr
    return res


def directPrediction(sample:ML.Sample, features=None, label_names=None, makeMixtureParams=None, model_class=None, model_regr=None, labelMaps=None, folder='.', markersize=None, textsize=None, alpha=None, cv_count=2, repForStdCalc=1, unknown_sample=None, test_sample=None, textColumn=None, unknown_data_names=None, fileName=None, plot_diff=True, relativeConfMatrix=True, explanationParams=None, random_state=0, save_model=False):
    """Make direct prediction of the labels. Plot cv result graph.

        :param sample:  sample (not mixtures) or DataFrame with descriptors and labels
        :param features: (list of strings or string) features to use (x), or 'spectra' or 'spectra_d_i1,i2,i3,...' (spectra derivatives together), or ['spType1 spectra_d_i1,i2', 'spType2 spectra_d_i1,i2', ...]
        :param label_names: all label names to predict
        :param makeMixtureParams: arguments for mixture.generateMixtureOfSample excluding sample. Sample is divided into train and test and then make mixtures separately. All labels become real (concentration1*labelValue1 + concentration2*labelValue2).
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
        :param test_sample: sample to test (if is not None, sample is used as train set)
        :param textColumn: if given, use to put text inside markers
        :param unknown_data_names: if given it is used to print unk names
        :param fileName: file to save result
        :param plot_diff: True/False - whether to plot error (smoothed difference between true and predicted labels)
        :param relativeConfMatrix: normalize confusion matrix for classification by sample size
    """
    assert repForStdCalc>=1
    if label_names is None: label_names = sample.labels
    if features is None: features = sample.features
    if textColumn is None: textColumn = sample.nameColumn
    mix = makeMixtureParams is not None
    sample = sample.copy()
    if unknown_sample is not None: unknown_sample = unknown_sample.copy()
    if mix:
        assert labelMaps is None, 'We decode all features to make mixture. Set labelMaps=None'
        sample.decodeAllLabels()
        if unknown_sample is not None: unknown_sample.decodeAllLabels()
    if labelMaps is None: labelMaps = sample.labelMaps
    os.makedirs(folder, exist_ok=True)
    qualityResult = getQuality(sample, features, label_names, makeMixtureParams=makeMixtureParams, model_class=model_class, model_regr=model_regr, m=repForStdCalc, cv_count=cv_count, testSample=test_sample, returnModels=True, random_state=random_state)
    cv_result = {}
    res = getXYFromSample(sample, features, label_names, textColumn if textColumn in sample.paramNames else None)
    text = None if textColumn is None or textColumn not in sample.paramNames else res[2]
    if mix: text = None
    if test_sample is not None:
        res = getXYFromSample(test_sample, features, label_names, textColumn if textColumn in test_sample.paramNames else None)
        text = None if textColumn is None or textColumn not in test_sample.paramNames else res[2]

    if unknown_sample is not None:
        res = getXYFromSample(unknown_sample, features, label_names, textColumn)
        unkX, unky = res[0], res[1]
    else: unkX, unky = None, None
    unk_text = None
    if textColumn is not None: unk_text = res[2]
    if unknown_data_names is not None: unk_text = unknown_data_names
    if unk_text is None and unknown_sample is not None: unk_text = [str(i) for i in range(unknown_sample.getLength())]
    if isinstance(unk_text, list): unk_text = np.array(unk_text)

    for label in qualityResult:
        il = np.where(np.array(label_names) == label)[0][0]
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
                    t = cv_result[label]['model'][problemType].predict(unkX)
                    # fix MultiOutput bug
                    if len(t.shape) == 3 and t.shape[0] == 1 and t.shape[1] == unkX.shape[0]: t = t[0]
                    cv_result[label]['predictionsForUnknown'][problemType] = t
            cv_result[label]['MAE'] = np.mean(np.abs(trueLabels['avgLabels'] - predictedLabels['avgLabels']))
        else:
            isClassification = ML.isClassification(trueLabels)
            if unknown_sample is not None:
                if isClassification:
                    prob = cv_result[label]['model'].predict_proba(unkX)
                    cv_result[label]['probPredictionsForUnknown'] = prob
                cv_result[label]['predictionsForUnknown'] = cv_result[label]['model'].predict(unkX)
            if label in sample.labelMaps:
                t = sample.decode(label,values=trueLabels)
                if ML.isOrdinal(t): cv_result[label]['MAE'] = np.mean(np.abs(t - sample.decode(label,values=predictedLabels)))
                else: cv_result[label]['MAE'] = "not applied"
            else: cv_result[label]['MAE'] = np.mean(np.abs(trueLabels - predictedLabels))
        if fileName is None:
            dns = ' + '.join(features) if isinstance(features, list) else features
            if len(dns)>30: dns = dns[:30]+f'_{len(dns)}'
            fileNam = folder + os.sep + label + ' ' + dns
        else:
            fileNam = folder + os.sep + fileName
        cv_result[label]['fileNam'] = fileNam

        # saving the model
        if save_model:
            import joblib
            joblib.dump(qualityResult[label]['model'], fileNam + '_model.pkl')

        # predictions for knows (to plot predictive strength)
        if mix:
            y_pred = qualityResult[label]['singlePred']
            y_true = qualityResult[label]['singleTrue']
        else:
            y_pred = qualityResult[label]['predictedLabels']
            y_true = qualityResult[label]['trueLabels']
        if test_sample is None:
            assert np.all(y_true == sample.params[label]), f'{y_true.tolist()}\n{sample.params[label].tolist()}'
        if label in sample.labelMaps:
            y_true = sample.decode(label=label, values=y_true)
            if not mix:
                y_pred = sample.decode(label=label, values=y_pred)
        cv_data = pd.DataFrame()
        if sample.nameColumn is not None:
            cv_data[sample.nameColumn] = text
        cv_data['true'], cv_data['pred'] = y_true, y_pred
        cv_result[label]['true_vs_pred'] = cv_data
        cv_data.to_csv(fileNam+'_cv.csv', index=False, sep=';')

        def plot(true, pred, quality, filePostfix='', unkn_true=None, unkn_pred=None):
            plotFileName = fileNam+'_'+filePostfix
            pred = pred.reshape(-1)
            true = true.reshape(-1)
            if ML.isClassification(true):
                labelMap = labelMaps[label] if label in labelMaps else None
                plotting.plotConfusionMatrix(true, pred, label, labelMap=labelMap, fileName=plotFileName + '.png', relativeConfMatrix=relativeConfMatrix)
            else:
                err = np.abs(true - pred)
                def plotMoreFunction(ax):
                    if unkn_true is not None:
                        ind = ~np.isnan(unkn_true)
                        utr, upr = unkn_true[ind], unkn_pred[ind]
                        if np.sum(ind) > 0:
                            ax.plot(utr, upr, marker="o", alpha=alpha, markersize=markersize, lw=0)
                            ut = unk_text[ind]
                            for i in range(len(ut)):
                                ax.text(utr[i], upr[i], str(ut[i]), ha='center', va='center', size=textsize)
                    ax.plot([pred.min(), pred.max()], [pred.min(), pred.max()], 'r', lw=2)
                quality_s = ''
                for m,qm in quality.items():
                    if utils.isArray(qm):
                        assert len(qm) == 2, f'Error: {qm} must be confidence interval'
                        quality_s += f' {m}=[{qm[0]:.2f}; {qm[1]:.2f}]'
                    else: quality_s += f' {m}={qm:.2f}'
                quality_s = quality_s.strip()
                plotting.scatter(pred, true, color=err, alpha=alpha, markersize=markersize, text_size=textsize, marker_text=text, xlabel='predicted ' + label, ylabel='true ' + label, title='CV result for label ' + label + f'. Quality: {quality_s}', fileName=plotFileName + '.png', plotMoreFunction=plotMoreFunction)
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
                    if problemType == 'avgLabels':
                        if unky is None:
                            plot(true, pred, cv_result[label]["quality"][problemType], filePostfix=problemType)
                        else:
                            plot(true, pred, cv_result[label]["quality"][problemType], filePostfix=problemType, unkn_true=unky[:,il], unkn_pred=cv_result[label]['predictionsForUnknown'][problemType])
                    else:
                        plot(true, pred, cv_result[label]["quality"][problemType], filePostfix=problemType)
        else:
            if unky is None:
                plot(trueLabels, predictedLabels, cv_result[label]["quality"])
            else:
                plot(trueLabels, predictedLabels, cv_result[label]["quality"], unkn_true=unky[:,il], unkn_pred=cv_result[label]['predictionsForUnknown'])
    if explanationParams is not None and len(features) == 1 and isinstance(features[0],str) and features[0].split()[-1]=='spectra' and ' ' in features[0] and features[0][:features[0].index(' ')] in sample.spTypes():
        assert isinstance(explanationParams, dict)
        spType = features[0][:features[0].index(' ')]
        e = sample.getEnergy(spType)
        e1, e2 = e[0], e[-1]
        h = explanationParams.get('h', (e2-e1)/20)
        n = explanationParams.get('n', 20)
        assert sample.nameColumn is not None and unknown_sample.nameColumn is not None
        known_names = sample.params[sample.nameColumn].tolist()
        unknown_names = [] if unknown_sample is None else unknown_sample.params[unknown_sample.nameColumn].tolist()
        if 'explainFor' in explanationParams:
            explainFor = explanationParams['explainFor']
        else:
            if unknown_sample is None: explainFor = known_names
            else: explainFor = unknown_names
        centers = np.linspace(e1+h, e2-h, n)
        for label in qualityResult:
            Y = qualityResult[label]['trueLabels']
            model = copy.deepcopy(qualityResult[label]['model'])
            predictions = {name:np.zeros(n) for name in explainFor}
            for i in range(n):
                c = centers[i]
                X = sample.limit([c-h, c+h], spType=spType, inplace=False).getSpectra(spType).to_numpy()
                model.fit(X,Y)
                unknown_sample_limited = None
                for name in explainFor:
                    if name in known_names: pred_x = X[sample.getIndByName(name)]
                    else:
                        if unknown_sample_limited is None: unknown_sample_limited = unknown_sample.limit([c-h, c+h], spType=spType, inplace=False).getSpectra(spType).to_numpy()
                        pred_x = unknown_sample_limited[unknown_sample.getIndByName(name)]
                    predictions[name][i] = model.predict(pred_x.reshape(1,-1))[0]
            for name in explainFor:
                toPlot = (centers, predictions[name], 'prediction')
                if name in known_names:
                    true = sample.params.loc[sample.getIndByName(name),label]
                    toPlot += ([centers[0], centers[-1]], [true,true], 'true')
                plotting.plotToFile(*toPlot, fileName=f'{folder}{os.sep}explanation{os.sep}{label} {name}.png')

    cv_result_for_print = copy.deepcopy(cv_result)
    # apply labelMaps
    for label in cv_result_for_print:
        if label not in labelMaps: continue
        lm = {labelMaps[label][l]: l for l in labelMaps[label]}
        def convert(vec):
            assert isinstance(vec, np.ndarray), str(vec)
            assert len(vec.shape) == 1, str(vec.shape)
            return np.array([lm[int(x)] for x in vec])
        r = cv_result_for_print[label]
        if mix:
            compLab, u_compLab = [], []
            for component in range(r['trueLabels']['componentLabels'].shape[1]):
                compLab.append(convert(r['trueLabels']['componentLabels'][:,component]))
                u_compLab.append(convert(r['predictionsForUnknown']['componentLabels'][:,component]))
            r['trueLabels']['componentLabels'] = np.array(compLab).T
            if unknown_sample is not None: r['predictionsForUnknown']['componentLabels'] = np.array(u_compLab).T
        else:
            r['trueLabels'] = convert(r['trueLabels'])
            if unknown_sample is not None: r['predictionsForUnknown'] = convert(r['predictionsForUnknown'])
    for label in cv_result_for_print:
        r = cv_result_for_print[label]
        with open(r['fileNam'] + '_unkn.txt', 'w', encoding='utf-8') as f:
            s = f"Prediction of {label} by " + r['features'] + '\n'
            s += "quality = " + str(r['quality']) + '\n'
            s += "quality_std = " + str(r['quality_std']) + '\n'
            s += "quality (MAE) = " + str(r['MAE']) + '\n'
            f.write(s)
            if unknown_sample is not None:
                def printPredictions(ps):
                    assert len(ps) == len(unk_text), f'{len(ps)} != {len(unk_text)}. ps = {ps}'
                    pred = ""
                    for i in range(len(unk_text)):
                        p = ps[i]
                        pred += f"{unk_text[i]}: {p}"
                        true = unknown_sample.params.loc[i, label]
                        if label in unknown_sample.labelMaps:
                            true = unknown_sample.decode(label, values=[true])[0]
                        if (not utils.is_numeric(true)) or not np.isnan(true):
                            pred += f"   true = {true}"
                            if utils.is_numeric(true) and utils.is_numeric(p): pred += f"   err = {np.abs(true - p)}"
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


def directPredictionParser(folder, knownOnly=False, labels=None):
    """
    Parse results of the directPrediction function.
    
    :returns: cv Dataframe, predictions Dataframe for unknown
    """

    if labels is None:
        labels = []
        fs = utils.findFile(folder=folder, postfix='_diff.txt', returnAll=True)
        for f in fs:
            with open(f) as file: s = file.readline()
            i = s.rfind(' for label ')
            assert i>=0
            i += len(' for label ')
            labels.append(s[i:].strip())
    unk_pred = None
    cv_pred = pd.DataFrame(columns=labels)
    for label in labels:
        # unknown
        if not knownOnly:
            if unk_pred is None: unk_pred= pd.DataFrame(columns=labels)
            fileName = utils.findFile(folder=folder, mask=f'{label}*_unkn.txt', check_unique=True)
            with open(fileName,'r') as f: s = f.read()
            i1 = s.index('predictions:')+len('predictions:')+1
            while s[i1] == '\n': i1 += 1
            i2 = s.index('\n\n',i1)
            s = s[i1:i2].strip()
            if 'prediction:' in s: s = s[s.index('\n')+1:]
            for line in s.split('\n'):
                unkName, pred = line.split(': ')
                if utils.is_str_float(pred):
                    pred = float(pred)
                if label in unk_pred.columns and unkName in unk_pred.index:
                    assert pd.isnull(unk_pred.loc[unkName, label]), f'Row {unkName} column {label} was already filled. Old value = {unk_pred.loc[unkName, label]} new value = {pred}'
                unk_pred.loc[unkName, label] = pred

        # known
        fileName = utils.findFile(folder=folder, mask=f'{label}*_cv.csv')
        d = pd.read_csv(fileName, sep=';')
        nameCol = d.columns[0]
        for i in range(len(d)):
            name = d.loc[i, nameCol]
            if f'true {label}' not in cv_pred.columns or name not in cv_pred.index or pd.isnull(
                    cv_pred.loc[name, f'true {label}']):
                cv_pred.loc[name, f'true {label}'] = d.loc[i, 'true']
            else:
                if cv_pred.loc[name, f'true {label}'] - d.loc[i, 'true'] > 1e-3:
                    print(fileName)
                    for ii in range(len(d)):
                        name = d.loc[ii, nameCol]
                        print('name =', name, 'true1 =', cv_pred.loc[name, f'true {label}'], 'true2 =', d.loc[ii, 'true'])
                assert cv_pred.loc[name, f'true {label}'] - d.loc[i, 'true'] <= 1e-3
            cv_pred.loc[name, f'pred {label}'] = d.loc[i, 'pred']
    return cv_pred, unk_pred


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
        result_file = open(output_file, 'w', encoding='utf-8')
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


def calcCalibrationData(expData, theoryData, componentNameColumn=None, folder=None, excludeColumnNames=None, multiplierOnlyColumnNames=None, shiftOnlyColumnNames=None, stableLinearRegression=True, plotMoreFunction=None, scatterExtraParams=None):
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
    :param plotMoreFunction: user defined function(ax, x,y,names, descriptor_name, [a,b]) to plot something more
    :param scatterExtraParams:
    :return: dict {descriptorName: [toDiv, toSub]} theory = toDiv*exp + toSub, then calibration:  newTheoryDescr = (theoryDescr - toSub) / toDiv
    """
    if excludeColumnNames is None: excludeColumnNames = []
    if multiplierOnlyColumnNames is None: multiplierOnlyColumnNames = []
    if shiftOnlyColumnNames is None: shiftOnlyColumnNames = []
    if scatterExtraParams is None: scatterExtraParams = {}
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
            def plotMoreFunction1(ax):
                ax.plot(x, y_regr, color='k')
                if plotMoreFunction is not None:
                    plotMoreFunction(ax, x,y,names, descriptor_name, [a,b])
            plotting.scatter(x,y,marker_text=names, fileName=folder+os.sep+descriptor_name+'.png', plotMoreFunction=plotMoreFunction1, title=f"{descriptor_name}: theory = {a:.3f}*exp + {b:.2f}", xlabel='exp', ylabel='theory', **scatterExtraParams)
            # fig, ax = plotting.createfig()
            # ax.scatter(x, y, 600, c='yellow')
            # for i in range(len(names)):
            #     ax.text(x[i], y[i], names[i], ha='center', va='center', size=7)
            # ax.plot(x, y_regr, color='k')
            # ax.set_xlabel('exp')
            # ax.set_ylabel('theory')
            # ax.set_xlim(np.min(x), np.max(x))
            # ax.set_ylim(np.min(y), np.max(y))
            # ax.set_title(f"{descriptor_name}: theory = {a:.3f}*exp + {b:.2f}")
            # plotting.savefig(folder+os.sep+descriptor_name+'.png', fig)
            # plotting.closefig(fig)
    return calibration


def calibrateSample(sample, calibrationDataForDescriptors=None, calibrationDataForSpectra=None, inplace=False):
    """
    Linear calibration of sample
    :param sample: theory sample
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
                sp = sample.getSpectrum(ind=i, spType=spType, returnIntensityOnly=True)
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
            if spType not in theorySample.spTypes(): continue
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
        theory.changeEnergy(newEnergy=combined.getEnergy(spType=spType), spType=spType, inplace=True)
    combined.unionWith(theory, inplace=True)
    return combined


def energyImportance(sample:ML.Sample, label_names=None, folder='energyImportance', model=None, spType=None, class_metric=None, regr_metric=None):
    if label_names is None: label_names = sample.labels
    def saveRes(e,q,fn):
        d = pd.DataFrame()
        if len(q.shape) == 1: i = np.argsort(-q)
        else: i = np.argsort(-q[0])
        d['energy'] = e[i]
        if len(q.shape) == 1: d['quality'] = q[i]
        else:
            for j in range(q.shape[0]): d[f'quality_d{j}'] = q[j][i]
        d.to_csv(fn, index=False)
    def plot(e, quality, spectra, label, title, filename):
        if len(quality.shape) == 1: quality = quality.reshape(1, -1)
        sample1 = sample.copy()
        sample1.addSpectrumType(spectra, energy=e, spType='temp sptype')
        def plotMore(ax):
            ax2 = ax.twinx()
            for i in range(quality.shape[0]):
                lb = '' if quality.shape[0] == 1 else f'd{i}'
                ax2.plot(e, quality[i], label=lb, lw=2)
            if quality.shape[0] != 1: ax2.legend()
        sample1.plot(colorParam=sample.params[label], spType='temp sptype', folder=filename, plotSampleParams=dict(title=title, plotMoreFunction=plotMore))

    # autocorrelation
    n = len(sample)
    spectra = sample.getSpectra(spType=spType).to_numpy()
    energy = sample.getEnergy(spType=spType)
    auto_mi = np.zeros((len(energy),len(energy)))
    for j in range(len(energy)):
        auto_mi[j] = sklearn.feature_selection.mutual_info_regression(spectra, spectra[:,j],random_state=0)
        auto_mi[j,j] = 0
    format = '%g'
    es = [(format % ei) for ei in energy]
    plotting.plotMatrix(auto_mi, fileName=f'{folder}/auto MI.png', ticklabelsX=es, ticklabelsY=es)

    if n < 100: min_samples_leaf = 1
    else: min_samples_leaf = 4
    modelIsNone = model is None
    denergy = energy[1:]-energy[:-1]
    dspectra = (spectra[:,1:]-spectra[:,:-1])/denergy
    d2spectra = (dspectra[:,1:]-dspectra[:,:-1])/denergy[:-1]
    spectras = [spectra, dspectra, d2spectra]
    energies = [energy, energy[:-1], energy[1:-1]]
    for label in label_names:
        classification = ML.isClassification(sample.params, label)
        if modelIsNone:
            n_estimators = 1000
            if classification:
                model = sklearn.ensemble.ExtraTreesClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf)
                qName = class_metric if class_metric is not None else 'accuracy'
            else:
                model = sklearn.ensemble.ExtraTreesRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf)
                qName = regr_metric if regr_metric is not None else 'R2-score'
        for isp,sp in enumerate(spectras):
            overall_quality = ML.score_cv(model, sp, sample.params[label], cv_count=10, returnPrediction=False)[qName]
            m = len(energies[isp])
            quality = np.zeros(m)
            model.fit(sp, sample.params[label])
            quality[:] = model.feature_importances_
            plot(energies[isp], quality, spectra=sp, label=label, title=f'Energy importance for label {label}. Overall quality = {overall_quality:.2f}', filename=f'{folder}/{label}_d{isp}.png')
            saveRes(energies[isp], quality, f'{folder}/{label}_d{isp}_data.txt')
        overall_quality = ML.score_cv(model, np.hstack(spectras), sample.params[label], cv_count=10, returnPrediction=False)[qName]
        m = len(energies[-1])
        sps = np.hstack((spectras[0][:, 1:-1], spectras[1][:, :-1], spectras[2]))
        assert sps.shape[1] == 3*m
        model.fit(sps, sample.params[label])
        f = model.feature_importances_
        quality = np.array([f[:m], f[m:2*m], f[2*m:]])
        plot(energies[-1], quality, spectra=spectras[0][:, 1:-1], label=label, title=f'Energy importance for label {label}. Overall quality = {overall_quality:.2f}', filename=f'{folder}/{label}_d012.png')
        saveRes(energies[-1], quality, f'{folder}/{label}_d012_data.txt')


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


def calcPreedge(sample:ML.Sample, specialParams=None, commonOptimizationParams=None, plotFolder=None, debug=False, inplace=False):
    """
    :param specialParams: dict{ name: dict of substractBase params for spectrum}, if name == 'all' - apply to all spectra
    :param commonOptimizationParams: list of dicts with keys: src - name of spectra to take params, dst - 'all' or list of names, useAsStart - True/False run or not optimization
    """
    def getParams(name):
        p = {}
        if specialParams is None: return True, p
        if 'all' in specialParams:
            assert len(specialParams) == 1
            p = copy.deepcopy(specialParams['all'])
        else:
            if name in specialParams:
                p = copy.deepcopy(specialParams[name])
        if 'peakInterval' in p or 'baseFitInterval' in p:
            auto = False
            assert 'edgeLevel' not in p, f'edgeLevel (auto) is set along with peakInterval and baseFitInterval (non auto) for {name}'
            assert 'peakInterval' in p and 'baseFitInterval' in p
        else:
            auto = True
        return auto, p
    if commonOptimizationParams is not None:
        for c in commonOptimizationParams:
            if c['dst'] == 'all': c['dst'] = list(set(sample.params.loc[:,sample.nameColumn].tolist()) - {c['src']})
        all_dst = []
        for c in commonOptimizationParams: all_dst += c['dst']
        assert len(set(all_dst)) == len(all_dst), f'Duplicate destination detected: {all_dst}'
    if not inplace: sample = copy.deepcopy(sample)
    if sample.getDefaultSpType() == 'default':
        sample.renameSpType('default', 'xanes')
    area, center = [], []
    preedgeSpectra = []
    xanesWoPreedge = []
    for i in range(len(sample)):
        sp = sample.getSpectrum(ind=i, spType='xanes')
        name = sample.params.loc[i,sample.nameColumn] if sample.nameColumn is not None else str(i)
        if debug: print(name)
        plotFileName = f'{plotFolder}/{name}.png' if plotFolder is not None else None
        auto, params = getParams(name)
        # print('auto =',auto,'params =',params)
        if auto:
            assert commonOptimizationParams is None
            res = curveFitting.subtractBaseAuto(sp.x, sp.y, plotFileName=plotFileName, debug=debug, **params)
        else:
            if 'plotFileName' not in params: params['plotFileName'] = plotFileName
            res = curveFitting.subtractBase(sp.x, sp.y, debug=debug, **params)
        peak = res['peak']
        area.append( utils.integral(peak.x, peak.y) )
        center.append( utils.integral(peak.x, peak.x*peak.y)/area[-1] )
        preedgeSpectra.append(peak)
        wo = copy.deepcopy(sp.y)
        ab = res['info']['peakInterval']
        ind = (sp.x >= ab[0]) & (sp.x <= ab[1])
        wo[ind] = res['base'].y
        xanesWoPreedge.append(wo)
    sample.addSpectrumType(preedgeSpectra, spType='pre-edge', interpArgs=dict(left=0, right=0))
    sample.addSpectrumType(xanesWoPreedge, spType='xanesWoPre-edge', energy=sample.getEnergy('xanes'))
    sample.addParam(paramName='pe area', paramData=area)
    sample.addParam(paramName='pe center', paramData=center)
    if not inplace: return sample


def resultSummary(sample:ML.Sample, labels=None, unknownNames=None, baseFolder='results', settings=None, wrapXTickLabelLength=10, figsize=None, postfix='', notExistIsError=True, fileExtension='.png', delFromCV=None, criticalQuantileIndivPlot=0.5, plotIndividualParams=None, plotIndividualExtraNames=None, plotSampleParams=None):
    """
    Gather unknown exp predictions from various methods. Settings is the list of dicts with keys: type (descriptor, direct, LCF), folder, label (str or list), count, measure, prefix ...
    """
    assert settings is not None
    if labels is None: labels = sample.labels
    if plotIndividualParams is None: plotIndividualParams = {}
    knownOnly = False
    if unknownNames is None:
        if sample.splitUnknown(labels)[1] is None: knownOnly = True
        else:
            unknownNames = sample.splitUnknown(labels)[1].params[sample.nameColumn]
    if delFromCV is None: delFromCV = []
    if plotIndividualExtraNames is None: plotIndividualExtraNames = []
    if plotSampleParams is None: plotSampleParams = {}

    def updateKnown(cv_pred, columnName, label, dataSrc, predName='pred'):
        # known
        if isinstance(dataSrc, str):
            if not os.path.exists(dataSrc):
                print('No file:', dataSrc)
                return
            d = pd.read_csv(dataSrc, sep=';')
        else:
            assert isinstance(dataSrc, pd.DataFrame)
            d = dataSrc
        if label not in cv_pred: cv_pred[label] = pd.DataFrame()
        nameCol = d.columns[0]
        for i in range(len(d)):
            name = d.loc[i, nameCol]
            if 'true' not in cv_pred[label].columns or name not in cv_pred[label].index or pd.isnull(
                    cv_pred[label].loc[name, 'true']):
                cv_pred[label].loc[name, 'true'] = d.loc[i, 'true']
            else:
                if cv_pred[label].loc[name, 'true'] - d.loc[i, 'true'] > 1e-3:
                    if isinstance(dataSrc, str): print(dataSrc)
                    for ii in range(len(d)):
                        name = d.loc[ii, nameCol]
                        print('name =', name, 'true1 =', cv_pred[label].loc[name, 'true'], 'true2 =', d.loc[ii, 'true'])
                assert cv_pred[label].loc[name, 'true'] - d.loc[i, 'true'] <= 1e-3
            cv_pred[label].loc[name, columnName] = d.loc[i, predName]

    def updateUnknown(r:pd.DataFrame, index, column, value):
        if knownOnly: return
        if column in r.columns:
            assert pd.isnull(r.loc[index, column]), f'Row {index} column {column} was already filled. Old value = {r.loc[index, column]} new value = {value}'
        r.loc[index, column] = value

    r = pd.DataFrame()
    if not knownOnly:
        r.index = pd.MultiIndex.from_product([unknownNames,labels], names=['unk', 'label'])
    inverseLabelMaps = sample.inverseLabelMaps()
    cv_pred = {}

    for setngs in settings:
        assert isinstance(setngs, dict)
        assert set(setngs.keys()) <= {'type', 'folder', 'prefix', 'label', 'descriptors'}, str(setngs)
        folder = baseFolder + os.sep + setngs['folder']
        lbs = setngs.get('label', labels)
        if isinstance(lbs,str): lbs = [lbs]
        prefix0 = setngs.get('prefix', '')
        typ = setngs['type']
        if typ == 'direct':
            cv, unk = directPredictionParser(folder, knownOnly=False, labels=None)
            for l in lbs:
                # unknown
                if not knownOnly:
                    fileName = utils.findFile(folder=folder, mask=f'{l}*_unkn.txt', check_unique=notExistIsError)
                    if fileName is None: continue
                    features = os.path.split(fileName)[-1][len(l)+1:-9]
                    if utils.is_str_float(features.split('_')[-1]): features = 'features'
                    prefix = prefix0+features if prefix0 == '' else prefix0
                    for unkName, row in unk.iterrows():
                        if unkName not in unknownNames: continue
                        updateUnknown(r, (unkName, l), prefix, value=row[l])

                # known
                fileName = utils.findFile(folder=folder, mask=f'{l}*_cv.csv')
                features = os.path.split(fileName)[-1][len(l)+1:-7]
                prefix = prefix0+features if prefix0 == '' else prefix0
                for name, row in cv.iterrows():
                    cv_pred[l].loc[name, 'true'] = row[f'true {l}']
                    cv_pred[l].loc[name, prefix] = row[f'pred {l}']
        elif typ == 'LCF':
            # unknown
            for unkName in unknownNames:
                if os.path.exists(folder+os.sep+unkName+os.sep+'label_maps'): comp = 2
                else: comp = 1
                prefix = f'LCF{comp} {prefix0}'
                fileName = sorted(os.listdir(folder + os.sep + unkName + os.sep + 'spectra'))[1]
                fileName = folder + os.sep + unkName + os.sep + 'spectra' + os.sep + fileName
                d = plotting.readPlottingFile(fileName)
                title = d['title']
                if comp == 1:
                    neighbour = title.split('. DistsToExp')[0][len('Candidate '):].strip()
                    i_neighbour = sample.getIndByName(neighbour)
                    for l in lbs:
                        v = sample.params.loc[i_neighbour, l]
                        if l in inverseLabelMaps: v = inverseLabelMaps[l][v]
                        updateUnknown(r, (unkName, l), prefix, v)
                else:
                    sc = title.split(', distsToExp')[0][len('Concentrations: '):].split(', ')
                    conc = [float(s.split('=')[1]) for s in sc]
                    comp = [s.split('=')[0][2:] for s in sc]
                    for l in lbs:
                        lbls = [sample.params.loc[sample.getIndByName(n), l] for n in comp]
                        if l in inverseLabelMaps:
                            lbls = [inverseLabelMaps[l][v] for v in lbls]
                        mean_l = np.dot(conc, lbls)
                        updateUnknown(r, (unkName, l), prefix, mean_l)
            # known
            for l in lbs:
                fileName = folder+os.sep+'CV'+os.sep+l+os.sep+'predictions.csv'
                updateKnown(cv_pred, columnName=prefix, label=l, dataSrc=fileName)
        elif typ == 'descriptor':
            measure = setngs.get('measure', '')
            if measure == '':
                if os.path.exists(f'{folder}{os.sep}{lbs[0]} accuracy.xlsx'): measure = 'accuracy'
                else: measure = 'R2-score'
            for l in lbs:
                fileName = f'{folder}{os.sep}{l} {measure}.xlsx'
                d = pd.read_excel(fileName, index_col='unknown')
                if 'count' in setngs:
                    descriptors = d.columns[2:2 + setngs['count']]
                else:
                    assert 'descriptors' in setngs
                    descriptors = setngs['descriptors']
                for dn in descriptors:
                    for unkName, row in d.iterrows():
                        if unkName == 'quality': continue
                        updateUnknown(r, (unkName,l), prefix0+dn, d.loc[unkName, dn])
                    fileName = f'{folder}{os.sep}{l} cv.csv'
                    updateKnown(cv_pred, columnName=prefix0+dn, label=l, dataSrc=fileName, predName=dn)
        else:
            assert False, f'Unknown type: '+typ
    os.makedirs(f'{baseFolder}{os.sep}summary{postfix}', exist_ok=True)
    r.to_excel(f'{baseFolder}{os.sep}summary{postfix}{os.sep}unknown.xlsx')
    known:ML.Sample = sample.splitUnknown(labels)[0]
    if len(delFromCV) > 0:  known.delRowByName(delFromCV, inplace=True)
    for l in cv_pred:
        if len(delFromCV)>0:  cv_pred[l].drop(delFromCV, axis=0, inplace=True)
        d = cv_pred[l].sort_values('true')
        d.to_excel(f'{baseFolder}{os.sep}summary{postfix}{os.sep}known {l}.xlsx')
        cols = d.columns[d.columns != 'true']
        m = d.loc[:, cols].to_numpy()
        m = m - d.loc[:,'true'].to_numpy().reshape(-1,1)
        vmax = (np.max(d.loc[:,'true']) - np.min(d.loc[:,'true'])) / 2
        # cmap = 'RdYlBu'
        cmap = plotting.symmetrical_colormap(cmap_settings=('Reds', None), new_name=None)
        plotting.plotMatrix(m, ticklabelsX=cols, ticklabelsY=d.index, fileName=f'{baseFolder}{os.sep}summary{postfix}{os.sep}known {l} pred-true{fileExtension}', title=f'Difference between predicted and true label values for {l}', wrapXTickLabelLength=wrapXTickLabelLength, figsize=figsize, cmap=cmap, annot=True, fmt='.1f', vmin=-vmax, vmax=vmax, interactive=True)
        # plot spectra with max errors
        m = np.abs(m)
        q = np.quantile(m, q=criticalQuantileIndivPlot, axis=0, keepdims=True)
        error_count = np.sum(m >= q, axis=1)
        ind = np.argsort(-error_count)
        print(f'Spectra with max errors count for label {l}:', end=' ')
        for ii in range(len(m)):
            i = ind[ii]
            name = d.index[i]
            if error_count[i] == error_count[ind[0]]:
                print(f'{name}', end=' ')
        print('')
        known.plot(f'{baseFolder}{os.sep}tmp_sample_plot{postfix}{os.sep}/{l}', colorParam=l, plotIndividualParams={'plot on sample': True}, plotSampleParams=plotSampleParams)
        os.makedirs(f'{baseFolder}{os.sep}summary{postfix}{os.sep}cv_errors {l}', exist_ok=True)
        ii = 0
        while True:
            i = ind[ii]
            name = d.index[i]
            if ii < 5 or error_count[i] == error_count[ind[0]] or name in plotIndividualExtraNames:
                for spType in known.spTypes():
                    shutil.copyfile(f'{baseFolder}{os.sep}tmp_sample_plot{postfix}{os.sep}{l}{os.sep}individ_{spType}{os.sep}{name}.png', f'{baseFolder}{os.sep}summary{postfix}{os.sep}cv_errors {l}{os.sep}{name} {spType}.png')
            ii += 1
            if ii >= len(m): break

    if utils.isJupyterNotebook():
        print('Predictions for unknown spectra from the file', f'{baseFolder}{os.sep}summary{postfix}{os.sep}unknown.xlsx')
        print(r)


def graphConnectedComponents(aNeigh):
    """
    Input example: myGraph = {0: [1,2,3], 1: [], 2: [1], 3: [4,5],4: [3,5], 5: [3,4,7], 6: [8], 7: [],8: [9], 9: []}
    Returns: [6, 8, 9], [0, 1, 2, 3, 4, 5, 7]
    """
    def findRoot(aNode, aRoot):
        while aNode != aRoot[aNode][0]:
            aNode = aRoot[aNode][0]
        return (aNode, aRoot[aNode][1])

    myRoot = {}
    for myNode in aNeigh.keys():
        myRoot[myNode] = (myNode, 0)
    for myI in aNeigh:
        for myJ in aNeigh[myI]:
            (myRoot_myI, myDepthMyI) = findRoot(myI, myRoot)
            (myRoot_myJ, myDepthMyJ) = findRoot(myJ, myRoot)
            if myRoot_myI != myRoot_myJ:
                myMin = myRoot_myI
                myMax = myRoot_myJ
                if myDepthMyI > myDepthMyJ:
                    myMin = myRoot_myJ
                    myMax = myRoot_myI
                myRoot[myMax] = (myMax, max(myRoot[myMin][1] + 1, myRoot[myMax][1]))
                myRoot[myMin] = (myRoot[myMax][0], -1)
    myToRet = {}
    for myI in aNeigh:
        if myRoot[myI][0] == myI:
            myToRet[myI] = []
    for myI in aNeigh:
        myToRet[findRoot(myI, myRoot)[0]].append(myI)
    return list(myToRet.values())


def independentFeatureSubset_old(sample:ML.Sample, features=None, labels=None, pair_MI_threshold=None, preliminary_label_MI_threshold=None, result_label_MI_threshold=None, debugInfoFolder=None):
    """
    :param preliminary_label_MI_threshold: for one label - MI threshold, for multiple - dict {label: threshold}
    :param result_label_MI_threshold: the same
    :param pair_MI_threshold: the same
    """
    if features is None: features = sample.features
    if isinstance(features, list): features = np.array(features)
    if labels is None: labels = sample.labels
    assert len(set(labels) & set(features)) == 0
    assert set(labels) <= set(sample.paramNames)
    assert set(features) <= set(sample.paramNames)
    for f in features:
        assert ML.isOrdinal(sample.params, f)
    if preliminary_label_MI_threshold is None: preliminary_label_MI_threshold = -np.inf
    if not isinstance(preliminary_label_MI_threshold, dict):
        preliminary_label_MI_threshold = {l: preliminary_label_MI_threshold for l in labels}
    if result_label_MI_threshold is None: result_label_MI_threshold = -np.inf
    if not isinstance(result_label_MI_threshold, dict):
        result_label_MI_threshold = {l: result_label_MI_threshold for l in labels}
    assert pair_MI_threshold is not None
    if not isinstance(pair_MI_threshold, dict):
        pair_MI_threshold = {l: pair_MI_threshold for l in labels}

    result = {}
    for label in labels:
        isClassification = not ML.isOrdinal(sample.params, label)
        if isClassification:
            label_MI = sklearn.feature_selection.mutual_info_classif(sample.params.loc[:,features], sample.params[label], random_state=0)
        else:
            label_MI = sklearn.feature_selection.mutual_info_regression(sample.params.loc[:,features], sample.params[label], random_state=0)
        ind = np.argsort(-label_MI)
        features = features[ind]
        label_MI = label_MI[ind]
        features1 = features[label_MI >= preliminary_label_MI_threshold[label]]
        label_MI1 = label_MI[label_MI >= preliminary_label_MI_threshold[label]]
        n = len(features1)
        features_MI = np.zeros((n, n))
        for i in range(n):
            features_MI[i] = sklearn.feature_selection.mutual_info_regression(sample.params.loc[:, features1], sample.params[features1[i]], random_state=0)
        features_MI1 = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                w = np.copy(label_MI1)
                w[i], w[j] = 1, 1
                power = 1
                features_MI1[i,j] = (np.mean( w*np.minimum(features_MI[i], features_MI[j])**power ))**(1/power)
        for i in range(n): features_MI1[i,i] = np.max(features_MI1)
        features_MI2 = np.copy(features_MI1)
        features_MI2[features_MI1 < pair_MI_threshold[label]] = 0
        features_MI2[features_MI1 >= pair_MI_threshold[label]] = 1
        nodes = {}
        inds = np.arange(n)
        for i in range(n):
            nodes[i] = inds[features_MI2[i] == 1].tolist()
        comp = graphConnectedComponents(nodes)
        ind = np.argsort([-label_MI1[np.min(c)] for c in comp])
        comp = [comp[ind[i]] for i in range(len(comp))]
        result[label] = [features1[np.min(c)] for c in comp if label_MI1[np.min(c)] >= result_label_MI_threshold[label]]
        if debugInfoFolder is not None:
            for i in range(n):
                features_MI[i, i], features_MI1[i, i] = 0, 0
            figsize = (20, 10)
            plotting.plotMatrix(features_MI, ticklabelsX=features1, ticklabelsY=features1, fileName=f'{debugInfoFolder}{os.sep}{label} MI matrix.png', title='Mutual information for best features', figsize=figsize)
            plotting.plotMatrix(features_MI1, ticklabelsX=features1, ticklabelsY=features1, fileName=f'{debugInfoFolder}{os.sep}{label} corrected MI matrix.png', title='Corrected mutual information for best features', figsize=figsize)
            plotting.plotMatrix(features_MI2, ticklabelsX=features1, ticklabelsY=features1, fileName=f'{debugInfoFolder}{os.sep}{label} graph matrix.png', title='Adjacency matrix', figsize=figsize)
            with open(f'{debugInfoFolder}{os.sep}{label} graph.txt','w') as f:
                for node in nodes:
                    f.write(f'{node}: {nodes[node]}\n')
                f.write('\nComponents:\n')
                for c in comp:
                    f.write(f'{c}\n')
                f.write(f'\nSorted features: ')
                for i in range(len(features)):
                    f.write(f'{features[i]}:{label_MI[i]:.3f}; ')
                f.write(f'\n\nResult features: {result[label]}\n')
    return result


def independentFeatureSubset(data:pd.DataFrame, features=None, labels=None, MIalgorithm='MI', featureFeatureMItype='regression', featureLabelMItype='auto', folder=None, neighbourCountForColor=None, logColor=False, scatterParams=None, perplexity=None, plotAxes='TSNE and PCA', debug=True):
    """
    :param features: list of feature column names
    :param labels: list of label names
    :param MIalgorithm: 'MI' (mutual information) or 'corr' (correlation - faster)
    :param featureFeatureMItype: 'regression' or 'classification' - use mutual_info_regression or mutual_info_classification for feature-feature similarity calculation
    :param featureLabelMItype: 'auto' (label dependent), 'regression' or 'classification', or dict(label:'regression/classification') - use mutual_info_regression or mutual_info_classification for feature-label similarity calculation
    :param plotAxes: 'TSNE and PCA' or 'PCA only'
    """

    def scale(data):
        return sklearn.preprocessing.StandardScaler().fit_transform(data)
    def similarity(f1,f2,typ):
        if isinstance(f1, pd.DataFrame):
            names = f1.columns
            f1 = f1.to_numpy()
        else:
            if len(f1.shape) == 2: names = [f'{j+1}' for j in range(f1.shape[1])]
        if len(f1.shape) == 1:
            f1 = f1.reshape(-1,1)
            names = ['0']
        if MIalgorithm == 'MI':
            if typ == 'regression':
                res = sklearn.feature_selection.mutual_info_regression(f1, f2, random_state=0)
            else:
                for j in range(f1.shape[1]): assert ML.isClassification(f1[:,j]), f'Feature {names[j]} is not polynominal: {f1[:,j].tolist()}'
                assert ML.isClassification(f2)
                res = sklearn.feature_selection.mutual_info_classif(f1, f2, random_state=0)
        else:
            if typ == 'regression':
                res = sklearn.feature_selection.r_regression(f1,f2)
            else:
                for j in range(f1.shape[1]): assert ML.isClassification(f1[:,j]), f'Feature {names[j]} is not polynominal'
                assert ML.isClassification(f2)
                res = np.array([sklearn.metrics.matthews_corrcoef(f1[:,j],f2) for j in range(f1.shape[1])])
                if f1.shape[1] == 1: res = res[0]
            res = np.abs(res)
            # print(res.shape)
        return res

    def sim2dist(similarityVector, M=None):
        if M is None:
            assert len(similarityVector) > 1
            M = np.max(similarityVector)
        dist = M - similarityVector
        if logColor: dist += 0.1*M
        return dist
    assert MIalgorithm in ['MI', 'corr']
    assert featureFeatureMItype in ['regression', 'classification']
    assert featureLabelMItype in ['auto', 'regression', 'classification']
    assert plotAxes in ['TSNE and PCA', 'PCA only']
    if features is None: features = data.columns
    if isinstance(features, list): features = np.array(features)
    assert len(set(labels) & set(features)) == 0
    assert set(labels) <= set(data.columns)
    assert set(features) <= set(data.columns)
    assert np.all(np.isfinite(data.loc[:,features]))
    if labels is None: labels = []
    if len(labels) > 0:
        assert np.all(np.isfinite(data.loc[:,labels]))
    if scatterParams is None: scatterParams = {}
    n = len(features)
    if np.all([f.startswith('e_') for f in features]):
        d = scale(data.loc[:,features])
        s = ML.Sample(params=data.loc[:, labels], spectra=d, energy=utils.getEnergy(data.loc[:,features]))
        if len(labels) == 0:
            s.plot(f'{folder}{os.sep}scaled spectra.png')
        else:
            for label in labels:
                s.plot(f'{folder}{os.sep}scaled spectra by {label}.png', colorParam=label)
    features_MI = np.zeros((n, n))
    for i in range(n):
        print(f'Calculate {i} from {n}')
        features_MI[i] = similarity(data.loc[:, features], data[features[i]], featureFeatureMItype)
        assert np.all(np.isfinite(features_MI[i]))
    figsize = (20, 10)
    plotting.plotMatrix(features_MI, ticklabelsX=features, ticklabelsY=features, fileName=f'{folder}{os.sep}MI matrix.png', title='Mutual information feature-feature', figsize=figsize)
    if neighbourCountForColor is None:
        neighbourCountForColor = max(1,int(n/20))
    r = np.partition(sim2dist(features_MI), neighbourCountForColor, axis=1)[:,neighbourCountForColor]
    # print(features_MI)
    # for i in range(len(r)): print(features[i], r[i])
    if logColor: r = np.log(r)
    # rows - descriptors
    descrAsObjects = scale(data.loc[:,features].to_numpy()).T
    pca_axes = pcaDescriptor(descrAsObjects, count=2, returnU=False)
    if plotAxes == 'PCA only':
        axes = pca_axes
    else:
        M = np.max(features_MI)
        def metric(f1,f2):
            mi = similarity(f1, f2, featureFeatureMItype)
            return sim2dist(mi,M)
        if perplexity is None:
            perplexity = min(30, len(descrAsObjects)//4)
            print(f'Set perplexity = {perplexity}')
        tsne = sklearn.manifold.TSNE(perplexity=perplexity, random_state=0, metric=metric)
        axes = tsne.fit_transform(descrAsObjects)
    def plotScatter(color, colorName):
        def plotScatterHelper(axes, plotAxesName):
            plotting.scatter(axes[:,0], axes[:,1], color=color, marker_text=features, fileName=f'{folder}{os.sep}{plotAxesName} {colorName}.png', title=f'Descriptors on {plotAxesName} plot colored by dist(MI) to {colorName}', **scatterParams)
        plotScatterHelper(axes, 'PCA' if plotAxes == 'PCA only' else 'TSNE')
        if plotAxes == 'TSNE and PCA':
            plotScatterHelper(pca_axes, 'PCA')

    plotScatter(color=r, colorName='neighbour')
    for label in labels:
        if len(np.unique(data[label])) == 1:
            print(f'Label {label} is constant = {data[label][0]}')
            continue
        if featureLabelMItype == 'auto':
            typ = 'classification' if ML.isClassification(data, label) else 'regression'
            if debug: print(f'For label {label} use {typ} similarity')
        else: typ = featureLabelMItype
        labMI = similarity(data.loc[:, features], data[label], typ)
        assert np.all(np.isfinite(labMI))
        plotScatter(color=sim2dist(labMI), colorName=label)


def separableXDescriptor(sample: ML.Sample, features, expIndexes, normalize=True):
    """Find two descriptors, that separate experiments most (first) and label values (second). Calculate these descriptors for all the objects inside sample

        :param sample:  sample with descriptors and labels
        :param features: (list of strings or string) features to use (x), or 'spectra' or 'spectra_d_i1,i2,i3,...' (spectra derivatives together), or ['spType1 spectra_d_i1,i2', 'spType2 spectra_d_i1,i2', ...]
        :param expIndexes: array or list of sample object indexes to separate
        :param normalize: normalize X by StandardScaler
        :returns: two column array with exp and label separable descriptors
    """
    assert len(expIndexes) >= 2
    for i in expIndexes: assert 0<=i<len(sample)
    X,_ = getXYFromSample(sample, features, label_names=None, textColumn=None)
    if normalize: X = sklearn.preprocessing.StandardScaler().fit_transform(X)
    d = scipy.spatial.distance.cdist(X[expIndexes], X[expIndexes])
    i,j = np.where(d == np.max(d))
    expDiff = X[expIndexes[i[0]]] - X[expIndexes[j[0]]]
    return np.dot(X, expDiff)


def separableYDescriptor(sample: ML.Sample, features, label, normalize=True, pairwiseTransformType='binary', debugFolder=None):
    """Find two descriptors, that separate experiments most (first) and label values (second). Calculate these descriptors for all the objects inside sample

        :param sample:  sample with descriptors and labels
        :param features: (list of strings or string) features to use (x), or 'spectra' or 'spectra_d_i1,i2,i3,...' (spectra derivatives together), or ['spType1 spectra_d_i1,i2', 'spType2 spectra_d_i1,i2', ...]
        :param label: label name
        :param normalize: normalize X by StandardScaler
        :param pairwiseTransformType: 'binary' or 'numerical' to find descriptor for label separation
        :returns: two column array with exp and label separable descriptors
    """
    assert isinstance(label, str)
    assert label in sample.params.columns
    X,y = getXYFromSample(sample, features, [label], None)
    if normalize: X = sklearn.preprocessing.StandardScaler().fit_transform(X)
    _,_,known_ind,_ = sample.splitUnknown(columnNames=label, returnInd=True)
    Xp, yp = ML.pairwiseTransform(X[known_ind,:],y[known_ind], maxSize=10000, randomSeed=0, pairwiseTransformType=pairwiseTransformType)
    if pairwiseTransformType == 'binary':
        model = sklearn.linear_model.LogisticRegression(fit_intercept=False, random_state=0)
    elif pairwiseTransformType == 'numerical':
        model = ML.pairwiseRidgeCV(X[known_ind,:],y[known_ind], alphas=(0.1,1,10,100,1000,100*1000), testRatio=0.1)
    else: raise Exception(f'Unknown pairwiseTransformType = {pairwiseTransformType}')
    model.fit(Xp,yp)
    coef = model.coef_.flatten()
    if debugFolder is not None and isinstance(features,str) and features.endswith(' spectra'):
        spType = features[:features.rfind(' ')]
        e = sample.getEnergy(spType)
        def plotMoreFunction(ax):
            ax2 = ax.twinx()
            ax2.plot(e, coef, c='k')
        sample.plot(debugFolder+os.sep+label, colorParam=label, plotSampleParams=dict(plotMoreFunction=plotMoreFunction, title='Projection base'))
        shutil.copyfile(debugFolder+os.sep+label+os.sep+f'plot_{spType}.png', debugFolder+os.sep+label+'.png')
        shutil.rmtree(debugFolder+os.sep+label)

    # oblique projection on two vectors: expDiff, coef
    # res = np.zeros((len(y),2))
    # for i in range(len(y)):
    #     res[i] = curveFitting.linearReg2(y_true=X[i,:], f1=expDiff, f2=coef)
    # orthogonal projection
    return np.dot(X, coef)


def findBestLabelMap(sample:ML.Sample, features, label):
    sample,_ = sample.splitUnknown(label)
    if label in sample.labelMaps: sample.decode(label)
    labelValues = sample.params[label]
    uniq_vals = np.unique(labelValues)
    n = len(uniq_vals)
    permutations = itertools.permutations(np.arange(n))
    X,_ = getXYFromSample(sample, features, [label], None)
    model = sklearn.linear_model.LogisticRegression(fit_intercept=False, random_state=0)
    qualities = []
    for perm in permutations:
        # direct and inverse permutations result in the symmetrical labelMaps, so we need only half of all permutations
        if perm[0] > perm[-1]: continue
        labelMap = {v:i for v,i in zip(uniq_vals,perm)}
        y1 = ML.encode(labelValues, labelMap)
        acc = ML.pairwiseCV(model, X, y1, pairwiseTransformType='binary', lossFunc='AUC')
        qualities.append([perm, acc])
    i = np.argmax([q[1] for q in qualities])
    print(qualities)
    best_perm = qualities[i][0]
    labelMap = {v:i for v,i in zip(uniq_vals,best_perm)}
    return labelMap

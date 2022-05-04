from . import utils, plotting, optimize, ML, descriptor
import numpy as np
import pandas as pd
import scipy, os, shutil, itertools, copy, warnings, sklearn, logging, sys
import matplotlib.pyplot as plt


def findConcentrations(energy, xanesArray, expXanes, fixConcentrations=None):
    """
    Concentration search
    :param energy: 1-d array with energy values (same for all xanes)
    :param xanesArray: 2-d array. One row - one xanes
    :param expXanes: experimental xanes for the same energy
    :param fixConcentrations: {index:value,...}
    :return: array of concentrations
    """
    def distToUnknown(mixture):
        return utils.rFactor(energy, mixture, expXanes)

    def makeMixture(c):
        return np.sum(xanesArray * c.reshape(-1, 1), axis=0)

    return findConcentrationsAbstract(distToUnknown, len(xanesArray), makeMixture, fixConcentrations=fixConcentrations)


def findConcentrationsAbstract(distToUnknown, componentCount, makeMixture, startConcentrations=None, fixConcentrations=None, trysGenerateMixtureOfSampleCount=1):
    """
    Concentration search
    :param distToUnknown: function(mixture) to return distance to unknown experiment
    :param componentCount: component count
    :param makeMixture: function(concentrations) to calculate mixture spectrum/descriptors
    :param startConcentrations: start point for optimization
    :param fixConcentrations: {index:value,...}
    :param trysGenerateMixtureOfSampleCount: 'all pure' or number. if 'all pure', try starting from all pure components and one - with all equal concentrations. If number - try starting from random concentrations. Choose best
    :return: array of optimized concentrations
    """
    n = componentCount
    assert n >= 2
    fn = 0 if fixConcentrations is None else len(fixConcentrations)
    fixInd = np.sort(np.array(list(fixConcentrations.keys()))) if fn>0 else np.array([])
    fixVal = np.array([fixConcentrations[i] for i in fixInd])
    assert np.all(fixInd < n)
    assert np.all(fixInd >= 0)
    assert len(np.unique(fixInd)) == len(fixInd), str(len(np.unique(fixInd))) + ' != ' + str(len(fixInd))
    assert np.sum(fixVal) <= 1

    notFixInd = np.sort(np.setdiff1d(np.arange(n), fixInd))

    if startConcentrations is not None:
        assert len(startConcentrations) == componentCount
        if fn > 0: assert np.all(startConcentrations[fixInd] == fixVal)
        startConcentrations = startConcentrations[notFixInd][:-1]

    def expand(partial_c):
        full_c = np.zeros(n)
        full_c[notFixInd[:-1]] = partial_c
        if fn > 0:
            full_c[fixInd] = fixVal
        full_c[notFixInd[-1]] = 1-np.sum(full_c)
        return full_c

    def func0(c):
        mixture = makeMixture(c)
        return distToUnknown(mixture)

    def func(partial_c):
        c = expand(partial_c)
        return func0(c)

    if fn == n: return func0(fixVal), fixVal
    if fn == n-1:
        res = np.zeros(n)
        res[notFixInd] = 1-np.sum(fixVal)
        res[fixInd] = fixVal
        return func0(res), res

    m = len(notFixInd)
    upperBound = 1-np.sum(fixVal) if n != m else 1
    constrains = ()
    if m >= 3:
        a = np.ones(m-1)
        constrains = (scipy.optimize.LinearConstraint(a, 0, upperBound, keep_feasible=True),)
    if trysGenerateMixtureOfSampleCount == 'all pure':
        assert startConcentrations is None
        results = []
        for i in range(m-1):
            startConcentrations = np.zeros(m-1)
            startConcentrations[i] = upperBound
            result = scipy.optimize.minimize(func, startConcentrations, bounds=[[0, upperBound]] * (m - 1), constraints=constrains)
            results.append(result)
        results.append(scipy.optimize.minimize(func, np.zeros(m - 1), bounds=[[0, upperBound]] * (m - 1), constraints=constrains))
        results.append(scipy.optimize.minimize(func, np.ones(m-1)*upperBound/m, bounds=[[0, upperBound]] * (m - 1), constraints=constrains))
        best_i = np.argmin([r.fun for r in results])
        result = results[best_i]
    else:
        assert isinstance(trysGenerateMixtureOfSampleCount, int)
        c = np.random.dirichlet(alpha=1 * np.ones(m), size=trysGenerateMixtureOfSampleCount)
        for try_i in range(trysGenerateMixtureOfSampleCount):
            if startConcentrations is None:
                if trysGenerateMixtureOfSampleCount == 1:
                    sConcentrations = np.ones(m-1)*upperBound/m
                else:
                    sConcentrations = (c[try_i]/upperBound)[:-1]
            else:
                assert np.sum(sConcentrations) <= upperBound
                assert trysGenerateMixtureOfSampleCount == 1, 'When start concentration is not None - all trys give the same result'
            result = scipy.optimize.minimize(func, sConcentrations, bounds=[[0,upperBound]]*(m-1), constraints=constrains)
    # if not result.success:
    #     warnings.warn("scipy.optimize.minimize can't find optimum. Result = "+str(result))
    c = result.x
    c = expand(c)
    return result.fun, c


def plotMixtureLabelMap(componentLabels, label_names, label_bounds, labelMaps, distsToExp, concentrations, componentNames, folder, fileNamePostfix=''):
    """For every label plot 2d map (label x label) with color - minimal distance between unknown experiment and mixture of knowns

    :param componentLabels: array of label tables (for each component - one table) with size componentCount x spectraCount x labelCount
    :param label_names: list of names of labels
    :param label_bounds: list of pairs [label_min, label_max]
    :param labelMaps: strings for values of categorical labels. dict labelName:{valueName:value,...}
    :param distsToExp: array of distances to experiment for each row of componentLabels tables (spectraCount values)
    :param concentrations: 2d array spectraCount x componentCount
    :param componentNames: 2d array spectraCount x componentCount
    :param folder: folder to save plots
    :param fileNamePostfix: to save plots
    """
    if labelMaps is None: labelMaps = {}
    os.makedirs(folder, exist_ok=True)
    spectraCount = concentrations.shape[0]
    component_count = concentrations.shape[1]
    assert np.all(np.abs(np.sum(concentrations, axis=1) - 1) < 1e-6), "Bad concentrations array: "+str(concentrations)
    assert len(label_bounds) == len(label_names)
    assert len(componentLabels.shape) == 3
    assert componentLabels.shape[-1] == len(label_names), f'{componentLabels.shape[-1]} != {len(label_names)}'
    assert componentLabels.shape[1] == spectraCount
    assert componentLabels.shape[0] == component_count
    assert len(distsToExp) == spectraCount
    # sort concentrations in descending order
    concentrations = np.copy(concentrations)
    componentNames = np.copy(componentNames)
    componentLabels = np.copy(componentLabels)
    for i in range(len(distsToExp)):
        ind = np.argsort(-concentrations[i, :])
        concentrations[i, :] = concentrations[i, ind]
        componentNames[i, :] = componentNames[i, ind]
        for k in range(len(label_names)):
            componentLabels[:, i, k] = componentLabels[ind, i, k]

    invLabelMaps = {}
    for label in labelMaps:
        invLabelMaps[label] = {v: k for k, v in labelMaps[label].items()}

    for label, i_lab in zip(label_names, range(len(label_names))):
        all_label_values = []
        for i in range(component_count):
            all_label_values += list(componentLabels[i,:,i_lab])
        isClassification = ML.isClassification(all_label_values)
        m, M = label_bounds[i_lab]
        d = (M - m) / 10
        linspaces = []
        for i in range(component_count):
            if isClassification:
                linspaces.append(np.unique(all_label_values))
            else:
                linspaces.append(np.linspace(m - d, M + d, 50))
        grid = list(np.meshgrid(*linspaces))
        grid_shape = grid[0].shape
        for i in range(len(grid)): grid[i] = grid[i].reshape(-1)
        label_grid = np.array(grid).T  # column <-> component
        sl = np.sort(all_label_values)
        rad = np.mean(sl[1:] - sl[:-1]) * 2
        if isClassification:
            notExist = np.max(distsToExp)+1
            plot_norm = np.zeros(label_grid.shape[0]) + notExist
        else:
            plot_norm = np.zeros(label_grid.shape[0]) + distsToExp[-1]*(sl[-1]-sl[0])/rad
        result_conc = np.zeros([plot_norm.shape[0], component_count])
        result_components = np.zeros([plot_norm.shape[0], component_count], dtype=object)
        for i_spectr in range(len(distsToExp)):
            lab = np.array([componentLabels[i_component, i_spectr, i_lab] for i_component in range(component_count)])
            if isClassification:
                ind0 = np.where(np.linalg.norm(lab - label_grid, axis=1) == 0)[0][0]
                ind = np.zeros(plot_norm.shape, int)
                if plot_norm[ind0] > distsToExp[i_spectr]:
                    ind[ind0] = 1
                    plot_norm[ind0] = distsToExp[i_spectr]
            else:
                ind = np.argmin(np.array([plot_norm, distsToExp[i_spectr]*(1 + np.linalg.norm(lab - label_grid, axis=1) / rad)]), axis=0)
                plot_norm = np.minimum(plot_norm, distsToExp[i_spectr]*(1 + np.linalg.norm(lab - label_grid, axis=1) / rad))
            for k in range(component_count):
                result_conc[ind == 1, k] = concentrations[i_spectr, k]
                result_components[ind == 1, k] = componentNames[i_spectr, k]
        label_grid_2d = []
        # result_conc_2d = []
        # result_components_2d = []
        for k in range(component_count):
            label_grid_2d.append(label_grid[...,k].reshape(grid_shape))
            # result_conc_2d.append(result_conc[..., k].reshape(grid_shape))
            # result_components_2d.append(result_components[..., k].reshape(grid_shape))
        plot_norm_2d = plot_norm.reshape(grid_shape)
        if component_count != 2:
            # take min by all other dimensions
            for i in range(component_count - 2):
                ind = np.argmin(plot_norm_2d, axis=-1)
                plot_norm_2d = np.take_along_axis(plot_norm_2d, np.expand_dims(ind, axis=-1), axis=-1)
                for k in range(component_count):
                    label_grid_2d[k] = label_grid_2d[k][..., 0]
                    # result_conc_2d[k] = np.take_along_axis(result_conc_2d[k], np.expand_dims(ind, axis=-1), axis=-1)
                    # result_components_2d[k] = np.take_along_axis(result_components_2d[k], np.expand_dims(ind, axis=-1), axis=-1)
            plot_norm_2d = plot_norm_2d[:,:,0]
        if fileNamePostfix == '': fileNamePostfix1 = '_' + label
        else: fileNamePostfix1 = fileNamePostfix
        if isClassification:
            ticklabels = np.unique(all_label_values)
            if label in labelMaps:
                invMap = {labelMaps[label][s]:s for s in labelMaps[label]}
                ticklabels = [invMap[v] for v in ticklabels]
            plotting.plotMatrix(plot_norm_2d, cmap='summer', ticklabels=ticklabels, title=f'notExist value = {notExist}', xlabel=label, ylabel=label, fileName=folder + os.sep + 'map' + fileNamePostfix1 + '.png')
        else:
            fig, ax = plotting.createfig(figsize=(6.2,4.8))
            minnorm = np.min(plot_norm_2d)
            maxnorm = np.max(plot_norm_2d)
            maxnorm = minnorm + 1 * (maxnorm - minnorm)  # 0.3*(maxnorm - minnorm)
            CF = ax.contourf(label_grid_2d[0], label_grid_2d[1], plot_norm_2d, cmap='plasma', extend='both', vmin=minnorm, vmax=maxnorm)
            cbar = fig.colorbar(CF, ax=ax, extend='max', orientation='vertical')
            ax.set_xlabel(label)
            ax.set_ylabel(label)
            plotting.savefig(folder + os.sep + 'map' + fileNamePostfix1 + '.png', fig, figdpi=300)
            plotting.closefig(fig)
        # save to file
        cont_data = pd.DataFrame()
        cont_data['1_' + label] = label_grid[:, 0]
        cont_data['2_' + label] = label_grid[:, 1]
        cont_data['norm'] = plot_norm
        for k in range(component_count):
            cont_data[f'concentration_{k}'] = result_conc[:,k]
            cont_data[f'component_{k}'] = result_components[:, k]
        cont_data.to_csv(folder + os.sep + 'map' + fileNamePostfix1 + '.csv', index=False)

        def makeMixtureString(labels, componentNames, concentrations):
            mixture = ''
            for k in range(component_count):
                lab = labels[k]
                if label in labelMaps:
                    lab = invLabelMaps[label][lab]
                else:
                    if lab == int(lab): lab = int(lab)
                    else: lab = f'{lab:.2}'
                mixture += f'{componentNames[k]}({lab})*{concentrations[k]:.2}'
                if k != component_count - 1: mixture += ' + '
            return mixture

        best_mixtures = {}
        ind = np.argsort(plot_norm)
        i = 0
        while i < len(plot_norm):
            ii = ind[i]
            mixture = makeMixtureString(label_grid[ii], result_components[ii], result_conc[ii])
            if mixture not in best_mixtures: best_mixtures[mixture] = plot_norm[ii]
            if len(best_mixtures) > 5: break
            i += 1
        best_mixtures = [{'norm': best_mixtures[m], 'mixture': m} for m in best_mixtures]
        best_mixtures = sorted(best_mixtures, key=lambda p: p['norm'])

        best_mixtures2 = []
        # add more results
        ind = np.argsort(distsToExp)
        ind = ind[:100]
        for i in ind:
            mixture = makeMixtureString(componentLabels[:,i,i_lab], componentNames[i], concentrations[i])
            best_mixtures2.append({'norm':distsToExp[i], 'mixture':mixture})

        with open(folder + os.sep + 'best_mix' + fileNamePostfix1 + '.txt', 'w') as f:
            f.write('norm: mixture = n1(label1)*c1')
            for j in range(2,component_count+1): f.write(f' + n{j}(label{j})*c{j}')
            f.write('\n')
            for p in best_mixtures:
                f.write(str(p['norm']) + ': ' + p['mixture'] + '\n')
            f.write('\n')
            for p in best_mixtures2:
                f.write(str(p['norm']) + ': ' + p['mixture'] + '\n')


def tryAllMixtures_old(unknownExp, componentCandidates, componentCount, energyRange, outputFolder, rFactorToleranceMultiplier=1.1, maxGraphNumToPlot=50, plotMixtureLabelMapParam=None):
    """
    Try to fit unknown spectrum by all possible linear combinations of componentCount spectra from componentCandidates. Plots sorted best results in outputFolder.

    :param unknownExp: Spectrum
    :param componentCandidates: {name:Spectrum}
    :param componentCount: >=1 number of components to compose the mixture
    :param energyRange: two-element tuple
    :param outputFolder: string
    :param rFactorToleranceMultiplier: float
    :param maxGraphNumToPlot: maximum number of plots to draw
    :param plotMixtureLabelMapParam: dict with keys: componentCandidatesLabels (dict name: labels), label_names, labelMaps, rFactorMultiplierToPlot (<1 - means plot all)
    :return:
    """
    def limitEnergyInterval(spectrum):
        ind = (energyRange[0] <= spectrum.energy) & (spectrum.energy <= energyRange[1])
        return spectrum.energy[ind], spectrum.intensity[ind]
    energy, xanesUnknown = limitEnergyInterval(unknownExp)
    componentCandidates = copy.deepcopy(componentCandidates)
    for compName in componentCandidates:
        sp = componentCandidates[compName]
        componentCandidates[compName] = np.interp(energy, sp.energy, sp.intensity)

    def fitBy1Plot(norms):
        # save results
        with open(outputFolder+os.sep+'mixture_statistics.txt', 'w') as sf:
            sf.write('R-Factor Name\n')
            for entry in norms:
                sf.write(str(entry["best-dist"])+' "'+entry["names"]+'"\n')

        i = 1
        for entry in norms:
            fname = os.path.join(outputFolder, f'{utils.zfill(i, len(componentCandidates))} {entry["names"]}')
            plotting.plotToFile(
                energy, xanesUnknown, 'target',
                energy, entry['spectrum'], entry["names"],
                title=f'Component: {entry["names"]}, R-Factor: {entry["best-dist"]:f}', fileName=fname)
            np.savetxt(fname+'.txt', [energy, xanesUnknown, entry['spectrum']], delimiter=',')
            i += 1

    def combineExps(spectra, coeffs):
        assert len(spectra) == len(coeffs)
        if len(coeffs) == 1:
            return spectra[0].reshape(-1)
        res = spectra[0]*coeffs[0]
        for i in range(1, len(coeffs)):
            res += spectra[i]*coeffs[i]
        return res

    def rfactorMetric(mixtureSpectrum):
        return np.sqrt(utils.integral(energy, (xanesUnknown - mixtureSpectrum) ** 2))

    def plot(spectrumUnknown, componentCandidates, componentNames, coeffs, title, fname):
        xanesCollection = []
        energy, intensity = limitEnergyInterval(spectrumUnknown)
        for name in componentNames:
            # (energy, xanes) = limitEnergyInterval(componentCandidates[name])
            xanesCollection.append(componentCandidates[name])

        xanesCombined = combineExps(xanesCollection, coeffs)
        plotting.plotToFile(
            energy, intensity, 'target',
            energy, xanesCombined, 'mixture', title=title, fileName=fname)

        np.savetxt(fname+'.txt', [energy, intensity, xanesCombined], delimiter=',')

    def writeResults(expUnknown, componentCandidates, data):
        os.makedirs(outputFolder, exist_ok=True)
        index = 1
        data = sorted(data, key = lambda n: n['best-dist'])
        with open(outputFolder+os.sep+'mixture_statistics.txt', 'w') as sf:
            sf.write('R-Factor ')
            for i in range(1,componentCount+1):
                sf.write('Name_'+str(i)+' ')
                sf.write('C_'+str(i))
                if i != componentCount: sf.write(' ')
            sf.write('\n')
            for entry in data:
                sf.write(str(entry['best-dist'])+' ')
                for i in range(componentCount):
                    sf.write('"'+entry['names'][i]+'" ')
                    sf.write(str(entry['best-concentrations'][i]))
                    if i != componentCount-1: sf.write(' ')
                sf.write('\n')
        for entry in data:
            title = 'Concentrations: {}, R-Factor: {:f}'.format(', '.join(['C_{}={:.2f}[{:.2f}, {:.2f}]'.format(name, bestConc, l, r)
                for ((l, r), name, bestConc)
                in zip(entry['intervals'], entry['names'], entry['best-concentrations'])]), entry['best-dist'])
            fname = outputFolder+os.sep+utils.zfill(index,len(data))+' '+' '.join('C_' + x for x in entry['names'])+'.png'
            plot(expUnknown, componentCandidates, entry['names'], entry['best-concentrations'], title, fname)
            index += 1
            if index > maxGraphNumToPlot: break

    # root function body
    data = tryAllMixturesAbstract(rfactorMetric, combineExps, componentCandidates, componentCount, outputFolder, distToleranceMultiplier=rFactorToleranceMultiplier)
    if componentCount == 1:
        fitBy1Plot(data)
    else:
        writeResults(unknownExp, componentCandidates, data)

    if plotMixtureLabelMapParam is not None and componentCount > 1:
        label_names = plotMixtureLabelMapParam['label_names']
        labelMaps = plotMixtureLabelMapParam['labelMaps']
        componentCandidatesLabelsDict = plotMixtureLabelMapParam['componentCandidatesLabels']
        componentCandidatesLabels = np.zeros([len(componentCandidates), len(label_names)])
        for i, name in zip(range(len(componentCandidates)), componentCandidates):
            componentCandidatesLabels[i,:] = componentCandidatesLabelsDict[name]
        rFactorMultiplierToPlot = plotMixtureLabelMapParam['rFactorMultiplierToPlot']
        bestRF = data[0]['best-dist']
        if rFactorMultiplierToPlot >= 1:
            data = [data[i] for i in range(len(data)) if data[i]['best-dist'] <= bestRF*rFactorMultiplierToPlot]
        n = len(data)
        min_l = np.min(componentCandidatesLabels, axis=0); max_l = np.max(componentCandidatesLabels, axis=0)
        label_bounds = [[min_l[i], max_l[i]] for i in range(len(label_names))]
        componentLabels = np.zeros([componentCount, n, len(label_names)])
        distsToExp = np.zeros(n)
        concentrations = np.zeros([n, componentCount])
        componentNames = np.zeros([n, componentCount], dtype=object)
        for i in range(n):
            for j in range(componentCount):
                componentLabels[j,i,:] = componentCandidatesLabelsDict[data[i]['names'][j]]
            distsToExp[i] = data[i]['best-dist']
            concentrations[i,:] = data[i]['best-concentrations']
            componentNames[i, :] = data[i]['names']
        plotMixtureLabelMap(componentLabels, label_names, label_bounds, labelMaps, distsToExp, concentrations, componentNames, outputFolder)


def tryAllMixturesAbstract(distToExperiment, makeMixture, componentCandidates, componentCount, outputFolder, distToleranceMultiplier=1.1):
    """
    Try to fit unknown spectrum by all possible linear combinations of componentCount spectra from componentCandidates. Plots sorted best results in outputFolder.

    :param distToExperiment: function which calculates distance to experimental spectrum distToExperiment(mixtureSpectrum)
    :param makeMixture: function, that creates mixture of spectra makeMixture(spectraList, concentraitions)
    :param componentCandidates: {name:Spectrum}
    :param componentCount: >=1 number of components to compose the mixture
    :param outputFolder: string
    :param distToleranceMultiplier: tolerance is determined from equation: distToExperiment = minDist*distToleranceMultiplier
    :returns: {'intervals': concentrationIntervas, 'best-concentrations': bestConcentrations, 'best-dist': minMetricValue, 'spectrum':mixtureSpectrum}
    """

    def fitBy1():
        os.makedirs(outputFolder, exist_ok=True)
        norms = []
        for name in componentCandidates:
            comp = componentCandidates[name]
            spectrum = makeMixture((comp,), [1])
            norms.append({
                'names': name,
                'spectrum': spectrum,
                'best-dist': distToExperiment(spectrum)
            })

        norms = sorted(norms, key=lambda x: x['names'])
        return norms

    def findIntervalsForConcentrations(components):
        def metric(concentrations):
            last = 1-np.sum(concentrations)
            mixtureSp = makeMixture(components, np.append(concentrations,last))
            return distToExperiment(mixtureSp)
        bnds = [(0, 1)] * (len(components) - 1)
        cons = ({'type': 'ineq', 'fun': lambda x: 1 - sum(x)})
        initial = [0.5] * (len(components) - 1)
        res = scipy.optimize.minimize(metric, initial, method='SLSQP', bounds=bnds, constraints=cons)
        if res.success:
            initialConcentrations = res.x
            minMetricValue = metric(res.x)
        else:
            raise Exception("Couldn't find initial concentrations")

        # use initial guess to find min and max for each concentration, for which r-factor is in certain bounds

        concentrationIntervas = []
        rfCons = ({'type': 'ineq', 'fun': lambda x: minMetricValue * distToleranceMultiplier - metric(x)},
                  {'type': 'ineq', 'fun': lambda x: 1 - sum(x)})
        for i in range(componentCount):
            def concentrationMetric(x):
                return np.append(x, 1 - sum(x))[i]

            def minMetric(x):
                return concentrationMetric(x)

            def maxMetric(x):
                return -concentrationMetric(x)

            minRes = scipy.optimize.minimize(minMetric, initialConcentrations, method='SLSQP', bounds=bnds, constraints=rfCons)
            maxRes = scipy.optimize.minimize(maxMetric, initialConcentrations, method='SLSQP', bounds=bnds, constraints=rfCons)

            concentrationIntervas.append((minMetric(minRes.x), -maxMetric(maxRes.x)))
        bestConcentrations = np.append(initialConcentrations, 1 - sum(initialConcentrations))
        mixtureSpectrum = makeMixture(components, bestConcentrations)
        return {'intervals': concentrationIntervas,
                'best-concentrations': bestConcentrations,
                'best-dist': minMetricValue,
                'spectrum':mixtureSpectrum}

    def getBestCoefficientsWithIntervals(componentCandidates):
        all_names = list(componentCandidates.keys())
        all_names.sort()
        components = [componentCandidates[name] for name in all_names]
        intervals = []
        componentCombinations = list(itertools.combinations(components, componentCount))
        names = list(itertools.combinations(all_names, componentCount))
        for i in range(len(componentCombinations)):
            intervalsRes = findIntervalsForConcentrations(componentCombinations[i])
            intervals.append({'names':names[i],
                              'intervals': intervalsRes['intervals'],
                              'best-concentrations': intervalsRes['best-concentrations'],
                              'best-dist': intervalsRes['best-dist'],
                              'spectrum': intervalsRes['spectrum']})
        return intervals

    # root function body
    if os.path.exists(outputFolder): shutil.rmtree(outputFolder)
    if componentCount == 1:
        data = fitBy1()
    else:
        data = getBestCoefficientsWithIntervals(componentCandidates)
    return data


def tryAllMixtures(unknownCharacterization, componentCount, mixtureTrysCount, optimizeConcentrations, optimizeConcentrationsTrysCount, singleComponentData, label_names, folder, fileNamePostfix='', spectraFolderPostfix='', labelMapsFolderPostfix='', label_bounds=None, labelMaps=None, componentNameColumn=None, calcDescriptorsForSingleSpectrumFunc=None, randomSeed=0, makeMixtureOfSpectra=None, plotSpectrumType=None, maxSpectraCountToPlot=50, unknownSpectrumToPlot=None):
    """
    Try to fit unknown spectrum/descriptors by all possible linear combinations of componentCount spectra from componentCandidates using different label values. Plots mixture label map, best mixture spectra

    :param unknownCharacterization: dict('type', ...)
                1) 'type':'distance function',  'function': function(mixtureSpectrum, mixtureDescriptors) to return distance to unknown experiment. mixtureDescriptors - result of calcDescriptorsForSingleSpectrumFunc
                2) 'type':'spectrum', 'spType': type of spectrum from singleComponentData sample (default - default), 'spectrum':unknownSpectrum (intensity only, len == len(energy[spType] in singleComponentData)
                3) 'type':'descriptors', 'paramNames':list of descriptor names to use for distance, 'paramValues':array of unknown spectrum descriptor values
    :param componentCount: >=1 number of components to compose the mixture
    :param mixtureTrysCount: string 'all combinations of singles', 'all combinations for each label' or number or dict {label:number}. In case of a number mixtures are chosen with uniform label distribution in a loop for each label
    :param optimizeConcentrations: bool. False - choose random concentrations, True - find optimal by minimizing distance to unknown
    :param optimizeConcentrationsTrysCount: 'all pure' or number. if 'all pure', try starting from all pure components and one - with all equal concentrations. If number - try starting from random concentrations. Choose best
    :param singleComponentData: Sample of different single component spectra and descriptors and labels
    :param label_names: list of names of labels
    :param folder: folder to save plots
    :param fileNamePostfix: to save plots
    :param labelMapsFolderPostfix:
    :param spectraFolderPostfix:
    :param label_bounds: list of pairs [label_min, label_max]. If None, derive from  singleComponentData
    :param labelMaps: strings for values of categorical labels. dict labelName:{valueName:value,...}
    :param componentNameColumn: if None use index
    :param calcDescriptorsForSingleSpectrumFunc: function(spectrum) used when unknownCharacterization type is 'descriptors'. Should return array of the same length as 'paramValues' or dict of descriptors (should contain subset paramValues). spectrum - one spectrum or dict of spectra for different spTypes (depends on singleComponentData)
    :param randomSeed: random seed
    :param makeMixtureOfSpectra: function(inds, concentrations) to calculate mixture of spectra by given concentrations and spectra indices (returns intensity only or dict {spType: intensity only} for multiple spectra types)
    :param plotSpectrumType: 'all' or name of spType inside sample to plot (when unknownCharacterization['type']=='spectrum' default to plot unknownCharacterization['spType'] only)
    :param maxSpectraCountToPlot: plot not more than maxSpectraCountToPlot spectra
    :param unknownSpectrumToPlot: Spectrum or array of the same length as singleComponentData or dict spType: Spectrum or array
    """
    assert 'default' not in label_names
    n = singleComponentData.getLength()
    if componentNameColumn is not None: assert len(np.unique(singleComponentData.params[componentNameColumn])) == n, 'Duplicate names'
    np.random.seed(randomSeed)
    # =================== setup runType and mixtureTrysCount ===================
    runType = 'each label'
    if mixtureTrysCount == 'all combinations of singles':
        runType = 'one'
        mixtureTrysCount = {'default': int(scipy.special.comb(n, componentCount, repetition=True))}
    elif mixtureTrysCount == 'all combinations for each label':
        mixtureTrysCount = {}
        for label in label_names:
            if ML.isClassification(singleComponentData.params, label):
                all_label_values = np.unique(singleComponentData.params[label])
                mixtureTrysCount[label] = int(scipy.special.comb(len(all_label_values), componentCount, repetition=True))
            else:
                mixtureTrysCount[label] = 100 if optimizeConcentrations else 1000
    elif isinstance(mixtureTrysCount, str):
        assert False, f'Unknown mixtureTrysCount string: {mixtureTrysCount}. It should be "all combinations of singles" or "all combinations for each label"'
    elif isinstance(mixtureTrysCount, int):
        mixtureTrysCount = {label:mixtureTrysCount for label in label_names}
    else: assert isinstance(mixtureTrysCount, dict) and len(mixtureTrysCount) == len(label_names)

    # =================== generate mixtures ===================
    label_names_surrogate = ['default'] if runType == 'one' else label_names
    if componentCount == 1:
        if runType == 'one':
            assert mixtureTrysCount['default'] <= n
            all_ind = {'default': np.random.choice(n, mixtureTrysCount['default'], replace=False).reshape(-1,1)}
        else:
            all_ind = {label: generateUniformLabelDistrib(mixtureTrysCount[label], componentCount, label, singleComponentData) for label in label_names_surrogate}
        mixtureData = {label: singleComponentData.takeRows(all_ind[label].reshape(-1)) for label in label_names_surrogate}
        if componentNameColumn is not None:
            componentNames = singleComponentData.params[componentNameColumn].to_numpy()
        else:
            componentNames = [str(i) for i in range(singleComponentData.getLength())]
    else:
        mixtureData = {}; componentLabels = {}; concentrations = {}; componentNames = {}
        for label in label_names_surrogate:
            makeUniformLabelDistrib = None if runType == 'one' else label
            mixtureData[label], componentLabels[label], concentrations[label], componentNames[label] = generateMixtureOfSample(mixtureTrysCount[label], componentCount, singleComponentData, label_names, addDescrFunc=None, makeMixtureOfSpectra=makeMixtureOfSpectra, makeMixtureOfLabel=None, dirichletAlpha=1, randomSeed=np.random.randint(1000000), returnNotMixLabels=True, componentNameColumn=componentNameColumn, makeUniformLabelDistrib=makeUniformLabelDistrib)

    # =================== concentration optimization ===================
    distsToExp = {label: np.zeros((mixtureTrysCount[label],1)) for label in label_names_surrogate}
    if unknownCharacterization['type'] == 'descriptors':
        std = np.std(singleComponentData.params.loc[:, unknownCharacterization['paramNames']].to_numpy(), axis=0).reshape(1, -1)
        # print('std = ', std, unknownCharacterization['paramNames'])
        assert set(unknownCharacterization['paramNames']) <= set(singleComponentData.paramNames)

    def distToUnknown(mixture):
        if isinstance(mixture, tuple):
            mixtureSpectrum, mixtureDescriptors = mixture
        else:
            mixtureSpectrum = mixture
            assert unknownCharacterization['type'] == 'spectrum', str(mixture)
        if unknownCharacterization['type'] == 'distance function':
            return unknownCharacterization['function'](mixtureSpectrum, mixtureDescriptors)
        elif unknownCharacterization['type'] == 'spectrum':
            spType = unknownCharacterization['spType'] if 'spType' in unknownCharacterization else singleComponentData.getDefaultSpType()
            unk_spectrum = unknownCharacterization['spectrum']
            energy = singleComponentData.getEnergy(spType)
            assert len(unk_spectrum) == len(energy), f'{len(unk_spectrum)} != {len(energy)}'
            if 'rFactorParams' not in unknownCharacterization: unknownCharacterization['rFactorParams'] = {}
            return utils.rFactor(energy, mixtureSpectrum[spType], unk_spectrum, **unknownCharacterization['rFactorParams'])
        else:
            assert unknownCharacterization['type'] == 'descriptors'
            assert len(unknownCharacterization['paramValues']) == len(unknownCharacterization['paramNames'])
            unkDescriptors = unknownCharacterization['paramValues'] / std
            if isinstance(mixtureDescriptors, np.ndarray):
                assert len(mixtureDescriptors) == len(unkDescriptors)
            else:
                assert isinstance(mixtureDescriptors, dict), 'mixtureDescriptors = '+str(mixtureDescriptors)
                mixtureDescriptors = np.array([mixtureDescriptors[pname] for pname in unknownCharacterization['paramNames']])
            mixtureDescriptors = mixtureDescriptors.reshape(1, -1) / std
            # print('unkDescriptors = ', unkDescriptors)
            # print('mixtureDescriptors = ', mixtureDescriptors)
            # print('diff = ', np.abs(mixtureDescriptors - unkDescriptors), np.linalg.norm(mixtureDescriptors - unkDescriptors))
            return np.linalg.norm(mixtureDescriptors - unkDescriptors)

    if optimizeConcentrations:
        assert componentCount > 1
        spectra = {spType: singleComponentData.getSpectra(spType).to_numpy() for spType in singleComponentData._spectra}
        if componentNameColumn is not None:
            assert singleComponentData.params.dtypes[componentNameColumn] == object, str(singleComponentData.params.dtypes[componentNameColumn]) + ' != object (str)'
            componentNameData = singleComponentData.params[componentNameColumn].to_numpy()
        for label in label_names_surrogate:
            for i in range(mixtureTrysCount[label]):
                def makeMixture(concentrations):
                    if componentNameColumn is None:
                        componentInds = componentNames[label][i]
                    else:
                        componentInds = np.array([np.where(componentNames[label][i,j] == componentNameData)[0][0] for j in range(componentCount)])
                    mix_spectra = {}
                    if makeMixtureOfSpectra is None:
                        for spType in spectra:
                            mix_spectra[spType] = np.dot(spectra[spType][componentInds, :].T, concentrations)
                    else:
                        ms = makeMixtureOfSpectra(componentInds, concentrations)
                        if isinstance(ms, dict):
                            mix_spectra = ms
                            assert len(ms[singleComponentData.getDefaultSpType()]) == len(singleComponentData.getEnergy())
                        else:
                            assert len(spectra) == 1
                            for spType in spectra:
                                assert len(ms) == len(singleComponentData.getEnergy(spType))
                                mix_spectra[spType] = ms
                    if calcDescriptorsForSingleSpectrumFunc is None:
                        return mix_spectra
                    else:
                        # print('concentrations = ', concentrations)
                        mix_spectra1 = mix_spectra if len(mix_spectra) > 1 else mix_spectra[list(mix_spectra.keys())[0]]
                        mixtureDescriptors = calcDescriptorsForSingleSpectrumFunc(mix_spectra1)
                        return mix_spectra, mixtureDescriptors
                old_c = copy.deepcopy(concentrations[label][i])
                c = copy.deepcopy(concentrations[label][i]) if optimizeConcentrationsTrysCount == 1 else None
                old_func_value = distToUnknown(makeMixture(concentrations[label][i]))
                distsToExp[label][i], concentrations[label][i] = findConcentrationsAbstract(distToUnknown, componentCount, makeMixture, fixConcentrations=None, trysGenerateMixtureOfSampleCount=optimizeConcentrationsTrysCount, startConcentrations=c)
                print(f'Were fun = {old_func_value}, conc = {old_c}. Now fun = {distsToExp[label][i,0]}, conc = {concentrations[label][i]}')
                # fill new mixture spectrum
                mix_spectra = makeMixture(concentrations[label][i])
                if isinstance(mix_spectra, tuple): mix_spectra = mix_spectra[0]
                for spType in mixtureData[label].spTypes():
                    mixtureData[label].getSpectra(spType).loc[i] = mix_spectra[spType]
                # !!!!!!!!!! Attention !!!!!!!!  Mixture descriptors we can't update, because calcDescriptorsForSingleSpectrumFunc can return not all descriptors. Also they were not calculated, because addDescrFunc=None in generateMixtureOfSample
    else:
        for label in label_names_surrogate:
            for i in range(len(distsToExp[label])):
                mixtureSpectrum = {spType:mixtureData[label].getSpectra(spType).loc[i].to_numpy() for spType in mixtureData[label].spTypes()}
                mixtureDescriptors = calcDescriptorsForSingleSpectrumFunc(mixtureSpectrum)
                distsToExp[label][i] = distToUnknown((mixtureSpectrum, mixtureDescriptors))

    # =================== combine data for different labels ===================
    def combine(dataDict):
        return np.vstack(tuple(dataDict[label] for label in label_names_surrogate))
    distsToExp = combine(distsToExp).reshape(-1)
    if componentCount > 1:
        componentLabels = combine(componentLabels)
        concentrations = combine(concentrations)
        componentNames = combine(componentNames)
    for label in label_names_surrogate:
        if label == label_names_surrogate[0]:
            mixtureData1 = mixtureData[label].copy()
        else:
            mixtureData1.unionWith(mixtureData[label])
    mixtureData = mixtureData1
    sorted_order = np.argsort(distsToExp)

    # =================== result spectra plotting ===================
    if plotSpectrumType is None and unknownCharacterization['type'] == 'spectrum':
        plotSpectrumType = unknownCharacterization['spType'] if 'spType' in unknownCharacterization else singleComponentData.getDefaultSpType()
    if plotSpectrumType == 'all': plotSpectrumTypes = singleComponentData.spTypes()
    else: plotSpectrumTypes = [plotSpectrumType]
    for ii in range(len(distsToExp)):
        i = sorted_order[ii]
        for spType in plotSpectrumTypes:
            energy = mixtureData.getEnergy(spType)
            unknownSpectrum = None
            if unknownSpectrumToPlot is None:
                if unknownCharacterization['type'] == 'spectrum':
                    if unknownCharacterization['spType'] == spType:
                        unknownSpectrum = unknownCharacterization['spectrum']
                        unknownEnergy = energy
            else:
                if isinstance(unknownSpectrumToPlot, dict):
                    u = unknownSpectrumToPlot[spType]
                else:
                    u = unknownSpectrumToPlot
                if isinstance(u, utils.Spectrum):
                    unknownSpectrum = u.intensity
                    unknownEnergy = u.energy
                else:
                    unknownSpectrum = u
                    unknownEnergy = energy
            intensity = mixtureData.getSpectra(spType).loc[i].to_numpy()
            if unknownSpectrum is not None:
                if componentCount > 1:
                    conc_string = ', '.join([f'C_{name}={bestConc:.2f}' for (name, bestConc) in zip(componentNames[i].reshape(-1).tolist(), concentrations[i].reshape(-1).tolist())])
                    title = f'Concentrations: {conc_string}, distsToExp: {distsToExp[i]:.4f}'
                else:
                    title = f'Candidate {componentNames[i]}. DistsToExp: {distsToExp[i]:.4f}'
                spectraFolder = folder + os.sep + 'spectra' + spectraFolderPostfix
                if not os.path.exists(spectraFolder):
                    os.makedirs(spectraFolder)
                fileName = spectraFolder + os.sep + f'{utils.zfill(ii,len(distsToExp))}_{spType}' + fileNamePostfix + '.png'
                plotting.plotToFile(energy, intensity, 'mixture', unknownEnergy, unknownSpectrum, 'experiment', title=title, fileName=fileName)
        if ii >= maxSpectraCountToPlot-1: break

    # =================== mixture label map plotting ===================
    if componentCount > 1:
        if label_bounds is None:
            label_bounds = [[np.min(singleComponentData.params[label]), np.max(singleComponentData.params[label])] for label in label_names]
        plotMixtureLabelMap(componentLabels=componentLabels, label_names=label_names, label_bounds=label_bounds, labelMaps=labelMaps, distsToExp=distsToExp, concentrations=concentrations, componentNames=componentNames, folder=folder+os.sep+'label_maps'+labelMapsFolderPostfix, fileNamePostfix=fileNamePostfix)


def plotBestMixtures2d(paramsNames, paramsBounds, spectraFunc, makeMixture, distToExperiment, maxPercentToBeBest=0.1):
    """
    Creates figures of mixtures for different concentrations of several components.

    :param paramsNames: [paramName1, paramName2, ...]
    :param paramsBounds: [ [x1, y1], [x2, y2], ...]
    :param spectraFunc: function which calculates spectrum by parameters
                        (mixture components are obtained by applying spectraFunc to different parameters). 
    :param makeMixture: function which creates mixture of spectra calculated by spectraFunc 
    :param distToExperiment: function which calculates distance to experimental spectrum
    :param maxPercentToBeBest: sets the proportion of figures from the total number that will be drawn
    :return:
    """
    # Common params for plotBestMixtures2d
    amountOfRandomStartsForMinimumSearch = 5
    amountOfRandomStartsForPlotting = 200
    n = len(paramsNames)
    
    def makeStartPointForOptimizing():
        initialParams = []
        # First component parameters
        for p in paramsBounds:
            randomParam = p[0] + np.random.random()*(p[1] - p[0])
            initialParams.append(randomParam)
        # Second component parameters
        for p in paramsBounds:
            randomParam = p[0] + np.random.random()*(p[1] - p[0])
            initialParams.append(randomParam)
        # Concentration
        initialParams.append(np.random.random())
        return np.array(initialParams)

    def targetFunction(x):
        x1 = x[:n].reshape(1, -1)
        x2 = x[n:-1].reshape(1, -1)
        concentraitions = np.array([x[-1], 1-x[-1]])
        spectrum1 = spectraFunc(x1)
        spectrum2 = spectraFunc(x2)
        mixtureSpectrum = makeMixture(spectrum1, spectrum2, concentraitions)
        return distToExperiment(mixtureSpectrum)
    
    firstSetOfOptimizedParameters = []
    # Find the best set of parameters of five launches
    bestMinimum = (np.inf, None)
    for i in range(amountOfRandomStartsForMinimumSearch):
        arg0 = makeStartPointForOptimizing()
        currentMinimum = scipy.optimize.minimize(targetFunction, arg0, bounds=paramsBounds + paramsBounds + [[0, 1]])
        firstSetOfOptimizedParameters.append((currentMinimum.fun, currentMinimum.x))
        if currentMinimum.fun < bestMinimum[0]:
            bestMinimum = (currentMinimum.fun, currentMinimum.x)
    
    lowerBound = bestMinimum[0] * (1 + maxPercentToBeBest)
    # Make custom distance function
    def makeNewDistanceFunction(x):
        currentDistance = targetFunction(x)
        return lowerBound if currentDistance < lowerBound else currentDistance
    
    # Make list of amountOfRandomStartsForPlotting minimums
    secondSetOfOptimizedParameters = []
    for i in range(amountOfRandomStartsForPlotting):
        if i % 10 == 0: print(f'{i}/{amountOfRandomStartsForPlotting}')
        arg0 = makeStartPointForOptimizing()
        currentMinimum = scipy.optimize.minimize(makeNewDistanceFunction, arg0, bounds=paramsBounds + paramsBounds + [[0, 1]])
        if currentMinimum.fun < bestMinimum[0] * (1 + 2 * maxPercentToBeBest):
            secondSetOfOptimizedParameters.append((currentMinimum.fun, currentMinimum.x))
    
    if not os.path.exists('mixtures'):
        os.mkdir('mixtures')
    # Make mixture plots of all pairs of parameters
    for i in range(len(paramsNames)):
        for j in range(i+1, len(paramsNames)):
            fig, ax = plotting.createfig()
            for minimumParams in secondSetOfOptimizedParameters:
                ax.plot([minimumParams[1][i], minimumParams[1][i+n]], [minimumParams[1][j], minimumParams[1][j+n]], linestyle=':', marker='.', color='k')
                ax.scatter(minimumParams[1][i], minimumParams[1][j], color='k', s=minimumParams[1][-1]*200)
                ax.scatter(minimumParams[1][i+n], minimumParams[1][j+n], color='k', s=(1-minimumParams[1][-1])*200)
            for minimumParams in firstSetOfOptimizedParameters:
                ax.plot([minimumParams[1][i], minimumParams[1][i+n]], [minimumParams[1][j], minimumParams[1][j+n]], linestyle='--', marker='.', color='r')
                ax.scatter(minimumParams[1][i], minimumParams[1][j], color='r', s=minimumParams[1][-1]*200)
                ax.scatter(minimumParams[1][i+n], minimumParams[1][j+n], color='r', s=(1-minimumParams[1][-1])*200)
                
            ax.set_xlabel(paramsNames[i])
            ax.set_ylabel(paramsNames[j])
            plotting.savefig('mixtures' + os.sep + paramsNames[i] + '_' + paramsNames[j] + '.png', fig)
            plotting.closefig(fig)
                

def findGlobalMinimumMixture(distToExperiment, spectraFuncs, makeMixture, trysCount, bounds, paramNames, componentNames=None, constraints=None, folderToSaveResult='globalMinimumSearchResult', fixParams=None, contourMapCalcMethod='fast', plotContourMaps='all', extraPlotFuncContourMaps=None):
    """
    Try several optimizations from random start point. Retuns sorted list of results and plot contours around minimum.

    :param distToExperiment: function which calculates distance to experimental spectrum distToExperiment(mixtureSpectrum, allArgs, *fun_args). allArgs = [[component1_params], ..., [componentN_params],[commonParams]]
    :param spectraFuncs: list of functions [func1, ...] which calculates component spectra by thier parameters. func1(component1_params, *commonParams)
    :param makeMixture: function, that creates mixture of spectra calculated by spectraFunc makeMixture(spectraList, concentrations, *commonParams)
    :param trysCount: number of attempts to find minimum
    :param bounds: list of N+1 lists of 2-element lists with parameter bounds (component params and common params)
    :param paramNames: [[component1_params], ..., [componentN_params], [commonParams]]. Parameters of different components can have similar names. All the parameters will be prepend in result by componentNames+'_' prefixes
    :param componentNames: list of component names
    :param constraints: additional constrains for ‘trust-constr’ scipy.optimize.minimize method (for [notFixedConcentrations[:-1],allArgs].flattern argument)
    :param folderToSaveResult: all result graphs and log are saved here
    :param fixParams: dict of paramName:value to fix (component params must have prefix 'componentName_')
    :param contourMapCalcMethod: 'fast' - plot contours of the target function; 'thorough' - plot contours of the min of target function by all arguments except axes
    :param plotContourMaps: 'all' or list of 1-element or 2-elements lists of axes names to plot contours of target function
    :param extraPlotFuncContourMaps: user defined function to plot something on result contours: func(ax, axisNamesList, xminDict)
    :return: sorted list of trysCount minimums of dicts with keys 'value', 'x' (have allArgs-like format), 'x1d' (flattened array of all param values), 'paramNames1d', 'concentrations', 'spectra', 'mixtureSpectrum'
    """
    N = len(spectraFuncs)
    if fixParams is None: fixParams = {}
    fixParams = copy.deepcopy(fixParams)
    if constraints is None: constraints = tuple()
    oneComponent = len(spectraFuncs) == 1
    if len(bounds) == N: bounds.append([])
    if len(paramNames) == N: paramNames.append([])
    assert len(componentNames) == N, f'{len(componentNames)} != {N}'
    assert len(bounds) == N+1, f'{len(bounds)} != {N+1}'
    assert len(paramNames) == N+1, f'{len(paramNames)} != {N+1}'
    if componentNames is None:
        componentNames = [f'c{i+1}' for i in range(N)]

    def getParamName(componentName, paramName):
        if paramName == '': return componentName
        if oneComponent:
            return paramName
        else:
            return componentName + '_' + paramName

    concentrationNames = []
    for componentName in componentNames:
        if not oneComponent:
            concentrationNames.append(getParamName(componentName, ''))
    notFixedConcentrations = [cn for cn in concentrationNames if cn not in fixParams]
    concentrationBound = 1 - np.sum([fixParams[c] for c in concentrationNames if c in fixParams])
    if len(notFixedConcentrations) == 1:  # fix it
        assert notFixedConcentrations[0] not in fixParams, 'You do not need to fix concentration'
        fixParams[notFixedConcentrations[0]] = concentrationBound
        notFixedConcentrations = []
    assert (len(notFixedConcentrations) >= 2) or (len(notFixedConcentrations) == 0)
    if not oneComponent and len(notFixedConcentrations) == 0:
        sum = 0
        for c in concentrationNames: sum += fixParams[c]
        assert np.abs(1-sum) < 1e-6, 'Sum of fixed concentrations != 1'
    fixParams1 = copy.deepcopy(fixParams)
    # we manually fix concentrations
    for p in concentrationNames:
        if p in fixParams1: del fixParams1[p]

    paramNames1d = copy.deepcopy(notFixedConcentrations[:-1])
    bounds1d = [[0,1]]*len(notFixedConcentrations[:-1])
    # paramNamesNotFixed = []
    for i_component in range(len(componentNames)):
        componentName = componentNames[i_component]
        toAdd = copy.deepcopy(paramNames[i_component])
        for i in range(len(toAdd)):
            name = getParamName(componentName, toAdd[i])
            paramNames1d.append(name)
        bounds1d += bounds[i_component]
    paramNames1d += paramNames[N]  # common params
    bounds1d += bounds[N]

    assert len(bounds1d) == len(paramNames1d), f'{len(bounds1d)} != {len(paramNames1d)}\n{bounds1d}\n{paramNames1d}'
    assert set(fixParams1.keys()) < set(paramNames1d), 'Some fixed param names are not present in all param names list\nFixed param names: ' + str(list(fixParams1.keys())) + '\nAll param names: ' + str(paramNames1d)
    assert len(np.unique(componentNames)) == len(componentNames), "Component names names should be different!"
    assert len(np.unique(paramNames1d)) == len(paramNames1d), "Combinations of component names and param names should be different!\n"+str(paramNames1d)

    def getParamIndFullList(paramName):
        res = np.where(np.array(paramNames1d) == paramName)[0]
        assert len(res) == 1, f'Parameter {paramName} is not present in list {paramNames1d}'
        return res[0]

    def getConcentration(fullParamList, componentInd):
        cname = concentrationNames[componentInd]
        if cname in fixParams:
            return fixParams[cname]
        if cname != notFixedConcentrations[-1]:
            i = getParamIndFullList(cname)
            return fullParamList[i]
        else:
            return 1 - np.sum(fullParamList[:len(notFixedConcentrations)-1])

    def getAllConcentrations(fullParamList):
        if oneComponent: return np.ones(1)
        concentraitions = np.zeros(N)
        for i in range(N):
            concentraitions[i] = getConcentration(fullParamList, i)
        return concentraitions

    def getComponentParamVector(fullParamList, componentInd):
        componentName = componentNames[componentInd]
        pp = np.zeros(len(paramNames[componentInd]))
        for i in range(len(paramNames[componentInd])):
            paramInd = getParamIndFullList(getParamName(componentName, paramNames[componentInd][i]))
            pp[i] = fullParamList[paramInd]
        return pp

    def getAllComponentsParams(fullParamList):
        return [getComponentParamVector(fullParamList, i) for i in range(N)]

    def getCommonParams(fullParamList):
        pp = np.zeros(len(paramNames[N]))
        for i in range(len(paramNames[N])):
            paramInd = getParamIndFullList(paramNames[N][i])
            pp[i] = fullParamList[paramInd]
        return pp

    def targetFunction(x):
        # for i in range(len(x)):
        #     assert bounds1d[i][0] <= x[i] <= bounds1d[i][1], f'{paramNames1d[i]} = {x[i]} not in {bounds1d[i]}'
        concentraitions = getAllConcentrations(x)
        spectra = []
        x_common = getCommonParams(x)
        for i in range(N):
            x_partial = getComponentParamVector(x,i)
            spectrum = spectraFuncs[i](x_partial, *x_common)
            spectra.append(spectrum)
        mixtureSpectrum = makeMixture(spectra, concentraitions, *x_common)
        return distToExperiment(mixtureSpectrum, getAllComponentsParams(x), *x_common)

    cons = tuple()
    if (not oneComponent) and (len(notFixedConcentrations) >= 2):
        a = np.zeros(len(paramNames1d))
        a[:len(notFixedConcentrations)-1] = 1
        cons = (scipy.optimize.LinearConstraint(a, 0, concentrationBound, keep_feasible=True),)
    result1d = optimize.findGlobalMinimum(targetFunction, trysCount, bounds1d, constraints=constraints+cons, fun_args=None, paramNames=paramNames1d, folderToSaveResult=folderToSaveResult, fixParams=fixParams1, contourMapCalcMethod=contourMapCalcMethod, plotContourMaps=plotContourMaps, extraPlotFunc=extraPlotFuncContourMaps)
    result = []
    for r in result1d:
        x1d = r['x']
        x_common = getCommonParams(x1d)
        x = []; spectra = []
        for i in range(N):
            x_partial = getComponentParamVector(x1d, i)
            x.append(x_partial)
            spectra.append(spectraFuncs[i](x_partial, *x_common))
        x.append(x_common)
        concentrations = getAllConcentrations(x1d)
        mixtureSpectrum = makeMixture(spectra, concentrations, *x_common)
        result.append({'value':r['value'], 'x':x, 'x1d':x1d, 'paramNames1d':paramNames1d, 'concentrations':concentrations, 'spectra':spectra, 'mixtureSpectrum':mixtureSpectrum})
    return result


def generateUniformLabelDistrib(size, componentCount, uniformLabelName, sample):
    all_ind = np.zeros(shape=(size, componentCount), dtype=np.int)
    label = sample.params.loc[:, uniformLabelName].to_numpy()
    if ML.isClassification(sample.params, uniformLabelName):
        label_values = np.unique(label)
        # print(label_values, label)
        inds_by_label_values = {lv:np.where(label == lv)[0] for lv in label_values}
        assert componentCount <= len(label_values), f'componentCount = {componentCount} > len(label_values) = {len(label_values)}'
        maxSize = scipy.special.comb(len(label_values), componentCount, repetition=True)
        index_combinations = utils.comb_index(len(label_values), componentCount, repetition=True)
        if maxSize <= size:
            # repeat combinations
            index_combinations = index_combinations[np.arange(size) % len(index_combinations), :]
        else:
            # take random subset
            index_combinations = index_combinations[np.random.choice(int(maxSize), size=size, replace=False), :]
        for i in range(size):
            for j in range(componentCount):
                label_value = label_values[index_combinations[i, j]]
                # take random sepctrum from the set with same label
                all_ind[i, j] = np.random.choice(inds_by_label_values[label_value])
    else:
        # label is real but not integer or nominal
        mi = np.min(label)
        ma = np.max(label)
        random_labels = np.random.rand(size, componentCount) * (ma - mi) + mi
        sorted_label_ind = np.argsort(label)
        sorted_label = label[sorted_label_ind]
        for i in range(size):
            for j in range(componentCount):
                _, idx = utils.find_nearest_in_sorted_array(sorted_label, random_labels[i, j])
                initial_index = sorted_label_ind[idx]
                all_ind[i, j] = initial_index
    # print(label[all_ind])
    return all_ind


def generateMixtureOfSample(size, componentCount, sample, label_names, addDescrFunc=None, makeMixtureOfSpectra=None, makeMixtureOfLabel=None, dirichletAlpha=1, randomSeed=0, returnNotMixLabels=False, componentNameColumn=None, makeUniformLabelDistrib=None):
    """
    Generates descriptor data for random mixtures of component combinations
    :param componentCount: count of mixture components
    :param size: mixture data size to generate
    :param sample: initial pure sample
    :param label_names: names of target variables that are not descriptors of spectra, calculated by addDescrFunc
    :param addDescrFunc: calculates descriptors for the given sample. Optionally returns goodSpecrtrumIndices (if some specrtra were bad and descriptors couldn't be built)
                        sample_with_descriptors, goodSpecrtrumIndices = addDescrFunc(sample)
    :param makeMixtureOfSpectra: function(inds, concentrations) to calculate mixture of spectra by given concentrations and spectra indices (returns intensity only or dict {spType: intensity only} for multiple spectra types)
    :param makeMixtureOfLabel: function(label_name, sample, inds, concentrations) to calculate label for mixture of spectra given concentrations and spectra indices
    :param dirichletAlpha: parameter of dirichlet distribution of concentrations (1 - uniform, >1 - mostly equal, <1 - some components prevail)
    :param randomSeed: random seed
    :param returnNotMixLabels: if true, along with mixture of labels returns:
        componentLabels: array of label tables (for each component - one table) with size componentCount x size x labelCount
        concentrations: 2d array size x componentCount (sorted in descendant order!!!!!!)
        componentNames: 2d array size x componentCount
    :param componentNameColumn: used when labels are not mixed (if None - use index)
    :param makeUniformLabelDistrib: name of label to have uniform result distribution (in labelValues^componentCount space) in mixture sample. If None - the mixture is uniform in terms of single component indices
    :return: new mixture sample with mixed labels and separate info about component labels if returnNotMixLabels=True
    """
    # print('Do not use CV for one mixture sample. Instead divide pure sample into 2 parts and generate 2 mixture samples: train and test.')
    data = sample.params.loc[:,label_names].to_numpy()
    spectra = {spType:sample.getSpectra(spType).to_numpy() for spType in sample._spectra}
    for j in range(len(label_names)):
        assert np.all(pd.notnull(data[:,j])), 'Null values for label '+str(label_names[j])+'\n'+str(data[:,j])
    assert np.all(pd.notnull(data))
    np.random.seed(randomSeed)
    c = np.random.dirichlet(alpha=dirichletAlpha*np.ones(componentCount), size=size)
    c = -np.sort(-c, axis=1)
    labelCount = len(label_names)
    n = sample.getLength()

    if makeUniformLabelDistrib is None:
        maxSize = scipy.special.comb(n, componentCount, repetition=True)
        if maxSize < 100000:
            index_combinations = utils.comb_index(n, componentCount, repetition=True)
            if maxSize <= size:
                # repeat combinations
                all_ind = index_combinations[np.arange(size) % len(index_combinations), :]
            else:
                # take random subset
                all_ind = index_combinations[np.random.choice(maxSize, size=size, replace=False), :]
        else:
            all_ind = np.random.randint(low=0, high=n-1, size=(size, componentCount))
    else:
        all_ind = generateUniformLabelDistrib(size, componentCount, makeUniformLabelDistrib, sample)

    if returnNotMixLabels:
        componentLabels = np.zeros((componentCount, size, labelCount))
        concentrations = np.zeros((size, componentCount))
        componentNames = np.zeros((size, componentCount), dtype=object)
    mix_data = np.zeros((size, len(label_names)))
    mix_spectra = {}
    for spType in spectra:
        mix_spectra[spType] = np.zeros((size, len(sample.getEnergy(spType))))
    for i in range(size):
        ind = all_ind[i]
        if makeMixtureOfSpectra is None:
            for spType in spectra:
                mix_spectra[spType][i] = np.dot(spectra[spType][ind,:].T, c[i])
        else:
            ms = makeMixtureOfSpectra(ind, c[i])
            if isinstance(ms, dict):
                for spType in spectra:
                    assert len(ms[spType]) == len(sample.getEnergy(spType))
                    mix_spectra[spType][i] = ms[spType]
            else:
                assert len(spectra) == 1
                for spType in spectra:
                    assert len(ms) == len(sample.getEnergy(spType))
                    mix_spectra[spType][i] = ms
        if makeMixtureOfLabel is None:
            mix_data[i] = np.dot(data[ind,:].T, c[i])
        else:
            for j in range(len(label_names)):
                mix_data[i,j] = makeMixtureOfLabel(label_names[j], sample, ind, c[i])
        if returnNotMixLabels:
            for j in range(componentCount):
                componentLabels[j,i] = data[ind[j],:]
                componentNames[i,j] = str(ind[j]) if componentNameColumn is None else str(sample.params.loc[ind[j],componentNameColumn])
            concentrations[i] = c[i]

    mix_data = pd.DataFrame(columns=label_names, data=mix_data, dtype='float64')
    mix_sample = ML.Sample(mix_data, mix_spectra, energy=sample._energy)
    if addDescrFunc is not None:
        res = addDescrFunc(mix_sample)
        if isinstance(res, tuple): mix_sample, goodSpectrumIndices = res
        else: mix_sample = res
    else: goodSpectrumIndices = np.arange(len(mix_data))
    mix_sample.paramNames = mix_sample.params.columns.to_numpy()
    if addDescrFunc is not None and set(mix_sample.paramNames) != set(sample.paramNames):
        assert set(mix_sample.paramNames) <= set(sample.paramNames), 'Set of initial sample and mixture descriptors are not equal. There are extra mixture descriptors: ' + str(set(mix_sample.paramNames)-set(sample.paramNames))+'.\nMay be you forget to list all labels in generateMixtureOfSample call?'
        # check dtype
        forgotten = set(sample.paramNames)-set(mix_sample.paramNames)
        for f in forgotten:
            assert sample.params[f].dtype != np.float64, f'Set of initial sample and mixture descriptors are not equal. Param {f} with dtype=float64 is absent in mixture data'
    if returnNotMixLabels:
        return mix_sample, componentLabels[:,goodSpectrumIndices,:], concentrations[goodSpectrumIndices,:], componentNames[goodSpectrumIndices,:]
    else:
        return mix_sample


def score_cv(model_regr, sample, features, label, label_names, makeMixtureParams, testRatio=0.2, repetitions=5, model_class=None):
    assert 'sample' not in makeMixtureParams
    assert 'size' not in makeMixtureParams
    assert 'randomSeed' not in makeMixtureParams
    assert 'returnNotMixLabels' not in makeMixtureParams
    assert 'makeUniformLabelDistrib' not in makeMixtureParams
    n = sample.getLength()
    testSize = int(n * testRatio)
    trainSize = int(n * (1-testRatio))
    sample = sample.copy()
    trueAvgLabels = None; predAvgLabels = None
    trueComponentLabels = None; predComponentLabels = None
    trueConcentrations = None; predConcentrations = None
    i_label = label_names.index(label)
    for ir in range(repetitions):
        indTrain, indTest = sklearn.model_selection.train_test_split(np.arange(n).reshape(-1,1), test_size=testRatio, random_state=ir, shuffle=True)
        sampleTrain = sample.takeRows(indTrain)
        sampleTest = sample.takeRows(indTest)
        mixSampleTrain, componentLabelsTrain, concentrationsTrain, _ = generateMixtureOfSample(size=trainSize, sample=sampleTrain, label_names=label_names, randomSeed=ir, returnNotMixLabels=True, makeUniformLabelDistrib=label, **makeMixtureParams)
        componentLabelsTrain = componentLabelsTrain[:,:,i_label].T
        xTrain, yTrain = descriptor.getXYFromSample(mixSampleTrain, features, label)

        mixSampleTest, componentLabelsTest, concentrationsTest, _ = generateMixtureOfSample(size=testSize, sample=sampleTest, label_names=label_names, randomSeed=ir, returnNotMixLabels=True, makeUniformLabelDistrib=label, **makeMixtureParams)
        componentLabelsTest = componentLabelsTest[:,:,i_label].T
        xTest, yTest = descriptor.getXYFromSample(mixSampleTest, features, label)

        model_regr_avgLabel = copy.deepcopy(model_regr)
        model_regr_avgLabel.fit(xTrain, yTrain)
        if predAvgLabels is None: predAvgLabels = model_regr_avgLabel.predict(xTest)
        else: predAvgLabels = np.append(predAvgLabels, model_regr_avgLabel.predict(xTest), axis=0)
        if trueAvgLabels is None: trueAvgLabels = yTest
        else: trueAvgLabels = np.append(trueAvgLabels, yTest, axis=0)

        model_regr_conc = copy.deepcopy(model_regr)
        model_regr_conc.fit(xTrain, concentrationsTrain)
        if predConcentrations is None: predConcentrations = model_regr_conc.predict(xTest)
        else: predConcentrations = np.append(predConcentrations, model_regr_conc.predict(xTest), axis=0)
        if trueConcentrations is None: trueConcentrations = concentrationsTest
        else: trueConcentrations = np.append(trueConcentrations, concentrationsTest, axis=0)
        assert np.all(predConcentrations.shape == trueConcentrations.shape), str(predConcentrations.shape)+' != '+str(trueConcentrations.shape)

        isClassification = ML.isClassification(componentLabelsTrain[:,0])
        if isClassification:
            assert model_class is not None
            model_comp_labels = copy.deepcopy(model_class)
        else:
            model_comp_labels = copy.deepcopy(model_regr)
        model_comp_labels.fit(xTrain, componentLabelsTrain)
        if predComponentLabels is None: predComponentLabels = model_comp_labels.predict(xTest)
        else: predComponentLabels = np.append(predComponentLabels, model_comp_labels.predict(xTest), axis=0)
        if trueComponentLabels is None: trueComponentLabels = componentLabelsTest
        else: trueComponentLabels = np.append(trueComponentLabels, componentLabelsTest, axis=0)

    conc_quality = [ML.scoreFast(trueConcentrations[:,j], predConcentrations[:,j]) for j in range(trueConcentrations.shape[1])]
    comp_quality = []
    for j in range(trueComponentLabels.shape[1]):
        if isClassification:
            accuracy = np.sum(trueComponentLabels[:,j] == predComponentLabels[:,j])/trueComponentLabels.shape[0]
            comp_quality.append(accuracy)
        else:
            r_score = ML.scoreFast(trueComponentLabels[:,j], predComponentLabels[:,j])
            comp_quality.append(r_score)
    qualities = {'avgLabels': ML.scoreFast(trueAvgLabels, predAvgLabels), 'concentrations': conc_quality, 'componentLabels': comp_quality}
    trueVals = {'avgLabels': trueAvgLabels, 'concentrations': trueConcentrations, 'componentLabels': trueComponentLabels}
    predVals = {'avgLabels': predAvgLabels, 'concentrations': predConcentrations, 'componentLabels': predComponentLabels}
    models = {'avgLabels': model_regr_avgLabel, 'concentrations': model_regr_conc, 'componentLabels': model_comp_labels}
    return qualities, trueVals, predVals, models


def plotMixtures(ax, axesNames, calcDescriptorsFunc, sample, mixtures, componentNameColumn=None, mixtureStepCount=200):
    """
    Used by plot_descriptors_2d as additionalMapPlotFunc
    :param ax: axes
    :param axesNames: descriptor names pair of ax
    :param calcDescriptorsForSingleSpectrumFunc: function(spectrum) should return array of the same length as 'sample.paramNames' or dict of descriptors. spectrum - one spectrum or dict of spectra for different spTypes
    :param sample: Sample with spectra and params
    :param mixtures: list of pairs of names (if componentNameColumn!=None) or indices
    :param componentNameColumn: if None use index
    """
    n = sample.getLength()
    componentNames = sample.params[componentNameColumn].to_numpy() if componentNameColumn is not None else np.arange(n)
    for mix in mixtures:
        if not isinstance(componentNames[0], type(mix[0])):
            componentNames = componentNames.astype(type(mix[0]))
        i1 = np.where(componentNames == mix[0])[0][0]
        i2 = np.where(componentNames == mix[1])[0][0]
        conc = np.linspace(0, 1, mixtureStepCount)
        desc2 = conc * data.loc[i1, 'pre-edge area'] + (1 - conc) * data.loc[i2, 'pre-edge area']
        a = data.loc[i1, 'pre-edge center'] * data.loc[i1, 'pre-edge area']
        b = data.loc[i2, 'pre-edge center'] * data.loc[i2, 'pre-edge area']
        desc1 = (conc * a + (1 - conc) * b) / desc2
        ax.scatter(desc1, desc2, 3, c='black')


def mcrRealization(matrix, diapFrom, diapTo, temperature, energy):
    """
    :param matrix: spectrum matrix
    :param diapFrom: start of spectrum range
    :param diapTo:  end of spectrum range
    :param temperature: vector of temperature
    :param energy: energy column

    """
    from pymcr.mcr import McrAR
    from pymcr.regressors import NNLS
    logger = logging.getLogger('pymcr')
    logger.setLevel(logging.DEBUG)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_format = logging.Formatter('%(message)s')
    stdout_handler.setFormatter(stdout_format)
    logger.addHandler(stdout_handler)
    mcrar = McrAR()
    totalComponents = len(matrix.T)
    if ((diapFrom < 1) or (diapFrom > totalComponents) or (diapTo < 0) or (diapTo > totalComponents)):
        print("ERROR")
        print("invalid range")
        print("The value must be no less than zero and no more than" + totalComponents)
    elif (diapFrom == diapTo and diapTo != totalComponents):
        diapTo += 1
    elif (diapTo < diapFrom):
        diap = diapTo
        diapTo = diapFrom
        diapFrom = diap
    elif (diapFrom == diapTo and diapTo == totalComponents):
        diapFrom -=1
    i = 0
    numberOfComponents = diapTo - diapFrom
    toPlotComponents = tuple()
    toPlotConcentrations = tuple()
    initial_spectra = matrix[:, diapFrom:diapTo].T
    mcrar = McrAR(c_regr=NNLS(), st_regr=NNLS(), c_constraints=[], st_constraints=[], tol_increase=0.1,
                    max_iter=100,
                    tol_n_above_min=5000)
    mcrar.fit(matrix.T, ST=initial_spectra, verbose=False)
    print('\nFinal MSE: {:.7e}'.format(mcrar.err[-1]))
   # print(mcrar.ST_opt_[i,:].T)
    while i < numberOfComponents:
        plt.plot(energy, initial_spectra[i,:].T, label= "initial")
        plt.plot(energy,mcrar.ST_opt_[i,:].T, label= "result")
        plt.xlabel('Энернгия')
        plt.ylabel('Интенсивность')
        plt.title('Компоненты')
        plt.legend()
        plt.show()

        plt.plot(temperature, mcrar.C_opt_[:,i], label= "C")
        plt.title('Зависимость концентраций от температуры')
        plt.legend()
        plt.show()
        #plotting.plotToFile(*toPlotConcentrations, fileName='mcr/graphs/concentrations/concentration' + str(i) + '.png')
        i += 1

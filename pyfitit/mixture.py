from . import utils, plotting, optimize, ML
import numpy as np
import pandas as pd
import scipy, os, shutil, itertools, copy, sklearn, logging, sys, sklearn.multioutput
import matplotlib.pyplot as plt


def findConcentrations(energy, xanesArray, expXanes, fixConcentrations=None, trysGenerateMixtureOfSampleCount=1):
    """
    Concentration search
    :param energy: 1-d array with energy values (same for all xanes)
    :param xanesArray: 2-d array. One row - one xanes
    :param expXanes: experimental xanes for the same energy
    :param fixConcentrations: {index:value,...}
    :return: optimized function value, array of optimal concentrations
    """
    def distToUnknown(mixture):
        return utils.rFactor(energy, mixture, expXanes)

    def makeMixture(c):
        return np.sum(xanesArray * c.reshape(-1, 1), axis=0)

    return findConcentrationsAbstract(distToUnknown, len(xanesArray), makeMixture, fixConcentrations=fixConcentrations, trysGenerateMixtureOfSampleCount=trysGenerateMixtureOfSampleCount)


def findConcentrationsNNLS(energy, xanesArray, expXanes):
    """
    Concentration search
    :param energy: 1-d array with energy values (same for all xanes)
    :param xanesArray: 2-d array. One row - one xanes
    :param expXanes: experimental xanes for the same energy
    :return: optimized function value, array of optimal concentrations
    """
    assert np.all(np.isfinite(energy))
    assert np.all(np.isfinite(xanesArray))
    assert np.all(np.isfinite(expXanes))
    assert len(energy) == xanesArray.shape[1], f'{len(energy)} != {xanesArray.shape[1]}'
    assert len(energy) == len(expXanes), f'{len(energy)} != {len(expXanes)}'
    nc = xanesArray.shape[0]
    if nc==2 and np.all(xanesArray[0]==xanesArray[1]):
        concentrations = np.array([1,0])
    else:
        nnls = sklearn.linear_model.LinearRegression(positive=True, fit_intercept=False)
        w = np.diff(energy)
        w = np.append(w, energy[-1]-energy[-2])
        X = np.append(xanesArray.T, np.ones((1,nc)), axis=0)
        w = np.append(w, np.max(w)*100)
        expXanes1 = np.append(expXanes,1).reshape(-1,1)
        def normalize(c):
            s = np.sum(c)
            assert s != 0, str(c)
            return c / np.sum(c)
        w = normalize(w)
        nnls.fit(X, expXanes1, sample_weight=w)
        concentrations = normalize(nnls.coef_.reshape(-1))
    error = utils.rFactor(energy, np.dot(xanesArray.T, concentrations), expXanes)
    return error, concentrations


def findConcentrationsAbstract(distToUnknown, componentCount, makeMixture, fixConcentrations=None, trysGenerateMixtureOfSampleCount=1):
    """
    Concentration search
    :param distToUnknown: function(mixture) to return distance to unknown experiment
    :param componentCount: component count
    :param makeMixture: function(concentrations) to calculate mixture spectrum/descriptors
    :param fixConcentrations: {index:value,...}
    :param trysGenerateMixtureOfSampleCount: Number. First try starting from all pure components and one - with all equal concentrations. Then try starting from random concentrations. Choose best
    :return: optimized function value, array of optimal concentrations
    """
    assert isinstance(trysGenerateMixtureOfSampleCount, int)
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
    if m >= 3:
        a = np.ones(m-1)
        constrains = (scipy.optimize.LinearConstraint(a, 0, upperBound, keep_feasible=True),)
    else:
        constrains = None
    res = optimize.findGlobalMinimum(func, trysGenerateMixtureOfSampleCount, bounds=[[0, upperBound]] * (m - 1), constraints=constrains, plotContourMaps=[], printOnline=False)
    return res[0]['value'], expand(res[0]['x'])


def plotMixtureLabelMap(componentLabels, label_names, label_bounds, labelMaps, distsToExp, concentrations, componentNames, folder, fileNamePostfix='', showInNotebook=False):
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
    assert componentLabels.shape[1] == spectraCount, f'{componentLabels.shape[1]} != {spectraCount}, componentLabels.shape = {componentLabels.shape}'
    assert componentLabels.shape[0] == component_count, f'{componentLabels.shape[0]} == {component_count}'
    assert len(distsToExp) == spectraCount, f'{len(distsToExp)} == {spectraCount}'
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

    for label, i_lab in zip(label_names, range(len(label_names))):
        all_label_values = []
        for i in range(component_count):
            all_label_values += list(componentLabels[i,:,i_lab])
        isClassification = ML.isClassification(all_label_values)
        if isClassification and label in labelMaps:
            assert set(all_label_values) <= set(labelMaps[label].keys()), f'Encoded componentLabels detected for label: {label}'
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
            notExist = np.max(distsToExp)*1
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
            ticklabels = [int(l) for l in ticklabels]
            plotting.plotMatrix(plot_norm_2d, cmap='summer', ticklabelsX=ticklabels, ticklabelsY=ticklabels, title=f'Distance to experiment for different label values of the components', xlabel=label, ylabel=label, fileName=folder + os.sep + 'map' + fileNamePostfix1 + '.png', interactive=showInNotebook)
        else:
            fig, ax = plotting.createfig(figsize=(6.2,4.8), interactive=showInNotebook)
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


def tryAllMixtures(unknownCharacterization, componentCount, mixtureTrysCount, singleComponentData: ML.Sample, labelNames, folder, optimizeConcentrations=True, optimizeConcentrationsTrysCount=None, fileNamePostfix='', spectraFolderPostfix='', labelMapsFolderPostfix='', labelBounds=None, labelMaps=None, componentNameColumn=None, calcDescriptorsForSingleSpectrumFunc=None, randomSeed=0, makeMixtureOfSpectra=None, plotSpectrumType=None, maxSpectraCountToPlot=50, unknownSpectrumToPlot=None, plotWrapperGenerator=None, customPlotter=None, doNotPlot=False, ignoreNotOrdinalLabels=False, showInNotebook=False, debug=False):
    """
    Try to fit unknown spectrum/descriptors by all possible linear combinations of componentCount spectra from componentCandidates using different label values. Plots mixture label map, best mixture spectra

    :param unknownCharacterization: dict('type', ...)
                1) 'type':'distance function',  'function': function(mixtureSpectrum, mixtureDescriptors) to return distance to unknown experiment. mixtureSpectrum - dict(spType=spectrum) mixtureDescriptors - result of calcDescriptorsForSingleSpectrumFunc
                2) 'type':'spectrum', 'spType': type of spectrum from singleComponentData sample (default - default), 'spectrum':unknownSpectrum (spectrum or intensity only), 'rFactorParams':dict - params of the utils.rFactorSp function
                3) 'type':'descriptors', 'paramNames':list of descriptor names to use for distance, 'paramValues':array of unknown spectrum descriptor values
    :param componentCount: >=1 number of components to compose the mixture
    :param mixtureTrysCount: string 'all combinations of singles', 'all combinations for each label' or number or dict {label:number}. In case of a number mixtures are chosen with uniform label distribution in a loop for each label
    :param optimizeConcentrations: bool. False - choose random concentrations, True - find optimal by minimizing distance to unknown
    :param optimizeConcentrationsTrysCount: number. First try starting from all pure components and one - with all equal concentrations. Then try starting from random concentrations. Choose best. Default: 2*componentCount+1
    :param singleComponentData: Sample of different single component spectra and descriptors and labels
    :param labelNames: list of names of labels
    :param folder: folder to save plots
    :param fileNamePostfix: to save plots
    :param labelMapsFolderPostfix:
    :param spectraFolderPostfix:
    :param labelBounds: list of pairs [label_min, label_max]. If None, derive from  singleComponentData
    :param labelMaps: strings for values of categorical labels. dict labelName:{valueName:value,...}
    :param componentNameColumn: if None use index
    :param calcDescriptorsForSingleSpectrumFunc: function(spectrum) used when unknownCharacterization type is 'descriptors'. Should return array of the same length as 'paramValues' or dict of descriptors (should contain subset paramValues). spectrum - one spectrum or dict of spectra for different spTypes (depends on singleComponentData)
    :param randomSeed: random seed
    :param makeMixtureOfSpectra: function(inds, concentrations) to calculate mixture of spectra by given concentrations and spectra indices (returns intensity only or dict {spType: intensity only} for multiple spectra types)
    :param plotSpectrumType: 'all' or name of spType inside sample to plot (when unknownCharacterization['type']=='spectrum' default to plot unknownCharacterization['spType'] only)
    :param maxSpectraCountToPlot: plot not more than maxSpectraCountToPlot spectra
    :param unknownSpectrumToPlot: Spectrum or array of the same length as singleComponentData or dict spType: Spectrum or array
    :param plotWrapperGenerator: func(mixSpectrum, unkSpectrum)->plotWrapper argument for plotToFile
    :param customPlotter: funct() to plot graphs for one mixture spectrum
    :param ignoreNotOrdinalLabels: if true - delete ordinal labels from label list
    """
    if labelNames is None: labelNames = []
    assert 'default' not in labelNames
    if componentCount == 1:
        optimizeConcentrations = False
        optimizeConcentrationsTrysCount = None
    else:
        if optimizeConcentrationsTrysCount is None:
            optimizeConcentrationsTrysCount = 2*componentCount+1
    if ignoreNotOrdinalLabels and componentCount > 1:
        labelNames = [l for l in labelNames if singleComponentData.isOrdinal(l)]
    nolabels = len(labelNames) == 0

    singleComponentDataDecoded = singleComponentData.copy()
    for l in list(singleComponentDataDecoded.labelMaps.keys()):
        singleComponentDataDecoded.decode(l)
    unknownCharacterization = copy.deepcopy(unknownCharacterization)
    if unknownCharacterization['type'] == 'spectrum':
        uparams = set(unknownCharacterization.keys())
        allparams = {'type', 'spType', 'spectrum', 'rFactorParams'}
        assert uparams <= allparams, f'Wrong parameter names in unknownCharacterization: {uparams-allparams}'

        def preprocessParams(unknownCharacterization):
            if 'spType' not in unknownCharacterization:
                unknownCharacterization['spType'] = singleComponentData.getDefaultSpType()
            sp = unknownCharacterization['spectrum']
            spType = unknownCharacterization['spType']
            sample_e = singleComponentData.getEnergy(spType)
            assert len(sample_e) > 0
            if isinstance(sp, np.ndarray):
                assert len(sp) == len(sample_e)
                sp = utils.Spectrum(sample_e, sp)
                unknownCharacterization['spectrum'] = sp
            interval = utils.intervalIntersection(sample_e[[0,-1]], sp.x[[0,-1]])
            assert interval is not None, f'Energy intervals are not intersect. Sample interval: {sample_e[[0,-1]]}. Spectrum interval: {sp.x[[0,-1]]}'
            rFactorParams = unknownCharacterization.get('rFactorParams',{})
            if 'interval' in rFactorParams:
                assert len(rFactorParams['interval']) == 2
                interval = utils.intervalIntersection(interval, rFactorParams['interval'])
                assert interval is not None, f'{interval} doesn\'t intersect with {rFactorParams["interval"]}'
            if 'rFactorParams' not in unknownCharacterization: unknownCharacterization['rFactorParams'] = {}
            unknownCharacterization['rFactorParams']['interval'] = interval
            return unknownCharacterization
        unknownCharacterization = preprocessParams(unknownCharacterization)
    n = len(singleComponentData)
    if componentNameColumn is not None: assert len(np.unique(singleComponentData.params[componentNameColumn])) == n, 'Duplicate names'
    allCombinationsOfSingles = mixtureTrysCount == 'all combinations of singles'
    label_names_surrogate = ['default'] if allCombinationsOfSingles or nolabels else labelNames
    # =================== setup mixtureTrysCount ===================
    if mixtureTrysCount == 'all combinations of singles':
        mixtureTrysCount = {'default': int(scipy.special.comb(n, componentCount, repetition=True))}
    elif mixtureTrysCount == 'all combinations for each label':
        mixtureTrysCount = {}
        for label in label_names_surrogate:
            if ML.isClassification(singleComponentData.params, label):
                all_label_values = np.unique(singleComponentData.params[label])
                mixtureTrysCount[label] = int(scipy.special.comb(len(all_label_values), componentCount, repetition=True))
            else:
                mixtureTrysCount[label] = 100 if optimizeConcentrations else 1000
    elif isinstance(mixtureTrysCount, str):
        assert False, f'Unknown mixtureTrysCount string: {mixtureTrysCount}. It should be "all combinations of singles" or "all combinations for each label"'
    elif isinstance(mixtureTrysCount, int):
        mixtureTrysCount = {label:mixtureTrysCount for label in label_names_surrogate}
    else: assert isinstance(mixtureTrysCount, dict) and len(mixtureTrysCount) == len(label_names_surrogate) and set(mixtureTrysCount.keys()) == set(label_names_surrogate)

    # =================== generate mixtures ===================
    if componentCount == 1:
        if allCombinationsOfSingles or nolabels:
            assert mixtureTrysCount['default'] <= n, f"{mixtureTrysCount['default']} > {n}"
            componentInds = {'default': np.random.choice(n, mixtureTrysCount['default'], replace=False)}
        else:
            componentInds = {label: generateUniformLabelDistrib(mixtureTrysCount[label], componentCount=1, uniformLabelName=label, sample=singleComponentData).reshape(-1) for label in label_names_surrogate}
        mixtureData = {label: singleComponentData.takeRows(componentInds[label]) for label in label_names_surrogate}
        if componentNameColumn is not None:
            componentNames = {label: singleComponentData.params.loc[componentInds[label],componentNameColumn].to_numpy() for label in label_names_surrogate}
        else:
            componentNames = {label: np.array([str(i) for i in componentInds[label]]) for label in label_names_surrogate}
    else:
        mixtureData = {}; componentLabels = {}; concentrations = {}; componentNames = {}
        for label in label_names_surrogate:
            makeUniformLabelDistrib = None if allCombinationsOfSingles or nolabels else label
            mixtureData[label], componentLabels[label], concentrations[label], componentNames[label] = generateMixtureOfSample(mixtureTrysCount[label], componentCount, singleComponentData, labelNames, addDescrFunc=None, makeMixtureOfSpectra=makeMixtureOfSpectra, makeMixtureOfLabel=None, dirichletAlpha=1, randomSeed=randomSeed, returnNotMixLabels=True, componentNameColumn=componentNameColumn, makeUniformLabelDistrib=makeUniformLabelDistrib)
            assert np.all(np.isfinite(concentrations[label]))

    # =================== concentration optimization ===================
    distsToExp = {label: np.zeros(mixtureTrysCount[label]) for label in label_names_surrogate}
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
        assert isinstance(mixtureSpectrum, dict), str(mixtureSpectrum)
        for spType in mixtureSpectrum:
            assert isinstance(mixtureSpectrum[spType], utils.Spectrum), str(mixtureSpectrum[spType])
        if unknownCharacterization['type'] == 'distance function':
            # print(mixtureSpectrum)
            return unknownCharacterization['function'](mixtureSpectrum, mixtureDescriptors)
        elif unknownCharacterization['type'] == 'spectrum':
            spType = unknownCharacterization['spType'] if 'spType' in unknownCharacterization else singleComponentData.getDefaultSpType()
            assert np.all(np.isfinite(mixtureSpectrum[spType].y))
            unk_spectrum = unknownCharacterization['spectrum']
            assert isinstance(unk_spectrum, utils.Spectrum), str(unk_spectrum)
            energy = singleComponentData.getEnergy(spType)
            assert len(energy)>0
            rFactorParams = unknownCharacterization.get('rFactorParams', {})
            return utils.rFactorSp(mixtureSpectrum[spType], unk_spectrum, **rFactorParams)
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
        spectra = {spType: singleComponentData.getSpectra(spType).to_numpy() for spType in singleComponentData.spTypes()}
        if componentNameColumn is not None:
            assert singleComponentData.params.dtypes[componentNameColumn] == object, str(singleComponentData.params.dtypes[componentNameColumn]) + ' != object (str)'
            componentNameData = singleComponentData.params[componentNameColumn].to_numpy()
        for label in label_names_surrogate:
            for i in range(mixtureTrysCount[label]):
                if componentNameColumn is None:
                    componentInds = [int(cn) for cn in componentNames[label][i]]
                else:
                    componentInds = np.array([np.where(componentNames[label][i,j] == componentNameData)[0][0] for j in range(componentCount)])

                def getComponents():
                    assert makeMixtureOfSpectra is None
                    res = {}
                    for spType in spectra:
                        res[spType] = spectra[spType][componentInds, :]
                    return res

                def makeMixture(concentrations):
                    assert np.all(np.isfinite(concentrations))
                    components = getComponents()
                    mix_spectra = {}
                    if makeMixtureOfSpectra is None:
                        for spType in spectra:
                            mix_spectra[spType] = np.dot(components[spType].T, concentrations)
                            assert np.all(np.isfinite(mix_spectra[spType])), f'concentrations = {concentrations}'
                    else:
                        ms = makeMixtureOfSpectra(componentInds, concentrations)
                        if isinstance(ms, dict):
                            mix_spectra = ms
                            assert len(ms[singleComponentData.getDefaultSpType()]) == len(singleComponentData.getEnergy())
                        else:
                            assert len(spectra) == 1
                            spType = list(spectra.keys())[0]
                            assert len(ms) == len(singleComponentData.getEnergy(spType))
                            mix_spectra[spType] = ms
                    # Add energy to mix spectra to avoid errors with energy intervals
                    for spType in mix_spectra:
                        e = singleComponentData.getEnergy(spType)
                        mix_spectra[spType] = utils.Spectrum(e,mix_spectra[spType])
                    if calcDescriptorsForSingleSpectrumFunc is None:
                        return mix_spectra, None
                    else:
                        # print('concentrations = ', concentrations)
                        mix_spectra1 = mix_spectra if len(mix_spectra) > 1 else mix_spectra[list(mix_spectra.keys())[0]]
                        mixtureDescriptors = calcDescriptorsForSingleSpectrumFunc(mix_spectra1)
                        return mix_spectra, mixtureDescriptors
                old_c = copy.deepcopy(concentrations[label][i])
                # c = copy.deepcopy(concentrations[label][i]) if optimizeConcentrationsTrysCount == 1 else None
                old_func_value = distToUnknown(makeMixture(old_c))
                rFactorParams = unknownCharacterization.get('rFactorParams',{})
                if unknownCharacterization['type'] == 'spectrum' and set(rFactorParams.keys())<={'interval'} and makeMixtureOfSpectra is None:
                    components = getComponents()
                    spType = unknownCharacterization['spType']
                    sample_e = singleComponentData.getEnergy(spType)
                    unk = unknownCharacterization['spectrum']
                    interval = utils.intervalIntersection(sample_e[[0,-1]], unk.x[[0,-1]])
                    assert interval is not None, f'{sample_e[[0,-1]]} doesn\'t intersect with {unk.x[[0,-1]]}'
                    if 'interval' in rFactorParams:
                        assert len(rFactorParams['interval']) == 2
                        interval = utils.intervalIntersection(interval, rFactorParams['interval'])
                        assert interval is not None, f'{interval} doesn\'t intersect with {rFactorParams["interval"]}'
                    ind = (interval[0]<=sample_e) & (sample_e<=interval[1])
                    unk = unk.changeEnergy(sample_e[ind])
                    distsToExp[label][i], concentrations[label][i] = findConcentrationsNNLS(sample_e[ind], components[spType][:,ind], unk.y)
                    # tmp = distToUnknown(makeMixture(concentrations[label][i]))
                    # tmp != distsToExp[label][i] because here we interpolate unknown, but in utils.rFactor - we interpolate theory
                    # assert abs(tmp-distsToExp[label][i]) < 1e-5, f'{tmp} {distsToExp[label][i]}'
                else:
                    distsToExp[label][i], concentrations[label][i] = findConcentrationsAbstract(distToUnknown, componentCount, makeMixture, fixConcentrations=None, trysGenerateMixtureOfSampleCount=optimizeConcentrationsTrysCount)
                if debug: print(f'Were fun = {old_func_value}, conc = {old_c}. Now fun = {distsToExp[label][i]}, conc = {concentrations[label][i]}')
                # fill new mixture spectrum
                mix_spectra = makeMixture(concentrations[label][i])
                if isinstance(mix_spectra, tuple): mix_spectra = mix_spectra[0]
                for spType in mixtureData[label].spTypes():
                    mde = mixtureData[label].getEnergy(spType)
                    assert len(mde) == len(mix_spectra[spType].x), f'{len(mde)} != {len(mix_spectra[spType].x)}.\nmixtureData.energy={mde}\ncomponents energy = {singleComponentData.getEnergy(spType)}'
                    mixtureData[label].getSpectra(spType).loc[i] = mix_spectra[spType].y
                mixtureData[label].params.loc[i,:] = np.nan
                for l in labelNames:
                    lv = np.sum([singleComponentDataDecoded.params.loc[componentInds[ic],l]*concentrations[label][i,ic] for ic in range(componentCount)])
                    mixtureData[label].params.loc[i, l] = lv
                # !!!!!!!!!! Attention !!!!!!!!  Mixture descriptors we can't update, because calcDescriptorsForSingleSpectrumFunc can return not all descriptors. Also they were not calculated, because addDescrFunc=None in generateMixtureOfSample
    else:
        for label in label_names_surrogate:
            for i in range(len(distsToExp[label])):
                # ind=i - is correct, because mixtureData was constructed as sample.take(componentInds)
                mixtureSpectrum = mixtureData[label].getSpectrum(ind=i, spType='all types', returnIntensityOnly=False)
                mixtureDescriptors = calcDescriptorsForSingleSpectrumFunc(mixtureSpectrum) if calcDescriptorsForSingleSpectrumFunc is not None else None
                distsToExp[label][i] = distToUnknown((mixtureSpectrum, mixtureDescriptors))

    # =================== combine data for different labels ===================
    def combine(dataDict):
        dim = len(dataDict[label_names_surrogate[0]].shape)
        if dim == 1:
            return np.concatenate(tuple(dataDict[label] for label in label_names_surrogate)).reshape(-1)
        elif dim == 2:
            return np.vstack(tuple(dataDict[label] for label in label_names_surrogate))
        else:
            assert dim == 3
            return np.concatenate(tuple(dataDict[label] for label in label_names_surrogate), axis=1)
    distsToExp = combine(distsToExp)
    if componentCount == 1:
        componentInds = combine(componentInds)
    else:
        componentLabels = combine(componentLabels)
        concentrations = combine(concentrations)
    componentNames = combine(componentNames)
    for il,label in enumerate(label_names_surrogate):
        md = mixtureData[label].copy()
        if label == label_names_surrogate[0]:
            mixtureData1 = md
        else:
            if md.nameColumn is not None:
                for i in range(len(md)):
                    md.params.loc[i,md.nameColumn] += f'_{il}'
            mixtureData1.unionWith(md, inplace=True)
    mixtureData = mixtureData1
    if componentCount == 1:
        for l in singleComponentData.labelMaps:
            mixtureData.decode(l)
    sorted_order = np.argsort(distsToExp)

    # =================== result building ===========================
    results = []
    for ii in sorted_order:
        if componentCount == 1: i = componentInds[ii]
        else: i = ii
        result = {'distsToExp': distsToExp[ii], 'spectrum': mixtureData.getSpectrum(ii, spType='all types'), 'componentNames': componentNames[ii], 'index':i, 'params': mixtureData.params.loc[ii].to_dict()}
        if componentCount > 1: result['concentrations'] = concentrations[i]
        else: result['concentrations'] = np.ones(1)
        results.append(result)

    # =================== result spectra plotting ===================
    if customPlotter is not None: customPlotter(results, locals())
    else:
        if plotSpectrumType is None and unknownCharacterization['type'] in ['spectrum', 'distance function']:
            plotSpectrumType = unknownCharacterization['spType'] if 'spType' in unknownCharacterization else singleComponentData.getDefaultSpType()
        if plotSpectrumType == 'all': plotSpectrumTypes = singleComponentData.spTypes()
        else: plotSpectrumTypes = [plotSpectrumType]

        def getUnknown(spType):
            energy = mixtureData.getEnergy(spType)
            unknownSpectrum = None
            if unknownSpectrumToPlot is None:
                if unknownCharacterization['type'] == 'spectrum' and unknownCharacterization['spType'] == spType:
                        unknownSpectrum = unknownCharacterization['spectrum'].y
                        unknownEnergy = unknownCharacterization['spectrum'].x
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
            if unknownSpectrum is None: return None
            return utils.Spectrum(unknownEnergy, unknownSpectrum)

        for ii in range(len(distsToExp)):
            if ii >= maxSpectraCountToPlot: break
            result = results[ii]
            for spType in plotSpectrumTypes:
                unknownSpectrum = getUnknown(spType)
                if unknownSpectrum is None: break
                d, cNames, conc, spectrum = result['distsToExp'], result['componentNames'], result['concentrations'], result['spectrum'][spType]
                if componentCount > 1:
                    conc_string = ', '.join([f'C_{name}={bestConc:.2f}' for (name, bestConc) in zip(cNames.reshape(-1).tolist(), conc.reshape(-1).tolist())])
                    title = f'Concentrations: {conc_string}, distsToExp: {d:.4f}'
                else:
                    title = f'Candidate {cNames}. DistsToExp: {d:.4f}'
                spectraFolder = folder + os.sep + 'spectra' + spectraFolderPostfix
                if not os.path.exists(spectraFolder):
                    os.makedirs(spectraFolder)
                fileName = spectraFolder + os.sep + f'{utils.zfill(ii,len(distsToExp))}_{spType}' + fileNamePostfix + '.png'
                if plotWrapperGenerator is not None:
                    plotWrapper = plotWrapperGenerator(spectrum, unknownSpectrum)
                else: plotWrapper = None
                def extraPlot(ax: plt.Axes):
                    rFactorParams = unknownCharacterization.get('rFactorParams')
                    interval = rFactorParams.get('interval',None)
                    if interval is not None:
                        ax.axvspan(xmin=interval[0], xmax=interval[-1], color='grey', alpha=0.2, zorder=-1)
                if not doNotPlot:
                    plotting.plotToFile(spectrum.x, spectrum.y, 'mixture', unknownSpectrum.x, unknownSpectrum.y, 'experiment', title=title, fileName=fileName, plotMoreFunction=extraPlot, plotWrapper=plotWrapper, showInNotebook=showInNotebook)

    # =================== mixture label map plotting ===================
    if componentCount > 1 and not doNotPlot and not nolabels:
        if labelBounds is None:
            labelBounds = [[np.min(singleComponentData.params[label]), np.max(singleComponentData.params[label])] for label in labelNames]
        plotMixtureLabelMap(componentLabels=componentLabels, label_names=labelNames, label_bounds=labelBounds, labelMaps=labelMaps, distsToExp=distsToExp, concentrations=concentrations, componentNames=componentNames, folder=folder + os.sep + 'label_maps' + labelMapsFolderPostfix, fileNamePostfix=fileNamePostfix, showInNotebook=showInNotebook)
    return results


def tryAllMixturesParser(folder):
    unk_names = os.listdir(folder)
    unk_names = list(set(unk_names) - {'CV'})
    d = pd.DataFrame(index=unk_names)
    for name in unk_names:
        lm_folder = folder+os.sep+name+os.sep+'label_maps'
        assert os.path.exists(lm_folder)
        bm_files = utils.findFile(mask='best_mix_*.txt', folder=lm_folder, returnAll=True)
        for fn in bm_files:
            with open(fn) as file: s = file.read()
            line = s.split('\n\n')[1].split('\n')[0]
            i = line.find(': ')
            dist = float(line[:i])
            line = line[i+2:]
            parts = line.split(' + ')
            comp_info = [None]*2
            for ip,part in enumerate(parts):
                compName, conc = part.split('*')
                i = compName.rfind('(')
                l = compName[i+1:-1]
                if utils.is_str_float(l): l = float(l)
                compName = compName[:i]
                comp_info[ip] = dict(name=compName, label=l, conc=float(conc))
            label = os.path.split(fn)[-1][len('best_mix_'):-4]
            labelPrediction = comp_info[0]['label']*comp_info[0]['conc'] + comp_info[1]['label']*comp_info[1]['conc']
            d.loc[name, 'distance'] = dist 
            d.loc[name, f'{label} prediction'] = labelPrediction
            for ip in [0,1]:
                d.loc[name, f'concentration {ip+1}'] = comp_info[ip]['conc']
                d.loc[name, f'component {ip+1} name'] =comp_info[ip]['name']
                d.loc[name, f'component {ip+1} {label}'] = comp_info[ip]['label']
    return d
            

def tryAllMixturesCV(**tryAllMixturesParams):
    singleComponentData = tryAllMixturesParams['singleComponentData']
    label_names = tryAllMixturesParams['labelNames']
    ignoreNotOrdinalLabels = tryAllMixturesParams.get('ignoreNotOrdinalLabels', False)
    componentCount = tryAllMixturesParams['componentCount']
    unknownCharacterization = tryAllMixturesParams['unknownCharacterization']
    folder0 = tryAllMixturesParams['folder']
    if ignoreNotOrdinalLabels and componentCount>1:
        label_names = [l for l in label_names if singleComponentData.isOrdinal(l)]
    for label in label_names:
        known0, _ = singleComponentData.splitUnknown(columnNames=[label])
        trueLab = known0.params[label]
        predLab = [None]*len(known0)
        isClassification = ML.isClassification(known0.params, label)
        if componentCount == 2: isClassification = False
        if isClassification: trueLab = trueLab.astype(int)
        for i in range(len(known0)):
            name = known0.params.loc[i, known0.nameColumn]
            exp = known0.getSpectrum(i, spType='all types')
            known = known0.takeRows(np.arange(len(known0)) != i)
            tryAllMixturesParams1 = copy.deepcopy(tryAllMixturesParams)
            tryAllMixturesParams1['singleComponentData'] = known
            unknownCharacterization1 = copy.deepcopy(unknownCharacterization)
            assert unknownCharacterization1['type'] == 'spectrum'
            spType = unknownCharacterization1['spType']
            unknownCharacterization1['spectrum'] = exp[spType]
            tryAllMixturesParams1['unknownCharacterization'] = unknownCharacterization1
            tryAllMixturesParams1['labelNames'] = [label]
            tryAllMixturesParams1['folder'] = folder0+os.sep+label+os.sep+name
            tryAllMixturesParams1['doNotPlot'] = True
            res = tryAllMixtures(**tryAllMixturesParams1)
            p = res[0]['params'][label]
            if isClassification and label in known0.labelMaps:
                predLab[i] = known0.labelMaps[label][p]
            else:
                if isClassification: predLab[i] = int(p)
                else: predLab[i] = p
        result = pd.DataFrame()
        result[known0.nameColumn] = known0.params[known0.nameColumn]
        if isClassification and label in known0.labelMaps:
            trueLab = known0.decode(label=label, values=trueLab)
            predLab = known0.decode(label=label, values=predLab)
        else:
            if label in known0.labelMaps: trueLab = known0.decode(label=label, values=trueLab)

        os.makedirs(f'{folder0}{os.sep}{label}', exist_ok=True)
        stat = open(f'{folder0}{os.sep}{label}{os.sep}stat.txt', 'w')
        stat.write(str(ML.calcAllMetrics(trueLab, np.array(predLab), isClassification)) + '\n')
        stat.write(str(known0.params[known0.nameColumn].tolist())+'\n')
        stat.write(f'true: {trueLab.tolist()}\n')
        stat.write(f'pred: {predLab}\n')
        result['true'] = trueLab
        result['pred'] = predLab
        if isClassification and label in known0.labelMaps:
            stat.write(f'Confusion matrix order: {sorted(list(set(trueLab) | set(predLab)))}\n')
            stat.write(str(sklearn.metrics.confusion_matrix(trueLab,predLab))+'\n')
        stat.close()
        result.to_csv(f'{folder0}{os.sep}{label}{os.sep}predictions.csv', index=False, sep=';')
        if utils.isJupyterNotebook():
            print('Label:', label)
            print(result)


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
                

def findGlobalMinimumMixture(distToExperiment, spectraFuncs, makeMixture, trysCount, bounds, paramNames, componentNames=None, constraints=None, folderToSaveResult=None, fixParams=None, contourMapCalcMethod='fast', plotContourMaps='all', extraPlotFuncContourMaps=None):
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


class LabelValuesLackError(Exception):
    pass


def generateUniformLabelDistrib(size, componentCount, uniformLabelName, sample):
    all_ind = np.zeros(shape=(size, componentCount), dtype=int)
    label = sample.params.loc[:, uniformLabelName].to_numpy()
    if ML.isClassification(sample.params, uniformLabelName):
        label_values = np.unique(label)
        # print(label_values, label)
        inds_by_label_values = {lv:np.where(label == lv)[0] for lv in label_values}
        if componentCount > len(label_values):
            raise LabelValuesLackError(f'componentCount = {componentCount} > len(label_values) = {len(label_values)}')
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
                # take random spectrum from the set with same label
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
        componentLabels: array of label tables (for each component - one table) with size componentCount x size x labelCount (if label in labelMaps, than decoded label is returned!)
        concentrations: 2d array size x componentCount (sorted in descendant order!!!!!!)
        componentNames: 2d array size x componentCount
    :param componentNameColumn: used when labels are not mixed (if None - use index)
    :param makeUniformLabelDistrib: name of label to have uniform result distribution (in labelValues^componentCount space) in mixture sample. If None - the mixture is uniform in terms of single component indices
    :return: new mixture sample with mixed labels and separate info about component labels if returnNotMixLabels=True
    """
    # print('Do not use CV for one mixture sample. Instead divide pure sample into 2 parts and generate 2 mixture samples: train and test.')
    sample = sample.copy()
    for l in label_names:
        if l in sample.labelMaps: sample.decode(label=l)
        assert ML.isOrdinal(sample.params, l), f'Can\'t mix not ordinal label {l}'
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
                all_ind = index_combinations[np.random.choice(int(maxSize), size=size, replace=False), :]
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
    mix_sample = ML.Sample(mix_data, mix_spectra, energy=sample._energy, meta={'labels':label_names})
    goodSpectrumIndices = None
    if addDescrFunc is not None:
        res = addDescrFunc(mix_sample)
        if isinstance(res, tuple): mix_sample, goodSpectrumIndices = res
        else: mix_sample = res
    if goodSpectrumIndices is None: goodSpectrumIndices = np.arange(len(mix_data))
    mix_sample.paramNames = mix_sample.params.columns.to_numpy()
    if addDescrFunc is not None and set(mix_sample.paramNames) != set(sample.paramNames):
        assert set(mix_sample.paramNames) <= set(sample.paramNames), 'Set of initial sample and mixture descriptors are not equal. There are extra mixture descriptors: ' + str(set(mix_sample.paramNames)-set(sample.paramNames))+'.\nMay be you forget to list all labels in generateMixtureOfSample call?'
        # check dtype
        forgotten = set(sample.paramNames)-set(mix_sample.paramNames)
        for f in forgotten:
            assert sample.params[f].dtype != np.float64, f'Set of initial sample and mixture descriptors are not equal. Param {f} with dtype=float64 is absent in mixture data.\nInitial sample params: {list(sample.paramNames)}\nMixture sample params: {list(mix_sample.paramNames)}'
    if len(sample.features) > 0:
        mix_sample.features = list(set(sample.features) & set(mix_sample.paramNames))
    if returnNotMixLabels:
        return mix_sample, componentLabels[:,goodSpectrumIndices,:], concentrations[goodSpectrumIndices,:], componentNames[goodSpectrumIndices,:]
    else:
        return mix_sample


def fit_mixture_models(singleComponentSample:ML.Sample, features, mixSampleSize, label, label_names, randomSeed, makeMixtureParams, model_regr, isClassification, model_class=None):
    singleComponentSample = singleComponentSample.copy()
    if set(singleComponentSample.labels) >= set(label_names):
        # delete unused labels
        for l in set(singleComponentSample.labels) - set(label_names):
            singleComponentSample.delParam(l)
    mixSampleTrain, componentLabelsTrain, concentrationsTrain, _ = generateMixtureOfSample(size=mixSampleSize, sample=singleComponentSample, label_names=label_names, randomSeed=randomSeed, returnNotMixLabels=True, makeUniformLabelDistrib=label, **makeMixtureParams)
    i_label = label_names.index(label)
    componentLabelsTrain = componentLabelsTrain[:, :, i_label].T
    from . import descriptor
    X, trueAvgLabels = descriptor.getXYFromSample(mixSampleTrain, features, label)
    model_regr_avgLabel = copy.deepcopy(model_regr)
    model_regr_avgLabel.fit(X, trueAvgLabels)
    model_regr_conc = sklearn.multioutput.MultiOutputRegressor(copy.deepcopy(model_regr))
    model_regr_conc.fit(X, concentrationsTrain)
    if isClassification:
        assert model_class is not None
        model_comp_labels = copy.deepcopy(model_class)
        if not isinstance(model_class, sklearn.ensemble.ExtraTreesClassifier):
            model_comp_labels = sklearn.multioutput.MultiOutputClassifier(model_comp_labels)
    else:
        model_comp_labels = copy.deepcopy(model_regr)
        if not isinstance(model_regr, sklearn.ensemble.ExtraTreesRegressor):
            model_comp_labels = sklearn.multioutput.MultiOutputRegressor(model_comp_labels)
    model_comp_labels.fit(X, componentLabelsTrain)
    return model_regr_avgLabel, model_regr_conc, model_comp_labels


def score_cv(model_regr, sample:ML.Sample, features, label, label_names, makeMixtureParams, testRatio=0.2, repetitions=5, model_class=None, random_state=0):
    assert 'sample' not in makeMixtureParams
    assert 'size' not in makeMixtureParams
    assert 'randomSeed' not in makeMixtureParams
    assert 'returnNotMixLabels' not in makeMixtureParams
    assert 'makeUniformLabelDistrib' not in makeMixtureParams
    def predict_with_fix(model, data):
        r = model.predict(data)
        if len(r.shape) == 3 and r.shape[0] == 1 and r.shape[1] == data.shape[0]:
            r = r[0]
        return r
    n = sample.getLength()
    testSize = n  # int(n * testRatio)
    trainSize = n  # int(n * (1-testRatio))
    sample = sample.copy()
    if set(sample.labels) >= set(label_names):
        # delete unused labels
        for l in set(sample.labels) - set(label_names):
            sample.delParam(l)
    trueAvgLabels, predAvgLabels = None, None
    trueComponentLabels, predComponentLabels = None, None
    trueConcentrations, predConcentrations = None, None
    unq = 8461047254
    singleTrue, singlePred = np.zeros(n)+unq, np.zeros(n)+unq
    i_label = label_names.index(label)
    isClassification = ML.isClassification(sample.params[label])
    all_indexes = np.arange(n)
    test_size = int(np.round(testRatio*n))
    not_used_indexes = copy.deepcopy(all_indexes)
    rng = np.random.default_rng(random_state)
    rng.shuffle(not_used_indexes)
    ir = 0
    while ir < repetitions or np.any(singleTrue == unq):
        indTest = not_used_indexes[:test_size]
        if len(indTest) < test_size:
            not_used_indexes = copy.deepcopy(all_indexes)
            not_used_indexes = np.setdiff1d(not_used_indexes, indTest)
            rng.shuffle(not_used_indexes)
            remain = test_size-len(indTest)
            indTest = np.append(indTest, not_used_indexes[:remain])
            not_used_indexes = not_used_indexes[remain:]
        else:
            not_used_indexes = not_used_indexes[test_size:]
        indTrain = np.setdiff1d(all_indexes, indTest)
        # indTrain, indTest = sklearn.model_selection.train_test_split(np.arange(n).reshape(-1,1), test_size=testRatio, shuffle=True)
        sampleTrain = sample.takeRows(indTrain)
        sampleTest = sample.takeRows(indTest)
        model_regr_avgLabel, model_regr_conc, model_comp_labels = fit_mixture_models(singleComponentSample=sampleTrain, features=features, mixSampleSize=trainSize, label=label, label_names=label_names, randomSeed=rng.integers(np.iinfo(np.int32).max, dtype=np.int32), makeMixtureParams=makeMixtureParams, model_regr=model_regr, isClassification=isClassification, model_class=model_class)

        # for cv picture for exp known spectra
        from . import descriptor
        singleX, singleY = descriptor.getXYFromSample(sampleTest, features, label)
        singleTrue[indTest] = singleY.flatten()
        singlePred[indTest] = model_regr_avgLabel.predict(singleX).flatten()

        mixSampleTest, componentLabelsTest, concentrationsTest, _ = generateMixtureOfSample(size=testSize, sample=sampleTest, label_names=label_names, randomSeed=rng.integers(np.iinfo(np.int32).max, dtype=np.int32), returnNotMixLabels=True, makeUniformLabelDistrib=label, **makeMixtureParams)
        componentLabelsTest = componentLabelsTest[:,:,i_label].T
        xTest, yTest = descriptor.getXYFromSample(mixSampleTest, features, label)

        if predAvgLabels is None: predAvgLabels = predict_with_fix(model_regr_avgLabel, xTest)
        else: predAvgLabels = np.append(predAvgLabels, predict_with_fix(model_regr_avgLabel, xTest), axis=0)
        if trueAvgLabels is None: trueAvgLabels = yTest
        else: trueAvgLabels = np.append(trueAvgLabels, yTest, axis=0)

        if predConcentrations is None: predConcentrations = predict_with_fix(model_regr_conc, xTest)
        else: predConcentrations = np.append(predConcentrations, predict_with_fix(model_regr_conc, xTest), axis=0)

        if trueConcentrations is None: trueConcentrations = concentrationsTest
        else: trueConcentrations = np.append(trueConcentrations, concentrationsTest, axis=0)
        assert np.all(predConcentrations.shape == trueConcentrations.shape), str(predConcentrations.shape)+' != '+str(trueConcentrations.shape)

        if predComponentLabels is None: predComponentLabels = predict_with_fix(model_comp_labels, xTest)
        else: predComponentLabels = np.append(predComponentLabels, predict_with_fix(model_comp_labels, xTest), axis=0)

        if trueComponentLabels is None: trueComponentLabels = componentLabelsTest
        else: trueComponentLabels = np.append(trueComponentLabels, componentLabelsTest, axis=0)
        ir += 1

    filled_i = singleTrue!=unq
    assert np.all(singleTrue[filled_i] == sample.params.loc[filled_i, label]), f'{singleTrue[filled_i].tolist()} !=\n{sample.params.loc[filled_i, label].tolist()}\nUnique: {np.unique(singleTrue[filled_i], return_counts=True)}\n{np.unique(sample.params.loc[filled_i, label], return_counts=True)}'
    conc_quality = [ML.calcAllMetrics(trueConcentrations[:,j], predConcentrations[:,j], classification=False) for j in range(trueConcentrations.shape[1])]
    comp_quality = []
    for j in range(trueComponentLabels.shape[1]):
        comp_quality.append(ML.calcAllMetrics(trueComponentLabels[:,j], predComponentLabels[:,j], isClassification))
    qualities = {'avgLabels': ML.calcAllMetrics(trueAvgLabels, predAvgLabels, classification=False), 'concentrations': conc_quality, 'componentLabels': comp_quality}
    trueVals = {'avgLabels': trueAvgLabels, 'concentrations': trueConcentrations, 'componentLabels': trueComponentLabels}
    predVals = {'avgLabels': predAvgLabels, 'concentrations': predConcentrations, 'componentLabels': predComponentLabels}
    singleCV = dict(singleTrue=singleTrue, singlePred=singlePred)
    return qualities, trueVals, predVals, singleCV


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


def mcr(spectra, paramValues, energy, method, methodParams=None, initialComponents=None, componentCount=None, folder=None, componentNames=None, plotFits=False):
    """
    :param spectra: spectrum matrix (each column is one spectrum)
    :param paramValues: one param value corresponds to one spectrum
    :param energy: energy
    :param method: 'McrAR' or 'FastICA'
    :param initialComponents: for McrAR - initial spectra of components to start optimization (each column is one spectrum)
    :param componentCount: for FastICA
    :param folder: folder to save result
    :returns: dict{'error':float, 'components': matrix (each column is one spectrum), 'concentrations': matrix (each row is concentration of one component)
    """
    assert len(spectra.shape) == 2
    assert spectra.shape[0] == len(energy)
    if folder is not None:
        assert spectra.shape[1] == len(paramValues)
    assert method in ['McrAR', 'FastICA']
    if componentNames is None: componentNames = [str(i+1) for i in range(spectra.shape[1])]
    if folder is not None:
        # plot PCA
        pca = sklearn.decomposition.PCA(n_components=3, random_state=0)
        mixing = pca.fit_transform(spectra.T)
        paramValuesStr = [str(p) for p in paramValues]
        plotting.scatter(mixing[:, 0], mixing[:, 1], color=mixing[:, 2], marker_text=paramValuesStr, fileName=f'{folder}/pca.png')
        # TODO: создать виджет, в котором пользователь может задавать число компонент и двигать точки на плоскости PCA.
        #  В смеси всегда есть главное изменение концентрации - это та линия, вдоль которой вытянут график PCA больше всего. Можно просто взять PCA1. Одна из ее крайних точек - это хорошее начальное приближение для доминирующей компоненты (какой край, можно понять по стороне к которой сужается область вариации проекции точек на ортогональное к главное компоненте измерение). Конечно, есть неопределенность - на концах вещества не обязательно чистые, поэтому пол-ль должен иметь возможность их растянуть (но не сблизить - иначе получим отрицательные концентрации).
        # Чтобы откорректировать наклон линии, нужно иметь возможность выбирать спектры из выборки для задания линии или сдвигать точки вдоль всех трех PCA-компонент.
        #  Обобщение на 3 компоненты: можно сдвигать точки в направлении первой компоненты, второй и третьей (высота).
    if method == 'McrAR':
        assert initialComponents is not None
        assert len(initialComponents.shape) == 2
        assert spectra.shape[0] == initialComponents.shape[0]
        if methodParams is None:
            methodParams = dict(tol_increase=0.1, max_iter=100, tol_n_above_min=5000)
        componentCount = initialComponents.shape[1]
        from pymcr.mcr import McrAR, _logger
        from pymcr.regressors import NNLS, OLS
        from pymcr.constraints import ConstraintNonneg, ConstraintNorm
        _logger.setLevel(logging.ERROR)
        _logger.propagate = False
        _logger.disabled = True
        mcrar = McrAR(c_regr=NNLS(), st_regr=OLS(), c_constraints=[ConstraintNonneg(), ConstraintNorm()], st_constraints=[], **methodParams)
        mcrar.fit(spectra.T, ST=copy.deepcopy(initialComponents).T, verbose=False)
        finalComponents = mcrar.ST_opt_.T  # each column is one spectrum
        concentrations = mcrar.C_opt_  # each column is concentrations for one component
        error = mcrar.err[-1]
    else:
        assert componentCount is not None
        if methodParams is None: methodParams = dict()
        ica = sklearn.decomposition.FastICA(n_components=componentCount, **methodParams)
        transpose = True
        if transpose:
            concentrations = ica.fit_transform(spectra.T) # each column is concentrations for one component
            finalComponents = ica.mixing_  # finalComponents - each column is one spectrum
            print('finalComponents.shape =',finalComponents.shape)
            print('concentrations.shape =', concentrations.shape)
            if methodParams.get('whiten','unit-variance'): mean = ica.mean_
            else: mean = np.array(0)
            error = np.mean((spectra - (np.dot(finalComponents[:,:componentCount], concentrations[:,:componentCount].T) + mean.T.reshape(-1,1)))**2)
        else:
            finalComponents = ica.fit_transform(spectra)  # finalComponents - each column is one spectrum
            concentrations = ica.mixing_  # each column is concentrations for one component
            print('finalComponents.shape =',finalComponents.shape)
            print('concentrations.shape =', concentrations.shape)
            if methodParams.get('whiten','unit-variance'): mean = ica.mean_
            else: mean = np.array(0)
            error = np.mean((spectra - (np.dot(finalComponents[:,:componentCount], concentrations[:,:componentCount].T) + mean))**2)
        print('mean.shape =', mean.shape)
        
    if folder is not None:
        print(f'\nFinal MSE: {error:.7e}')
        toPlot0, toPlot1, allConc = tuple(), tuple(), tuple()
        for i in range(componentCount):
            toPlot_separate_comp = (energy, initialComponents[:,i], "initial") if method == 'McrAR' else tuple()
            toPlot_separate_comp += (energy, finalComponents[:,i], "result")
            plotting.plotToFile(*toPlot_separate_comp, fileName=f'{folder}/comp_{componentNames[i]}.png')
            if method == 'McrAR':
                toPlot0 += (energy, initialComponents[:,i].T, f"{componentNames[i]}")
            toPlot1 += (energy, finalComponents[:,i], f"{componentNames[i]}")
            allConc += (paramValues, concentrations[:, i], f"{componentNames[i]}")
        if method == 'McrAR':
            plotting.plotToFile(*toPlot0, fileName=f'{folder}/all_initial.png')
        plotting.plotToFile(*toPlot1, fileName=f'{folder}/all_result.png')
        plotting.plotToFile(*allConc, fileName=f'{folder}/concentrations.png')
        if plotFits:
            for j in range(len(paramValues)):
                fit = np.dot(finalComponents, concentrations[j, :])
                plotting.plotToFile(energy, fit, 'fit', energy, spectra[:,j], 'exact', fileName=f'{folder}/fits/{paramValues[j]}.png')
    assert np.all(np.isfinite(concentrations))
    assert np.all(np.isfinite(finalComponents))
    c = concentrations.T
    if np.max(np.abs(np.sum(c, axis=0)-1))>0.3:
        print('MCR failed. Concentrations don\'t sum to 1')
        return None
    return {'error':error, 'components':finalComponents, 'concentrations': c}
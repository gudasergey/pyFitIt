from . import utils, plotting, optimize, ML
import numpy as np
import pandas as pd
import scipy, os, shutil, itertools, copy, warnings
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
    assert len(xanesArray.shape) == 2
    n = xanesArray.shape[0]
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
        approxXanes = np.sum(xanesArray*c.reshape(-1,1), axis=0)
        return np.sqrt(utils.integral(energy, (approxXanes-expXanes)**2))

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
    result = scipy.optimize.minimize(func, np.ones(m-1)*upperBound/m, bounds=[[0,upperBound]]*(m-1), constraints=constrains)
    if not result.success:
        warnings.warn("scipy.optimize.minimize can't find optimum. Result = "+str(result))
    c = result.x
    c = expand(c)
    return result.fun, c


def calcAndPlotMixtureLabelMap(data, predictByFeatures, label_names, mix_features_func, concentrations, folder, use):
    """For every label plot 2d map (label x label) with color - minimal distance between unknown row and mixture of knowns

    :param data: data base of descriptor values
    :param predictByFeatures: list of predictors to use for distance calculation
    :param label_names: names of data columns which are labels
    :param mix_features_func: f(component_descriptors, concentrations) -> descriptors for mixture (component_descriptors[component][descr_ind])
    :param concentrations: array or component count - to find best
    :param folder: [description]
    :param use: 'exp&theory', 'unknown', 'exp'
    """
    nMean = data.loc[:, predictByFeatures].mean(axis=0).to_numpy()
    nStd = data.loc[:, predictByFeatures].std(axis=0).to_numpy()

    def normalize(x): return (x - nMean) / nStd

    unk_ind = np.isin(data.Ind, unknownExperiments)
    unknownData = data.loc[unk_ind].reset_index(drop=True)
    data = data.loc[~unk_ind]
    data.reset_index(drop=True, inplace=True)

    if os.path.exists(folder): shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

    labels = {}
    if use == 'unknown':
        m = np.min(data.loc[:, predictByFeatures], axis=0)
        M = np.max(data.loc[:, predictByFeatures], axis=0)
        m, M = m - 0.2 * (M - m), M + 0.2 * (M - m)
        linspaces = []
        for i in range(len(m)): linspaces.append(np.linspace(m[i], M[i], 10))
        grid = list(np.meshgrid(*linspaces))
        for i in range(len(grid)): grid[i] = grid[i].reshape(-1)
        descriptor_values = np.array(grid).T
        quality, predictions, models = getQuality(data, predictByFeatures, m=5, returnModels=True)
        for label in label_names:
            if label not in data.columns: continue
            labels[label] = models[label].predict(descriptor_values)
    elif use in ['exp&theory', 'exp']:
        ind = np.arange(data.shape[0]) if use == 'exp&theory' else data.Ind < 25
        descriptor_values = data.loc[ind, predictByFeatures].to_numpy()
        names = data.loc[ind, 'Ind'].to_numpy()
        for label in label_names:
            if label not in data.columns: continue
            labels[label] = data.loc[ind, label].to_numpy()
    else: assert False, 'Unknown use value'

    if isinstance(concentrations, int) and concentrations > 1:
        component_count = concentrations
        concentrations = None
        conc_search = True
    else:
        component_count = len(concentrations)
        conc_search = False
        assert abs(np.sum(concentrations) - 1) < 1e-6
    result = {}
    for unk_exp in unknownExperiments:
        if unk_exp not in result: result[unk_exp] = {}
        unk_desc = unknownData.loc[unknownData.Ind == unk_exp, predictByFeatures].to_numpy()
        for label in labels:
            if label not in result[unk_exp]:
                result[unk_exp][label] = [pd.DataFrame(columns=predictByFeatures + ['C']) for i in range(component_count)] + [pd.DataFrame(columns=predictByFeatures + ['norm'])]
            m = np.min(data[label])
            M = np.max(data[label])
            d = (M - m) / 10
            linspaces = []
            for i in range(component_count):
                if isClassification(data, label):
                    linspaces.append(np.unique(data[label]))
                else:
                    linspaces.append(np.linspace(m - d, M + d, 50))
            grid = list(np.meshgrid(*linspaces))
            grid_shape = grid[0].shape
            for i in range(len(grid)): grid[i] = grid[i].reshape(-1)
            label_grid = np.array(grid).T
            plot_norm = np.zeros(label_grid.shape[0]) + 100
            sl = np.sort(data[label])
            rad = np.mean(sl[1:] - sl[:-1]) * 2
            print(label, 'rad =', rad)

            if conc_search:
                result_conc = np.zeros(plot_norm.shape)
                result_components = np.zeros((plot_norm.shape[0], component_count), dtype=np.int)
            # permutations - because concentrations are sorted in descending order
            combinations = itertools.permutations(range(len(descriptor_values)), component_count)
            for ic in combinations:
                ic = list(ic)
                desc = descriptor_values[ic, :]
                lab = labels[label][ic]
                if conc_search:
                    assert component_count == 2, 'TODO'
                    large_conc = np.linspace(0.5, 1, 10)
                    concentrations = np.zeros(component_count)
                    nrms = np.zeros(large_conc.size)
                    for i in range(len(large_conc)):
                        concentrations[0] = large_conc[i]
                        concentrations[1] = 1 - concentrations[0]
                        mix_desc = mix_features_func(desc, concentrations)
                        nrms[i] = np.linalg.norm(normalize(unk_desc) - normalize(mix_desc))
                    best_i = np.argmin(nrms)
                    concentrations[0] = large_conc[best_i]
                    concentrations[1] = 1 - concentrations[0]
                mix_desc = mix_features_func(desc, concentrations)
                nrm = np.linalg.norm(normalize(unk_desc) - normalize(mix_desc))
                if conc_search:
                    ind = np.argmin(np.array([plot_norm, nrm + np.linalg.norm(lab - label_grid, axis=1) / rad]), axis=0)
                    result_conc[ind == 1] = concentrations[0]
                    for k in range(len(ic)):
                        result_components[ind == 1, k] = names[ic[k]]
                plot_norm = np.min(np.array([plot_norm, nrm + np.linalg.norm(lab - label_grid, axis=1) / rad]), axis=0)  # result[unk_exp][label][-1].append({'norm':nrm})  # result[unk_exp][label][-1].loc[-1,predictByFeatures] = mix_desc  # for i in range(component_count):  #     result[unk_exp][label][i].append({'C':concentrations[i]})  #     result[unk_exp][label][i].loc[-1,predictByFeatures] = desc[i]
            if label_grid.shape[1] != 2:
                assert False, 'TODO - take min by all other dimensions'
            fig, ax = plt.subplots(figsize=plotting.figsize)
            # colorMap = truncate_colormap('hsv', minval=0, maxval=np.max(plot_norm))
            minnorm = np.min(plot_norm);
            maxnorm = np.max(plot_norm)
            maxnorm = minnorm + 0.3 * (maxnorm - minnorm)  # 0.3*(maxnorm - minnorm)
            CF = ax.contourf(label_grid[:, 0].reshape(grid_shape), label_grid[:, 1].reshape(grid_shape), plot_norm.reshape(grid_shape), cmap='plasma', extend='both', vmin=minnorm, vmax=maxnorm)
            cbar = fig.colorbar(CF, ax=ax, extend='max', orientation='vertical')
            if label in labelMaps:
                cbarTicks = [None] * len(labelMaps[label])
                for name in labelMaps[label]:
                    cbarTicks[labelMaps[label][name]] = name
                ax.set_xticks(sorted(list(labelMaps[label].values())))
                ax.set_yticks(sorted(list(labelMaps[label].values())))
                ax.set_xticklabels(cbarTicks)
                ax.set_yticklabels(cbarTicks)
            if isClassification(data, label):
                n = len(np.unique(data[label]))
                for i in range(len(plot_norm)):
                    ax.scatter([label_grid[i, 0]], [label_grid[i, 1]], (300 / n) ** 2, c=[plot_norm[i]], cmap='plasma', vmin=minnorm, vmax=maxnorm, marker='s')
            if not conc_search:
                ax.set_xlabel(label + ', conc = ' + ('%.2f' % concentrations[0]))
                ax.set_ylabel(label + ', conc = ' + ('%.2f' % concentrations[1]))
            fig.savefig(folder + os.sep + str(unk_exp) + '_' + label + '.png', dpi=plotting.dpi)
            plt.close(fig)
            # save to file
            cont_data = pd.DataFrame()
            cont_data['1_' + label + '_' + ('%.2f' % concentrations[0])] = label_grid[:, 0]
            cont_data['2_' + label + '_' + ('%.2f' % concentrations[1])] = label_grid[:, 1]
            cont_data['norm'] = plot_norm
            if conc_search:
                cont_data['main_concentration'] = result_conc
                for k in range(component_count):
                    cont_data[f'component_{k}'] = result_components[:, k]
                best_mixtures = {}
                ind = np.argsort(plot_norm)
                i = 0
                while i < len(plot_norm):
                    ii = ind[i]
                    mixture = ''
                    for k in range(component_count):
                        mixture += f'{result_components[ii, k]}({label_grid[ii, k]})'
                        if k != component_count - 1: mixture += ' + '
                    if mixture not in best_mixtures: best_mixtures[mixture] = plot_norm[ii]
                    if len(best_mixtures) > 5: break
                    i += 1
                best_mixtures = [{'norm': best_mixtures[m], 'mixture': m} for m in best_mixtures]
                best_mixtures = sorted(best_mixtures, key=lambda p: p['norm'])
                with open(folder + os.sep + str(unk_exp) + '_' + label + '_best_mix.txt', 'w') as f:
                    f.write('norm: mixture = n1(label1) + n2(label2)\n')
                    for p in best_mixtures:
                        f.write(str(p['norm']) + ': ' + p['mixture'] + '\n')
            cont_data.to_csv(folder + os.sep + str(unk_exp) + '_' + label + '.csv', index=False)  # if label=='Type of ligands_new': exit(0)


def plotMixtureLabelMap(componentLabels, label_names, label_bounds, labelMaps, distsToExp, concentrations, componentNames, folder):
    """For every label plot 2d map (label x label) with color - minimal distance between unknown experiment and mixture of knowns

    :param componentLabels: array of label tables (for each component - one table) with size componentCount x spectraCount x labelCount
    :param label_names: list of names of labels
    :param label_bounds: list of pairs [label_min, label_max]
    :param labelMaps: strings for values of categorical labels. dict labelName:{valueName:value,...}
    :param distsToExp: distance to experiment for each row of componentLabels tables (spectra count values)
    :param concentrations: 2d array spectraCount x componentCount
    :param componentNames: 2d array spectraCount x componentCount
    :param folder: folder to save plots
    """

    os.makedirs(folder, exist_ok=True)
    component_count = concentrations.shape[1]
    assert np.all(np.abs(np.sum(concentrations, axis=1) - 1) < 1e-6), "Bad concentrations array: "+str(concentrations)
    assert len(label_bounds) == len(label_names)
    assert componentLabels.shape[-1] == len(label_names)
    assert componentLabels.shape[0] == component_count
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
        fig, ax = plt.subplots(figsize=plotting.figsize)
        # colorMap = truncate_colormap('hsv', minval=0, maxval=np.max(plot_norm))
        minnorm = np.min(plot_norm_2d)
        maxnorm = np.max(plot_norm_2d)
        maxnorm = minnorm + 1 * (maxnorm - minnorm)  # 0.3*(maxnorm - minnorm)
        CF = ax.contourf(label_grid_2d[0], label_grid_2d[1], plot_norm_2d, cmap='plasma', extend='both', vmin=minnorm, vmax=maxnorm)
        cbar = fig.colorbar(CF, ax=ax, extend='max', orientation='vertical')
        if label in labelMaps:
            cbarTicks = [None] * len(labelMaps[label])
            for name in labelMaps[label]:
                cbarTicks[labelMaps[label][name]] = name
            ax.set_xticks(sorted(list(labelMaps[label].values())))
            ax.set_yticks(sorted(list(labelMaps[label].values())))
            ax.set_xticklabels(cbarTicks)
            ax.set_yticklabels(cbarTicks)
        if isClassification:
            n = len(np.unique(all_label_values))
            lg0 = label_grid_2d[0].reshape(-1)
            lg1 = label_grid_2d[1].reshape(-1)
            pn = plot_norm_2d.reshape(-1)
            for i in range(len(pn)):
                if pn[i] == notExist:
                    ax.scatter([lg0[i]], [lg1[i]], (300 / n) ** 2, c='k', marker='s')
                else:
                    ax.scatter([lg0[i]], [lg1[i]], (300 / n) ** 2, c=[pn[i]], cmap='plasma', vmin=minnorm, vmax=maxnorm, marker='s')
        ax.set_xlabel(label)
        ax.set_ylabel(label)
        fig.savefig(folder + os.sep + 'map_' + label + '.png', dpi=plotting.dpi)
        plt.close(fig)
        # save to file
        cont_data = pd.DataFrame()
        cont_data['1_' + label] = label_grid[:, 0]
        cont_data['2_' + label] = label_grid[:, 1]
        cont_data['norm'] = plot_norm
        cont_data.to_csv(folder + os.sep + 'map_' + label + '.csv', index=False)  # if label=='Type of ligands_new': exit(0)
        for k in range(component_count):
            cont_data[f'concentration_{k}'] = result_conc[:,k]
            cont_data[f'component_{k}'] = result_components[:, k]
        cont_data.to_csv(folder + os.sep + 'map_' + label + '.csv', index=False)  # if label=='Type of ligands_new': exit(0)

        best_mixtures = {}
        ind = np.argsort(plot_norm)
        i = 0
        while i < len(plot_norm):
            ii = ind[i]
            mixture = ''
            for k in range(component_count):
                lab = label_grid[ii, k]
                if label in labelMaps:
                    lab = invLabelMaps[label][lab]
                mixture += f'{result_components[ii, k]}({lab})'
                if k != component_count - 1: mixture += ' + '
            if mixture not in best_mixtures: best_mixtures[mixture] = plot_norm[ii]
            if len(best_mixtures) > 5: break
            i += 1
        best_mixtures = [{'norm': best_mixtures[m], 'mixture': m} for m in best_mixtures]
        best_mixtures = sorted(best_mixtures, key=lambda p: p['norm'])
        with open(folder + os.sep + 'best_mix_' + label + '.txt', 'w') as f:
            f.write('norm: mixture = n1(label1) + n2(label2)\n')
            for p in best_mixtures:
                f.write(str(p['norm']) + ': ' + p['mixture'] + '\n')


def tryAllMixtures(unknownExp, componentCandidates, componentCount, energyRange, outputFolder, rFactorToleranceMultiplier=1.1, maxGraphNumToPlot=50, plotMixtureLabelMapParam=None):
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
            fig, ax = plt.subplots(1, figsize=(15, 10))        
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
            fig.savefig('mixtures' + os.sep + paramsNames[i] + '_' + paramsNames[j] + '.png')
                

def findGlobalMinimumMixture(distToExperiment, spectraFuncs, makeMixture, trysCount, bounds,  paramNames, componentNames=None, constraints=None, folderToSaveResult='globalMinimumSearchResult', fixParams=None, contourMapCalcMethod='fast', plotContourMaps='all', extra_plot_func=None):
    """
    Try several optimizations from random start point. Retuns sorted list of results and plot contours around minimum.

    :param distToExperiment: function which calculates distance to experimental spectrum distToExperiment(mixtureSpectrum, allArgs, *fun_args). allArgs = [[component1_params], ..., [componentN_params],[commonParams]]
    :param spectraFuncs: list of functions [func1, ...] which calculates component spectra by thier parameters. func1(component1_params, *commonParams)
    :param makeMixture: function, that creates mixture of spectra calculated by spectraFunc makeMixture(spectraList, concentraitions, *commonParams)
    :param trysCount: number of attempts to find minimum
    :param bounds: list of N+1 lists of 2-element lists with parameter bounds (component params and common params)
    :param paramNames: [[component1_params], ..., [componentN_params], [commonParams]]. Parameters of different components can have similar names. All the parameters will be prepend in result by componentNames+'_' prefixes
    :param componentNames: list of component names
    :param constraints: additional constrains for ‘trust-constr’ scipy.optimize.minimize method (for [notFixedConcentrations[:-1],allArgs].flattern argument)
    :param folderToSaveResult: all result graphs and log are saved here
    :param fixParams: dict of paramName:value to fix (component params must have prefix 'componentName_')
    :param contourMapCalcMethod: 'fast' - plot contours of the target function; 'thorough' - plot contours of the min of target function by all arguments except axes
    :param plotContourMaps: 'all' or list of 1-element or 2-elements lists of axes names to plot contours of target function
    :param extra_plot_func: user defined function to plot something on result contours: extra_plot_func(ax, axisNamesList)
    :return: sorted list of trysCount minimums of dicts with keys 'value', 'x' (have allArgs-like format), 'x1d' (flattened array of all param values), 'paramNames1d', 'concentrations', 'spectra', 'mixtureSpectrum'
    """
    N = len(spectraFuncs)
    if fixParams is None: fixParams = {}
    if constraints is None: constraints = tuple()
    oneComponent = len(spectraFuncs) == 1
    if len(bounds) == N: bounds.append([])
    if len(paramNames) == N: paramNames.append([])
    assert len(componentNames) == N
    assert len(bounds) == N+1
    assert len(paramNames) == N+1
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
    result1d = optimize.findGlobalMinimum(targetFunction, trysCount, bounds1d, constraints=constraints+cons, fun_args=None, paramNames=paramNames1d, folderToSaveResult=folderToSaveResult, fixParams=fixParams1, contourMapCalcMethod=contourMapCalcMethod, plotContourMaps=plotContourMaps, extra_plot_func=extra_plot_func)
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


def generateMixtureOfSample(size, componentCount, sample, label_names, addDescrFunc, makeMixtureOfSpectra=None, makeMixtureOfLabel=None, dirichletAlpha=1, randomSeed=0):
    """
    Generates descriptor data for random mixtures of component combinations
    :param componentCount: count of mixture components
    :param size: mixture data size to generate
    :param sample: initial pure sample
    :param label_names: names of target variables that are not descriptors of spectra, calculated by addDescrFunc
    :param addDescrFunc: function, that calculates descriptors for the given sample
    :param makeMixtureOfSpectra: function(sample, inds, concentrations) to calculate mixture of spectra by given concentrations and spectra indices (return intensity only)
    :param makeMixtureOfLabel: function(label_name, sample, inds, concentrations) to calculate label for mixture of spectra given concentrations and spectra indices
    :param dirichletAlpha: parameter of dirichlet distribution of concentrations (1 - uniform, >1 - mostly equal, <1 - some components prevail)
    :param randomSeed: random seed
    :return: new mixture sample
    """
    spectra = sample.spectra.to_numpy()
    data = sample.params.loc[:,label_names].to_numpy()
    np.random.seed(randomSeed)
    c = np.random.dirichlet(alpha=dirichletAlpha*np.ones(componentCount), size=size)
    n = spectra.shape[0]
    all_ind = np.random.randint(low=0, high=n-1, size=(size, componentCount))
    mix_spectra = np.zeros((size, len(sample.energy)))
    mix_data = np.zeros((size, len(label_names)))
    for i in range(size):
        ind = all_ind[i]
        if makeMixtureOfSpectra is None:
            mix_spectra[i] = np.dot(spectra[ind,:].T, c[i])
        else:
            ms = makeMixtureOfSpectra(sample, ind, c[i])
            assert len(ms) == len(sample.energy)
            mix_spectra[i] = ms
        if makeMixtureOfLabel is None:
            mix_data[i] = np.dot(data[ind,:].T, c[i])
        else:
            for j in range(len(label_names)):
                mix_data[i,j] = makeMixtureOfLabel(label_names[j], sample, ind, c[i])
    mix_data = pd.DataFrame(columns=label_names, data=mix_data, dtype=np.float)
    mix_sample = ML.Sample(mix_data, mix_spectra, energy=sample.energy)
    addDescrFunc(mix_sample)
    mix_sample.paramNames = mix_sample.params.columns.to_numpy()
    assert set(mix_sample.paramNames) == set(sample.paramNames), 'Set of initial sample and mixture descriptors are not equal. Initial: '+str(sample.paramNames)+' mixture: '+str(mix_sample.paramNames)+'. May be you forget to list all labels in generateMixtureOfSample call?'
    return mix_sample

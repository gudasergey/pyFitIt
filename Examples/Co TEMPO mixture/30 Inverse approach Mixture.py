#!/opt/anaconda/bin/python -u
# -*- coding: utf-8 -*-
import sys
sys.path.append("../../..")
from pyfitit import *


def fitByOneComponent(project_list, inverseEstimators, spectra, folderToSaveResult='results/fitByOneComponent'):
    os.makedirs(folderToSaveResult, exist_ok=True)
    r_factors = {}; params = {}
    for project, estimator in zip(project_list, inverseEstimators):
        nt = len(spectra.params)
        fmins = np.zeros(nt); geoms = [None]*nt
        for temperature, i in zip(spectra.params, range(nt)):
            # print('Было: ', len(estimator.exp.spectrum.energy))
            estimator.exp.spectrum = spectra.getSpectrumByParam(temperature)
            # print('Стало: ', len(estimator.exp.spectrum.energy))
            fmins[i], geoms[i] = inverseMethod.findGlobalL2NormMinimum(1, estimator, folderToSaveResult+'/'+project.name+'/'+str(temperature), calcXanes=None, fixParams=None, plotMaps=[])
        r_factors[project.name] = fmins
        params[project.name] = geoms
    graphs = tuple()
    for name, r_factor in r_factors.items():
        graphs += (spectra.params, r_factor, name+' r-factor')
    plotting.plotToFileAndSaveCsv(*graphs, folderToSaveResult+'/r-factors')

    graphs = tuple()
    # print('params = ', params)
    for projectName, geoms in params.items():
        for paramName in geoms[0]:
            # print('paramName = ', paramName)
            p_values = np.zeros(nt)
            for i in range(nt):
                # print('geoms[i] = ', geoms[i])
                p_values[i] = geoms[i][paramName]
            graphs += (spectra.params, p_values, projectName+'_'+paramName)
    plotting.plotToFileAndSaveCsv(*graphs, folderToSaveResult+'/params')

def genMolecule():
    # HT_2ligands=-0.016299  HT_tempo=0.0085353  LT_2ligands=0.054304  LT_tempo=-0.19722
    proj = loadProject('../Co_common/Co_common.py', expFile='../exp_without_dupl_increase.dat', label='300')
    m = proj.moleculeConstructor({'2ligands':0.054304, 'tempo':-0.19722})
    m.export_xyz('molecule_117.xyz')
    m = proj.moleculeConstructor({'2ligands':-0.016299, 'tempo':0.0085353})
    m.export_xyz('molecule_300.xyz')

# project_list = [
#     loadProject('../CoII_HS/CoII_HS.py', expFile='../exp_without_dupl_increase.dat', label='300'),
#     loadProject('../CoIII_LS/CoIII_LS.py', expFile='../exp_without_dupl._increasedat', label='117')]
# sample_list = [readSample('../CoII_HS/grid_36'), readSample('../CoIII_LS/grid_36')]

genMolecule()
exit(0)

project_list = [
    loadProject('../Co_common/Co_common.py', expFile='../exp_without_dupl_increase.dat', label='300'),
    loadProject('../Co_common/Co_common.py', expFile='../exp_without_dupl_increase.dat', label='117')]
project_list[0].name = 'common_300'
project_list[1].name = 'common_117'
sample_list = [readSample('../Co_common/IHS_100'), readSample('../Co_common/IHS_100')]

for project in project_list:
    project.maxSpectrumPoints = 500

inverseEstimators = [constructInverseEstimator('RBF', project, project.FDMNES_smooth, CVcount=10, folderToSaveCVresult='results/inverseEstimator_CVresult') for project in project_list]
for i in range(len(project_list)):
    inverseEstimators[i].fit(sample_list[i])

# inverseMethod.findGlobalL2NormMinimumMixture(1, inverseEstimators, fixParams={}, folderToSaveResult='results/inverseMethod_mixture_Results')

spectra = readSpectra('../exp_without_dupl_increase.dat')
spectra.parseLabels()

inverseMethod.findGlobalL2NormMinimumMixtureUniform(5, spectra, inverseEstimators, concInterpPoints=[117,130,140,150,163,173,180,190,210,220,230,240,260,280,300], fixParams={}, folderToSaveResult='results/invMeth_mixUnif_Results')

# fitByOneComponent(project_list, inverseEstimators, spectra, folderToSaveResult='results/fitBy_1_Comp_common_decrease')

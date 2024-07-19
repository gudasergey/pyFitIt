# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %matplotlib notebook
import sys
import pandas as pd
sys.path.append("../../..")
sys.path.append("/opt/pyfitit")
from pyfitit import *
initPyfitit()

# %% [markdown]
# # Spectrum Routines

# %%
# Read spectrum
exp_spectrum = readSpectrum('data/exp_ground.txt', energyColumn=0, intensityColumn=1)

# %%
# Read sample from the given folder. sample.params - params DataFrame, sample.spectra - spectra DataFrame, sample.paramNames, sample.energy
sample = readSample('data/sample')

# %%
# Plot sample
sample.plot(folder='results/sample_plot', colorParam='centralRing1_Shift', plotSampleParams=dict(sortByColors=True, xlim=[7270,7300]),
            plotIndividualParams={}, maxIndivPlotCount=10)

# %%
# Save sample
sample.saveToFolder('results/sample')

# %%
# Construct sample from spectra in a folder and table with labels
params = pd.read_excel('data/sampleConstruction/exp_data.xlsx')
spectra = []
for name in params['name']:
    s = readSpectrum(f'data/sampleConstruction/spectra/{name}.nor', energyColumn=0, intensityColumn=3)
    spectra.append(s)
sample = Sample(params=params, spectra=spectra, meta={'nameColumn':'name', 'labels': ['CN', 'OxState', 'NCl', 'NOx', 'OxType']}, encodeLabels=True)
sample.limit([4950, 5200], inplace=True)
sample.saveToFolder('results/constructedSample', plot=True, colorParam='CN')

# %% [markdown]
# # Spectra Calculation

# %%
# Generate fdmnes input files for the given xyz file
molecule = Molecule('xyz/Fe_terpy1.xyz')
fdmnes.generateInput(molecule, energyRange='-10 0.1 15 1 30 3 54 4 200 5 250', radius=5, folder='results/fdmnes_calc_xyz', Green=True, Adimp=0.2, additional='Sym\n1\n\n')

# %%
# Generate feff input files for the given xyz file
molecule = Molecule('xyz/Fe_terpy1.xyz')
feff.generateInput(molecule, folder='results/feff_calc_xyz', feffVersion='9', separatePotentials=True, SCF=5, XANES='9.0 0.1 1.5', FMS='7.0  0')

# %%
# Run feff in one folder
feff.runLocal(folder='results/feff_calc_xyz', feffVersion='9')
# and read result
spectrum = feff.parseOneFolder('results/feff_calc_xyz')

# %%
# Calculate spectra for all xyz in the given folder
p = {'Energy range': '-10 0.1 15 1 30 3 54 4 200 5 250', 'radius': '5', 'Adimp': '0.2', 'Green': True, 'additional': 'Sym\n1\n\n'}
calcForAllxyzInFolder(xyzfolder='xyz', spectralProgram='fdmnes', continueCalculation=True, calcSampleInParallel=2, generateOnly=False, calcFolder='results/xyz_calc', outputFolder='results/xyz_calc_result', **p)

# %% [markdown]
# # XAS preprocessing

# %%
# Smooth spectrum by FDMNES convolution
theory_spectrum = fdmnes.parseOneFolder('data/fdmnes_fdm_5')
smoothParams = {'Gamma_hole': 4.23, 'Ecent': 53, 'Elarg': 24, 'Gamma_max': 24, 'Efermi': 7109, 'shift': -152}
smoothed_spectrum, norm = smoothInterpNorm(smoothParams, theory_spectrum, smoothType='fdmnes', expSpectrum=exp_spectrum, fitNormInterval=[7100,7200])
print('norm =', norm)
plotToFile(smoothed_spectrum.x, smoothed_spectrum.y, 'smoothed', fileName='results/fdmnes_convolution.png')

# %%
# Smooth sample
smoothed_sample = sample.copy()
smoothParams = {'Gamma_hole': 4.23, 'Ecent': 53, 'Elarg': 24, 'Gamma_max': 24, 'Efermi': 7109, 'shift': 152}
smoothed_sample.spectra = smoothDataFrame(smoothParams, sample.spectra, smoothType='fdmnes', expSpectrum=exp_spectrum, fitNormInterval=[7100,7200], norm=0.031)

# %%
# Smooth of experimental spectrum keeping the peak height
s = readSpectrum(f'data/LC42.nor', energyColumn=0, intensityColumn=3)
ss = smoothExp(s, noiseSampleIntervals=[4990,1,[4900,4965], np.inf,5,[5040,5200]], noiseConfidenceLevel=0.9999, debugFileName=f'results/exp_smooth_debug.png')
plotToFile(s.x, s.y, 'init', ss.x, ss.y, 'smoothed', fileName=f'results/exp_smooth.png', showInNotebook=True)

# %%
# Pre-edge extraction (fully automatic)
s = readSpectrum(f'data/LC39.nor', energyColumn=0, intensityColumn=3)
result = subtractBaseAuto(s.x, s.y, plotFileName=f'results/preedge_auto/LC39.png', debug=False)

# %%
# Pre-edge extraction (with turning)
s = readSpectrum(f'data/LC39.nor', energyColumn=0, intensityColumn=3)
result = subtractBase(s.x, s.y, peakInterval=[4965, 4974], baseFitInterval=[4955, 4980], plotFileName=f'results/preedge_manual/LC39.png')

# %%
# TODO: adf smooth


# %%
# Mback normalization
spectrum = readSpectrum('data/TiCl4THF2.txt')
e = spectrum.x
flat_spectrum = mback(spectrum, pre=[4902, 4952], post=[5000, e[-1]], e0=4983, element='Ti', edge='K', deg=3)
plotToFile([e[0], e[-1]], [1, 1], '1', e, spectrum.y, 'initial', e, flat_spectrum.y, 'flat', fileName='results/mback.png', xlim=[4900, 5300])

# %%
# Larch Autobk
spectrum = readSpectrum('data/TiCl4THF2.txt')
# Be carefull, for spectrum with pre-edge autobk sometimes doesn't properly estimate e0 and edge_step. Better to specify these parameters
flat_spectrum, chik, extra = autobk(spectrum, rbkg=1, kmin=0, kmax=None, kweight=2)
k = chik.x
plotToFile([0,20],[0,0],'zero', k, chik.y*k**2, 'chi(k)', fileName='results/autobk.png')
R, ftchi = convert2RSpace(k, chik.y*k**2)
plotToFile([R[0],R[-1]],[0,0],'zero', R, np.abs(ftchi), 'abs', R, np.real(ftchi), 'real', R, np.imag(ftchi), 'imag', fileName='results/autobk_Rspace.png', xlim=[0,5])

# %% [markdown]
# # Mixtures

# %%
# Find concentration
Mn = readSpectrum('data/Mn/001_Mn_metal.dat')
MnO = readSpectrum('data/Mn/002_MnO.dat')
Mn2O3 = readSpectrum('data/Mn/003_Mn2O3.dat')
MnO2 = readSpectrum('data/Mn/004_MnO2.dat')
mix = readSpectrum('data/Mn/005_Mn_mixture.dat')
xanesArray = np.array([Mn.y, MnO.y, Mn2O3.y, MnO2.y])
err, c = mixture.findConcentrations(Mn.x, xanesArray, mix.y, fixConcentrations=None, trysGenerateMixtureOfSampleCount=1)
print('error =', err)
print('concentrations =', c)
approx = np.dot(c,xanesArray).flatten()
plotToFile(Mn.x, Mn.y, 'Mn', MnO.x, MnO.y, 'MnO', Mn2O3.x, Mn2O3.y, 'Mn2O3', MnO2.x, MnO2.y, 'MnO2', mix.x, mix.y, 'mix', Mn.x, approx, 'approx', fileName='results/Mn.png')

# %%
# Fit target spectrum by all mixtures of the spectra from the database with known labels
sample = readSample('data/sample_Fe')
sample,_ = sample.splitUnknown('CN')
unk = readSpectrum('data/exp_ground.txt')
# mixtureTrysCount: 'all combinations of singles', 'all combinations for each label' or number
for componentCount in [1,2]:
    tryAllMixtures(unknownCharacterization=dict(type='spectrum', spectrum=unk, rFactorParams={'interval':[7100,7250]}), componentCount=componentCount, mixtureTrysCount=100, singleComponentData=sample, labelNames=['CN', 'avgDist'], folder=f'results/tryAllMixtures {componentCount}')

# %% [markdown]
# # Descriptors

# %%
# Descriptor calculation
sample = readSample('data/sampleDescriptors')
addDescriptors(sample, [
    {'type': 'pca', 'usePrebuiltData': False, 'energyInterval': [4960, 5180]},
    {'type': 'efermi'},
    {'type': 'max', 'energyInterval': [4950,5030]},
    {'type': 'polynom', 'deg': 3}
])
print(sample.paramNames)
# 'Compound' 'Activity' 'Selectivity' 'pca1' 'pca2' 'pca3' 'efermi_e' 'efermi_slope' 'efermi1_e' 'efermi1_slope' 'max_e' 'max_i' 'max_d2' 'polynom_0' 'polynom_1' 'polynom_2' 'polynom_3'

# %%
# Scatter plots
label = 'Activity'
known, unknown = sample.splitUnknown(columnNames=[label])
plotDescriptors2d(known.params, descriptorNames=['pca1', 'efermi_e'], labelNames=[label], textColumn='Compound', unknown=unknown.params, folder_prefix='results/descriptors')

# %%
# Pre-edge spectrum and descriptors
sample = readSample('data/sampleDescriptors')
calcPreedge(sample, plotFolder='results/pre-edge_descriptors', debug=False, inplace=True)
label = 'Activity'
sample.saveToFolder('results/sample_with_pre-edge', plot=True, colorParam=label)
print(sample.paramNames)
known, unknown = sample.splitUnknown(columnNames=[label])
plotDescriptors2d(known.params, descriptorNames=['pe area', 'pe center'], labelNames=[label], textColumn='Compound', unknown=unknown.params, folder_prefix='results/pre-edge descriptors')


# %%

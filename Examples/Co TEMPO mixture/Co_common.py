# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Import libraries

# +
import sys

sys.path.append("../../..")
from pyfitit import *
import os
def getProjectFolder(): return os.path.dirname(os.path.realpath(__file__))

initPyfitit()


# -

# ## Structural information

def moleculeConstructor(project, params):
    projectFolder = getProjectFolder()
    
#Modify 1.1. Name of the XYZ structure file.
    m = Molecule(join(projectFolder,'Co_common.xyz'))
    
#Modify 1.2. Split molecule into parts.
    m.setParts('0','1-36','37-72','73-113')
    
#Modify 1.3. Define deformations
    deformation = 'tempo'
    axis = normalize((m.atom[73]+m.atom[74])/2 - m.atom[0])
    m.part[3].shift(axis*params[deformation])
    
    deformation = '2ligands'
    axis1 = normalize((m.atom[1]+m.atom[2])/2 - m.atom[0])
    axis2 = normalize((m.atom[37]+m.atom[38])/2 - m.atom[0])
    m.part[1].shift(axis1*params[deformation])
    m.part[2].shift(axis2*params[deformation])
    
    if not m.checkInteratomicDistance(minDist = 0.8):
        print('Warning: there are atoms with distance < minDist')
    return m



# ## Parameters of the project

#Modify 1.4. Name of the file with experiment.
def projectConstructor(expFile, label):
    project = Project()
    project.name = 'common'
        
    filePath = join(getProjectFolder(), expFile)
    
#Modify 1.5. load experimental data
    spectra = readSpectra(expFile)
    project.spectrum = spectra.getSpectrumByLabel(label)

#Modify 1.6. Number of spectrum points for machine learning
    project.maxSpectrumPoints = 100
    
#Modify 1.7. Energy interval for fitting
    a = 7710; b = 7820
    project.intervals = {
      'fit_norm': [a, b],
      'fit_smooth': [a, b],
      'fit_geometry': [a, b],
      'plot': [a, b]
    }
#Modify 1.8. Ranges of deformations
    project.geometryParamRanges = {
        'tempo': [-0.3, 0.1], 
        '2ligands': [-0.3, 0.1]
    }
#Modify 1.9. Parameters of FDMNES calculation
    project.FDMNES_calc = {
        'Energy range': '-15 0.02 8 0.1 18 0.5 30 2 54 3 117',
        'Green': False,
        'radius': 5.5,
    }
#Modify 1.10. Default parameters for convolution.
    # все общее
    project.FDMNES_smooth = { 'Gamma_hole':1.091, 'Ecent':48.163, 'Elarg':58.028, 'Gamma_max':23.725, 'Efermi':7703.3, 'shift':-162.73, 'norm':0.027846 }
    # размазка со всеми параметрами общими, кроме Efermi
    # project.FDMNES_smooth = { "Gamma_hole": 2.238139803781936, "Ecent": 49.7667899034184, "Elarg": 56.471856911829214, "Gamma_max": 22.372379847891583, "Efermi": 7707.926860475672, "shift": -162.76654114008548, 
    # "norm":0.027909
    # }
    # размазка подобранная для T=180
    # project.FDMNES_smooth = { 'shift': -161.15911328101794, 'Gamma_hole': 2.607389955288546, 'Ecent': 35.03280708868292,'Elarg': 27.930877208236065, 'Gamma_max': 24.985300066464198, 'Efermi': 7706.913786719198}
    # размазка подобранная для T=300
    # project.FDMNES_smooth = { "shift": -162.94193198022396, "Gamma_hole": 0.8148877964604098, "Ecent": 42.962767326160304, "Elarg": 57.778471807815095, "Gamma_max": 24.161143774914624, "Efermi": 7705.930968019785 }
    # анти размазка - подобрана для T=117
    # project.FDMNES_smooth = { 'shift': -159.34156562810927, 'Gamma_hole': 4.92817003317672, 'Ecent': 28.88261745264056, 'Elarg': 1.0000011037043097, 'Gamma_max': 24.999997847882852, 'Efermi': 7716.774594992829 }
    project.moleculeConstructor = MethodType(moleculeConstructor, project)
    return project



# ===============

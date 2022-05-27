# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Import libraries

import sys
sys.path.append("../../..")
from pyfitit import *
import os
def getProjectFolder(): return os.path.dirname(os.path.realpath(__file__))

initPyfitit()


# ## Structural information

# +
def moleculeConstructor(project, params):
    projectFolder = getProjectFolder()
    
#Modify 1.1. Name of the XYZ structure file.
    m = Molecule(join(projectFolder,'Fe_terpy.xyz'))
    
#Modify 1.2. Split molecule into parts.
    m.setParts('0','1-9','10-19','20-29','30-38','39-48','49-58')
    
#Modify 1.3. Define deformations
    deformation = 'centralRing1_Shift'
    axis = normalize(m.atom[1] - m.atom[0])
    m.part[1].shift(axis*params[deformation])
    
    deformation = 'sideRings1_Shift'
    axis = normalize(m.atom[1] - m.atom[0])
    m.part[2].shift(axis*params[deformation])
    m.part[3].shift(axis*params[deformation])
    
    deformation = 'sideRings1_Elong'
    axis1 = normalize(m.atom[10] - m.atom[0])
    axis2 = normalize(m.atom[20] - m.atom[0])
    m.part[2].shift(axis1*params[deformation])
    m.part[3].shift(axis2*params[deformation])
    
    deformation = 'centralRing2_Shift'
    axis = normalize(m.atom[30] - m.atom[0])
    m.part[4].shift(axis*params[deformation])
    
    deformation = 'sideRings2_Shift'
    axis = normalize(m.atom[30] - m.atom[0])
    m.part[5].shift(axis*params[deformation])
    m.part[6].shift(axis*params[deformation])
    
    deformation = 'sideRings2_Elong'
    axis1 = normalize(m.atom[39] - m.atom[0])
    axis2 = normalize(m.atom[49] - m.atom[0])
    m.part[5].shift(axis1*params[deformation])
    m.part[6].shift(axis2*params[deformation])
    
    
    if not m.checkInteratomicDistance(minDist = 0.8):
        print('Warning: there are atoms with distance < minDist')
    return m

# -

# ## Parameters of the project

# +
#Modify 1.4. Name of the file with experiment.
def projectConstructor(expFile='exp_ground.txt'):
    project = Project()
    project.name = 'Feterpy'
        
    filePath = join(getProjectFolder(), expFile)
    
#Modify 1.5. load experimental data
    project.spectrum = readSpectrum(filePath, energyColumn=0, intensityColumn=1, skiprows = 1)

#Modify 1.6. Number of spectrum points for machine learning
    project.maxSpectrumPoints = 100
    
#Modify 1.7. Energy interval for fitting
    a = 7113; b = 7178
    project.intervals = {
      'fit_norm': [a, b],
      'fit_smooth': [a, b],
      'fit_geometry': [a, b],
      'plot': [a, b]
    }
#Modify 1.8. Ranges of deformations
    project.geometryParamRanges = {
        'centralRing1_Shift': [-0.3, 0.5], 
        'sideRings1_Shift': [-0.3, 0.5], 
        'sideRings1_Elong': [-0.3, 0.5], 
        'centralRing2_Shift': [-0.3, 0.5], 
        'sideRings2_Shift': [-0.3, 0.5], 
        'sideRings2_Elong': [-0.3, 0.5]
    }
#Modify 1.9. Parameters of FDMNES calculation
    project.FDMNES_calc = {
        'Energy range': '-15 0.02 8 0.1 18 0.5 30 2 54 3 117',
        'Green': False,
        'radius': 5,
    }
#Modify 1.10. Default parameters for convolution.
    project.FDMNES_smooth = {
        'Gamma_hole': 4.23,
        'Ecent': 53,
        'Elarg': 24,
        'Gamma_max': 24,
        'Efermi': 7109,
        'shift': -152,
        'norm': 0.0316
    }
    project.moleculeConstructor = MethodType(moleculeConstructor, project)
    return project
    


# -

# ===============

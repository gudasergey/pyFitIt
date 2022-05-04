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

# ## 2.1 Import libraries and project file

import sys
sys.path.append("../../..")
from pyfitit import *
project = loadProject('FeterpyProject.py')

# ## 2.2 Generate input files for XANES training set

# +
smoothParams = project.FDMNES_smooth

smoothConfig = {'smoothParams': smoothParams, 'smoothType': 'fdmnes', 'expSpectrum': project.spectrum, 
                'fitNormInterval': project.intervals['fit_norm']}

sampleAdaptively(paramRanges=project.geometryParamRanges, 
         moleculeConstructor=project.moleculeConstructor, 
         spectrCalcParams=project.FDMNES_calc,
         maxError=0.01,
         spectralProgram='fdmnes',
         samplePreprocessor=smoothConfig,
         workingFolder='sample_calc', 
         seed=0,
         outputFolder='sample_result',
         runConfiguration={'runType':'local', 'calcSampleInParallel':2, 'recalculateErrorsAttemptCount':1})
# -

# ## 2.3 Generate input files for supplementary XANES training set (compare different machine learning methods)

# +
folderCompare = 'sample_compareMethods'
generateInputFiles(project.geometryParamRanges, project.moleculeConstructor, sampleCount=20, 
   method='line', spectralProgram='fdmnes', spectrCalcParams = project.FDMNES_calc, 
   folder=folderCompare,
   lineEdges={'start':{'centralRing1_Shift': 0, 'sideRings1_Shift': 0, 'sideRings1_Elong': 0, 
                       'centralRing2_Shift': 0, 'sideRings2_Shift': 0, 'sideRings2_Elong': 0}, 
              'end':{'centralRing1_Shift': -0.3, 'sideRings1_Shift': 0.5, 'sideRings1_Elong': 0.5, 
                     'centralRing2_Shift': 0.5, 'sideRings2_Shift': -0.3, 'sideRings2_Elong': -0.3}})

#  Attention. Start xanes calculation on local computer (can be too long)
calcSpectra(spectralProgram='fdmnes', runType='local', calcSampleInParallel=4, folder=folderCompare)

# Collect results into two files: params.txt and spectra.txt
collectResults(folder=folderCompare, outputFolder=folderCompare+'_result')
# -

saveNotebook()

# ## 2.4 Save this file as python script and execute remotely on cluster.

saveAsScript('sample.py')

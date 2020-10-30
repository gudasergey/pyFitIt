# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## 2.1 Import libraries and project file

import sys
sys.path.append("../../..")
from pyfitit import *
project = loadProject('FeterpyProject.py')

# ## 2.2 Generate input files for XANES training set

folder = 'sample'
sampleAdaptively(paramRanges=project.geometryParamRanges, 
                 moleculeConstructor=project.moleculeConstructor,
                 maxError=0.15,
                 spectrCalcParams=project.FDMNES_calc,
                 spectralProgram='fdmnes',
                 workingFolder=folder,
                 runType='local',
                 calcSampleInParallel=16,
                 outputFolder=folder+'_result')

# ## 2.3 Generate input files for supplementary XANES training set (compare different machine learning methods)

# +
# folderCompare = 'sample_compareMethods'
# generateInputFiles(project.geometryParamRanges, project.moleculeConstructor, sampleCount=20, 
#    method='line', spectralProgram='fdmnes', spectrCalcParams = project.FDMNES_calc, 
#    folder=folderCompare,
#    lineEdges={'start':{'centralRing1_Shift': 0, 'sideRings1_Shift': 0, 'sideRings1_Elong': 0, 
#                        'centralRing2_Shift': 0, 'sideRings2_Shift': 0, 'sideRings2_Elong': 0}, 
#               'end':{'centralRing1_Shift': -0.3, 'sideRings1_Shift': 0.5, 'sideRings1_Elong': 0.5, 
#                      'centralRing2_Shift': 0.5, 'sideRings2_Shift': -0.3, 'sideRings2_Elong': -0.3}})
# -

saveNotebook()

# ## 2.4 Save this file as python script and execute remotely on cluster.

saveAsScript('sample.py')

# ## 2.5 Attention. Start xanes calculation on local computer (can be too long)

# +
# calcSpectra(spectralProgram='fdmnes', runType='local', calcSampleInParallel=4, folder=folder)
# calcSpectra(spectralProgram='fdmnes', runType='local', calcSampleInParallel=4, folder=folderCompare)
# -

# ## 2.6 Collect results into two files: params.txt and xanes.txt

# +
# collectResults(folder=folder, outputFolder=folder+'_result')
# collectResults(folder=folderCompare, outputFolder=folderCompare+'_result')

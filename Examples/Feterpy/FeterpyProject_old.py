import numpy as np
import pandas as pd
import os
from pyfitit import *

def getProjectFolder(): return os.path.dirname(os.path.realpath(__file__))

def moleculeConstructor(project, params):
    projectFolder = getProjectFolder()
    m = Molecule(projectFolder+'/Fe_terpy_standard.xyz')
    m.setParts('0', '1-9', '10-19', '20-29', '30-38', '39-48', '49-58')

    part = 1
    vc = m.part[part][0]
    vc = vc / norm(vc)
    m.part[part].shift(vc*params['nearCentralRingShift'])

    part = 2
    v = m.part[part][0]
    v = v / norm(v)
    axis = cross(v,vc)
    m.part[part].shift(v*params['nearSideRingsShift'])
    m.part[part].rotate(axis, [0,0,0], params['nearSideRingsTurn']/180*pi)

    part = 3
    v = m.part[part][0]; v = v/norm(v)
    axis = cross(v,vc)
    m.part[part].shift(v*params['nearSideRingsShift'])
    m.part[part].rotate(axis, [0,0,0], params['nearSideRingsTurn']/180*pi)

    part = 4
    vc = m.part[part][0]; vc = vc/norm(vc)
    m.part[part].shift(vc*params['remoteCentralRingShift'])

    part = 5
    v = m.part[part][0]; v = v/norm(v)
    axis = cross(v,vc)
    m.part[part].shift(v*(params['nearSideRingsShift']+params['remoteSideRingsShiftAdd']))
    m.part[part].rotate(axis, [0,0,0], params['remoteSideRingsTurn']/180*pi)

    part = 6
    v = m.part[part][0]; v = v/norm(v)
    axis = cross(v,vc)
    m.part[part].shift(v*(params['nearSideRingsShift']+params['remoteSideRingsShiftAdd']))
    m.part[part].rotate(axis, [0,0,0], params['remoteSideRingsTurn']/180*pi)
    return m

# expType = 'lowspin-ground' or 'highspin-excited'
def projectConstructor(expType):
    project = Project()
    project.name = 'Feterpy'
    folder = getProjectFolder()
    experiment_data = pd.read_csv(folder+'/exp_Feterpy_'+expType+'.txt', sep="\t", decimal=",", header=1).values
    exp_e = experiment_data[:, 0];
    ind = (exp_e>=7100) & (exp_e<=7350)
    exp_e = exp_e[ind]
    exp_xanes = experiment_data[ind, 1]
    # exp_xanes /= np.mean(exp_xanes[-3:]) - нельзя нормировать, когда размазку подбираем совместно
    project.spectrum = Spectrum(exp_e, exp_xanes)
    a = 7112; b = 7200
    project.intervals = {'fit_norm':[a, b], 'fit_smooth':[a, b], 'fit_geometry':[a, b], 'plot':[a, b]}
    a=-0.5; b = 0.5
    project.geometryParamRanges = {'nearSideRingsShift':[a,b], 'nearCentralRingShift':[a,b], 'remoteSideRingsShiftAdd':[0,b-a], 'remoteCentralRingShift':[a,b], 'nearSideRingsTurn':[-20,7], 'remoteSideRingsTurn':[-20,7]}
    project.FDMNES_calc = {'Energy range': '-15 0.02 8 0.1 18 0.5 30 2 54 3 117', 'Green': False, 'radius': 5}
    project.FDMNES_smooth = {
        'Gamma_hole': 1.6,
        'Ecent': 50,
        'Elarg': 50,
        'Gamma_max': 15,
        'Efermi': 0,
        'shift': 7113,
    }
    project.moleculeConstructor = MethodType(moleculeConstructor, project)
    return project

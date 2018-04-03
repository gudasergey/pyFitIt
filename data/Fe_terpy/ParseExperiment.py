import numpy as np
import pandas as pd
import os
from experiment import Experiment
from smoothLib import DefaultSmoothParams
from optimize import param
from molecula import Molecula
from utils import Xanes
import math

def moleculaConstructor(params):
    folder = os.path.dirname(os.path.realpath(__file__))
    m = Molecula(folder+'/Fe_terpy.xyz', None, checkSymmetry = False, lastGroupIsPart = True, partCenter = 'zero')

    part = 1
    vc = m.mol.loc[m.parts[part].ind[0], ['x','y','z']].values; vc = vc/np.linalg.norm(vc)
    m.move_mol_part(part, vc*params['nearCentralRingShift'])

    part = 2
    v = m.mol.loc[m.parts[part].ind[0], ['x','y','z']].values; v = v/np.linalg.norm(v)
    m.move_mol_part(part, v*params['nearSideRingsShift'])
    axis = np.ravel(np.cross(np.reshape(v, [1, 3]),np.reshape(vc, [1, 3])))
    m.turn_mol_part(part, axis, params['nearSideRingsTurn']/180*math.pi)
    part = 3
    v = m.mol.loc[m.parts[part].ind[0], ['x','y','z']].values; v = v/np.linalg.norm(v)
    m.move_mol_part(part, v*params['nearSideRingsShift'])
    axis = np.ravel(np.cross(np.reshape(v, [1, 3]),np.reshape(vc, [1, 3])))
    m.turn_mol_part(part, axis, params['nearSideRingsTurn']/180*math.pi)

    part = 4
    vc = m.mol.loc[m.parts[part].ind[0], ['x','y','z']].values; vc = vc/np.linalg.norm(vc)
    m.move_mol_part(part, vc*params['remoteCentralRingShift'])

    part = 5
    v = m.mol.loc[m.parts[part].ind[0], ['x','y','z']].values; v = v/np.linalg.norm(v)
    m.move_mol_part(part, v*(params['nearSideRingsShift']+params['remoteSideRingsShiftAdd']))
    axis = np.ravel(np.cross(np.reshape(v, [1, 3]),np.reshape(vc, [1, 3])))
    m.turn_mol_part(part, axis, params['remoteSideRingsTurn']/180*math.pi)
    part = 6
    v = m.mol.loc[m.parts[part].ind[0], ['x','y','z']].values; v = v/np.linalg.norm(v)
    m.move_mol_part(part, v*(params['nearSideRingsShift']+params['remoteSideRingsShiftAdd']))
    axis = np.ravel(np.cross(np.reshape(v, [1, 3]),np.reshape(vc, [1, 3])))
    m.turn_mol_part(part, axis, params['remoteSideRingsTurn']/180*math.pi)

    return m

def parse():
    folder = os.path.dirname(os.path.realpath(__file__))
    experiment_data = pd.read_csv(folder+'/exp_Feterpy_lowspin-ground.txt', sep="\t", decimal=",", header=1).values
    exp_e = experiment_data[:, 0];
    ind = (exp_e>=7100) & (exp_e<=7350)
    exp_e = exp_e[ind]
    exp_xanes = experiment_data[ind, 1]
    exp_xanes /= np.mean(exp_xanes[-3:])
    fit_intervals = {'norm':[exp_e[0], exp_e[-1]], 'smooth':[exp_e[0], exp_e[-1]], 'geometry':[exp_e[0], exp_e[-1]]}
    a=-0.5; b = 0.5
    geometryParamRanges = {'nearSideRingsShift':[a,b], 'nearCentralRingShift':[a,b], 'remoteSideRingsShiftAdd':[0,b-a], 'remoteCentralRingShift':[a,b], 'nearSideRingsTurn':[-20,7], 'remoteSideRingsTurn':[-20,7]}
    exp = Experiment('Feterpy', Xanes(exp_e, exp_xanes), fit_intervals, geometryParamRanges)
    exp.defaultSmoothParams = DefaultSmoothParams(7113)
    exp.defaultSmoothParams.fdmnesSmoothHeader = '''  7112.000   26  1  1  9.5222797E-03 -6.2329036E+00  0.0000000E+00  1  1  7.2647627E+03  0.0000000E+00  0.0000000E+00  2.6654847E+00   1 = E_edge, Z, n_edge, j_edge, Abs_before_edge, VO_interstitial, E_cut, ninitl, ninit1, Epsii, UnitCell_Volume, Surface_ref, f0_forward, natomsym
    Energy    <xanes>    '''
    exp.moleculaConstructor = moleculaConstructor
    return exp

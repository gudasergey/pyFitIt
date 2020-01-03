import numpy as np
import pandas as pd
import os
from experiment import Experiment
from smoothLib import DefaultSmoothParams
from optimize import param, setValue, value
from molecula import Molecula, MoleculaPart
from utils import Xanes, Exafs
import math, copy
import geometry
from types import MethodType

join = os.path.join

def moleculaConstructor(self, params):
    ok = True
    folder = os.path.dirname(os.path.realpath(__file__))
    M = Molecula(join(folder,'CeO2_Cu_parts update.xyz'), None, checkSymmetry = False, lastGroupIsPart = True, partCenter = 'first atom')
    Cu_old = M.mol.loc[0,['x','y','z']].values
    def Cu(): return M.mol.loc[0,['x','y','z']].values
    O_old = {i:M.mol.loc[[M.parts[i].ind[0]],['x','y','z']].values for i in range(1,9)}
    def O(i): return M.mol.loc[[M.parts[i].ind[0]],['x','y','z']].values
    c5678 = (O(5)+O(6)+O(7)+O(8))/4
    if self.fitType == '1':
        ok = ok & M.move_mol_part(0, (c5678-Cu())*params['d1'])
        O12 = (O(2)-O(1))/np.linalg.norm(O(2)-O(1))
        ok = ok & M.move_mol_part(0, O12*params['d2'])
        O57 = (O(7)-O(5))/np.linalg.norm(O(7)-O(5))
        ok = ok & M.move_mol_part(5, O57*params['d3']/2)
        ok = ok & M.move_mol_part(7, -O57*params['d3']/2)
        O68 = (O(8)-O(6))/np.linalg.norm(O(8)-O(6))
        ok = ok & M.move_mol_part(6, O68*params['d4']/2)
        ok = ok & M.move_mol_part(8, -O68*params['d4']/2)
        O56 = (O_old[6]-O_old[5])/np.linalg.norm(O_old[6]-O_old[5])
        ok = ok & M.move_mol_part(5, O56*params['d5'])
        O78 = (O_old[8]-O_old[7])/np.linalg.norm(O_old[8]-O_old[7])
        ok = ok & M.move_mol_part(7, O78*params['d5'])
    elif self.fitType == '2':
        M = Molecula(join(folder,'CeO2_Cu_parts_fit2 update.xyz'), None, checkSymmetry = False, lastGroupIsPart = True, partCenter = 'first atom')
        ok = ok & M.move_mol_part(0, (c5678-Cu())*params['d1'])
        O12 = (O(2)-O(1))/np.linalg.norm(O(2)-O(1))
        ok = ok & M.move_mol_part(0, O12*params['d2'])
        O68 = (O_old[8]-O_old[6])/np.linalg.norm(O_old[8]-O_old[6])
        ok = ok & M.move_mol_part(5, O68*params['d3']/2)
        ok = ok & M.move_mol_part(6, -O68*params['d3']/2)
    elif self.fitType == '3':
        M = Molecula(join(folder,'CeO2_Cu_parts_fit3 update.xyz'), None, checkSymmetry = False, lastGroupIsPart = True, partCenter = 'first atom')
        ok = ok & M.move_mol_part(0, (c5678-Cu())*params['d1'])
        O67 = (O_old[7]-O_old[6])/np.linalg.norm(O_old[7]-O_old[6])
        ok = ok & M.move_mol_part(5, O67*params['d2']/2)
        ok = ok & M.move_mol_part(6, -O67*params['d2']/2)
    elif self.fitType == '4a':
        c24 = (O(2)+O(4))/2
        dir = (c24-Cu())/np.linalg.norm(c24-Cu())
        ok = ok & M.move_mol_part(0, dir*params['d1'])
        ok = ok & M.move_mol_part(5, dir*params['d1'])
        ok = ok & M.move_mol_part(7, dir*params['d1'])
        O24 = (O(4)-O(2))/np.linalg.norm(O(4)-O(2))
        ok = ok & M.move_mol_part(2, O24*params['d2']/2)
        ok = ok & M.move_mol_part(4, -O24*params['d2']/2)
        CuO5 = (O(5)-Cu()); dCuO5 = np.linalg.norm(O(5)-Cu()); CuO5 /= dCuO5
        CuO7 = (O(7)-Cu()); dCuO7 = np.linalg.norm(O(7)-Cu()); CuO7 /= dCuO7
        ok = ok & M.move_mol_part(5, CuO5*(-dCuO5 + params['d3']))
        ok = ok & M.move_mol_part(7, CuO7*(-dCuO7 + params['d3']))
        axis = np.cross(O(5)-Cu(), O(7)-Cu())
        M.parts[5].center = Cu()
        angle = (1-(params['d4']-60)/(120-60))*(10.5+49.5) - 49.5
        ok = ok & M.turn_mol_part(5, axis, angle/180*np.pi/2, checkDistance=False)
        M.parts[7].center = Cu()
        ok = ok & M.turn_mol_part(7, axis, -angle/180*np.pi/2, checkDistance=False)
    elif self.fitType == '4b':
        M = Molecula(join(folder,'CeO2_Cu_parts_fit4b update.xyz'), None, checkSymmetry = False, lastGroupIsPart = True, partCenter = 'first atom')
        c24 = (O(2)+O(4))/2
        dir = (c24-Cu())/np.linalg.norm(c24-Cu())
        ok = ok & M.move_mol_part(0, dir*params['d1'])
        O24 = (O(4)-O(2))/np.linalg.norm(O(4)-O(2))
        ok = ok & M.move_mol_part(2, O24*params['d2']/2)
        ok = ok & M.move_mol_part(4, -O24*params['d2']/2)
    else: assert False, 'Unknown fit type'
    if ok: return M
    else: return None

# fitType = '1','2','3','4a','4b'   expType = 'initial', 'final'
def parse(fitType, expType):
    assert (fitType=='1') and (expType=='initial') or (fitType=='2') and (expType=='final') or (fitType=='3') and (expType=='final') or (fitType=='4a') or (fitType=='4b'), 'Wrong fit/exp types combination'
    folder = os.path.dirname(os.path.realpath(__file__))
    experiment_data = pd.read_csv(folder+os.sep+'exp_'+expType+'.nor', sep=r"\s+", decimal=".", skiprows=26, header=None).values
    exp_e = experiment_data[:, 0];
    exp_xanes = experiment_data[:, 1]

    shift = 8976; af = shift; b = shift+120
    fit_intervals = {'norm':[af, b], 'smooth':[af, b], 'geometry':[af, b], 'plot':[shift-8,b], 'exafs':[1.5,7.5]}
    if fitType == '1':
        geometryParamRanges = {'d1':[0,1], 'd2':[-0.15,0.15], 'd3':[-0.2,0.2], 'd4':[-0.2,0.2], 'd5':[-0.2,0.2]}
    elif fitType == '2':
        geometryParamRanges = {'d1':[0,1], 'd2':[-0.25,0.25], 'd3':[-0.2,0.2]}
    elif fitType == '3':
        geometryParamRanges = {'d1':[0,1], 'd2':[-0.2,0.2]}
    elif fitType == '4a':
        geometryParamRanges = {'d1':[0,0.75], 'd2':[-0.15,0.15], 'd3':[1.75,2.1], 'd4':[60,120]}
    elif fitType == '4b':
        geometryParamRanges = {'d1':[0,0.75], 'd2':[-0.15,0.15]}
    else: assert False, 'Unknown fit type'
    fdmnesEnergyRange='0 0.1 6 0.01 8 0.05 12 0.3 30 0.5 40 1.0 124 4.0 150'
    project = Experiment('CeO2_Cu_'+fitType+'_'+expType, Xanes(exp_e, exp_xanes), fit_intervals, geometryParamRanges, fdmnesEnergyRange=fdmnesEnergyRange)
    project.defaultSmoothParams = DefaultSmoothParams(shift)
    project.moleculaConstructor = MethodType(moleculaConstructor, project)
    project.fitType = fitType
    project.expType = expType
    return project

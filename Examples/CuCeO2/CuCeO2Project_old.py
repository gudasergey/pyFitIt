import numpy as np
import pandas as pd
from pyfitit import *

def getProjectFolder(): return os.path.dirname(os.path.realpath(__file__))

def moleculeConstructor(self, params):
    ok = True
    folder = getProjectFolder()
    M = Molecule(join(folder,'CeO2_Cu update.xyz'))
    M.setParts('0', '1', '2', '3', '4', '5', '6', '7', '8', '9-141')
    Cu_old = M.atom[0]
    def Cu(): return M.atom[0]
    O_old = {i:M.part[i][0] for i in range(1,9)}
    def O(i): return M.part[i][0]
    c5678 = (O(5)+O(6)+O(7)+O(8))/4
    if self.fitType == '1':
        M.part[0].shift((c5678-Cu())*params['d1'])
        O12 = (O(2)-O(1))/norm(O(2)-O(1))
        M.part[0].shift(O12*params['d2'])
        O57 = (O(7)-O(5))/norm(O(7)-O(5))
        M.part[5].shift(O57*params['d3']/2)
        M.part[7].shift(-O57*params['d3']/2)
        O68 = (O(8)-O(6))/norm(O(8)-O(6))
        M.part[6].shift(O68*params['d4']/2)
        M.part[8].shift(-O68*params['d4']/2)
        O56 = (O_old[6]-O_old[5])/norm(O_old[6]-O_old[5])
        M.part[5].shift(O56*params['d5'])
        O78 = (O_old[8]-O_old[7])/norm(O_old[8]-O_old[7])
        M.part[7].shift(O78*params['d5'])
    elif self.fitType == '2':
        M = Molecula(join(folder,'CeO2_Cu_fit2 update.xyz'))
        M.part[0].shift((c5678-Cu())*params['d1'])
        O12 = (O(2)-O(1))/norm(O(2)-O(1))
        M.part[0].shift(O12*params['d2'])
        O68 = (O_old[8]-O_old[6])/norm(O_old[8]-O_old[6])
        M.part[5].shift(O68*params['d3']/2)
        M.part[6].shift(-O68*params['d3']/2)
    elif self.fitType == '3':
        M = Molecula(join(folder,'CeO2_Cu_fit3 update.xyz'))
        M.part[0].shift((c5678-Cu())*params['d1'])
        O67 = (O_old[7]-O_old[6])/norm(O_old[7]-O_old[6])
        M.part[5].shift(O67*params['d2']/2)
        M.part[6].shift(-O67*params['d2']/2)
    elif self.fitType == '4a':
        c24 = (O(2)+O(4))/2
        dir = (c24-Cu())/norm(c24-Cu())
        M.part[0].shift(dir*params['d1'])
        M.part[5].shift(dir*params['d1'])
        M.part[7].shift(dir*params['d1'])
        O24 = (O(4)-O(2))/norm(O(4)-O(2))
        M.part[2].shift(O24*params['d2']/2)
        M.part[4].shift(-O24*params['d2']/2)
        CuO5 = (O(5)-Cu()); dCuO5 = norm(O(5)-Cu()); CuO5 /= dCuO5
        CuO7 = (O(7)-Cu()); dCuO7 = norm(O(7)-Cu()); CuO7 /= dCuO7
        M.part[5].shift(CuO5*(-dCuO5 + params['d3']))
        M.part[7].shift(CuO7*(-dCuO7 + params['d3']))
        axis = cross(O(5)-Cu(), O(7)-Cu())
        angle = (1-(params['d4']-60)/(120-60))*(10.5+49.5) - 49.5
        M.part[5].rotate(axis, Cu(), angle/180*np.pi/2)
        M.part[7].rotate(axis, Cu(), -angle/180*np.pi/2)
    elif self.fitType == '4b':
        M = Molecula(join(folder,'CeO2_Cu_fit4b update.xyz'))
        c24 = (O(2)+O(4))/2
        dir = (c24-Cu())/norm(c24-Cu())
        M.part[0].shift(dir*params['d1'])
        O24 = (O(4)-O(2))/norm(O(4)-O(2))
        M.part[2].shift(O24*params['d2']/2)
        M.part[4].shift(-O24*params['d2']/2)
    else: assert False, 'Unknown fit type'
    if not M.checkInteratomicDistance(minDist = 0.8):
        print('Warning: there are atoms with distance < minDist')
    return M

# fitType = '1','2','3','4a','4b'   expType = 'initial', 'final'
def projectConstructor(fitType, expType):
    assert (fitType=='1') and (expType=='initial') or (fitType=='2') and (expType=='final') or (fitType=='3') and (expType=='final') or (fitType=='4a') or (fitType=='4b'), 'Wrong fit/exp types combination'
    project = Project()
    project.name = 'CeO2_Cu_'+fitType+'_'+expType
    project.fitType = fitType
    project.expType = expType
    folder = getProjectFolder()
    experiment_data = pd.read_csv(folder+os.sep+'exp_'+expType+'.nor', sep=r"\s+", decimal=".", skiprows=26, header=None).values
    exp_e = experiment_data[:, 0];
    exp_xanes = experiment_data[:, 1]
    project.spectrum = Spectrum(exp_e, exp_xanes)

    shift = 8976; af = shift; b = shift+120
    project.intervals = {'fit_norm':[af, b], 'fit_smooth':[af, b], 'fit_geometry':[af, b], 'plot':[shift-8,b], 'fit_exafs':[1.5,7.5]}
    if fitType == '1':
        project.geometryParamRanges = {'d1':[0,1], 'd2':[-0.15,0.15], 'd3':[-0.2,0.2], 'd4':[-0.2,0.2], 'd5':[-0.2,0.2]}
    elif fitType == '2':
        project.geometryParamRanges = {'d1':[0,1], 'd2':[-0.25,0.25], 'd3':[-0.2,0.2]}
    elif fitType == '3':
        project.geometryParamRanges = {'d1':[0,1], 'd2':[-0.2,0.2]}
    elif fitType == '4a':
        project.geometryParamRanges = {'d1':[0,0.75], 'd2':[-0.15,0.15], 'd3':[1.75,2.1], 'd4':[60,120]}
    elif fitType == '4b':
        project.geometryParamRanges = {'d1':[0,0.75], 'd2':[-0.15,0.15]}
    else: assert False, 'Unknown fit type'
    project.FDMNES_calc = {'Energy range':'0 0.1 6 0.01 8 0.05 12 0.3 30 0.5 40 1.0 124 4.0 150'}
    project.FDMNES_smooth = {
        'Gamma_hole': 1.6,
        'Ecent': 50,
        'Elarg': 50,
        'Gamma_max': 15,
        'Efermi': 0,
        'shift': shift,
    }
    project.moleculeConstructor = MethodType(moleculeConstructor, project)
    return project

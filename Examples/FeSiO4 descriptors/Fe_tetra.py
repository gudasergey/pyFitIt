import sys
sys.path.append("../../..")
from pyfitit import *
import os
def getProjectFolder(): return os.path.dirname(os.path.realpath(__file__))


def moleculeConstructor(project, params):
    projectFolder = getProjectFolder()
    m = Molecule(join(projectFolder,'Fe.xyz'))
    t = Molecule(join(projectFolder,'tetra.xyz'))
    if project.CN == 2:
        t.rotate([0,1,0],[0,0,0],params['psi']/180*pi)
        t1 = t.copy()
        t2 = t.copy()
        t1.shift([params['r1'], 0, 0])
        m.unionWith(t1)
        t2.shift([params['r1']+params['d12'], 0, 0])
        t2.rotate([0,0,1], [0,0,0], params['phi1']/180*pi)
        m.unionWith(t2)
    elif project.CN == 3:
        m = Molecule(join(projectFolder,'17_croped.xyz'))
        m.setParts('0', '1-5', '6-10', '11-15', '16-20') # Cr, 4 groups of 1 Si & 4 Oxygens
        for i in [2,3,4]:
            # shifting 
            shiftV = m.part[i][1] - m.part[0][0]
            r_Cr_i = norm(shiftV)
            shiftV = normalize(shiftV)
            shiftParam = params['r1'] if i == 2 else params['r2']
            m.part[i].shift((shiftParam - r_Cr_i) * shiftV)
            # rotating
            rotationAxis = cross(m.part[1][1] - m.part[0][0], m.part[i][1] - m.part[0][0])
            center = m.part[0][0]
            angle = params['phi']/180*pi
            u = m.part[1][1] - m.part[0][0] # group at the top (vertical axis)
            v = m.part[i][1] - m.part[0][0] # i-th bottom group axis
            c = dot(u,v)/norm(u)/norm(v)
            originalAngle = np.arccos(np.clip(c, -1, 1)) 
            m.part[i].rotate(rotationAxis, center, angle - originalAngle)
        m.deleteAtomsAt(1,2,3,4,5)
    elif project.CN == 4:
        t1 = t.copy()
        t2 = t.copy()
        t3 = t.copy()
        t4 = t.copy()
        
        t1.shift([params['r1'], 0, 0])
        t2.shift([params['r1'], 0, 0])
        t3.rotate([0,1,0], [0,0,0], pi) # rotating so closest oxygen faces Cr
        t3.shift([-params['r2'], 0, 0])
        t4.rotate([0,1,0], [0,0,0], pi)
        t4.shift([-params['r2'], 0, 0])
        
        t1.rotate([0,1,0], [0,0,0], params['phi1']/360*pi)
        m.unionWith(t1)
        t2.rotate([0,-1,0], [0,0,0], params['phi1']/360*pi)
        m.unionWith(t2)
        t3.rotate([0,0,1], [0,0,0], params['phi2']/360*pi)
        m.unionWith(t3)
        t4.rotate([0,0,-1], [0,0,0], params['phi2']/360*pi)
        m.unionWith(t4)
    elif project.CN == 5:
        # t.rotate([0,1,0],[0,0,0],params['psi']/180*pi)
        t1 = t.copy()
        t2 = t.copy()
        t3 = t.copy()
        t4 = t.copy()
        t5 = t.copy()
        t1.shift([params['r1'], 0, 0])
        t2.shift([params['r2'], 0, 0])
        t3.shift([params['r2']+params['d2'], 0, 0])
        t4.shift([params['r2'], 0, 0])
        t5.shift([params['r2']+params['d2'], 0, 0])
        t1.rotate([0,1,0], [0,0,0], pi/2)
        m.unionWith(t1)
        t2.rotate([0,0,1], [0,0,0], pi/2)
        m.unionWith(t2)
        t3.rotate([0,0,1], [0,0,0], pi)
        m.unionWith(t3)
        t4.rotate([0,0,1], [0,0,0], -pi/2)
        m.unionWith(t4)
        m.unionWith(t5)
    elif project.CN == 6:
        # t.rotate([0,1,0],[0,0,0],params['psi']/180*pi)
        t1 = t.copy()
        t2 = t.copy()
        t3 = t.copy()
        t4 = t.copy()
        t5 = t.copy()
        t6 = t.copy()
        t1.shift([params['r1'], 0, 0])
        t6.shift([params['r1'], 0, 0])
        t2.shift([params['r2'], 0, 0])
        t3.shift([params['r2']+params['d2'], 0, 0])
        t4.shift([params['r2'], 0, 0])
        t5.shift([params['r2']+params['d2'], 0, 0])
        t1.rotate([0,1,0], [0,0,0], pi/2)
        m.unionWith(t1)
        t6.rotate([0,1,0], [0,0,0], -pi/2)
        m.unionWith(t6)
        t2.rotate([0,0,1], [0,0,0], pi/2)
        m.unionWith(t2)
        t3.rotate([0,0,1], [0,0,0], pi)
        m.unionWith(t3)
        t4.rotate([0,0,1], [0,0,0], -pi/2)
        m.unionWith(t4)
        m.unionWith(t5)

    # return molecule after check
    if not m.checkInteratomicDistance(minDist=0.8):
        print('Warning: there are atoms with distance < minDist')
    return m


def projectConstructor(CN, exp='experiments/Fe2O3_alpha.dat'):
    assert CN in [2,3,4,5,6]
    assert fdmnes.useEpsiiShift, 'Cr samples are bad when Epsii is off, because of huge pre-edge for some cases'
    project = Project()

    project.name = 'Fe'+str(CN)
    project.CN = CN
    
    filePath = join(getProjectFolder(), exp)
    # load experimental data.
    # Specify energy column number (numbering starts from zero)
    # Specify intensity column number
    # Specify number of lines to skip: skiprows = 1
    if exp[-3:] == 'nor':
        # print('read nor')
        project.spectrum = readSpectrum(filePath, energyColumn=0, intensityColumn=3, skiprows=38)
    elif exp[-3:] == 'dat':
        # print('read dat')
        project.spectrum = readSpectrum(filePath, energyColumn=0, intensityColumn=1, skiprows=0, decimal=',')
    else:
        assert False, "Unknown exp spectrum extension"
    # Number of spectrum points to use for machine learning prediction (bigger is slower)
    # project.maxSpectrumPoints = 200
    project.useFullSpectrum = True
    
    # specify part of experimental spectrum for fitting
    a = 7110; b = 7180
    project.intervals = {
      'fit_norm': [a, b],
      'fit_smooth': [a, b],
      'fit_geometry': [a, b],
      'plot': [a, b]
    }
    # specify ranges of deformations
    if CN == 2:
        project.geometryParamRanges = {
            'psi': [0,70],
            'r1': [1.8, 2.3],
            'd12': [0, 0.2],
            'phi1': [120,180]
        }
    elif CN == 3:
        project.geometryParamRanges = {
            'r1': [1.8, 2.3],
            'r2': [1.8, 2.3],
            'phi': [80, 135],
        }    
    elif CN == 4:
        project.geometryParamRanges = {
            'r1': [1.8, 2.3],
            'r2': [1.8, 2.3],
            'phi1': [65, 180],
            'phi2': [65, 180],
        }
    elif (CN == 5) or (CN == 6):
        project.geometryParamRanges = {
            'r1': [1.8, 2.3],
            'r2': [1.8, 2.3],
            'd2': [0, 0.3]
        } 
    # specify parameters of calculation by FDMNES
    project.FDMNES_calc = {
        'Energy range': '-5 0.1 18 0.5 30 2 54 3 130',
        'Green': False,
        'radius': 5,
    }
    # specify convolution parameters and spectrum shift
    project.FDMNES_smooth = {
        'Gamma_hole': 3.47,
        'Ecent': 28.72,
        'Elarg': 56.44,
        'Gamma_max': 21.4,
        'Efermi': 7118,
        'norm': 0.0301
    }
    shifts = {2:-150, 3:-150, 4:-148.6, 5:-148, 6:-148}
    project.FDMNES_smooth['shift'] = shifts[CN]
    project.moleculeConstructor = MethodType(moleculeConstructor, project)
    return project

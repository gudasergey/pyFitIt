import os

from . import fdmnes
import json
import numpy as np

def generateInput(molecula, radius=5, folder='', Adimp=None, Quadrupole=False, Convolution='', Absorber=1, Green=False, Edge='K', cellSize=1.0, electronTransfer=None, **other):
    return fdmnes.generateInput(molecula, radius, folder, Adimp, Quadrupole, Convolution, Absorber, Green, Edge, cellSize, electronTransfer, **other)


def parse_all_folders(parentFolder, printOutput=True):
    return fdmnes.parse_all_folders(parentFolder, printOutput)

def runLocal(folder = '.'):
    fileName = 'geometryParams.txt'
    with open(folder+os.sep+fileName, 'r') as f: params = json.load(f)
    p = params[0][1]
    value = p**3*np.sin(p*10) + 2
    header = '''  5989.000   24  1  1  2.4460094E-02 -1.0339471E+00  0.0000000E+00  1  1  6.1292230E+03  0.0000000E+00  0.0000000E+00  1.3711528E+00  1.0000000E+00  0.0000000E+00 = E_edge, Z, n_edge, j_edge, Abs_before_edge, VO_interstitial, E_cut, ninitl, ninit1, Epsii, UnitCell_Volume, Surface_ref, f0_forward, natomsym_f, abs_u_i
    Energy    <xanes>    \n'''
    with open(folder+os.sep+'out.txt', 'w') as f:
        f.write(header)
        for e in range(10):
            f.write('  %.4f  0.0\n' % e)

        for e in range(10, 100):
            f.write('  %.4f  %f\n' % (e, value))


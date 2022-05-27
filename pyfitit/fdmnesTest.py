import os
from . import fdmnes
import json
import numpy as np


def generateInput(molecula, radius=5, folder='', Adimp=None, Quadrupole=False, Convolution='', Absorber=1, Green=False, Edge='K', cellSize=1.0, electronTransfer=None, **other):
    return fdmnes.generateInput(molecula, radius, folder, Adimp, Quadrupole, Convolution, Absorber, Green, Edge, cellSize, electronTransfer, **other)


def parseOneFolder(d):
    return fdmnes.parseOneFolder(d)


def get_good_folders(parentFolder):
    return fdmnes.getGoodFolders(parentFolder)


def getParams(filename):
    return fdmnes.getParams(filename)


def peak(p1,p2,a,b,w1=1,w2=1):
    return 1 / (0.02 + w1**2*np.abs(p1 - a) ** 2 + w2**2*np.abs(p2 - b) ** 2)

global_counter = 1
def runLocal(folder='.'):
    global global_counter
    fileName = 'params.txt'
    with open(folder+os.sep+fileName, 'r') as f: params = json.load(f)
    params = {params[i][0]:params[i][1] for i in range(len(params))}
    p1 = params['centralRing1_Shift']
    p2 = params['sideRings1_Shift']
    # value = 1/(0.02+np.abs(p1-0.1)**2+np.abs(p2-0.1)**2)
    value = peak(p1,p2,0.1,0.1) + peak(p1,p2,0.4,0.4,4,4) + peak(p1,p2,-0.1,0.1,3,3)
    header = '''  7112.000   26  1  1  9.5222797E-03 -4.9956723E+00 -3.2821184E-01  1  1  7.2650319E+03  0.0000000E+00  0.0000000E+00  2.6457996E+00   1 = E_edge, Z, n_edge, j_edge, Abs_before_edge, VO_interstitial, E_cut, ninitl, ninit1, Epsii, UnitCell_Volume, Surface_ref, f0_forward, natomsym
    Energy    <xanes>    \n'''
    with open(folder+os.sep+'out.txt', 'w') as f:
        f.write(header)
        for e in range(10):
            f.write('  %.4f  0.0\n' % e)
        last = 100 if global_counter % 5 == 0 else 90
        if p1<0 and p2<0: last = 80
        for e in range(10, last):
            f.write('  %.4f  %f\n' % (e, value))
    global_counter += 1


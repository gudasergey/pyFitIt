import numpy as np
import os, sys, subprocess, tempfile, json, shutil, traceback
from pyfitit import utils


def generateInput(params, folder='', **other):
    if folder == '':
        if not os.path.exists("./tmp"): os.makedirs("./tmp")
        folder = tempfile.mkdtemp(dir='./tmp')
    else:
        folder = utils.fixPath(folder)
        if not os.path.exists(folder): os.makedirs(folder)

    # Create input file for ansys
    E = params['E']
    v = params['v']
    sigma = params['SigMat']
    with open('program_template.txt', 'r') as f:
        program = f.read()
    program = program.replace('%E_param%', str(E) + 'e9')
    program = program.replace('%v_param%', str(v))
    program = program.replace('%sigmat_param%', str(sigma) + 'e9')
    with open(folder+os.sep+'complete_program.txt', 'w') as f:
        f.write(program)
    s = utils.readSpectrum('check.txt', guess=False, intensityColumn=2, skiprows=1, checkIncreaseEnergy=False)
    s.energy += (E/1000)**2 + (1-v**2) + sigma/15
    # shutil.copyfile('check.txt', folder+os.sep+'indentation_curve.txt')
    s.save(folder+os.sep+'indentation_curve.txt')
    return folder


def parse_one_folder(d):
    s = utils.readSpectrum(d + os.sep + 'indentation_curve.txt', guess=True, checkIncreaseEnergy=False)
    return canonize(s)


def runLocal(folder='.'):
    cmd = ['C:\\Program Files\\ANSYS Inc\\v192\\ansys\\bin\\winx64\\ansys192.exe', '-p', 'ansys',
           '-b', 'list', '-i', 'complete_program.txt', '-o', 'output.txt', '-dir', os.path.abspath(folder),
           '-j', 'indentation_curve', '-s', 'noread', '-l', 'en-us', '-t']
    current_path = folder + os.sep + 'complete_program.txt'

    # proc = subprocess.Popen(cmd, workingFolder=folder, stdout=subprocess.PIPE)
    # proc.wait()
    # if proc.returncode != 0:
    #    raise Exception('Error while executing Ansys calculation')
    # return proc.stdout.read()
    return ''


def canonize(curve):
    x = curve.energy
    y = curve.intensity
    i_max = np.argmax(y)
    x1 = x[:i_max+1]
    y1 = y[:i_max+1]
    x2 = x[i_max + 1:]
    y2 = y[i_max + 1:]
    max_y = y[i_max]
    ind = np.argsort(x1)
    x1 = x1[ind]
    y1 = y1[ind]
    ind = np.argsort(-x2)
    x2 = x2[ind]
    y2 = y2[ind]
    new_x = np.concatenate((y1, max_y + (max_y - y2)))
    new_y = np.concatenate((x1, x2))
    curve.energy = new_x
    curve.intensity = new_y
    return curve

import numpy as np
import pandas as pd
from io import StringIO
import os, sys, subprocess, tempfile, json, glob, shutil, traceback, pickle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    current_path = os.path.abspath(__file__)
    utils_path = os.path.split(current_path)[0] + os.sep + 'utils.py'
    from importlib.machinery import SourceFileLoader
    utils = SourceFileLoader('abc', utils_path).load_module()
else:
    from . import utils  


# если строка folder пустая - создает внутри папки ./tmp и возвращает путь к созданной папке
# energyRange - строка в формате FDMNES
# electronTransfer = ['lineInFDMNESInputAtom1', 'lineInFDMNESInputAtom2', ...]
# пока поддерживается только переброс электронов сразу всем атомам заданного типа
def generateInput(nanoparticle_components, folder='', **other):

    if folder == '':
        if not os.path.exists("./tmp"): os.makedirs("./tmp")
        folder = tempfile.mkdtemp(dir='./tmp')
    else:
        folder = utils.fixPath(folder)
        if not os.path.exists(folder): os.makedirs(folder)
    with open(folder+os.sep+'project.json', 'w') as f:
        json.dump(other, f)    
        
    return folder


def getParams(fileName):
    with open(fileName, 'r') as f: params0 = json.load(f)
    return [p[0] for p in params0], [p[1] for p in params0]


def parse_one_folder(d):
    spectrum_file = d + os.sep + 'spectrum.txt'
    wavelengths_file = d + os.sep + 'wavelengths.txt'
    if os.path.exists(spectrum_file):
        try:
            absorb = np.loadtxt(spectrum_file).ravel()
        except:
            raise Exception('Can\'t parse output file in folder '+d)
    if os.path.exists(wavelengths_file):
        try:
            energy = np.loadtxt(wavelengths_file).ravel()
        except:
            raise Exception('Can\'t parse output file in folder '+d)
    
    return utils.Spectrum(energy, absorb)


def parse_all_folders(parentFolder, printOutput=True):
    subfolders = [f for f in os.listdir(parentFolder) if os.path.isdir(os.path.join(parentFolder, f))]
    subfolders.sort()
    badFolders = []; allXanes = {}
    output = ''
    for i in range(len(subfolders)):
        d = subfolders[i]
        try:
            res = parse_one_folder(os.path.join(parentFolder, d))
            if res is not None: allXanes[d] = res
            else: output += 'Can\'t read output in folder '+d
        except:
            output += traceback.format_exc()+'\n'
            badFolders.append(d)
    if len(allXanes) == 0:
        if printOutput: print('None good folders')
        for i in range(len(badFolders)): badFolders[i] = os.path.join(parentFolder, badFolders[i])
        return None, None, badFolders
    else:
        if output != '': print(output)

    goodFolders = list(set(subfolders) - set(badFolders))
    goodFolders.sort()
    if len(goodFolders) == 0:
        if printOutput: print('None good folders')
        for i in range(len(badFolders)): badFolders[i] = os.path.join(parentFolder, badFolders[i])
        return None, None, badFolders
    allEnergies = np.array([ allXanes[folder].energy for folder in goodFolders ])
    n = len(goodFolders)
    if n == 1: allEnergies.reshape(1,-1)
    energies = np.median(allEnergies, axis=0)
    
    paramNames, _ = getParams(os.path.join(parentFolder, goodFolders[0], 'geometryParams.txt'))
    df_xanes = np.zeros([n, energies.size])
    df_params = np.zeros([n, len(paramNames)])
    for i in range(n):
        d = goodFolders[i]
        _, params = getParams(os.path.join(parentFolder, d, 'geometryParams.txt'))
        df_params[i,:] = np.array(params)
        df_xanes[i,:] = allXanes[d].intensity
    df_xanes = pd.DataFrame(data=df_xanes, columns=['e_'+str(e) for e in energies])
    df_params = pd.DataFrame(data=df_params, columns=paramNames)
    for i in range(len(badFolders)): badFolders[i] = os.path.join(parentFolder, badFolders[i])
    return df_xanes, df_params, badFolders


def runLocal(folder = '.'):
    proc = subprocess.Popen(["/opt/anaconda/bin/python", os.path.abspath(__file__)], cwd=folder, stdout=subprocess.PIPE)
    proc.wait()
    if proc.returncode != 0:
      raise Exception('Error while executing pyGDM calculation')
    return proc.stdout.read()


def runCluster(folder = '.', memory=5000, nProcs=None):
    proc = subprocess.Popen(["run-cluster-and-wait", "-m", str(memory), "/opt/anaconda/bin/python", os.path.abspath(__file__)], cwd=folder, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.wait()


if __name__ == '__main__':
    
    def loadProject(projectFile, *params0, **params1):
        projectFile = utils.fixPath(projectFile)
        ProjectModule = SourceFileLoader(utils.randomString(10), projectFile).load_module()
        return ProjectModule.projectConstructor(*params0, **params1)
    
    if not utils.isLibExists('pyfitit'):
        sys.path.append(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])
        
    if utils.isLibExists("pyGDM2"):
        from pyGDM2 import structures
        from pyGDM2 import materials
        from pyGDM2 import fields
        from pyGDM2 import core
        from pyGDM2 import linear
        from pyGDM2 import tools
    
    with open('project.json', 'r') as f:
        project_json = json.load(f)
    
    params = getParams('geometryParams.txt')
    params = {params[0][i]: params[1][i] for i in range(len(params[0]))}
        
    project = loadProject(project_json['project'], **project_json['projectArgs'])
    geometry, step, material, n1, n2, norm, wavelengths = project.moleculeConstructor(params)
    print('Geometry: ', geometry.shape)
    field_generator = fields.planewave
    avg_polarization = None

    line = {0: 'r-', 90: 'b-'}
    for theta_val in [0, 90]:
        struct = structures.struct(step, geometry, material[theta_val], n1, n2, norm)
        kwargs = dict(theta=[theta_val], kSign=[-1])
        efield = fields.efield(field_generator, wavelengths=wavelengths, kwargs=kwargs)
        sim = core.simulation(struct, efield)
        E = core.scatter(sim)
        search_dict = dict(theta=80, kSign=-1, wavelength=750)
        idx = tools.get_closest_field_index(sim, search_dict)
        a_ext, a_scat, a_abs = linear.extinct(sim, idx)
        field_kwargs = tools.get_possible_field_params_spectra(sim)
        for i, conf in enumerate(field_kwargs):
            print("config", i, ":", conf)
        config_idx = 0
        wl, spectrum = tools.calculate_spectrum(sim, field_kwargs[config_idx], linear.extinct)
        plt.plot(wl, spectrum.T[2], line[theta_val], label='theta={}'.format(theta_val))
        avg_polarization = spectrum.T[2] if theta_val == 0 else avg_polarization + 2* spectrum.T[2]
        
    plt.plot(wl, avg_polarization, 'g-', label='Average')
    plt.xlabel("wavelength (nm)")
    plt.ylabel("cross section (nm^2)")
    plt.legend(loc='best', fontsize=8)

    plt.savefig('./img.png')
    np.savetxt('./spectrum.txt', avg_polarization)
    np.savetxt('./wavelengths.txt', wavelengths)

    

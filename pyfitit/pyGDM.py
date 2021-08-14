import numpy as np
import pandas as pd
import os, sys, subprocess, tempfile, json, traceback
import matplotlib.pyplot as plt

if __name__ == '__main__':
    current_path = os.path.abspath(__file__)
    utils_path = os.path.split(current_path)[0] + os.sep + 'utils.py'
    from importlib.machinery import SourceFileLoader
    utils = SourceFileLoader('abc', utils_path).load_module()
else:
    from . import utils  


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
    spectra_file = d + os.sep + 'spectra.txt'
    #wavelengths_file = d + os.sep + 'wavelengths.txt'
    if os.path.exists(spectra_file):
        try:
            spectra = np.loadtxt(spectra_file)
        except:
            raise Exception('Can\'t parse output file in folder '+d)
        
    ext_spectrum = utils.Spectrum(spectra[0], spectra[1])
    abs_spectrum = utils.Spectrum(spectra[0], spectra[3])
    return {'ext': ext_spectrum, 'abs': abs_spectrum}


def runLocal(folder = '.'):
    proc = subprocess.Popen(["/opt/anaconda/bin/python", os.path.abspath(__file__)], cwd=folder, stdout=subprocess.PIPE)
    proc.wait()
    if proc.returncode != 0:
      raise Exception('Error while executing pyGDM calculation')
    return proc.stdout.read()


def runCluster(folder = '.', memory=5000, nProcs=1):
    proc = subprocess.Popen(["run-cluster-and-wait", "-m", str(memory), "-t", str(nProcs), "/opt/anaconda/bin/python", os.path.abspath(__file__)], cwd=folder, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.wait()


if __name__ == '__main__':
    
    def loadProject(projectFile, *params0, **params1):
        projectFile = utils.fixPath(projectFile)
        ProjectModule = SourceFileLoader(utils.randomString(10), projectFile).load_module()
        return ProjectModule.projectConstructor(*params0, **params1)
    
    
    def calc_spectrum(geometry, material, theta):
        struct = structures.struct(step, geometry, material[theta], n1, n2, norm)
        kwargs = dict(theta=[theta], kSign=[-1])
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
        return spectrum.T[0], spectrum.T[1], spectrum.T[2]
    
    
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
    geometry, step, materials, n1, n2, norm, wavelengths, dimeter_less_2 = project.moleculeConstructor(params)
    if dimeter_less_2:
        print('WARNING! Diamter of nanoparticle is less than 2 nm!')
        exit(0)
    print('Geometry: ', geometry.shape)
    field_generator = fields.planewave
    
    # Initialize array for wavelength, extinction and absorption spectra
    result_spectra = np.zeros((4, len(wavelengths)))
    result_spectra[0] = wavelengths
    # Calculate polarization along the first dimension (the longest size)
    if os.path.exists('./polarization_ext_1.txt'):
        result_spectra[1] += np.loadtxt('./polarization_ext_1.txt')
        result_spectra[2] += np.loadtxt('./polarization_scat_1.txt')
        result_spectra[3] += np.loadtxt('./polarization_abs_1.txt')
    else:
        ext_sp, scat_sp, abs_sp = calc_spectrum(geometry, materials['X'], 0)
        result_spectra[1] += ext_sp
        result_spectra[2] += scat_sp
        result_spectra[3] += abs_sp
        np.savetxt('./polarization_ext_1.txt', ext_sp)
        np.savetxt('./polarization_scat_1.txt', scat_sp)
        np.savetxt('./polarization_abs_1.txt', abs_sp)
    # Calculate polarization along the 2nd and 3rd dimensions
    geometry_z = structures.rotate(geometry, 90, axis='y')
    geometry_z[:, 2] -= np.min(geometry[:, 2])
    # Theta angle = 0 (polarization along OX)
    if os.path.exists('./polarization_ext_2.txt'):
        result_spectra[1] += np.loadtxt('./polarization_ext_2.txt')
        result_spectra[2] += np.loadtxt('./polarization_scat_2.txt')
        result_spectra[3] += np.loadtxt('./polarization_abs_2.txt')
    else:
        ext_sp, scat_sp, abs_sp = calc_spectrum(geometry, materials['Z'], 0)
        result_spectra[1] += ext_sp
        result_spectra[2] += scat_sp
        result_spectra[3] += abs_sp
        np.savetxt('./polarization_ext_2.txt', ext_sp)
        np.savetxt('./polarization_scat_2.txt', scat_sp)
        np.savetxt('./polarization_abs_2.txt', abs_sp)
    # Theta angle = 90 (polarization along OY)
    if os.path.exists('./polarization_ext_3.txt'):
        result_spectra[1] += np.loadtxt('./polarization_ext_3.txt')
        result_spectra[2] += np.loadtxt('./polarization_scat_3.txt')
        result_spectra[3] += np.loadtxt('./polarization_abs_3.txt')
    else:
        ext_sp, scat_sp, abs_sp = calc_spectrum(geometry, materials['Z'], 90)
        result_spectra[1] += ext_sp
        result_spectra[2] += scat_sp
        result_spectra[3] += abs_sp
        np.savetxt('./polarization_ext_3.txt', ext_sp)
        np.savetxt('./polarization_scat_3.txt', scat_sp)
        np.savetxt('./polarization_abs_3.txt', abs_sp)
    # Save calculated spectra and plot them
    np.savetxt('./spectra.txt', result_spectra)    
    plt.plot(wavelengths, result_spectra[1], 'g-', label='Extinction')
    plt.plot(wavelengths, result_spectra[2], 'b-', label='Scattering')
    plt.plot(wavelengths, result_spectra[3], 'r-', label='Absorption')
    plt.xlabel("wavelength (nm)")
    plt.ylabel("cross section (nm^2)")
    plt.legend(loc='best', fontsize=8)
    plt.savefig('./img.png')

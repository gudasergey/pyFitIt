import numpy as np
import os, subprocess, tempfile, json, shutil, re
from pyfitit import utils


def cartesian2spherical(points):
    sqr_sum = points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2
    theta_angle = np.arccos(points[:, 2] / np.sqrt(sqr_sum))
    phi_angle = np.arctan(points[:, 1] / points[:, 0])
    radius = np.sqrt(sqr_sum)
    new_points = np.zeros(points.shape)
    new_points[:, 0] = theta_angle
    new_points[:, 1] = phi_angle
    new_points[:, 2] = radius
    return new_points
    
    
def spherical2cartesian(points):
    # points: x, y, z
    x = points[:, 2]*np.sin(points[:, 0])*np.cos(points[:, 1])
    y = points[:, 2]*np.sin(points[:, 0])*np.sin(points[:, 1])
    z = points[:, 2]*np.cos(points[:, 0])
    new_points = np.zeros(points.shape)
    new_points[:, 0] = x
    new_points[:, 1] = y
    new_points[:, 2] = z
    return new_points

    
def distance(atom1, atom2):
    return np.sqrt((atom1[0]-atom2[0])**2 + (atom1[1]-atom2[1])**2 + (atom1[2]-atom2[2])**2)
    
    
def dist_to_pd_atoms(C, O, Pd):
        C_distances = np.linalg.norm(Pd-C, axis=1)
        O_distances = np.linalg.norm(Pd-O, axis=1)
        return np.min(C_distances), np.min(O_distances)
    
    
def CO_on_surface(C, O, Pd):
    Pd_C, Pd_O = dist_to_pd_atoms(C, O, Pd)
    if Pd_C <= 1.62 or Pd_O <= 1.92:
        return True
    else:
        return False


def generate_CO(Ctheta, Cphi, r_pdc, Pd):
    ro = 1.128
    halfcell = np.array([15, 15, 15])
    radii = np.linspace(10, 0, 1000)
    for i in range(len(radii)):
        C = spherical2cartesian(np.array([Ctheta, Cphi, radii[i]]).reshape(1, -1)) + halfcell
        O = spherical2cartesian(np.array([Ctheta, Cphi, radii[i]+ro]).reshape(1, -1)) + halfcell
        if CO_on_surface(C, O, Pd):
            C = spherical2cartesian(np.array([Ctheta, Cphi, radii[i-1]+r_pdc]).reshape(1, -1)) + halfcell
            O = spherical2cartesian(np.array([Ctheta, Cphi, radii[i-1]+r_pdc+ro]).reshape(1, -1)) + halfcell
            return C.flatten(), O.flatten()
        

def generate_CO_improved(Ctheta, Cphi, Otheta, Ophi, r_pdc, Pd):
    # Ctheta and Cphi define the position of C atom around Pd nanocluster at the r_pdc distance
    # Otheta and Ophi define the position of O atom around C atom
    ro = 1.128
    halfcell = np.array([15, 15, 15])
    radii = np.linspace(10, 0, 1000)
    for i in range(len(radii)):
        C = spherical2cartesian(np.array([Ctheta, Cphi, radii[i]]).reshape(1, -1)) + halfcell
        O = spherical2cartesian(np.array([Otheta, Ophi, ro]).reshape(1, -1)) + C
        if CO_on_surface(C, O, Pd):
            C = spherical2cartesian(np.array([Ctheta, Cphi, radii[i-1]+r_pdc]).reshape(1, -1)) + halfcell
            O = spherical2cartesian(np.array([Otheta, Ophi, ro]).reshape(1, -1)) + C
            return C.flatten(), O.flatten()
        
    
def generate_CO_cartesian(C, ang, Pd):
    Pd_C = np.min(np.linalg.norm(Pd-C, axis=1))
    if Pd_C <= 1.62:
        return None, None
    ro = 1.128
    halfcell = np.array([15, 15, 15])
    C_cartesian = C - halfcell
    C_spherical = cartesian2spherical(C_cartesian)
    O_spherical = np.array([C_spherical[0][0], ang, ro]).reshape(1, -1)
    O = spherical2cartesian(O_spherical) + C
    Pd_O = np.min(np.linalg.norm(Pd-O, axis=1))
    if Pd_C <= 1.62 or Pd_O <= 1.92:
        return None, None
    else:
        return C.flatten(), O.flatten()


def prepare_vasp_inputs(C, O, folder):
    # Generate POSCAR
    with open('vasp_inputs/POSCAR_template', 'r') as f_template: poscar = f_template.read()
    f_new = open(folder+os.sep+'POSCAR', 'w')
    f_new.write(poscar)
    f_new.write(f'{C[0]:<11f} {C[1]:<11f} {C[2]:<11f} F F F\n')
    f_new.write(f'{O[0]:<11f} {O[1]:<11f} {O[2]:<11f} F F F')
    f_new.close()
    # Copy other vasp inputs
    shutil.copyfile('vasp_inputs/INCAR', folder+os.sep+'INCAR')
    shutil.copyfile('vasp_inputs/KPOINTS', folder+os.sep+'KPOINTS')
    shutil.copyfile('vasp_inputs/POTCAR', folder+os.sep+'POTCAR')
    

def generate_rdf(r_min, r_max, step, folder, both_atoms=False):
    from pymatgen.io.vasp import Xdatcar
    from vasppy.rdf import RadialDistributionFunction
    bins = int((r_max - r_min) / step)
    xd = Xdatcar(folder+os.sep+'POSCAR')
    
    rdf_C_Pd = RadialDistributionFunction.from_species_strings(structures=xd.structures,
                                                                species_i='C', species_j='Pd',
                                                                nbins=bins, r_min=r_min, r_max=r_max)
    f = open(folder + os.sep + 'RDF_C', 'w')
    for i in range(len(rdf_C_Pd.r)):
        f.write(str(rdf_C_Pd.r[i])+' ')
        f.write(str(rdf_C_Pd.smeared_rdf()[i])+'\n')
    f.close()
    
    if both_atoms:
        rdf_O_Pd = RadialDistributionFunction.from_species_strings(structures=xd.structures,
                                                                species_i='O', species_j='Pd',
                                                                nbins=bins, r_min=r_min, r_max=r_max)
        f = open(folder + os.sep + 'RDF_O', 'w')
        for i in range(len(rdf_O_Pd.r)):
            f.write(str(rdf_O_Pd.r[i])+' ')
            f.write(str(rdf_O_Pd.smeared_rdf()[i])+'\n')
        f.close()
        
        
def generateInputNew(params, folder='', **other):
    if folder == '':
        if not os.path.exists("./tmp"): os.makedirs("./tmp")
        folder = tempfile.mkdtemp(dir='./tmp')
    else:
        folder = utils.fixPath(folder)
        if not os.path.exists(folder): os.makedirs(folder)
    
    Pd = np.loadtxt('pd_nanocluster.txt')
    # Create input file for vasp
    Ctheta = params['Ctheta']
    Cphi = params['Cphi']
    Otheta = params['Otheta']
    Ophi = params['Ophi']
    r_pdc = params['r_pdc']
        
    with open(folder+os.sep+'geometryParams.txt', 'w') as f: json.dump([['Ctheta', Ctheta], ['Cphi', Cphi],
                                                                        ['Otheta', Otheta], ['Ophi', Ophi],
                                                                        ['r_pdc', r_pdc]], f)
    # generate CO molecule at the r_pdc distance from Pd nanocluster
    C, O = generate_CO_improved(Ctheta, Cphi, Otheta, Ophi, r_pdc, Pd)
    # Copy all vasp inputs and generate POSCAR
    prepare_vasp_inputs(C, O, folder)
    # Generate RDF by POSCAR file
    generate_rdf(0, 7, 0.05, folder, both_atoms=True)
    # Save the inromation about arrangement of CO molecule (which atom is closer to Pd)
    Pd_C, Pd_O = dist_to_pd_atoms(C, O, Pd)
    molecule_ending = open(folder+os.sep+'molecule_end.txt', 'w')
    if Pd_C < Pd_O:
        molecule_ending.write('C')
    else:
        molecule_ending.write('O')
    molecule_ending.close()
    return folder


def generateInput(params, folder='', **other):
    if folder == '':
        if not os.path.exists("./tmp"): os.makedirs("./tmp")
        folder = tempfile.mkdtemp(dir='./tmp')
    else:
        folder = utils.fixPath(folder)
        if not os.path.exists(folder): os.makedirs(folder)

    Pd = np.loadtxt('pd_nanocluster.txt')
    # Create input file for vasp
    theta = params['theta']
    phi = params['phi']
    r_pdc = params['r_pdc']

    # generate CO molecule at the r_pdc distance from Pd nanocluster
    C, O = generate_CO(theta, phi, r_pdc, Pd)
    # Copy all vasp inputs and generate POSCAR
    prepare_vasp_inputs(C, O, folder)
    # Generate RDF by POSCAR file
    generate_rdf(0, 7, 0.05, folder, both_atoms=True)
    # Save the inromation about arrangement of CO molecule (which atom is closer to Pd)
    Pd_C, Pd_O = dist_to_pd_atoms(C, O, Pd)
    molecule_ending = open(folder + os.sep + 'molecule_end.txt', 'w')
    if Pd_C < Pd_O:
        molecule_ending.write('C')
    else:
        molecule_ending.write('O')
    molecule_ending.close()
    return folder


def parse_energy(path_to_file):
    with open(path_to_file, 'r') as f: content = f.read()
    if 'General timing and accounting informations for this job' in content:
        return float(re.findall(r'[\S\s]+energy\(sigma->0\)\s*=\s*([-+]?\d*\.\d*)', content)[-1])
    else:
        return None


def parseOneFolder(folder):
    if not os.path.exists(folder + os.sep + 'OUTCAR') or not os.path.exists(folder + os.sep + 'RDF_C'):
        return None
    rdf = utils.readSpectrum(folder + os.sep + 'RDF_C')
    energy = parse_energy(folder+os.sep+'OUTCAR') - (-173.54598)
    if energy is None:
        return None
    else:
        return {'rdf': rdf, 'energy': energy}


def runLocal(folder='.'):
    output, returncode = utils.runCommand("vasp --wait -k 1 -b 1 -c 1 -m 10000", folder, outputTxtFile=None)
    if returncode != 0:
       print('Error while executing VASP calculation!\n'+output)
       return "Error"
    else:
        return output

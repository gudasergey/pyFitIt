import numpy as np
import pandas as pd
from io import StringIO
import subprocess, os, tempfile, json
from . import optimize, utils

def distance(x, y, z):
    return (x ** 2 + y ** 2 + z ** 2) ** 0.5
# если строка folder пустая - создает внутри папки ./tmp и возвращает путь к созданной папке
# energyRange - строка в формате FDMNES

def generateInput(molecula, folder = '', temperature=None, debyeTemperature=None, **other):
    if 'Debye Temperature' in other:
        debyeTemperature = other['Debye Temperature']
    if folder == '':
        if not os.path.exists("./tmp"): os.makedirs("./tmp")
        folder = tempfile.mkdtemp(dir='./tmp')
    else:
        os.makedirs(folder, exist_ok=True)

    with open(folder + '/feff.inp', 'w') as f:
        f.write
        f.write('EDGE      K\n')
        f.write('S02       1.0\n')
        f.write('CONTROL 1 1 1 1 1 1\n')
        f.write('PRINT 1 0 0 0 0 3\n\n')
        f.write('EXCHANGE  0\n')
        f.write('SCF       5.0\n\n')
        f.write('DEBYE ' + str(temperature) + ' ' + str(debyeTemperature) + '\n\n')
        f.write('RPATH     5.0\n')
        f.write('EXAFS     20\n')
        #center = molecula.mol.loc[0,['x','y','z']].values
        ls = []
        d = dict()
        for i in range(molecula.atom.shape[0]):
            a = molecula.atom[i, :]
            # pn - proton number
            x, y, z, pn, atom_name = a[0], a[1], a[2], molecula.atomNumber[i], molecula.atomName[i]
            if atom_name not in d:
                d[atom_name] = pn
            ls.append((x, y, z, pn, atom_name, distance(x, y, z)))
        f.write('POTENTIALS\n') #TODO
        f.write('*ipot\tZ\ttag\n')
        sorted_d = sorted(d.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        if len(sorted_d) > 0:
            f.write('0\t{0}\t{1}\n'.format(sorted_d[0][1], sorted_d[0][0]))
        else:
            f.write('NONE\n')
        equal_mol_count = np.sum(molecula.atomNumber == sorted_d[0][1])
        if equal_mol_count > 1:
            j = -1
            flag = False
        else:
            j = 0
            flag = True
        for tag, Z in sorted_d:
            if flag:
                flag = False
                continue
            j += 1
            f.write('{0}\t{1}\t{2}\n'.format(j, Z, tag))
        j = 0
        dict_ipot = dict()
        for tag, _ in sorted_d:
            dict_ipot[tag] = j
            j += 1
        f.write('\nATOMS\n*x\ty\tz\tipot\ttag\tdistance\n')
        ls.sort(key=lambda x: x[5])
        for x in ls:
            #f.write('{0[0]: f}\t{1[1]: f}\t{2[2]: f}\t{3[3]: 3}\t{4[4]}\t{5[5]:f}\n'.format(x))
            f.write('{x: f}\t{y: f}\t{z: f}\t{ipot: 3}\t{tag}\t{distance:f}\n'.format(x=x[0], y=x[1], z=x[2], ipot=dict_ipot[x[4]], tag=x[4]+'1.1', distance=x[5]))
        '''
        for i in range(molecula.mol.shape[0]):
            a = molecula.mol.loc[i, :]
            x, y , z = a['x'], a['y'], a['z']
            f.write('{0: f}\t{1: f}\t{2: f}\t{3: 3}\t{4}\t{5}\n'.format(x, y, z, a['proton_number'], a['atom_name'] + '1.0', distance(x, y, z)))
        '''
        f.write('\nEND\n')

    #with open(folder + '/feff.txt', 'w') as f:
        #f.write('1\nin.txt')
    return folder

def runLocal(folder = '.'):
    feff = 'feff85L.exe' if os.name == 'nt' else 'feff85L'
    proc = subprocess.Popen([feff, 'feff.inp'], cwd=folder, stdout=subprocess.PIPE)
    proc.wait()
    if proc.returncode != 0:
      raise Exception('Error while executing "'+feff+'" command')
    return proc.stdout.read()

def runCluster(folder = '.', memory=5000, nProcs = 1):
    proc = subprocess.Popen(["run-cluster-and-wait", "-m", str(memory), '-n', str(nProcs), "feff85L feff.inp"], cwd=folder, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.wait()

def calc_header_size(fileName):
    with open(fileName, 'r') as f:
        lines = f.read().split('\n')
    matches = [i for i in range(len(lines)) if lines[i].find('#       k          chi') >= 0] # '      k          chi'
    assert len(matches) == 1, "wrong feff output file format"
    return matches[0] + 1



def getParams(fileName):
    with open(fileName, 'r') as f: params0 = json.load(f)
    return [p[0] for p in params0], [p[1] for p in params0]

def parse_one_folder(d):
    exafsFile = d+os.sep+'chi.dat'
    if not os.path.isfile(exafsFile):
        raise Exception('Error: in folder ' + d + ' there is no output file with energies')
    skip_header = calc_header_size(exafsFile)
    exafs = pd.read_csv(exafsFile, skiprows = skip_header, header=None, sep='\s+')
    energies = exafs.loc[:,0].ravel()
    exafs = exafs.loc[:,1].ravel()
    return utils.Exafs(energies, exafs)

import numpy as np
import pandas as pd
from io import StringIO
import subprocess, os, tempfile, json
from . import optimize, utils


def generateInputUniqueAbsorber(molecule, folder, temperature, debyeTemperature, additional, absorber, edge, exchange, scf, rpath, exafs, **other):
    os.makedirs(folder, exist_ok=True)
    with open(folder + '/feff.inp', 'w') as f:
        f.write
        f.write(f'EDGE      {edge}\n')
        f.write('S02       1.0\n')
        f.write('CONTROL 1 1 1 1 1 1\n')
        f.write('PRINT 1 0 0 0 0 0\n\n')
        f.write(f'EXCHANGE  {exchange}\n')
        f.write(f'SCF       {scf}\n\n')
        f.write(f'DEBYE {temperature} {debyeTemperature}\n\n')
        f.write(f'RPATH     {rpath}\n')
        f.write(f'EXAFS     {exafs}\n')
        if additional != '': f.write(additional + '\n')
        if absorber is None: absorber = np.argmax(molecule.az)
        assert 0 <= absorber <= len(molecule.az)-1, f'Wrong absorber index {absorber}'
        f.write('POTENTIALS\n')
        f.write('*ipot\tZ\ttag\n')
        uniq_az, uniq_ind = np.unique(molecule.az, return_index=True)
        ipot = np.zeros(len(molecule.az), dtype=int)
        f.write(f'0\t{molecule.az[absorber]}\t{molecule.atomName[absorber]}\n')
        absorberNamesakes = np.sum(molecule.az[absorber] == molecule.az) > 1
        ip = 1
        for ii in range(len(uniq_az)):
            i = len(uniq_az)-1-ii
            if (not absorberNamesakes) and molecule.az[absorber] == uniq_az[i]: continue
            ind = uniq_ind[i]
            f.write(f'{ip}\t{molecule.az[ind]}\t{molecule.atomName[ind]}\n')
            ipot[molecule.az == molecule.az[ind]] = ip
            ip += 1
        ipot[absorber] = 0
        f.write('\nATOMS\n*x\ty\tz\tipot\ttag\tdistance\n')
        for i in range(len(molecule.az)):
            x,y,z = molecule.atom[i]
            d = np.linalg.norm(molecule.atom[i]-molecule.atom[absorber])
            f.write(f'{x}\t{y}\t{z}\t{ipot[i]}\t{molecule.atomName[i]}\t{d}\n')
        f.write('\nEND\n')


def generateInput(molecule, folder='', temperature=0, debyeTemperature=0, additional='', absorbers=None, edge='K', exchange=0, scf=5, rpath=5, exafs=20, **other):
    """
    Generate input for feff8.5 calculation

    :param molecule:
    :param folder:
    :param temperature:
    :param debyeTemperature:
    :param additional:
    :param absorbers: one index (from 0) of absorber atom or list of indexes. Default: take heaviest atom
    """

    if 'Debye Temperature' in other:
        debyeTemperature = other['Debye Temperature']
    if folder == '':
        if not os.path.exists("./tmp"): os.makedirs("./tmp")
        folder = tempfile.mkdtemp(dir='./tmp')
    os.makedirs(folder, exist_ok=True)
    if isinstance(absorbers, list):
        with open(folder+os.sep+'absorbers.txt', 'w') as f: f.write(' '.join(map(str, absorbers)))
        for absorber in absorbers:
            generateInputUniqueAbsorber(molecule, folder+os.sep+f'absorber_{absorber}', temperature, debyeTemperature, additional, absorber, edge, exchange, scf, rpath, exafs, **other)
    else:
        generateInputUniqueAbsorber(molecule, folder, temperature, debyeTemperature, additional, absorbers, edge, exchange, scf, rpath, exafs, **other)
    return folder


def getAbsorberFolders(folder, returnAbsorberNum=False):
    if not os.path.exists(folder+os.sep+'absorbers.txt'):
        if returnAbsorberNum: return [folder], [0]
        else: return [folder]
    with open(folder+os.sep+'absorbers.txt') as f: ab = f.read().strip().split(' ')
    fs = [folder+os.sep+f'absorber_{a}' for a in ab]
    if returnAbsorberNum: return fs, list(map(int,ab))
    else: return fs


def runLocal(folder):
    feff = 'feff85L.exe' if os.name == 'nt' else 'feff85L'
    output = ''
    for f in getAbsorberFolders(folder):
        proc = subprocess.Popen([feff, 'feff.inp'], cwd=f, stdout=subprocess.PIPE)
        stdoutdata, stderrdata = proc.communicate()
        if proc.returncode != 0:
            raise Exception('Error while executing "' + feff + '" command:\n' + str(stderrdata))
        output += str(stdoutdata) + '\n\n'
    return output


def runCluster(folder, memory=5000, nProcs=1):
    for f in getAbsorberFolders(folder):
        proc = subprocess.Popen(["run-cluster-and-wait", "-m", str(memory), '-n', str(nProcs), "feff85L feff.inp"], cwd=f, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.wait()


def getParams(fileName):
    with open(fileName, 'r') as f: params0 = json.load(f)
    return [p[0] for p in params0], [p[1] for p in params0]


def parse_one_folder(folder, multipleAbsorber=False):
    results = {}
    ds, absorbers = getAbsorberFolders(folder, returnAbsorberNum=True)
    for i, d in enumerate(ds):
        exafsFile = d+os.sep+'chi.dat'
        if not os.path.isfile(exafsFile):
            raise Exception('Error: in folder ' + d + ' there is no output file chi.dat')
        s = utils.readSpectrum(exafsFile, guess=True)
        results[absorbers[i]] = utils.Exafs(s.energy, s.intensity)

    # make all spectra have the same point count
    enNums = [len(results[a].k) for a in absorbers]
    maxEnInd = np.argmax(enNums)
    maxEnNum = enNums[maxEnInd]
    me = results[absorbers[maxEnInd]]
    for a in absorbers:
        if len(results[a].k) != maxEnNum:
            results[a] = utils.Exafs(me.k, np.interp(me.k, results[a].k, results[a].chi))
    if multipleAbsorber: return results
    else:
        assert len(results) == 1, 'Multiple absorbers detected. Set multipleAbsorber=True'
        return list(results.values())[0]

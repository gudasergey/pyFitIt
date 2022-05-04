import numpy as np
import subprocess, os, tempfile, json, shutil
from . import utils


def generateInputUniqueAbsorber(molecule, folder, additional, absorber, feffVersion, **cards):
    os.makedirs(folder, exist_ok=True)
    feffVersion = str(feffVersion)
    cards = {k.upper():cards[k] for k in cards}
    with open(folder + os.sep + 'feff.inp', 'w') as f:
        if 'HOLE' not in cards: cards['HOLE'] = '1 1.0'
        if 'NLEG' not in cards: cards['NLEG'] = '4'
        if 'CRITERIA' not in cards: cards['CRITERIA'] = '0 0'
        if 'CONTROL' not in cards: cards['CONTROL'] = '1 1 1 1' if feffVersion == '6' else '1 1 1 1 1 1'
        if 'PRINT' not in cards: cards['PRINT'] = '0 1 1 0' if feffVersion == '6' else '0 0 0 0 0 3'
        for key in cards: f.write(f'{key:<10}{cards[key]:>10}\n\n')
        if additional != '': f.write(additional + '\n')
        if absorber is None: absorber = np.argmax(molecule.az)
        assert 0 <= absorber <= len(molecule.az)-1, f'Wrong absorber index {absorber}'
        f.write('POTENTIALS\n')
        fmt = "{:>5}{:>5}{:>5}\n"
        f.write(fmt.format('*ipot','Z','tag'))
        uniq_az, uniq_ind = np.unique(molecule.az, return_index=True)
        ipot = np.zeros(len(molecule.az), dtype=int)
        f.write(fmt.format(0, molecule.az[absorber], molecule.atomName[absorber]))
        absorberNamesakes = np.sum(molecule.az[absorber] == molecule.az) > 1
        ip = 1
        for ii in range(len(uniq_az)):
            i = len(uniq_az)-1-ii
            if (not absorberNamesakes) and molecule.az[absorber] == uniq_az[i]: continue
            ind = uniq_ind[i]
            f.write(fmt.format(ip, molecule.az[ind], molecule.atomName[ind]))
            ipot[molecule.az == molecule.az[ind]] = ip
            ip += 1
        ipot[absorber] = 0
        f.write('\nATOMS\n')
        fmt = "{:>12.7f}{:>12.7f}{:>12.7f}{:>5}{:>5}{:>7.3f}\n"
        f.write("{:>12}{:>12}{:>12}{:>5}{:>5} {}\n".format('*x','y','z','ipot','tag','distance'))
        for i in range(len(molecule.az)):
            x,y,z = molecule.atom[i]
            d = np.linalg.norm(molecule.atom[i]-molecule.atom[absorber])
            f.write(fmt.format(x, y, z, ipot[i], molecule.atomName[i], d))
        f.write('\nEND\n')


def generateInput(molecule, folder='', additional='', absorbers=None, feffVersion='8.5', **cards):
    """
    Generate input for feff calculation

    :param molecule:
    :param folder:
    :param additional: extra text in input feff file
    :param absorbers: one index (from 0) of absorber atom or list of indexes. Default: take heaviest atom
    :param feffVersion: 6 or 8.5
    """
    if folder == '':
        if not os.path.exists("./tmp"): os.makedirs("./tmp")
        folder = tempfile.mkdtemp(dir='./tmp')
    os.makedirs(folder, exist_ok=True)
    if isinstance(absorbers, list):
        with open(folder+os.sep+'absorbers.txt', 'w') as f: f.write(' '.join(map(str, absorbers)))
        for absorber in absorbers:
            generateInputUniqueAbsorber(molecule, folder + os.sep +f'absorber_{absorber}', additional, absorber, feffVersion=feffVersion, **cards)
    else:
        generateInputUniqueAbsorber(molecule, folder, additional, absorbers, feffVersion=feffVersion, **cards)
    return folder


def getAbsorberFolders(folder, returnAbsorberNum=False):
    if not os.path.exists(folder+os.sep+'absorbers.txt'):
        if returnAbsorberNum: return [folder], [0]
        else: return [folder]
    with open(folder+os.sep+'absorbers.txt') as f: ab = f.read().strip().split(' ')
    fs = [folder+os.sep+f'absorber_{a}' for a in ab]
    if returnAbsorberNum: return fs, list(map(int,ab))
    else: return fs


feff6_exe = ''
feff85_exe = ''


def findFEFF(feffVersion):
    global feff85_exe, feff6_exe
    feffVersion = str(feffVersion)
    assert feffVersion in ['6', '8.5']
    if feffVersion == '6':
        if feff6_exe != '': return feff6_exe
        exe = 'feff6l.exe' if os.name == 'nt' else 'feff6l'
        tr = os.path.split(os.path.abspath(__file__))[0]+os.sep+'bin'+os.sep+'Feff6ldist'+os.sep+exe
        if os.path.exists(tr):
            feff6_exe = '"'+tr+'"'
            return feff6_exe
    feff_exe = feff6_exe if feffVersion == '6' else feff85_exe
    if feff_exe != '': return feff_exe
    # search for feff
    exe_list = ['feff6l.exe', 'feff6l', 'feff6L', 'feff6'] if feffVersion == '6' else ['feff85L.exe', 'feff85.exe', 'feff85L', 'feff85']
    for exe in exe_list:
        if shutil.which(exe) is not None:
            feff_exe = exe
            break
    if feff_exe == '' and utils.isLibExists('larch'): feff_exe = 'larch'
    assert feff_exe != '', f"Can't find feff v{feffVersion} on your computer"
    if feffVersion == '6': feff6_exe = feff_exe
    else: feff85_exe = feff_exe
    return feff_exe


def runFEFFHelper(folder, feffVersion='8.5'):
    feff_exe = findFEFF(feffVersion)
    if feff_exe == 'larch':
        if feffVersion == '6':
            from larch.xafs import feff6l
            feff6l(folder=folder, feffinp='feff.inp', verbose=False)
        else:
            from larch.xafs import feff8l
            feff8l(folder=folder, feffinp='feff.inp', verbose=False)
        output = ''
    else:
        output, returncode = utils.runCommand(f"{feff_exe} feff.inp", folder, outputTxtFile='output.txt')
        if returncode != 0:
            raise Exception('Error while executing "' + feff_exe + '" command:\n' + output)
        output = output + '\n\n'
    return output


def runLocal(folder, feffVersion='8.5'):
    output = ''
    for f in getAbsorberFolders(folder):
        assert os.path.exists(f+os.sep+'feff.inp'), f"Can't find file " + f+os.sep+'feff.inp'
        output += runFEFFHelper(f, feffVersion)
    return output


def runCluster(folder, memory=5000, nProcs=1):
    for f in getAbsorberFolders(folder):
        output, returncode = utils.runCommand(f"run-cluster-and-wait -m {memory} -n {nProcs} feff85L feff.inp", f, outputTxtFile=None)


def isSuccessful(folder):
    o = folder+os.sep+'output.txt'
    assert os.path.exists(o), 'No output file'
    with open(o) as f: s = f.read()
    if 'error' in s and 'STOP' in s: return False
    else: return True


def getParams(fileName):
    with open(fileName, 'r') as f: params0 = json.load(f)
    return [p[0] for p in params0], [p[1] for p in params0]


def parseOneFolder(folder, multipleAbsorber=False):
    results = {}
    ds, absorbers = getAbsorberFolders(folder, returnAbsorberNum=True)
    for i, d in enumerate(ds):
        exafsFile = d+os.sep+'chi.dat'
        if not os.path.exists(exafsFile):
            raise Exception('Error: in folder ' + d + ' there is no output file chi.dat')
        s = utils.readSpectrum(exafsFile, guess=True, energyColumn=0, intensityColumn=1, xName='k', yName='chi')
        if s.k[0] == 0: s.changeEnergy(s.k[1:], inplace=True)
        results[absorbers[i]] = s
    # make all spectra have the same point count
    enNums = [len(results[a].k) for a in absorbers]
    maxEnInd = np.argmax(enNums)
    maxEnNum = enNums[maxEnInd]
    me = results[absorbers[maxEnInd]]
    for a in absorbers:
        if len(results[a].k) != maxEnNum:
            results[a] = utils.Spectrum(me.k, np.interp(me.k, results[a].k, results[a].chi), xName='k', yName='chi')
    if multipleAbsorber: return results
    else:
        assert len(results) == 1, 'Multiple absorbers detected. Set multipleAbsorber=True'
        return list(results.values())[0]

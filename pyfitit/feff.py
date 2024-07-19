import numpy as np
import os, tempfile, json, shutil, copy, re
from . import utils, exafs


def getPotentials(molecule, absorber, separatePotentials, feffVersion):
    """Generate ipot column for the atoms in the molecule and ipot list"""
    if feffVersion == '6': max_pot_count = 8
    elif feffVersion == '8.5': max_pot_count = 12
    else:
        assert feffVersion == '9'
        max_pot_count = 16
    if separatePotentials:
        molecule = copy.deepcopy(molecule)
        max_rad = 5
        dist = np.linalg.norm(molecule.atom-molecule.atom[[absorber]], axis=1)
        sort_ind = np.argsort(dist)
        pot_count = 1  # absorber
        H_presented = False
        atom_variety_size = len(np.unique(molecule.az))
        for i in sort_ind:
            max_count = (max_pot_count-1 if H_presented else max_pot_count) - atom_variety_size
            if pot_count >= max_count or dist[i] > max_rad: break
            if i == absorber: continue
            if molecule.az[i] == 1:
                H_presented = True
                continue
            pot_count += 1
            molecule.atomName[i] = molecule.atomName[i]+'_'+str(pot_count)
            # print(molecule.atomName[i])
        return getPotentials(molecule, absorber, separatePotentials=False, feffVersion=feffVersion)

    def getScmtFmx(z):
        if z==1: return -1,2
        else: return -1,8

    uniq_atomName, uniq_ind = np.unique(molecule.atomName, return_index=True)
    ipot = np.zeros(len(molecule.atomName), dtype=int)
    ipotList = [[0, molecule.az[absorber], molecule.atomName[absorber], *getScmtFmx(molecule.az[absorber])]]
    absorberNamesakes = np.sum(molecule.az[absorber] == molecule.az) > 1
    ip = 1
    for ii in range(len(uniq_atomName)):
        i = len(uniq_atomName) - 1 - ii
        if (not absorberNamesakes) and molecule.atomName[absorber] == uniq_atomName[i]: continue
        ind = uniq_ind[i]
        ipotList.append([ip, molecule.az[ind], molecule.atomName[ind], *getScmtFmx(molecule.az[ind])])
        ipot[molecule.atomName == molecule.atomName[ind]] = ip
        ip += 1
    ipot[absorber] = 0
    return ipot, ipotList


def generateInputUniqueAbsorber(molecule, folder, additional, absorber, feffVersion, separatePotentials, **cards):
    os.makedirs(folder, exist_ok=True)
    feffVersion = str(feffVersion)
    assert feffVersion in ['6', '8.5', '9']
    cards = {k.upper():cards[k] for k in cards}
    with open(folder + os.sep + 'feff.inp', 'w') as f:
        if feffVersion == '9':
            if 'EDGE' not in cards: cards['EDGE'] = 'K 1.0'
            if 'EXCHANGE' not in cards: cards['EXCHANGE'] = '0'
            if 'XANES' in cards and 'DIMS' not in cards:
                cards['DIMS'] = '300 8'
                if separatePotentials is None: separatePotentials = True
        else:
            if 'HOLE' not in cards: cards['HOLE'] = '1 1.0'
        if 'XANES' not in cards:
            if 'NLEG' not in cards: cards['NLEG'] = '4'
            if 'CRITERIA' not in cards: cards['CRITERIA'] = '0 0'
        if 'CONTROL' not in cards: cards['CONTROL'] = '1 1 1 1' if feffVersion == '6' else '1 1 1 1 1 1'
        if 'PRINT' not in cards: cards['PRINT'] = '0 1 1 0' if feffVersion == '6' else '0 0 0 0 0 3'
        for key in cards: f.write(f'{key:<10}{cards[key]:>10}\n\n')
        if additional != '': f.write(additional + '\n')
        if absorber is None: absorber = np.argmax(molecule.az)
        assert 0 <= absorber <= len(molecule.az)-1, f'Wrong absorber index {absorber}'
        f.write('POTENTIALS\n')
        fmt = "{:>5}{:>5}{:>7}{:>7}{:>7}\n"
        f.write(fmt.format('*ipot','Z','tag','l_scmt','l_fms'))
        if separatePotentials is None: separatePotentials = False
        ipot, ipotList = getPotentials(molecule, absorber, separatePotentials, feffVersion=feffVersion)
        for ip in ipotList:
            f.write(fmt.format(*ip))
            
        max_atom_count = 250
        sort_ind = np.argsort(np.linalg.norm(molecule.atom - molecule.atom[[absorber]], axis=1))
        sort_ind = sort_ind[:max_atom_count]
        f.write('\nATOMS\n')
        fmt = "{:>12.7f}{:>12.7f}{:>12.7f}{:>5}{:>5}{:>7.3f}\n"
        f.write("{:>12}{:>12}{:>12}{:>5}{:>5} {}\n".format('*x','y','z','ipot','tag','distance'))
        for i in sort_ind:
            x,y,z = molecule.atom[i]
            d = np.linalg.norm(molecule.atom[i]-molecule.atom[absorber])
            f.write(fmt.format(x, y, z, ipot[i], molecule.atomName[i], d))
        f.write('\nEND\n')


def generateInput(molecule, folder='', additional='', absorbers=None, feffVersion='8.5', separatePotentials=None, **cards):
    """
    Generate input for feff calculation

    :param molecule:
    :param folder:
    :param additional: extra text in input feff file
    :param absorbers: one index (from 0) of absorber atom or list of indexes. Default: take heaviest atom
    :param feffVersion: 6 or 8.5 or 9
    :param separatePotentials: True/False - generate different potentials for close to absorber atoms
    """
    if folder == '':
        if not os.path.exists("./tmp"): os.makedirs("./tmp")
        folder = tempfile.mkdtemp(dir='./tmp')
    os.makedirs(folder, exist_ok=True)
    if isinstance(absorbers, list):
        with open(folder+os.sep+'absorbers.txt', 'w') as f: f.write(' '.join(map(str, absorbers)))
        for absorber in absorbers:
            generateInputUniqueAbsorber(molecule, folder + os.sep +f'absorber_{absorber}', additional, absorber, feffVersion=feffVersion, separatePotentials=separatePotentials, **cards)
    else:
        generateInputUniqueAbsorber(molecule, folder, additional, absorbers, feffVersion=feffVersion, separatePotentials=separatePotentials, **cards)
    return folder


def getAbsorberFolders(folder, returnAbsorberNum=False):
    if not os.path.exists(folder+os.sep+'absorbers.txt'):
        if returnAbsorberNum: return [folder], [0]
        else: return [folder]
    with open(folder+os.sep+'absorbers.txt') as f: ab = f.read().strip().split(' ')
    fs = [folder+os.sep+f'absorber_{a}' for a in ab]
    if returnAbsorberNum: return fs, list(map(int,ab))
    else: return fs


feff_executables = {'6': '', '8.5': '', '9': ''}


def findFEFF(feffVersion):
    global feff_executables
    feffVersion = str(feffVersion)
    assert feffVersion in ['6', '8.5', '9']
    if feff_executables[feffVersion] != '': return feff_executables[feffVersion]
    if feffVersion == '6':
        exe = 'feff6l.exe' if os.name == 'nt' else 'feff6l'
        tr = os.path.split(os.path.abspath(__file__))[0]+os.sep+'bin'+os.sep+'Feff6ldist'+os.sep+exe
        if os.path.exists(tr):
            feff_executables[feffVersion] = '"'+tr+'"'
            return feff_executables[feffVersion]
    feff_exe = feff_executables[feffVersion]
    if feff_exe != '': return feff_exe
    # search for feff
    if feffVersion == '6':
        exe_list = ['feff6l.exe', 'feff6l', 'feff6L', 'feff6']
    elif feffVersion == '8.5':
        exe_list = ['feff85L.exe', 'feff85.exe', 'feff85L', 'feff85']
    else:
        exe_list = ['feff9', 'feff9.1', 'feff9.6']
    for exe in exe_list:
        if shutil.which(exe) is not None:
            feff_exe = exe
            break
    if feff_exe == '' and utils.isLibExists('larch') and feffVersion != '9': feff_exe = 'larch'
    assert feff_exe != '', f"Can't find feff v{feffVersion} on your computer"
    feff_executables[feffVersion] = feff_exe
    return feff_executables[feffVersion]


def runFEFFHelper(folder, feffVersion):
    feff_exe = findFEFF(feffVersion)
    if feff_exe == 'larch':
        if feffVersion == '6':
            from larch.xafs import feff6l
            feff6l(folder=folder, feffinp='feff.inp', verbose=False)
        else:
            assert feffVersion == '8.5'
            from larch.xafs import feff8l
            feff8l(folder=folder, feffinp='feff.inp', verbose=False)
        output = ''
    else:
        output, returncode = utils.runCommand(f"{feff_exe} feff.inp", folder, outputTxtFile='output.txt')
        if returncode != 0:
            raise Exception('Error while executing "' + feff_exe + '" command:\n' + output)
        output = output + '\n\n'
    return output


def runLocal(folder, feffVersion):
    output = ''
    for f in getAbsorberFolders(folder):
        assert os.path.exists(f+os.sep+'feff.inp'), f"Can't find file " + f+os.sep+'feff.inp'
        output += runFEFFHelper(f, feffVersion)
    return output


def runCluster(folder, feffVersion, memory, nProcs):
    feff_exe = findFEFF(feffVersion)
    output = ''
    for f in getAbsorberFolders(folder):
        output, returncode = utils.runCommand(f"run-cluster-and-wait -m {memory} -n {nProcs} {feff_exe} feff.inp", f, outputTxtFile=None)
        if returncode != 0:
            raise Exception(f'Error while executing "{feff_exe}" command in the folder {folder}:\n' + output)
        output = output + '\n\n'
    return output


def isSuccessful(folder):
    o = folder+os.sep+'output.txt'
    assert os.path.exists(o), 'No output file'
    with open(o) as f: s = f.read()
    if 'error' in s and 'STOP' in s: return False
    else: return True


def getParams(fileName):
    with open(fileName, 'r') as f: params0 = json.load(f)
    return [p[0] for p in params0], [p[1] for p in params0]


def is_xmu(folder):
    feffinp = folder+os.sep+'feff.inp'
    assert os.path.exists(feffinp), feffinp
    with open(feffinp, 'r') as f: s = f.read()
    for w in ['XANES', 'DANES', 'FPRIME', 'XNCD']:
        if re.search(r'^\s*'+w+r'\s', s, re.IGNORECASE | re.MULTILINE): return True
    return False


def parseOneFolder(folder, multipleAbsorber=False, xanesProcessingParams=None, returnEfermi=False):
    """
    :param xanesProcessingParams: dict with keys 'return': {'mu', 'mu0', 'chi(k)', 'chi(e)', 'mu normalized'(default)}, 'normalizeCrossFadeInterval':[75,125], 'edgeSearchLevel':0.5
    """
    results, Efermi = {}, {}
    ds, absorbers = getAbsorberFolders(folder, returnAbsorberNum=True)
    for i, d in enumerate(ds):
        if is_xmu(d):
            xanesFile = d+os.sep+'xmu.dat'
            if not os.path.exists(xanesFile):
                raise Exception('Error: in folder ' + d + ' there is no output file xmu.dat')
            defaultParams = {'return':'mu normalized', 'normalizeCrossFadeInterval':[75,125], 'edgeSearchLevel':0.5}
            if xanesProcessingParams is None:
                xanesProcessingParams = defaultParams
            assert set(xanesProcessingParams.keys()) <= {'return', 'normalizeCrossFadeInterval', 'edgeSearchLevel'}
            assert 'return' in xanesProcessingParams
            assert xanesProcessingParams['return'] in {'mu', 'mu0', 'chi(k)', 'chi(e)', 'mu normalized'}
            if xanesProcessingParams['return'] == 'mu normalized' and len(xanesProcessingParams) < len(defaultParams):
                for k in defaultParams:
                    if k not in xanesProcessingParams: xanesProcessingParams[k] = defaultParams[k]
            if xanesProcessingParams['return'] in ['mu', 'mu normalized']:
                s = utils.readSpectrum(xanesFile, guess=True, energyColumn=0, intensityColumn=3, xName='energy', yName='intensity')
                if xanesProcessingParams['return'] == 'mu normalized':
                    edge_e = s.inverse(xanesProcessingParams['edgeSearchLevel'], 'min')
                    norm = utils.readSpectrum(xanesFile, guess=True, energyColumn=0, intensityColumn=4)
                    a,b = xanesProcessingParams['normalizeCrossFadeInterval']
                    a += edge_e
                    b += edge_e
                    from . import curveFitting
                    s.y = curveFitting.crossfade(s.x, s.y, s.y-norm.y+1, a, b)
            elif xanesProcessingParams['return'] == 'mu0':
                s = utils.readSpectrum(xanesFile, guess=True, energyColumn=0, intensityColumn=4, xName='energy', yName='intensity')
            elif xanesProcessingParams['return'] == 'chi(e)':
                s = utils.readSpectrum(xanesFile, guess=True, energyColumn=0, intensityColumn=5, xName='energy', yName='intensity')
            elif xanesProcessingParams['return'] == 'chi(k)':
                s = utils.readSpectrum(xanesFile, guess=True, energyColumn=2, intensityColumn=5)
                s = exafs.Exafs(s.x, s.y)
            else: assert False
            if returnEfermi:
                tmp = utils.readSpectrum(xanesFile, guess=True, energyColumn=0, intensityColumn=2)
                energy, k = tmp.x, tmp.y
                ind = np.where(k==0)[0][0]
                Efermi[absorbers[i]] = energy[ind]
        else:
            assert not returnEfermi, 'Can\'t get Efermi when there is no xmu.dat file'
            exafsFile = d+os.sep+'chi.dat'
            if not os.path.exists(exafsFile):
                raise Exception('Error: in folder ' + d + ' there is no output file chi.dat')
            s = utils.readSpectrum(exafsFile, guess=True, energyColumn=0, intensityColumn=1)
            if s.x[0] == 0: s = s.changeEnergy(s.x[1:])
            s = exafs.Exafs(s.x, s.y)
        results[absorbers[i]] = s
    # make all spectra have the same point count
    enNums = [len(results[a].x) for a in absorbers]
    maxEnInd = np.argmax(enNums)
    maxEnNum = enNums[maxEnInd]
    me = results[absorbers[maxEnInd]]
    for a in absorbers:
        if len(results[a].x) != maxEnNum:
            newY = np.interp(me.x, results[a].x, results[a].y)
            if me.xName == 'k': results[a] = exafs.Exafs(me.x, newY)
            else: results[a] = utils.Spectrum(me.x, newY, xName=me.xName, yName=me.yName)
    if not multipleAbsorber:
        assert len(results) == 1, 'Multiple absorbers detected. Set multipleAbsorber=True'
        results = list(results.values())[0]
        if returnEfermi: Efermi = list(Efermi.values())[0]
    if returnEfermi: return results, Efermi
    else: return results

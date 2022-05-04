import json, glob, os, parsy, subprocess, re, tempfile
import time

from scipy.spatial import distance
import numpy as np
import pandas as pd
from io import StringIO
from . import utils, plotting, molecule

cycleStartStr = " Coordinates in Geometry Cycle"
floatParser = parsy.regex(r"[-+]?\d*\.\d+|\d+").map(float)
intParser = parsy.regex(r"[-+]?\d+").map(int)
newLine = parsy.regex(r'\n')
untilNewLine = parsy.regex(r'[^\n]*')


def parseLogfile(fname):

    @parsy.generate
    def parseCycles():
        cycles = yield ((parsy.string(cycleStartStr).should_fail("") >> parsy.any_char).at_least(1) >> parseCycle.optional()).many()
        return cycles

    @parsy.generate
    def parseCycle():
        num = yield parsy.string(cycleStartStr) >> parsy.whitespace >> intParser # cycle number
        yield newLine >> parsy.regex(r'[^\n]*') >> newLine # column names
        rows = yield parseRow.many() # table rows
        return rows

    @parsy.generate
    def parseRow():
        num = yield parsy.whitespace.many() >> intParser
        atom = yield parsy.string('.') >> parsy.regex(r'[a-zA-Z.]*')
        yield parsy.whitespace.many()
        x = yield floatParser
        yield parsy.whitespace.many()
        y = yield floatParser
        yield parsy.whitespace.many()
        z = yield floatParser
        return {'atom':atom, 'num':num, 'x':x, 'y':y, 'z':z}

    with open(fname) as f:
        content = f.read()

    cycles = parseCycles.parse(content)
    cycles = [x for x in cycles if x is not None] # get rid of the last null

    return cycles


def enumerateParsedXYZ(table):
    num = 1
    for atom in table:
        atom['num'] = num
        num += 1


def trimAtomNames(table):
    for atom in table:
        atom['atom']=atom['atom'].strip()
        atom['atom']=atom['atom'].split('.', 1)[0] # remove everything after dot
        

def savexyz(atoms, file):
    with open(file, 'w') as output:
        output.write(str(len(atoms)))
        output.write('\n\n')
        for a in atoms:
            i = a['atom'].find('.')
            name = a['atom'] if i<0 else a['atom'][:i]
            print("{0:>2}{1:10.6g}{2:10.6g}{3:10.6g}".format(name, a['x'], a['y'], a['z']), file=output)


def nearestAtomsDistChange(logFileName, mainAtom, firstN=5):
    """
    Check ADF geometry optimization

    :param logFileName:
    :param mainAtom: name of central atom (Cr, Fe, ...)
    :param firstN: number of checked atoms close to the central one
    """
    def getPos(atom):
        return atom['x'],atom['y'],atom['z']

    def getInterAtomicDistances(cycle):
        main = getPos(cycle[0])
        res = map(lambda x: {'atom':x['atom'], 'dist':distance.euclidean(getPos(x), main)}, cycle[1:])
        return list(res)

    logfileFolder = os.path.split(logFileName)[0]
    name = os.path.splitext(os.path.split(logFileName)[-1])[0]
    # logFileName = utils.findFile(logfileFolder, 'logfile', check_unique=False)
    cycles = parseLogfile(logFileName)
    savexyz(cycles[0], logfileFolder+os.sep+f'xyz_{name}_cycle_0.xyz')
    savexyz(cycles[-1], logfileFolder+os.sep+f'xyz_{name}_cycle_{len(cycles)}.xyz')
    assert all(map(lambda x: x[0]['atom'] == mainAtom, cycles))  # assert first atoms to be mainAtom
    for c in cycles:
        for a in c: a['atom'] = str(a['num'])+'.'+a['atom']

    inputFileName = logfileFolder+os.sep+name+'.job'
    if not os.path.exists(inputFileName):
        inputFileName = logfileFolder+os.sep+name+'.run'
        if not os.path.exists(inputFileName):
            print('Can\'t find input .run or .job file for log',logFileName)
            return None
    originalMolecule = extractMoleculeFromFile(inputFileName)
    originalAtomNames = [str(i+1)+'.'+originalMolecule.atomName[i] for i in range(len(originalMolecule.atomName))]
    # enumerateParsedXYZ(originalMolecule)
    # trimAtomNames(originalMolecule)
    # trimAtomNames(cycles[-1])
    assert all(map(lambda x: x[0] == x[1]['atom'], zip(originalAtomNames, cycles[-1]))), "Atoms in original molecule and optimized are different"
    dist = np.linalg.norm(originalMolecule.atom - originalMolecule.atom[0], axis=1)
    ind = np.argsort(dist)[1:]

    # find appropriate atoms in the last cycle
    lastDist = getInterAtomicDistances(cycles[-1])
    sortedDist = []
    if firstN > len(ind): firstN = len(ind)
    for i in range(firstN):
        try:
            elem = next(x for x in lastDist if x['atom'] == originalAtomNames[ind[i]])
        except StopIteration as e:
            print(originalAtomNames[ind[i]]+' from the first cycle was not found in the last cycle for experiment ' + logfileFolder+os.sep+name)
            raise e
        sortedDist.append(elem)
    lastDist = sortedDist

    firstDist = [{'atom':originalAtomNames[ind[i]], 'dist':dist[ind[i]]} for i in range(len(ind))]
    zipped = list(zip(firstDist, lastDist))[:firstN]
    res = map(lambda x: {'diff%': (abs(x[0]['dist'] - x[1]['dist']) / max(x[0]['dist'], x[1]['dist'])) * 100, 'atom':x[0]['atom']}, zipped)
    assert all(map(lambda x: x[0]['atom'] == x[1]['atom'], zipped)), "Nearest atoms have different names"

    return list(res)


def generateInput(molecule, cards, lowest=None, folder=''):
    if folder == '':
        if not os.path.exists("./tmp"): os.makedirs("./tmp")
        folder = tempfile.mkdtemp(dir='./tmp')
    else:
        # if os.path.exists(folder):shutil.rmtree(folder)
        assert not os.path.exists(folder), 'Folder '+folder+' exists!'
        os.makedirs(folder, exist_ok=True)

    lowestRe = r'^lowest.*?\d+$'
    if lowest is not None:
        newLowest = 'lowest '+str(lowest)
        assert re.search(lowestRe, cards, flags=re.MULTILINE) is not None, 'Can\'t find lowest string in cards to replace by '+newLowest
        cards = re.sub(lowestRe, newLowest, cards, flags=re.MULTILINE)
    # delete ZORA for spectrum calc
    if re.search(lowestRe, cards, flags=re.MULTILINE) is not None:
        cards = re.sub(r'^.*?ZORA.*?\n', '', cards, flags=re.MULTILINE)

    with open(folder + '/input.run', 'w') as f:
        f.write('#! /bin/sh\n\n"$ADFBIN/adf" <<eor\nATOMS\n')
        mol = molecule
        #ids = np.arange(1,mol.atom.shape[0]+1)
        for i in range(len(mol.atom)):
            f.write('{} {} {} {} {}\n'.format(i+1, mol.atomName[i], mol.atom[i][0],mol.atom[i][1],mol.atom[i][2]))
        f.write('END\n\n')
        f.write(cards+'\n\n')
        f.write('eor\n\n')


def convertRunToJob(folder):
    runFile = utils.findFile(folder,'.run')
    runFile = os.path.split(runFile)[1]
    proc = subprocess.Popen(['convertRun2Job', runFile], cwd=folder, stdout=subprocess.PIPE)
    return os.path.splitext(runFile)[0]+'.job'


def runLocal(folder='.'):
    jobFile = convertRunToJob(folder)
    proc = subprocess.Popen(['.'+os.sep+jobFile], cwd=folder, stdout=subprocess.PIPE)
    proc.wait()
    if proc.returncode != 0:
      raise Exception('Error while executing "'+jobFile+'" command')
    return proc.stdout.read()


def runCluster(folder='.', memory=5000, nProcs=6):
    jobFile = convertRunToJob(folder)
    proc = subprocess.Popen(["run-cluster-and-wait", "-m", str(memory), '-n', str(nProcs), '.'+os.sep+jobFile], cwd=folder, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.wait()


def getParams(fileName):
    with open(fileName, 'r') as f: params0 = json.load(f)
    return [p[0] for p in params0], [p[1] for p in params0]


def parseOutFile(fileName):
    with open(fileName, 'r') as f: s = f.read()
    i = s.find('All individual components to the origin independent oscillator strength')
    if i < 0: return None
    i = s.find('----------\n',i)
    if i < 0 : i = s.find('----------\r\n',i)
    if i < 0: return None
    i = s.find('\n', i)+1
    j = s.find('End of higher-order oscillator strengths', i)
    if j<0: return None
    s = s[i:j]
    return pd.read_csv(StringIO(s), header=None, names=['N','colon', 'E', 'fmu', 'fQ', 'fm', 'fOmu', 'fMmu','ftot'], sep=r'\s+')


def extractMoleculeFromFile(fileName, fileType='.run'):
    assert fileType[0] == '.', f'fileType must be a file extension start with dot: .???'
    with open(fileName, 'r') as f: s = f.read()
    i = re.search(r"\n\s*Atoms\s*\n", s,  re.IGNORECASE)
    assert i is not None
    i = i.end()
    j = re.search(r"\n\s*END\s*\n", s[i:],  re.IGNORECASE)
    assert j is not None
    j = j.start()
    s = s[i:i+j]
    if fileType == '.out': return molecule.Molecule.fromXYZcontent(s)
    p = ''
    lines = s.split('\n')
    for line in lines:
        # print(line)
        if fileType in ['.run', '.job']:
            p += ' '.join(line.strip().split(' ')[1:]) + '\n'
        else:
            assert False, f'Unknown file type {fileType}'
    return molecule.Molecule.fromXYZcontent(p)


def parse_one_folder(folder, makePiramids=False):
    xanesFile = utils.findFile(folder,'.out', check_unique=False)
    if xanesFile is None:
        raise Exception('Error: in folder '+folder+' there is no output file')
    try:
        table = parseOutFile(xanesFile)
        energy = table.loc[:,'E'].ravel()
        absorb = table.loc[:,'ftot'].ravel()
    except:
        raise Exception('Can\'t parse output file in folder '+folder)
    if makePiramids:
        energy, absorb = utils.makePiramids(energy, absorb, 0.01)
    return utils.Spectrum(energy, absorb), None


def nextWord(s, i0, ignoreNewLine=False):
    """
    Find word from position i
    :param s: text string
    :param i0: current position
    :param ignoreNewLine: pass newline, when search for word
    :return: word, positionJustAfterWord. None, None if end of string or newline encountered
    """
    if i0 >= len(s): return None, None
    allwhitespace = [' ', '\t', '\n', '\r']
    whitespace = [' ', '\t']
    if ignoreNewLine: whitespace = allwhitespace
    i = i0
    while s[i] in whitespace:
        i += 1
        if i >= len(s): return None, None
    if s[i] in allwhitespace: return None, None  # pass newline and ignoreNewLine=False
    k = i
    while s[k] not in allwhitespace:
        k += 1
        if k >= len(s): return s[i:k], k
    word = s[i:k]
    return word, k


def nextLine(s, i0):
    """
    Find first element of next line in string s from position i
    :return: element position (None is end of string encountered)
    """
    assert i0 >= 0
    if (i0 is None) or (i0 >= len(s)): return None
    i = i0
    while s[i] != '\n':
        i += 1
        if i >= len(s): return None
    i += 1
    if i >= len(s): return None
    return i


def parseNumberSpinMO_Excitations(output, MOcount):
    # read MOs responsible for excitations and their spin
    i = output.rfind('Major MO -> MO transitions for the above excitations')
    for l in range(9): i = nextLine(output, i)
    end_i = output.find('\n \n\n', i)
    number = [];    spin = [];    MO = []
    j = 1
    twoSpins = True
    while True:
        num, i = nextWord(output, i)
        assert num[:-1] == str(j), f'Wrong NumberSpinMO line format: {num[:-1]} != {j} \n' + output[i:i+300]
        num = int(num[:-1])
        number.append(num)

        sp, i = nextWord(output, i)
        if sp not in ['Alph', 'Beta']:
            sp = 'Alph'   # only one spin (restriction calculation)
            twoSpins = False
        spin.append(sp)

        i = output.find('->', i)
        i += 2
        mo, i = nextWord(output, i)
        MO.append(mo)

        i = output.find('\n\n\n', i)
        if i > end_i or i < 0: break
        i += 3
        j += 1
        if j > MOcount: break
    return number, spin, MO, twoSpins


def getNumberSpinMO_DOS(output, MOcount):
    # find mo_a values
    i = output.find('Irreducible Representations, including subspecies')
    assert i >= 0
    i = nextLine(output, i)
    i = nextLine(output, i)
    i2 = output.find('\n\n', i)
    assert i2>0
    syms = output[i:i2].strip().split('\n')
    for i in range(len(syms)): syms[i] = syms[i].strip()
    number = [];    spin = [];     MO = []
    for sym in syms:
        for l in range(MOcount):
            number.append(l + 1)
            spin.append('Alph')
            MO.append(str(l + 1) + sym)
        for l in range(MOcount):
            number.append(MOcount + l + 1)
            spin.append('Beta')
            MO.append(str(l + 1) + sym)
    return number, spin, MO


def findMin(s, start_ind, *substr):
    i = len(s)
    for ss in substr:
      i1 = s.find(ss, start_ind)
      if i1 >= 0 and i1 < i: i = i1
    if i == len(s): return -1
    else: return i


def parseExcitationsDOS_oneFile(filename, MOcount, fragment, fragment_l, DOS, printOutput=False):
    with open(filename, 'r') as file: output = file.read()
    if DOS:
        number, spin, MO = getNumberSpinMO_DOS(output, MOcount)
    else:
        number, spin, MO, twoSpins = parseNumberSpinMO_Excitations(output, MOcount)
        if printOutput: print(len(MO), 'transitions found')
    df = pd.DataFrame()
    df['Number'] = np.array(number)
    df['Spin'] = np.array(spin)
    df['MO'] = np.array(MO)

    i = output.rfind("List of all MOs, ordered by energy, with the most significant SFO gross populations")
    i1 = output.find(" *** SPIN 1 ***", i)
    if i1 < 0:
        if not DOS:
            assert not twoSpins
        twoSpins = False
        df = df.loc[df['Spin']=='Alph']
        df.reset_index(inplace=True)
        assert len(np.unique(df['Spin'])) == 1, str(df['Spin'])
        for l in range(13): i = nextLine(output, i)
    else:
        if not DOS:
            assert twoSpins
        twoSpins = True
        i = i1
        for l in range(6): i = nextLine(output, i)
    k = findMin(output, i, "\n\n", "\n \n")
    assert k >= 0
    spin_str = [output[i:k]]
    word, i = nextWord(output, k, ignoreNewLine=True)
    if twoSpins:
        assert word == '***'
        assert output[i + 1:i + 7] == 'SPIN 2'
        for l in range(6): i = nextLine(output, i)
        k = findMin(output, i, "\n \n", "\n\n", "\n  pauli")
        assert k >= 0
        spin_str.append(output[i:k])
    spin_df = []
    for s in spin_str:
        s = np.array([s], dtype=str)
        s = s.view('U1')
        i = 0
        lastHead = s[i:i + 30]
        spaces = np.array([' '] * 30)
        while i is not None:
            if np.all(s[i:i + 30] == spaces): s[i:i + 30] = lastHead
            else: lastHead = s[i:i + 30]
            i = nextLine(s, i)
            # if i > 100000: break
        s = s.tostring()[::4].decode()
        data = pd.read_csv(StringIO(s), sep=r'\s+', header=None, skipinitialspace=True, names=['E1', 'Occ1', 'MO_N', 'MO_A', 'percent', 'SFO_N', 'SFO_L', 'E2', 'Occ2', 'Fragment_N', 'Fragment_L'])
        dt = data.dtypes
        assert dt['E1'] == np.float64 and dt['Occ1'] == np.float64 and dt['MO_N'] == np.int64 and dt['MO_A'] == object and dt['percent'] == object and dt['SFO_N'] == np.int64 and dt['SFO_L'] == object and dt['E2'] == np.float64 and dt['Occ2'] == np.float64 and dt['Fragment_N'] == np.int64 and dt['Fragment_L'] == object, str(dt)
        assert data.shape[1] == 11
        for c in data.columns:
            if np.sum(pd.isna(data[c])) > 0:
                ind = np.where(pd.isna(data[c]))[0][0]
                print('There is NaN in column ' + c + '. Last good lines:')
                print(data.loc[ind - 3:ind + 3])
                exit(1)
        spin_df.append(data)
    #        print(spin_df[1])
    if printOutput and not DOS: print(data['SFO_L'])

    orbitals = ["D:xy", "D:xz", "D:yz", "D:x2-y2", "D:z2"]
    if DOS:
        # initialize arrays to store energies and occupations for MOs
        energy = np.zeros(df.shape[0])
        occ = np.zeros(df.shape[0])
    for orb in orbitals:
        n = df.shape[0]
        percent = np.zeros(n)
        t0 = time.time()
        for i in range(n):
            mo = df['MO'][i]
            j = re.search(r"[^\d]", mo).start()
            mo_n = int(mo[:j])
            mo_a = mo[j:]
            spin = df['Spin'][i]
            assert spin in ['Alph', 'Beta']
            if not twoSpins: assert spin == 'Alph', spin
            sdf = spin_df[0] if spin == 'Alph' else spin_df[1]
            ind = (sdf['MO_N'] == mo_n) & (sdf['MO_A'] == mo_a) & (sdf['SFO_N'] == 1) & (sdf['SFO_L'] == orb) & (sdf['Fragment_L'] == fragment_l) & (sdf['Fragment_N'] == fragment)
            #if mo == '24AA' and orb == 'D:x2-y2':
                #print(sdf.dtypes)
                #print(mo, mo_n, mo_a, orb, fragment, fragment_l)
                #print(sdf.loc[359:361])
                #iii = 360
                #print((sdf.loc[iii,'MO_N'] == mo_n) & (sdf.loc[iii,'MO_A'] == mo_a) & (sdf.loc[iii,'SFO_N'] == 1) & (sdf.loc[iii,'SFO_L'] == orb) & (sdf.loc[iii,'Fragment_L'] == fragment_l) & (sdf.loc[iii,'Fragment_N'] == fragment))
                #exit(0)
            sum_ind = np.sum(ind)
            assert sum_ind <= 1
            if sum_ind == 1:
                p = sdf['percent'][ind].values[0]
                percent[i] = float(p[:-1])
                if DOS:
                    e = sdf['E1'][ind].values[0]
                    o = sdf['Occ1'][ind].values[0]
                    assert (energy[i] == 0) or (energy[i] == e)
                    energy[i] = e
                    occ[i] = o
            else:
                assert sum_ind == 0, str(sdf.loc[ind])
            if printOutput:
                pdone = i * 100 // (n - 1)
                if pdone != (i - 1) * 100 // (n - 1):
                    t = time.time()
                    if pdone > 0:
                        print('Orbital ' + orb + '.', pdone, '% done. Left: ', (100 - pdone) * (t - t0) // pdone, 's')
        df[orb] = percent
        if printOutput: print(df[orb])
    if DOS:
        df['E'] = energy
        df['Occ'] = occ
        # change order of columns
        df = df[['E', 'Number', 'Spin', 'MO', "D:xy", "D:xz", "D:yz", "D:x2-y2", "D:z2", 'Occ']]
        df.loc[df['Spin'] == 'Beta', orbitals] = -df.loc[df['Spin'] == 'Beta', orbitals]
    return df


def parseExcitationsDOS(folder, MOcount, fragment, fragment_l, DOS, printOutput=False):
    """
    Parse all ADF output files from folder.

    :param folder:
    :param MOcount:
    :param fragment: fragment number according to FRAGMENTS section in .out file
    :param fragment_l: name of metallic atom
    :param DOS: True - parse DOS, False - parse excitations
    """
    fnames = glob.glob(folder + os.sep + '*.out')
    from natsort import natsorted
    fnames = natsorted(fnames)
    print(fnames)

    for filename in fnames:
        if printOutput: print('File', filename)
        df = parseExcitationsDOS_oneFile(filename, MOcount, fragment, fragment_l, DOS, printOutput)
        if DOS:
            df.to_csv(os.path.splitext(filename)[0] + '_DOS_3d.csv', index=False)
        else:
            df.to_csv(os.path.splitext(filename)[0] + '_excitations_3d.csv', index=False)


def parse_ADFEmis(fileName, returnDOS=False):
    with open(fileName, 'r') as f: s = f.read()
    s = s.strip()
    lines = s.split('\n')
    header = lines[0]
    words = header.strip().split(' ')
    energyCol = words.index('E_ADFEmis')
    intensityCol = words.index('Intensity')
    DOS_exists = 'd_DOS' in words
    if DOS_exists:
        dosEnCol = words.index('E_DOS')
        dosCol = words.index('d_DOS')
    lines = lines[1:]
    energy = []; intensity = []; DOS_en = []; DOS = []
    for l in lines:
        words = l.strip().split(' ')
        if len(words)<4: break
        energy.append(float(words[energyCol]))
        intensity.append(float(words[intensityCol]))
        if DOS_exists and len(words)>max(dosEnCol,dosCol):
            DOS_en.append(float(words[dosEnCol]))
            DOS.append(float(words[dosCol]))
    spectrum = utils.Spectrum(np.array(energy), np.array(intensity))
    dos_sp = None
    if DOS_exists:
        DOS_en = np.array(DOS_en); DOS = np.array(DOS)
        DOS_en,DOS = utils.fixMultiValue(DOS_en, DOS, gatherOp='sum')
        DOS_en, DOS = utils.makeBars(DOS_en, DOS, base=0)
        dos_sp = utils.Spectrum(DOS_en, DOS)
    if returnDOS: return spectrum, dos_sp
    else: return spectrum


def spectrumExcitations(directory, N_points, Elarge, Gamma_max, Gamma_hole):
    fnames = glob.glob(directory + '/*.out')
    from natsort import natsorted
    fnames = natsorted(fnames)
    print(fnames)

    for f1 in fnames:
        file = open(f1, 'r')
        output = file.read()
        file.close()

        # prepare the array of transitions Exc. We use findr to find first occurence from the end, since excitations section is printed several times
        i = output.rfind('No.            E/eV        f(mu)            f(Q)           f(m)          f(O-mu)        f(M-mu)         f(tot)')
        i = output.find('\n', i) + 1
        i = output.find('\n', i) + 1
        j = output.find('  End of higher-order oscillator strengths (X-ray spectroscopy)', i)
        excitations = output[i:j - 3]

        Exc = pd.read_csv(StringIO(excitations), sep='\s+', names=['Number', 'Energy', '3', '4', '5', '6', '7', 'Intensity'])
        print(Exc)
        Exc_numpy = np.zeros((N_points, 2))

        Efermi = Exc['Energy'][1]  # set Fermi energy
        print(Efermi)
        Ei = Efermi - 15  # initial energy
        Ef = Efermi + 55  # final energy in convoluted spectrum
        Nexc = len(Exc.index)
        print(Nexc)

        for i in range(0, N_points):
            print(str(int(i / N_points * 100)) + '%')
            Exc_numpy[i, 0] = Ei + (i) / N_points * (Ef - Ei)
            for j in range(1, Nexc):
                conv = Gamma_hole + 2 / np.pi * np.arctan((Exc_numpy[i, 0] - Efermi) / Elarge) * Gamma_max
                Exc_numpy[i, 1] = Exc_numpy[i, 1] + (Exc['3'] + Exc['4'])[j] * conv / ((Exc['Energy'][j] - Exc_numpy[i, 0]) ** 2 + conv ** 2)

        plotting.plotToFile(Exc_numpy[:,0], Exc_numpy[:,1], 'spectrum', fileName=directory + '_spectrum_' + str(Gamma_hole) + 'eV.png', save_csv=True)
        # file = open(filename.split('.')[0]+'_excitations.txt', 'w') #use only number 15 from 15.out
        Exc.to_csv(directory + '_excitations.txt', line_terminator='\n', index=False)

import json, glob, os, parsy, subprocess, re, tempfile
import time

from scipy.spatial import distance
import numpy as np
import pandas as pd
from io import StringIO
from . import utils

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


def parseXYZFile(fname):

    @parsy.generate
    def parseHeader():
        num = yield intParser # number of atoms
        yield newLine >> untilNewLine >> newLine # comment
        return num

    @parsy.generate
    def parseRow():
        atom = yield parsy.regex(r'\s*[a-zA-Z.]*')
        yield parsy.whitespace.many()
        x = yield floatParser
        yield parsy.whitespace.many()
        y = yield floatParser
        yield parsy.whitespace.many()
        z = yield floatParser
        return {'atom':atom, 'x':x, 'y':y, 'z':z}

    @parsy.generate
    def parseContent():
        yield parseHeader.optional()
        table = yield parseRow.many()
        yield parsy.whitespace.many()
        return table

    with open(fname, encoding='utf-8') as f:
        content = f.read()

    table = parseContent.parse(content)
    return table


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


def nearestAtomsDistChange(originalMoleculePath, logfileFolder, firstN=5):
    def getPos(atom):
        return atom['x'],atom['y'],atom['z']

    def getInterAtomicDistances(cycle):
        cr = getPos(cycle[0])
        res = map(lambda x: {'atom':str(x['num'])+'.'+x['atom'], 'dist':distance.euclidean(getPos(x), cr)}, cycle[1:])
        return list(res)

    logFname = utils.findFile(logfileFolder, 'logfile', check_unique=False)
    if logFname is None: return None
    cycles = parseLogfile(logFname)
    savexyz(cycles[0], logfileFolder+os.sep+'atoms_cycle_first.xyz')
    savexyz(cycles[-1], logfileFolder+os.sep+'atoms_cycle_last.xyz')
    savexyz(cycles[-1], logfileFolder+os.sep+os.path.split(logfileFolder)[1]+'.xyz')
    assert all(map(lambda x: x[0]['atom'] == 'Cr', cycles)) # assert first atoms to be Cr

    originalMolecule = parseXYZFile(originalMoleculePath)
    enumerateParsedXYZ(originalMolecule)
    trimAtomNames(originalMolecule)
    trimAtomNames(cycles[-1])
    assert all(map(lambda x: x[0]['atom'] == x[1]['atom'], zip(originalMolecule, cycles[-1]))), "Atoms in original molecule and optimized are different"
    firstDist = sorted(getInterAtomicDistances(originalMolecule), key=lambda x: x['dist'])

    # find appropriate atoms in the last cycle
    lastDist = getInterAtomicDistances(cycles[-1])
    sortedDist = []
    for i in range(firstN):
        try:
            elem = next(x for x in lastDist if x['atom'] == firstDist[i]['atom'])
        except StopIteration as e:
            print(firstDist[i]['atom']+' from the first cycle was not found in the last cycle for experiment ' + os.path.split(logfileFolder)[1])
            raise e
        sortedDist.append(elem)
    lastDist = sortedDist

    zipped = list(zip(firstDist, lastDist))[:firstN]
    res = map(lambda x: {'diff%': (abs(x[0]['dist'] - x[1]['dist']) / max(x[0]['dist'], x[1]['dist'])) * 100, 'atom':x[0]['atom']}, zipped)
    assert all(map(lambda x: x[0]['atom'] == x[1]['atom'], zipped)), "Nearest atoms have different names"

    return list(res)


def generateInput(molecule, cards, lowest=None, folder = ''):
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


def runLocal(folder = '.'):
    jobFile = convertRunToJob(folder)
    proc = subprocess.Popen(['.'+os.sep+jobFile], cwd=folder, stdout=subprocess.PIPE)
    proc.wait()
    if proc.returncode != 0:
      raise Exception('Error while executing "'+jobFile+'" command')
    return proc.stdout.read()


def runCluster(folder = '.', memory=5000, nProcs = 6):
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


def parseRunFile(fileName, xyz_result_filename):
    with open(fileName, 'r') as f: s = f.read()
    i = s.find('\nATOMS\n')
    assert i >= 0
    i += len('\nATOMS\n')
    j = s.find('\nEND', i)
    assert j >= 0
    s = s[i:j]
    p = ''
    lines = s.split('\n')
    for line in lines:
        # print(line)
        p += ' '.join(line.strip().split(' ')[1:]) + '\n'
    with open(xyz_result_filename, 'w') as xyzf:
        xyzf.write(str(len(lines))+'\n\n'+p)
    return 0

def parse_one_folder(folder, makePiramids = False):
    xanesFile = utils.findFile(folder,'.out', check_unique = False)
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


def parse_all_folders(parentFolder, printOutput=True):
    df_rows = []
    energies0 = np.zeros(1)
    atomColumnNames = []
    subfolders = [f for f in os.listdir(parentFolder) if os.path.isdir(os.path.join(parentFolder,f))]
    subfolders.sort()
    badFolders = []; allXanes = {}
    for i in range(len(subfolders)):
        d = subfolders[i]
        xanesFile = utils.findFile(os.path.join(parentFolder, d),'.out', check_unique = False)
        if xanesFile is None:
            print('Error: in folder '+d+' there is no output file')
            badFolders.append(d)
            continue
        table = parseOutFile(xanesFile)
        if table is None:
            print('Can\'t parse xanes table in .out file of folder '+d)
            badFolders.append(d)
            continue
        allXanes[d] = table
    if len(allXanes)==0: print('None good folders'); return None, None, None
    energyCount = np.array([ x.shape[0] for x in allXanes.values() ])
    maxEnergyCount = np.max(energyCount)
    for d in allXanes:
        if allXanes[d].shape[0] != maxEnergyCount:
            print('Error: in folder '+d+' there are less energies '+str(allXanes[d].shape[0]))
            badFolders.append(d)
    goodFolders = list(set(subfolders) - set(badFolders))
    if len(goodFolders)==0: print('None good folders'); return None, None, None
    energies = allXanes[goodFolders[0]].loc[:,'E'].ravel()
    paramNames, _ = getParams(os.path.join(parentFolder, goodFolders[0], 'geometryParams.txt'))
    n = len(goodFolders)
    df_xanes = np.zeros([n, energies.size])
    df_params = np.zeros([n, len(paramNames)])
    for i in range(n):
        d = goodFolders[i]
        _, params = getParams(os.path.join(parentFolder, d, 'geometryParams.txt'))
        df_params[i,:] = np.array(params)
        df_xanes[i,:] = allXanes[d].loc[:,'ftot'].ravel()
    df_xanes = pd.DataFrame(data=df_xanes, columns=['e_'+str(e) for e in energies])
    df_params = pd.DataFrame(data=df_params, columns=paramNames)
    return df_xanes, df_params, badFolders


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
    if (i0 is None) or (i0 >= len(s)): return None
    i = i0
    while s[i] != '\n':
        i += 1
        if i >= len(s): return None
    i += 1
    if i >= len(s): return None
    return i


def parseExcitations_DOS(folder, MOcount, fragment, printOutput=False):
    fnames = glob.glob(folder+os.sep+'*.out')
    fnames = natsorted(fnames)
    print(fnames)
     
    for filename in fnames:
        if printOutput: print('File', filename)
        with open(filename, 'r') as file: output = file.read()
        # read MOs responsible for excitations and their spin
        i = output.rfind('Major MO -> MO transitions for the above excitations')
        for l in range(9): i = nextLine(output, i)
        number = []; spin = []; MO = []
        j = 1
        while True:
            num, i = nextWord(output, i)
            if num[:-1] != str(j): break
            num = int(num[:-1])
            number.append(num)

            sp, i = nextWord(output, i)
            spin.append(sp)

            i = output.find('->', i)
            i += 2
            mo, i = nextWord(output, i)
            MO.append(mo)

            for l in range(7): i = nextLine(output, i)
            j += 1
            if j > MOcount: break

        if printOutput: print(j-1, 'transitions found')
        df = pd.DataFrame()
        df['Number'] = np.array(number)
        df['Spin'] = np.array(spin)
        df['MO'] = np.array(MO)
        
        i = output.rfind("List of all MOs, ordered by energy, with the most significant SFO gross populations")
        i = output.find(" *** SPIN 1 ***", i)
        for l in range(6): i = nextLine(output, i)
        k = output.find("\n\n", i)
        spin_str = [output[i:k]]
        word, i = nextWord(output, k, ignoreNewLine=True)
        assert word == '***'
        assert output[i+1:i+7] == 'SPIN 2'
        for l in range(6): i = nextLine(output, i)
        k1 = output.find("\n \n", i)
        k2 = output.find("\n  pauli", i)
        k = min(k1,k2)
        spin_str.append(output[i:k])
        spin_df = []
        for s in spin_str:
            s = np.array([s], dtype=str)
            s = s.view('U1')
            i = 0
            lastHead = s[i:i+30]
            spaces = np.array([' ']*30)
            while i is not None:
                if np.all(s[i:i+30] == spaces): s[i:i+30] = lastHead
                else: lastHead = s[i:i+30]
                i = nextLine(s, i)
                # if i > 100000: break
            s = s.tostring()[::4].decode()
            data = pd.read_csv(StringIO(s), sep=r'\s+', header=None, skipinitialspace=True, names=['E1','Occ1','MO_N', 'MO_A', 'percent', 'SFO_N', 'SFO_L', 'E2', 'Occ2', 'Fragment_N', 'Fragment_L'])
            for c in data.columns:
                if np.sum(pd.isna(data[c]))>0:
                    ind = np.where(pd.isna(data[c]))[0][0]
                    print('There is NaN in column '+c+'. Last good lines:')
                    print(data.loc[ind-3:ind+3])
                    exit(1)
            spin_df.append(data)
#        print(spin_df[1])

        orbitals = ["D:xy", "D:xz", "D:yz", "D:x2-y2", "D:z2"]
        #initialize arrays to store energies and occupations for MOs
        energy = np.zeros(df.shape[0])
        occ = np.zeros(df.shape[0])
        for orb in orbitals:
            n = df.shape[0]
            percent = np.zeros(n)
            t0 = time.time()
            for i in range(n):
                mo = df['MO'][i]
                mo_n = int(mo[:-1])
                mo_a = mo[-1].upper()
                spin = df['Spin'][i]
                assert spin in ['Alph', 'Beta']
                sdf = spin_df[0] if spin == 'Alph' else spin_df[1]
                # ind0 = (sdf['MO_N'] == mo_n) & (sdf['SFO_N'] == 1) & (sdf['SFO_L'] == orb)
                # if np.sum(ind0) > 0:
                #     print(sdf.loc[ind0])
                ind = (sdf['MO_N'] == mo_n) & (sdf['MO_A'] == mo_a) & (sdf['SFO_N'] == 1) & (sdf['SFO_L'] == orb) & (sdf['Fragment_L'] == 'Cr') & (sdf['Fragment_N'] == fragment)
                sum_ind = np.sum(ind)
                assert sum_ind <= 1
                if sum_ind == 1:
                    p = sdf['percent'][ind].values[0]
                    percent[i] = float(p[:-1])
                    e = sdf['E1'][ind].values[0]
                    o = sdf['Occ1'][ind].values[0]
                    assert (energy[i]==0) or (energy[i]==e)
                    energy[i] = e
                    occ[i] = o
                else:
                    assert sum_ind==0, str(sdf.loc[ind])
                if printOutput:
                    pdone = i*100 // (n-1)
                    if pdone != (i-1)*100 // (n-1):
                        t = time.time()
                        if pdone > 0:
                            print('Orbital ' + orb +'.', pdone,'% done. Left: ',(100-pdone)*(t-t0)//pdone,'s')
            df[orb] = percent
        df['E'] = energy
        df['Occ'] = occ
        #change order of columns
        df = df[['E', 'Number', 'Spin', 'MO', "D:xy", "D:xz", "D:yz", "D:x2-y2", "D:z2",'Occ']]
        df.loc[df['Spin'] == 'Beta', orbitals] = -df.loc[df['Spin'] == 'Beta', orbitals]
        df.to_csv(filename[:-4]+'_DOS_3d.csv', index=False)
        #df.to_csv(os.path.splitext(filename)[0]+'_DOS_3d.csv', index=False)

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
        DOS_en_unique = np.unique(DOS_en)
        DOS_unique = np.zeros(len(DOS_en_unique))
        for i in range(len(DOS_en_unique)):
            DOS_unique[i] = np.sum(DOS[DOS_en == DOS_en_unique[i]])
        # make bars
        de = np.min(DOS_en_unique[1:]-DOS_en_unique[:-1])
        DOS_en_unique = np.concatenate((DOS_en_unique, DOS_en_unique-de/3, DOS_en_unique+de/3))
        DOS_unique = np.concatenate((DOS_unique, np.zeros(len(DOS_unique)), np.zeros(len(DOS_unique))))
        ind = np.argsort(DOS_en_unique)
        DOS_en_unique = DOS_en_unique[ind]
        DOS_unique = DOS_unique[ind]
        
        dos_sp = utils.Spectrum(DOS_en_unique, DOS_unique)
    if returnDOS: return spectrum, dos_sp
    else: return spectrum

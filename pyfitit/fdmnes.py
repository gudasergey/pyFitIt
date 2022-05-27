import numpy as np
import pandas as pd
from io import StringIO
import os, subprocess, tempfile, json, glob, shutil, traceback
from . import utils, geometry
from .molecule import atom_names, Molecule

useEpsiiShift = True


# если строка folder пустая - создает внутри папки ./tmp и возвращает путь к созданной папке
# energyRange - строка в формате FDMNES
# electronTransfer = ['lineInFDMNESInputAtom1', 'lineInFDMNESInputAtom2', ...]
# пока поддерживается только переброс электронов сразу всем атомам заданного типа
def generateInput(molecule, radius=5, folder='', Adimp=None, Quadrupole=False, Convolution='', Absorber=1, Green=False, Edge='K', cellSize=1.0, electronTransfer=None, additional='', crystal=False, **other):
    if 'Energy range' in other: energyRange = other['Energy range']
    else: energyRange = '-15 0.02 8 0.1 18 0.5 30 2 54 3 117'
    if 'energyRange' in other: energyRange = other['energyRange']
    if folder == '':
        if not os.path.exists("./tmp"): os.makedirs("./tmp")
        folder = tempfile.mkdtemp(dir='./tmp')
    else:
        folder = utils.fixPath(folder)
        if not os.path.exists(folder): os.makedirs(folder)

    with open(folder + '/in.txt', 'w') as f:
        f.write('Filout\n')
        f.write('out\n\n')
        f.write('Radius\n')
        if isinstance(radius, str): f.write(radius+'\n\n')
        else: f.write('%.2f\n\n' % radius)
        if Green: f.write('Green\n\n')
        if Quadrupole: f.write('Quadrupole\n\n')
        f.write('Absorber\n')
        f.write(str(Absorber)+'\n\n')
        f.write('Range\n')
        f.write(energyRange+'\n\n')
        if Adimp is not None: f.write('Adimp\n'+str(Adimp)+'\n\n')
        if Edge != 'K': f.write('Edge\n'+Edge+'\n\n')
        if additional != '' and additional is not None: f.write(additional)
        if electronTransfer is not None:
            f.write('Atom\n')
            for e in electronTransfer: f.write(e+'\n')
            f.write('\n')
        if molecule.latticeVectors is None:
            if crystal: f.write('Crystal\n')
            else: f.write('Molecule\n')
            c = str(cellSize)
            f.write(c+' '+c+' '+c +' 90 90 90\n')
            atom = molecule.atom
        else:
            assert crystal
            assert cellSize == 1
            f.write('Crystal\n')
            d = np.linalg.norm(molecule.latticeVectors, axis=1)
            angle1 = geometry.calcAngle(molecule.latticeVectors[0], molecule.latticeVectors[1])/np.pi*180
            angle2 = geometry.calcAngle(molecule.latticeVectors[1], molecule.latticeVectors[2])/np.pi*180
            angle3 = geometry.calcAngle(molecule.latticeVectors[0], molecule.latticeVectors[2])/np.pi*180
            f.write(f'{d[0]} {d[1]} {d[2]} {angle1} {angle2} {angle3}\n')
            # FDMNES require atom coordinates to be in latticeVectors basis, but xyz extended format express coordinates in Decart basis
            P = np.linalg.inv(molecule.latticeVectors.T)
            atom = np.dot(P, molecule.atom.T).T
        for i in range(atom.shape[0]):
            a = atom[i, :]; az = molecule.az[i]
            if electronTransfer is None: atomInd = az
            else:
                s = str(az)+' '
                indexes = [j for j,e in enumerate(electronTransfer) if e[:len(s)] == s]
                assert len(indexes)==1, 'az = '+str(az)+' indexes = '+str(indexes)
                atomInd = 1 + indexes[0]
            f.write('{0}\t{1}\t{2}\t{3}\t! {4}\n'.format(atomInd, a[0], a[1], a[2], atom_names[az]))
        f.write('\n\n')
        if Convolution != '': f.write(Convolution)
        f.write('\nEnd\n')

    with open(folder + '/fdmfile.txt', 'w') as f: f.write('1\nin.txt')
    return folder


# folder - папка с рассчитанными fdmnes файлами
def generateConvolutionInput(folder, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi):
    with open(folder + '/in_conv.txt', 'w') as f:
        f.write('Calculation\n')
        f.write('out.txt\n\n') # входной файл
        f.write('Convolution\n\n')
        f.write('Gamma_hole\n') #нулевой член ширины гауссиана (не зависящий по энергии) в eV
        f.write(str(Gamma_hole)+'\n\n')
        f.write('Ecent\n') # центр арктангенсойды (относительно уровня Ферми) - прибавляем к EFermi
        f.write(str(Ecent)+'\n\n')
        f.write('Elarg\n') # растягивание по горизонтали
        f.write(str(Elarg)+'\n\n')
        f.write('Gamma_max\n') # растягивание по вертикали
        f.write(str(Gamma_max)+'\n\n')
        f.write('Efermi\n') # EFermi - точка, от которой отсчитывается арктангенсойда. Все что левее - ЗАНУЛЯЕТСЯ!
        f.write(str(Efermi)+'\n\n')
        #f.write('Estart\n') # Начиная с какого значения рисовать спектр (по умолчанию - с EFermi)
        #f.write(str(Estart)+'\n\n')  # у нас нет карты Energpho, поэтому задаем в обычных ev
    with open(folder + '/fdmfile.txt', 'w') as f: f.write('1\nin_conv.txt')


def smooth(folder, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi):
    generateConvolutionInput(folder, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi)
    try:
        runLocal(folder)
        xanes = np.genfromtxt(folder+'/out_conv.txt', skip_header=1)
    except Exception as e:
        print('Error while reading out_conv.txt in folder '+folder)
        raise e
    energies = xanes[:,0].ravel()
    xanesVal = xanes[:,1].ravel()
    return energies, xanesVal


def parseAtomPositions(bavfilename):
    f = open(bavfilename, 'r')
    bav = f.read()
    f.close()
    i = bav.find('    Z  Typ       posx           posy           posz')
    if i < 0:
        i = bav.find('    Z         x              y              z      Typ')
        if i < 0:
            print('Error: file '+bavfilename+' doesn\'t contain atom positions')
            return None
    i = bav.find("\n",i)+1
    j = bav.find("\n\n",i)
    moleculaText = bav[i:j]
    molecula = pd.read_csv(StringIO(moleculaText), sep='\s+', names=['proton_number', 'Typ', 'x', 'y', 'z'])
    return molecula


def parseEfermi(folder):
    bavfilename = folder+'/out_bav.txt'
    if not os.path.isfile(bavfilename):
        raise Exception('Error: in folder '+folder+' there is no output file with energies')
    f = open(bavfilename, 'r')
    bav = f.read()
    f.close()
    i = bav.find("Last cycle, XANES calculation")
    i = bav.find('E_Fermi =',i) + len('E_Fermi =')
    j = bav.find('eV',i)
    return float(bav[i:j])


def stripFdmnesFile(fileName):
    with open(fileName, 'r') as f: s = f.read().strip()
    lines = map(lambda a: a.strip(), s.split("\n"))
    lines = [l for l in lines if (l != '') and (l[0] != '!')]
    return lines


def parseInput(d):
    d = utils.fixPath(d)
    fdmfile = d+os.sep+'fdmfile.txt'
    if not os.path.isfile(fdmfile):
        raise Exception('Error: in folder '+d+' there is no fdmfile.txt')
    flines = stripFdmnesFile(fdmfile)
    count = int(flines[0].split()[0])
    lastFile = utils.fixPath(d+os.sep+flines[count])
    if not os.path.isfile(lastFile):
        raise Exception('Error: the last file '+lastFile+' in fdmfile.txt doesn\'t exist')
    lines = stripFdmnesFile(lastFile)
    lines_down = list(map(lambda a: a.lower(), lines))

    i = lines_down.index('filout') if 'filout' in lines_down else None
    if i is None:
        i = lines_down.index('fileout') if 'fileout' in lines_down else None
    if i is None: filout = 'fdmnes_out'
    else: filout = lines[i + 1]

    i = lines_down.index('absorber') if 'absorber' in lines_down else None
    if i is None: absorber = None
    else: absorber = lines[i+1].strip().split()

    Energpho = False
    if 'energpho' in lines_down: Energpho = True
    return {'filout':filout, 'energpho':Energpho, 'absorber':absorber}


def parseOutFile(xanesFile, Energpho):
    if not os.path.isfile(xanesFile):
        raise Exception('Error: the filout file '+xanesFile+' doesn\'t exist')
    xanes = np.genfromtxt(xanesFile, skip_header=2)
    energies = xanes[:,0].ravel()
    xanesVal = xanes[:,1].ravel()

    # parse energy shift
    with open(xanesFile,'r') as f: s = f.read()
    i = s.find('\n')
    if i<0: raise Exception('Error: can\'t find header in output file: '+xanesFile)
    s = s[:i]
    values, names = s.split('=')
    names = list(map(lambda w: w.strip(), names.strip().split(',')))
    i = names.index('Epsii')
    values = list(map(lambda w: w.strip(), values.strip().split()))
    Epsii = float(values[i])
    i = names.index('E_edge')
    E_edge = float(values[i])

    if Energpho: energies -= E_edge

    if useEpsiiShift:
        energies += Epsii
    return utils.Spectrum(energies, xanesVal)


def parseOneFolder(d, absorberCompositionFunc=None):
    """
    Parse one fdmnes folder
    :param d: folder
    :param absorberCompositionFunc: function(dict{n->spectrum}) to calculate result spectrum of multiple absorbers. Default:average. We have to use spectra instead of intensities only because in some cases some of spectra may be shifted due to different valence
    :return: spectrum
    """
    d = utils.fixPath(d)
    if not os.path.exists(d):
        raise Exception('Error: folder '+d+' doesn\'t exist')
    inp = parseInput(d)
    filout = inp['filout']
    Energpho = inp['energpho']
    absorber = inp['absorber']
    if absorber is not None and len(absorber) > 1:
        partials = {}
        for partialFile in glob.glob(d+os.sep+filout+'_*.txt'):
            name = os.path.splitext(os.path.split(partialFile)[1])[0]
            postfix = name[name.rindex('_')+1:]
            if postfix == 'bav': continue
            n = int(postfix)
            partials[n] = parseOutFile(partialFile, Energpho)
        if len(partials) != len(absorber):
            raise Exception(f'Error: in folder {d} only {len(partials)} outputs, but {len(absorber)} absorbers')
        energyCount = np.array([partials[a].intensity.shape[0] for a in partials])
        assert len(np.unique(energyCount)) == 1, f'Outputs with different energy counts are detected in folder {d}'
        allEnergies = np.array([partials[a].energy for a in partials])
        if useEpsiiShift:
            common_e = np.median(allEnergies, axis=0)
            common_e = np.sort(common_e)
        else: common_e = allEnergies[0]
        sum = np.zeros(common_e.size)
        for a in partials:
            partials[a] = partials[a].changeEnergy(common_e)
            sum += partials[a].intensity
        if absorberCompositionFunc is None: return utils.Spectrum(common_e, sum/len(partials))
        else: return absorberCompositionFunc(partials)
    else:
        xanesFile = d + os.sep + filout + '.txt'
        return parseOutFile(xanesFile, Energpho)


def parseConvolution(folder):
    xanes = np.genfromtxt(folder+'/out_conv.txt', skip_header=1)
    energies = xanes[:,0].ravel()
    xanesVal = xanes[:,1].ravel()
    return energies, xanesVal


def getParams(fileName):
    with open(fileName, 'r') as f: params0 = json.load(f)
    return [p[0] for p in params0], [p[1] for p in params0]


def getGoodFolders(parentFolder):
    subfolders = [f for f in os.listdir(parentFolder) if os.path.isdir(os.path.join(parentFolder, f))]
    subfolders.sort()
    badFolders = []
    allXanes = {}
    output = ''
    for i in range(len(subfolders)):
        d = subfolders[i]
        try:
            res = parseOneFolder(os.path.join(parentFolder, d))
            if res is not None:
                allXanes[d] = res
            else:
                output += 'Can\'t read output in folder ' + d
        except:
            output += traceback.format_exc() + '\n'
            badFolders.append(d)
    if len(allXanes) == 0:
        for i in range(len(badFolders)): badFolders[i] = os.path.join(parentFolder, badFolders[i])
        return []
    energyCount = np.array([x.intensity.shape[0] for x in allXanes.values()])
    maxEnergyCount = np.max(energyCount)
    for d in allXanes:
        if allXanes[d].intensity.shape[0] != maxEnergyCount:
            badFolders.append(d)
    goodFolders = list(set(subfolders) - set(badFolders))
    goodFolders.sort()
    return goodFolders


fdmnes_exe = ''


def findFDMNES():
    global fdmnes_exe
    if fdmnes_exe != '': return fdmnes_exe
    exe = 'fdmnes_win64.exe' if os.name == 'nt' else 'fdmnes_linux64'
    exe = os.path.split(os.path.abspath(__file__))[0] + os.sep + 'bin' + os.sep + 'fdmnes' + os.sep + exe
    if os.path.exists(exe):
        fdmnes_exe = '"' + exe + '"'
        if os.name != 'nt' and not os.access(fdmnes_exe, os.X_OK):
            code = os.system(f"chmod +x {fdmnes_exe}")
            if code != 0:
                raise Exception(f'Add execution permission to the file {fdmnes_exe}')
        return fdmnes_exe
    exe_list = ['fdmnes.exe', 'fdmnes_win32.exe', 'fdmnes_win64.exe'] if os.name == 'nt' else ['fdmnes', 'fdmnes_linux64', 'fdmnes_linux']
    for exe in exe_list:
        if shutil.which(exe) is not None:
            fdmnes_exe = exe
            break
    assert fdmnes_exe != '', 'Can\'t find fdmnes executable'
    return fdmnes_exe


def runLocal(folder='.'):
    fdmnes = findFDMNES()
    output, returncode = utils.runCommand(fdmnes, folder, outputTxtFile='output.txt')
    if returncode != 0:
        raise Exception('Error while executing "'+fdmnes+'" command:\n'+output)
    return output


def runCluster(folder='.', memory=5000, nProcs=1):
    output, returncode = utils.runCommand(f"run-cluster-and-wait -m {memory} -n {nProcs} fdmnes", folder, outputTxtFile=None)


def calcForAllxyzInFolder(xyzfolder, calcFolder, continueCalculation, nProcs=6, memory=10000, calcSampleInParallel=5, recalculateErrorsAttemptCount=2, generateOnly=False, **fdmnesParams):
    if os.path.exists(calcFolder):
        if not continueCalculation: shutil.rmtree(calcFolder)
    os.makedirs(calcFolder, exist_ok=True)
    generateAbsorber = False
    if 'Absorber' in fdmnesParams:
        absorber = fdmnesParams['Absorber']
        if absorber[:3] == 'all':
            generateAbsorber = True
            assert len(absorber.split(' ')) == 2
            absorber = absorber.split(' ')[1]
    for f in glob.glob(xyzfolder+'/*.xyz'):
        m = Molecule(f)
        d = os.path.splitext(os.path.split(f)[1])[0]
        if generateAbsorber:
            ind = np.where(m.atomName == absorber)[0]
            assert len(ind) > 0
            ind = ind + 1
            fdmnesParams['Absorber'] = ' '.join(str(i) for i in ind)
        generateInput(m, folder=os.path.join(calcFolder,d), **fdmnesParams)
        with open(os.path.join(calcFolder,d,"geometryParams.txt"), "w") as text_file: text_file.write("[[\"dummy\",0]]")
        m.export_xyz(os.path.join(calcFolder,d,"molecule.xyz"))

    if generateOnly: return
    from . import sampling
    sampling.calcSpectra('fdmnes', runType='run-cluster', nProcs=nProcs, memory=memory, calcSampleInParallel=calcSampleInParallel, folder=calcFolder, recalculateErrorsAttemptCount=recalculateErrorsAttemptCount, continueCalculation=continueCalculation)


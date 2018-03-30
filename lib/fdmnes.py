import os
import numpy as np
import pandas as pd
from io import StringIO
import subprocess
import tempfile
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import optimize
import utils

# если строка folder пустая - создает внутри папки ./tmp и возвращает путь к созданной папке
# energyRange - строка в формате FDMNES
def generateInput(molecula, energyRange, radius = 6, folder = '', Quadrupole=False, Convolution = '', Absorber = 1, Green = False, Edge = 'K', cellSize=1.0):
    if folder == '':
        if not os.path.exists("./tmp"): os.makedirs("./tmp")
        folder = tempfile.mkdtemp(dir='./tmp')
    else:
        if not os.path.exists(folder): os.makedirs(folder)

    with open(folder + '/in.txt', 'w') as f:
        f.write('Filout\n')
        f.write('out\n\n')
        f.write('Radius\n')
        f.write('%.2f\n\n' % radius)
        if Green: f.write('Green\n\n')
        if Quadrupole: f.write('Quadrupole\n\n')
        f.write('Absorber\n')
        f.write(str(Absorber)+'\n\n')
        f.write('Range\n')
        f.write(energyRange+'\n\n')
        if Edge != 'K': f.write('Edge\n'+Edge+'\n\n')
        f.write('Molecule\n')
        c = str(cellSize)
        f.write(c+' '+c+' '+c +' 90 90 90\n')
        center = molecula.mol.loc[0,['x','y','z']].values
        for i in range(molecula.mol.shape[0]):
            a = molecula.mol.loc[i, :]
            f.write('{0}\t{1}\t{2}\t{3}\t! {4}\n'.format(a['proton_number'], a['x'], a['y'], a['z'], a['atom_name']))
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

def parse_Efermi(folder):
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

def parse_one_folder(d):
    xanesFile = d+'/out.txt'
    if not os.path.isfile(xanesFile):
        raise Exception('Error: in folder '+d+' there is no output file with energies')
    xanes = np.genfromtxt(xanesFile, skip_header=2)
    energies = xanes[:,0].ravel()
    xanesVal = xanes[:,1].ravel()
    if os.path.exists(d+'/out_bav.txt'): molecula = parseAtomPositions(d+'/out_bav.txt')
    else: molecula = None
    return utils.Xanes(energies, xanesVal, d), molecula

def parse_convolution(folder):
    xanes = np.genfromtxt(folder+'/out_conv.txt', skip_header=1)
    energies = xanes[:,0].ravel()
    xanesVal = xanes[:,1].ravel()
    # with open(folder+'/in_conv.txt', 'r') as f: s=f.read()
    # i = s.find('Estart')
    # if i<0: raise Exception('Cant find Estart in in_conv.txt')
    # i = s.find('\n',i)+1
    # j = s.find('\n',i)
    # Estart = float(s[i:j])
    return energies, xanesVal

# если parseConvolution=True, тогда возвращает вместо обычного xanes - convolution
def parse_all_folders(parentFolder, paramNames, parseConvolution = False):
    df_rows = []
    energies0 = np.zeros(1)
    atomColumnNames = []
    subfolders = os.listdir(parentFolder)
    subfolders.sort()
    for d in subfolders:
        if not os.path.isdir(parentFolder+'/'+d): continue
        xanesFile = parentFolder+'/'+d+'/out.txt' if not parseConvolution else parentFolder+'/'+d+'/out_conv.txt'
        if not os.path.isfile(xanesFile):
            print('Error: in folder '+d+' there is no output file with energies')
            continue
        skip_header = 2 if not parseConvolution else 1
        xanes = np.genfromtxt(xanesFile, skip_header=2)
        molecula = parseAtomPositions(parentFolder+'/'+d+'/out_bav.txt')
        energies = xanes[:,0].ravel()
        with open(parentFolder+'/'+d+'/params.txt', 'r') as f: params = json.load(f)
        if energies0.shape[0] == 1:  # первая итерация цикла
            energies0 = np.copy(energies)
            for ai in range(molecula.shape[0]):
                z = 'atom'+str(ai+1)+'_'+str(int(molecula.loc[ai,'proton_number']))
                atomColumnNames.extend([z+'_x',z+'_y',z+'_z'])
        else:
            if energies0.shape[0] != energies.shape[0]:
                print('Error different number of energies in '+subfolders[0]+' and '+d)
                # exit(1)
        xanes = xanes[:,1].ravel()
        atom_coords = molecula.loc[:,['x','y','z']].values.ravel('C') # C - row-major order, F - column-major order
        params = np.array([params[p] for p in paramNames])
        df_rows.append({'xanes':xanes, 'atom_coords':atom_coords, 'folder':d, 'params':params})
    df_xanes = np.zeros([len(df_rows), df_rows[0]['xanes'].shape[0]])
    df_atom_coords  = np.zeros([len(df_rows), df_rows[0]['atom_coords'].shape[0]])
    df_params = np.zeros([len(df_rows), df_rows[0]['params'].shape[0]])
    i = 0
    for row in df_rows:
        df_xanes[i,:] = row['xanes']
        df_atom_coords[i,:] = row['atom_coords']
        df_params[i,:] = row['params']
        i += 1
    df_xanes = pd.DataFrame(data=df_xanes, columns=['e_'+str(e) for e in energies0])
    df_atom_coords = pd.DataFrame(data=df_atom_coords, columns=atomColumnNames)
    df_params = pd.DataFrame(data=df_params, columns=paramNames)
    return df_xanes, df_atom_coords, df_params

def runLocal(folder = '.'):
    return subprocess.Popen(["fdmnes_11"], cwd=folder, stdout=subprocess.PIPE).stdout.read()

def runCluster(folder = '.', memory=5000, nProcs = 1):
    proc = subprocess.Popen(["run-cluster-and-wait", "-m", str(memory), '-n', str(nProcs), "fdmnes_11"], cwd=folder, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.wait()

def plotToFolder(folder, exp, xanes0, smoothed_xanes, append=None):
    exp_e = exp.xanes.energy
    exp_xanes = exp.xanes.absorb
    shiftIsAbsolute = exp.defaultSmoothParams.shiftIsAbsolute
    search_shift_level = exp.defaultSmoothParams.search_shift_level
    fit_interval = exp.fit_intervals['norm']
    fig, ax = plt.subplots()
    fdmnes_en = smoothed_xanes.energy
    fdmnes_xan = smoothed_xanes.absorb
    with open(folder+'/args_smooth.txt', 'r') as f: smooth_params = json.load(f)
    shift = optimize.value(smooth_params,'shift')
    if not shiftIsAbsolute: shift += utils.getInitialShift(exp_e, exp_xanes, fdmnes_en, fdmnes_xan, search_shift_level)
    fdmnes_xan0 = utils.fit_arg_to_experiment(xanes0.energy, exp_e, xanes0.absorb, shift, lastValueNorm=True)
    ax.plot(exp_e-shift, fdmnes_xan0, label='initial')
    ax.plot(exp_e-shift, fdmnes_xan, label='convolution')
    ax.plot(exp_e-shift, exp_xanes, c='k', label="Experiment")
    if append is not None:
        e_fdmnes = exp_e-shift
        ax2 = ax.twinx()
        ax2.plot(e_fdmnes, append, c='r', label='Smooth width')
        ax2.legend()
        # ax3 = ax.twinx()
        # ax3.plot(e_fdmnes[1:], (append[1:]-append[:-1])/(e_fdmnes[1:]-e_fdmnes[:-1]), c='g', label='Diff smooth width')
        # ax3.legend()
    ax.set_xlim([fit_interval[0]-shift, fit_interval[1]-shift])
    ax.set_ylim([0, np.max(exp_xanes)*1.2])
    ax.set_xlabel("Energy")
    ax.set_ylabel("XANES")
    ax.legend()
    fig.set_size_inches((16/3*2, 9/3*2))
    fig.savefig(folder+'/xanes.png')
    plt.close(fig)

# fitBy = {'FixedNorm', 'RegressionMultOnly', 'Regression'}
def plotDiffToFolder(folder0, exp_e0, exp_xanes0, folder, exp_e, exp_xanes, fit_interval, shiftIsAbsolute=True, search_shift_level=None, fitBy='RegressionMultOnly'):
    fig, ax = plt.subplots()
    figDiff, axDiff = plt.subplots()
    fdmnes_en_init, fdmnes_xan_init, _ = parse_one_folder(folder)
    fdmnes_en_init0, fdmnes_xan_init0, _ = parse_one_folder(folder0)
    fdmnes_en_conv, fdmnes_xan_conv = parse_convolution(folder)
    fdmnes_en_conv0, fdmnes_xan_conv0 = parse_convolution(folder0)
    with open(folder+'/args_smooth.txt', 'r') as f: smooth_params = json.load(f)
    with open(folder0+'/args_smooth.txt', 'r') as f: smooth_params0 = json.load(f)
    shift = optimize.value(smooth_params,'shift')
    shift0 = optimize.value(smooth_params0,'shift')
    assert shift == shift0
    if not shiftIsAbsolute:
        shift += utils.getInitialShift(exp_e, exp_xanes, fdmnes_en_conv, fdmnes_xan_conv, search_shift_level)
        shift0 += utils.getInitialShift(exp_e0, exp_xanes0, fdmnes_en_conv0, fdmnes_xan_conv0, search_shift_level)
    fdmnes_xan_init = utils.fit_arg_to_experiment(fdmnes_en_init, exp_e, fdmnes_xan_init, shift, lastValueNorm=True)
    fdmnes_xan_init0 = utils.fit_arg_to_experiment(fdmnes_en_init0, exp_e0, fdmnes_xan_init0, shift0, lastValueNorm=True)
    axDiff.plot(exp_e-shift, fdmnes_xan_init-fdmnes_xan_init0, label='initDiff') # здесь важно чтобы shift==shift0
    ax.plot(exp_e-shift, fdmnes_xan_init, label='init')
    ax.plot(exp_e-shift, fdmnes_xan_init0, label='initBare')

    if fitBy == 'RegressionMultOnly':
        fdmnes_xan_conv = utils.fit_arg_to_experiment(fdmnes_en_conv, exp_e, fdmnes_xan_conv, shift)
        fdmnes_xan_conv0 = utils.fit_arg_to_experiment(fdmnes_en_conv0, exp_e0, fdmnes_xan_conv0, shift0)
        fdmnes_xan_conv = utils.fit_by_regression_mult_only(exp_e, exp_xanes, fdmnes_xan_conv, fit_interval)
        fdmnes_xan_conv0 = utils.fit_by_regression_mult_only(exp_e0, exp_xanes0, fdmnes_xan_conv0, fit_interval)
    elif fitBy == 'Regression':
        fdmnes_xan_conv = utils.fit_arg_to_experiment(fdmnes_en_conv, exp_e, fdmnes_xan_conv, shift)
        fdmnes_xan_conv0 = utils.fit_arg_to_experiment(fdmnes_en_conv0, exp_e0, fdmnes_xan_conv0, shift0)
        fdmnes_xan_conv = utils.fit_by_regression(exp_e, exp_xanes, fdmnes_xan_conv, fit_interval)
        fdmnes_xan_conv0 = utils.fit_by_regression(exp_e0, exp_xanes0, fdmnes_xan_conv0, fit_interval)
    elif fitBy == 'FixedNorm':
        norm = optimize.value(smooth_params, 'norm')
        norm0 = optimize.value(smooth_params0, 'norm')
        fdmnes_xan_conv = utils.fit_to_experiment_by_norm_or_regression_mult_only(exp_e, exp_xanes, fit_interval, fdmnes_en_conv, fdmnes_xan_conv, shift, norm)
        fdmnes_xan_conv0 = utils.fit_to_experiment_by_norm_or_regression_mult_only(exp_e0, exp_xanes0, fit_interval, fdmnes_en_conv0, fdmnes_xan_conv0, shift0, norm0)
    else: assert False, 'Unknown fitBy = '+fitBy
    axDiff.plot(exp_e-shift, fdmnes_xan_conv-fdmnes_xan_conv0, label='convDiff') # здесь важно чтобы shift==shift0
    ax.plot(exp_e-shift, fdmnes_xan_conv, label='conv')
    ax.plot(exp_e-shift, fdmnes_xan_conv0, label='convBare')
    axDiff.plot(exp_e-shift, exp_xanes-exp_xanes0, c='k', label="ExpDiff") # здесь важно чтобы shift==shift0
    ax.plot(exp_e-shift, exp_xanes, label="Exp")
    ax.plot(exp_e-shift, exp_xanes0, label="ExpBare")

    axDiff.set_xlim([fit_interval[0]-shift, fit_interval[1]-shift]) # здесь важно чтобы shift==shift0
    ax.set_xlim([fit_interval[0]-shift, fit_interval[1]-shift])
    m = np.min(exp_xanes-exp_xanes0); M = np.max(exp_xanes-exp_xanes0); mM = M-m
    axDiff.set_ylim([m - mM/5, M + mM/5])
    ax.set_ylim([0, np.max(exp_xanes)*1.5])
    axDiff.set_xlabel("Energy")
    ax.set_xlabel("Energy")
    axDiff.set_ylabel("XANES Diff")
    ax.set_ylabel("XANES")
    axDiff.legend()
    ax.legend()
    figDiff.set_size_inches((16/3*2, 9/3*2))
    fig.set_size_inches((16/3*2, 9/3*2))
    figDiff.savefig(folder+'/xanesDiff.png')
    fig.savefig(folder+'/xanes.png')
    plt.close(figDiff)
    plt.close(fig)

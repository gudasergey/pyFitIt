#!/opt/anaconda/bin/python -u
import sys, os, subprocess, shutil, tempfile, socket, copy, re, urllib, glob, random, string, json
import numpy as np
import pandas as pd
import scipy, scipy.optimize
from io import StringIO
from . import utils

def generateInput(molecule, struct_name, folder = ''):
    if folder == '':
        if not os.path.exists("./tmp"): os.makedirs("./tmp")
        folder = tempfile.mkdtemp(dir='./tmp')
    else:
        # if os.path.exists(folder):shutil.rmtree(folder)
        assert not os.path.exists(folder), 'Folder '+folder+' exists!'
        os.makedirs(folder, exist_ok=True)

    molecule.export_struct(struct_name + '.struct', a=15, b=15, c=15, alpha=90, beta=90, gamma=90)
    shutil.copy(struct_name + '.struct', folder+'/'+struct_name+'.struct')

   
def runLocal(folder = '.', w2auto_run_func=None):
    assert w2auto_run_func is not None
    w2auto_run_func(folder)

def parse_one_folder(folder):
    xanesFile = folder + '/spectrum.txt'
    if os.path.exists(xanesFile):
        try:
            xanes = np.genfromtxt(xanesFile, skip_header=2)
            energy = xanes[:,0].ravel()
            absorb = xanes[:,1].ravel()
        except:
            raise Exception('Can\'t parse output file in folder '+folder)
    return utils.Spectrum(energy, absorb)


def getParams(fileName):
    with open(fileName, 'r') as f: params0 = json.load(f)
    return [p[0] for p in params0], [p[1] for p in params0]


def parse_all_folders(parentFolder, printOutput=True):
    subfolders = [f for f in os.listdir(parentFolder) if os.path.isdir(os.path.join(parentFolder,f))]
    subfolders.sort()
    badFolders = []; allXanes = {}
    output = ''
    for i in range(len(subfolders)):
        d = subfolders[i]
        
        struct_file_name = utils.findFile(os.path.join(parentFolder, d),'.struct').split('.')[0].split('/')[-1]        
        path_to_wannier = os.path.join(parentFolder, d, 'w2webEmulator/caseBaseDir/'+struct_file_name+'_Bandstructure/'+struct_file_name+'/')
        wannier_folders = [w for w in os.listdir(path_to_wannier) if os.path.isdir(os.path.join(path_to_wannier, w))]
        assert len(wannier_folders) == 1, 'There must be only 1 wannier folder in ' + path_to_wannier
        xanesFileFolder = path_to_wannier + wannier_folders[0].split('/')[-1] + '/XTLS/1/xcodes'
        if os.path.exists(os.path.abspath(xanesFileFolder)):
            res = parse_one_folder(xanesFileFolder)
            if res is not None: 
                allXanes[d] = res
            else: output += 'Can\'t read spectrum.txt in folder '+d
        else:
            badFolders.append(d)
    
    if len(allXanes) == 0:
        if printOutput: print('None good folders')
        for i in range(len(badFolders)): badFolders[i] = os.path.join(parentFolder, badFolders[i])
        return None, None, badFolders
    else:
        if output != '': print(output)
        
    energyCount = np.array([ x.intensity.shape[0] for x in allXanes.values() ])
    maxEnergyCount = np.max(energyCount)
    for d in allXanes:
        if allXanes[d].intensity.shape[0] != maxEnergyCount:
            print('Error: in folder '+d+' there are less energies '+str(allXanes[d].intensity.shape[0]))
            badFolders.append(d)
    goodFolders = list(set(subfolders) - set(badFolders))
    goodFolders.sort()
    if len(goodFolders) == 0:
        if printOutput: print('None good folders')
        for i in range(len(badFolders)): badFolders[i] = os.path.join(parentFolder, badFolders[i])
        return None, None, badFolders
    allEnergies = np.array([ allXanes[folder].energy for folder in goodFolders ])    
    n = len(goodFolders)
    if n == 1: allEnergies.reshape(1,-1)
    energies = np.median(allEnergies, axis=0)

    #if useEpsiiShift:
    #    maxShift = np.max(allEnergies[:,0]) - np.min(allEnergies[:,0])
    #    if printOutput: print('Max energy shift between spectra: {:.2}'.format(maxShift))
    
    paramNames, _ = getParams(os.path.join(parentFolder, goodFolders[0], 'geometryParams.txt'))
    df_xanes = np.zeros([n, energies.size])
    df_params = np.zeros([n, len(paramNames)])
    for i in range(n):
        d = goodFolders[i]
        _, params = getParams(os.path.join(parentFolder, d, 'geometryParams.txt'))
        df_params[i,:] = np.array(params)
        #if useEpsiiShift:
        #    df_xanes[i, :] = np.interp(energies, allXanes[d].energy, allXanes[d].intensity)
        #else:
        df_xanes[i,:] = allXanes[d].intensity
    df_xanes = pd.DataFrame(data=df_xanes, columns=['e_'+str(e) for e in energies])
    df_params = pd.DataFrame(data=df_params, columns=paramNames)
    for i in range(len(badFolders)): badFolders[i] = os.path.join(parentFolder, badFolders[i])
    return df_xanes, df_params, badFolders











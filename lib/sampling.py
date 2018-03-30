import design # install: pip install py-design, than in the case of import errors "_design" you need to copy library .so in the upper folder
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
import tempfile
from distutils.dir_util import copy_tree, remove_tree
import copy
import math
import fdmnes
import shutil

# ranges - dictionary with geometry parameters region {'paramName':[min,max], 'paramName':[min,max], ...}
# method - IHL, random, grid (in case of grid, sampleCount must be a list of points count through each dimension)
# xanesCalcParams - for FDMNES: {'energyRange':'...', 'radius':6, 'Green':False}, add here Gamma_hole, Ecent, Elarg, Gamma_max, Efermi if parseConvolution=True
def sample(ranges, moleculaConstructor, sampleCount, method='IHL', threadCount = 0, parseConvolution=False, outputFolder='.', xanesCalcParams = {}):
    paramNames = [k for k in ranges]
    N = len(paramNames)
    leftBorder = np.array([ranges[k][0] for k in paramNames])
    rightBorder = np.array([ranges[k][1] for k in paramNames])
    if method == 'IHL':
        points = (design.ihs(sampleCount, N) - 0.5) / sampleCount # row - is one point
        for j in range(N):
            points[:,j] = leftBorder[j] + points[:,j]*(rightBorder[j]-leftBorder[j])
    elif method == 'random': points = np.random.rand(sampleCount, N)
    else:
        assert method == 'grid', 'Unknown method'
        assert len(sampleCount) == N, 'sampleCount must be a list of point count over dimensions'
        coords = [np.linspace(leftBorder[j], rightBorder[j], sampleCount[j]) for j in range(N)]
        repeatedCoords = np.meshgrid(coords)
        sz = np.prod(sampleCount)
        points = np.zeros(sz, N)
        for j in range(N): points[:,] = repeatedCoords[j].flatten()
    prettyPoints = []
    x1 = copy.deepcopy(ranges)
    for i in range(sampleCount):
        for j in range(N): x1[paramNames[j]] = points[i,j]
        prettyPoints.append(copy.deepcopy(x1))
    def calculateXANES(geometryParams):
        molecula = moleculaConstructor(geometryParams)
        folder = fdmnes.generateInput(molecula, xanesCalcParams['energyRange'], radius=xanesCalcParams['radius'], folder='')
        # folder = fdmnes.generateInput(newMol, '0 20 117', radius=5, folder='')
        with open(folder+'/geometryParams.txt', 'w') as f: f.write(str(geometryParams))
        print('folder=',folder, str(geometryParams))
        molecula.export_xyz(folder+'/molecule.xyz')
        fdmnes.runCluster(folder, 10000)
        if parseConvolution:
            fdmnes.smooth(folder, xanesCalcParams['Gamma_hole'], xanesCalcParams['Ecent'], xanesCalcParams['Elarg'], xanesCalcParams['Gamma_max'], xanesCalcParams['Efermi'])
        return folder
    if threadCount>0:
        threadPool = ThreadPool(threadCount)
        folders = threadPool.map(calculateXANES, prettyPoints)
    else:
        folders = [calculateXANES(prettyPoints[i]) for i in range(sampleCount)]
    parentFolder = tempfile.mkdtemp(dir='./tmp', prefix='features_')
    os.makedirs(parentFolder+'/xanesCalculations')
    folderNameSize = 1+math.floor(0.5+math.log(sampleCount,10))
    i = 0
    newFolderNames = []
    for folder in folders:
        newFolderName = parentFolder+'/xanesCalculations/'+str(i).zfill(folderNameSize)
        i += 1
        newFolderNames.append(newFolderName)
        remove_tree(folder+'/out_bav.txt') #TODO: don't delete is there was an error in calculations
        copy_tree(folder, newFolderName)
        remove_tree(folder)
    df_xanes, df_atom_coords, df_params = fdmnes.parse_all_folders(parentFolder, paramNames, parseConvolution = False)
    df_xanes.to_csv(outputFolder+'/xanes.txt', sep=' ', index=False)
    df_atom_coords.to_csv(outputFolder+'/atoms.txt', sep=' ', index=False)
    df_params.to_csv(outputFolder+'/params.txt', sep=' ', index=False)
    if parseConvolution:
        df_xanes, _, _ = fdmnes.parse_all_folders(parentFolder+'/xanesCalculations', paramNames, parseConvolution = True)
        df_xanes.to_csv(outputFolder+'/xanes_conv.txt', sep=' ', index=False)
    shutil.make_archive(outputFolder+'/xanesCalculations', 'zip', parentFolder+'/xanesCalculations')

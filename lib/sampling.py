import design # install: pip install py-design, than in the case of import errors "_design" you need to copy library .so in the upper folder
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
import tempfile
from distutils.dir_util import copy_tree, remove_tree
import copy
from minimize import setValue
import math
import fdmnes

# x - точка с областью расчета в формате модуля minimize (она включается в число всех расчетных точек нулевой)
# Func - функция расчета значения в одной точке x, возвращает папку с проведенным экспериментом
# method - как выбираем точки: IHL, random, grid (в случае grid, sampleCount долно быть списком кол-ва точек вдоль каждого измерения)
def buildFeatures(x, Func, sampleCount, method='IHL', threadCount = 0, parseConvolution=False, includeXvalue = False):
    N = len(x)
    leftBorder = np.array([k['leftBorder'] for k in x])
    rightBorder = np.array([k['rightBorder'] for k in x])
    if method == 'IHL':
        points = (design.ihs(sampleCount, N) - 0.5) / sampleCount # строка - это одна точка
        for j in range(N):
            points[:,j] = leftBorder[j] + points[:,j]*(rightBorder[j]-leftBorder[j])
        if includeXvalue:
            x0 = np.array([k['value'] for k in x])
            points = np.vstack([x0,points])
            sampleCount += 1
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
    x1 = copy.deepcopy(x)
    for i in range(sampleCount):
        for j in range(N): x1[j]['value'] = points[i,j]
        prettyPoints.append(copy.deepcopy(x1))
    if threadCount>0:
        threadPool = ThreadPool(threadCount)
        folders = threadPool.map(Func, prettyPoints)
    else:
        folders = [Func(prettyPoints[i]) for i in range(sampleCount)]
    parentFolder = tempfile.mkdtemp(dir='./tmp', prefix='features_')
    folderNameSize = 1+math.floor(0.5+math.log(sampleCount,10))
    i = 0
    newFolderNames = []
    for folder in folders:
        newFolderName = parentFolder+'/'+str(i).zfill(folderNameSize)
        i += 1
        newFolderNames.append(newFolderName)
        copy_tree(folder, newFolderName)
        remove_tree(folder)
    paramNames = [k['paramName'] for k in x]
    df_xanes, df_atom_coords, df_params = fdmnes.parse_all_folders(parentFolder, paramNames, parseConvolution = False)
    df_xanes.to_csv(parentFolder+'/xanes.txt', sep=' ', index=False)
    df_atom_coords.to_csv(parentFolder+'/atoms.txt', sep=' ', index=False)
    df_params.to_csv(parentFolder+'/params.txt', sep=' ', index=False)
    if parseConvolution:
        df_xanes, _, _ = fdmnes.parse_all_folders(parentFolder, paramNames, parseConvolution = True)
        df_xanes.to_csv(parentFolder+'/xanes_conv.txt', sep=' ', index=False)

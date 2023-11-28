import copy
import os, time, random
from collections import defaultdict
from . import utils


def getMPIinfo():
    if 'PMI_RANK' in os.environ:
        id = int(os.environ['PMI_RANK'])
        count = int(os.environ['PMI_SIZE'])
    elif 'OMPI_COMM_WORLD_RANK' in os.environ:
        id = int(os.environ['OMPI_COMM_WORLD_RANK'])
        count = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    else: id, count = 0,1
    return id, count


barrierCounter = defaultdict(int)
mpi_id, mpi_count = getMPIinfo()
rand = random.Random(mpi_id)
debug = True


def updateNodes(filename, firstRun):
    open(filename, 'a').close()
    time.sleep(rand.random())
    with open(filename) as f: s = f.read()
    old_nodes = s.split(' ') if len(s)>0 else []
    for n in old_nodes:  assert utils.is_str_float(n)
    old_nodes = [int(n) for n in old_nodes]
    if firstRun and mpi_id in old_nodes:
        if debug: print('program is running in the same folder once more')
        old_nodes = []
    new_nodes = copy.deepcopy(old_nodes)
    if mpi_id not in new_nodes:
        if debug: print(f'append node {mpi_id} to {filename}')
        new_nodes.append(mpi_id)
        new_nodes = sorted(new_nodes)
        with open(filename, 'w') as f: f.write(' '.join([str(n) for n in new_nodes]))
    return old_nodes, new_nodes


def safeRun(func, attemptCount):
    error = True
    i = 0
    res = None
    while error and i <= attemptCount:
        try:
            res = func()
            error = False
        except:
            if debug:
                print(f'There was error in the safeRun:')
                import traceback
                print(traceback.format_exc())
            time.sleep(rand.random())
        i += 1
    return ('success',res) if i <= attemptCount else ('failure',None)


def barrier(folder, name):
    if mpi_count == 1: return
    id = barrierCounter[name]
    barrierCounter[name] += 1
    filename = folder+os.sep+f'barrier_{name}_{id}'
    time.sleep(rand.random())
    res, _ = safeRun(lambda: updateNodes(filename, firstRun=True), 10)
    assert res == 'success', f"Can't update barrier file {filename}"
    nodes = []
    while len(nodes) < mpi_count:
        res, nodeInfo = safeRun(lambda: updateNodes(filename, firstRun=False), 10)
        _,nodes = nodeInfo
        assert res == 'success', f"Can't update barrier file {filename}"
        time.sleep(rand.random()+1)

import numpy as np
from parsy import regex, generate, whitespace, string
import math, scipy, copy, os, io
from . import utils, geometry
if utils.isLibExists("wbm"):
    from . import wbm

uintParser = regex(r"\d+").map(int)
pi = np.pi
cross = np.cross
norm = np.linalg.norm
dot = np.dot
def normalize(v): return v/norm(v)


atom_proton_numbers = {'H':1, 'He':2, \
    'Li':3, 'Be':4, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Ne':10, \
    'Na':11, 'Mg':12, 'Al':13, 'Si':14, 'P':15, 'S':16, 'Cl':17, 'Ar':18,\
    'K':19, 'Ca':20, 'Sc':21, 'Ti':22, 'V':23, 'Cr':24, 'Mn':25, 'Fe':26, 'Co':27, 'Ni':28,\
    'Cu':29, 'Zn':30, 'Ga':31, 'Ge':32, 'As':33, 'Se':34, 'Br':35, 'Kr':36,\
    'Rb':37, 'Sr':38, 'Y':39, 'Zr':40, 'Nb':41, 'Mo':42, 'Tc':43, 'Ru':44, 'Rh':45, 'Pd':46,\
    'Ag':47, 'Cd':48, 'In':49, 'Sn':50, 'Sb':51, 'Te':52, 'I':53, 'Xe':54,\
    'Cs':55, 'Ba':56, 'La':57, 'Ce':58, 'Pr':59, 'Nd':60, 'Pm':61, 'Sm':62, 'Eu':63,\
    'Gd':64, 'Tb':65, 'Dy':66, 'Ho':67, 'Er':68, 'Tm':69, 'Yb':70, 'Lu':71,\
    'Hf':72, 'Ta':73, 'W':74, 'Re':75, 'Os':76, 'Ir':77, 'Pt':78,\
    'Au':79, 'Hg':80, 'Tl':81, 'Pb':82, 'Bi':83, 'Po':84, 'At':85, 'Rn':86,\
    'Fr':87, 'Ra':88, 'Ac':89, 'Th':90, 'Pa':91, 'U':92, 'Np':93, 'Pu':94, 'Am':95,\
    'Cm':96, 'Bk':97, 'Cf':98, 'Es':99, 'Fm':100, 'Md':101, 'No':102, 'Lr':103,\
    'Rf':104, 'Db':105, 'Sg':106, 'Bh':107, 'Hn':108, 'Mt':109\
    }
atom_names = {v: k for k, v in atom_proton_numbers.items()}


@generate
def rangeParser():
    lower = yield uintParser
    yield whitespace.many() >> string('-') >> whitespace.many()
    upper = yield uintParser
    return list(range(lower, upper + 1))


@generate
def uintToList():
    num = yield uintParser
    return [num]


@generate
def partsParser():
    rangeOrNumber = whitespace.many() >> (rangeParser | uintToList)
    rangesList = yield rangeOrNumber.sep_by(string(','))
    return flatten(rangesList)


def flatten(array):
    return [item for sublist in array for item in sublist]


def parseXYZFile(fname):
    with open(fname, encoding='utf-8') as f:
        content = f.read()
    return parseXYZContent(content)


def parseXYZContent(content):
    intParser = regex(r"[-+]?\d+").map(int)
    floatParser = regex(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?").map(float)
    newLine = regex(r'\n')
    untilNewLine = regex(r'[^\n]*')

    @generate
    def parseHeader():
        num = yield intParser # number of atoms
        yield newLine >> untilNewLine >> newLine # comment
        return num

    @generate
    def parseRow():
        atom = yield regex(r'\s*[a-zA-Z0-9.]*')
        try:
            atomNumber = int(atom)
            atom = atom_names[atomNumber]
        except ValueError:
            pass
        yield whitespace.many()
        x = yield floatParser
        yield whitespace.many()
        y = yield floatParser
        yield whitespace.many()
        z = yield floatParser
        return np.array([x, y, z]), atom.strip()

    @generate
    def parseContent():
        yield parseHeader.optional()
        table = yield parseRow.many()
        yield whitespace.many()
        atomCoords = [a[0] for a in table]
        atomNames = [a[1] for a in table]
        return np.array(atomCoords), np.array(atomNames)

    table, atomNames = parseContent.parse(content)
    latticeVectors = None
    if 'VEC1' in atomNames:
        # this is a crystal
        latticeVectors = []
        for d in range(1,4):
            vName = f'VEC{d}'
            assert vName in atomNames
            i = np.where(atomNames == vName)[0][0]
            latticeVectors.append(table[i])
            table = np.delete(table, i, axis=0)
            atomNames = np.delete(atomNames, i, axis=0)
        latticeVectors = np.array(latticeVectors)
    atomNumbers = np.array([atom_proton_numbers[an] for an in atomNames])
    return table, atomNumbers, atomNames, latticeVectors


class Molecule:
    def __init__(self, fileName=None):
        self.atom, self.atomNumber, self.atomName, self.az, self.latticeVectors = None, None, None, None, None
        if fileName is not None:
            self.fileName = utils.fixPath(fileName)
            self.construct(*parseXYZFile(self.fileName))

    def construct(self, atom, atomNumber, atomName, latticeVectors):
        self.atom, self.atomNumber, self.atomName, self.latticeVectors = atom, atomNumber, atomName, latticeVectors
        self.az = self.atomNumber  # alias
        self.setParts('0-' + str(len(self.atom) - 1))

    @classmethod
    def fromXYZcontent(cls, content):
        m = cls()
        m.construct(*parseXYZContent(content))
        return m

    def setParts(self, *parts):
        # partsData should be a 2d-array, each of inner array contains indices of atoms that belong to the corresponding part
        # e.g. partsData[i] has all the indices of atoms from part <i>
        self.partsData = list(map(partsParser.parse, parts))

        self.assignParts()

    def assignParts(self):
        f = lambda i: MoleculePart(self, i)
        self.part = list(map(f, range(len(self.partsData))))

    def copy(self):
        return copy.deepcopy(self)

    def rotate(self, axis, center, angle):
        self.rotate__Impl(axis, center, angle, range(len(self.atom)))

    def rotate_xyz_axes(self, center, angles):
        self.rotate_xyz_axes__Impl(center, angles, range(len(self.atom)))

    def shift(self, shift):
        for i in range(len(self.partsData)):
            self.shift__Impl(shift, i)

    def tween(self, start, final, percent):
        """
        Perform linear motion tween of self using atom shifts final-start
        :param start: Start molecule with the same atom sequence as self
        :param final: Final molecule with the same atom sequence as self
        :param percent: tween percent. 0 (self), 1 (final). May be negative or > 1
        """
        assert np.all(self.az == final.az)
        assert np.all(self.az == start.az)
        shifts = final.atom - start.atom
        self.atom += percent*shifts

    def rotate__Impl(self, axis, center, angle, atomIndices):
        # поворачивает часть молекулы вокруг оси, заданной вектором axis. Положительное направление поворота определяется по правилу закручивающегося в направлении оси буравчика
        newMol = np.copy(self.atom)
        cphi = math.cos(angle)
        sphi = math.sin(angle)
        if isinstance(axis, list): axis = np.array(axis, dtype=float)
        if isinstance(center, list): center = np.array(center, dtype=float)
        axis = axis / np.linalg.norm(axis)
        # https://ru.wikipedia.org/wiki/%D0%9C%D0%B0%D1%82%D1%80%D0%B8%D1%86%D0%B0_%D0%BF%D0%BE%D0%B2%D0%BE%D1%80%D0%BE%D1%82%D0%B0
        x = axis[0]; y = axis[1]; z = axis[2]
        M = np.matrix([[cphi+(1-cphi)*x*x, (1-cphi)*x*y-sphi*z, (1-cphi)*x*z+sphi*y],\
            [(1-cphi)*y*x+sphi*z, cphi+(1-cphi)*y*y, (1-cphi)*y*z-sphi*x],\
            [(1-cphi)*z*x-sphi*y, (1-cphi)*z*y+sphi*x, cphi+(1-cphi)*z*z]])
        for i in atomIndices:
            newMol[i] = np.ravel(np.dot(M, newMol[i] - center)) + center
        self.atom = newMol

    def rotate_xyz_axes__Impl(self, axis, center, angles, atomIndices):
        coords = np.copy(self.atom)
        for icoord in range(3):
            # вращаем вокруг оси № icoord
            phi = angles[icoord]
            cphi = math.cos(phi)
            sphi = math.sin(phi)
            inds = np.arange(3)
            inds = np.delete(inds,icoord)
            c = center[inds]
            for i in atomIndices:
                atom = coords[i, ...]
                sign = 1-(icoord//2)*2
                atom[inds] = geometry.turnCoords(atom[inds]-c, cphi, sphi*sign) + c
                coords[i, ...] = atom
        self.atom = coords

    def shift__Impl(self, shift, partInd):
        # сдвигает часть молекулы на вектор shift
        shift = np.array(shift,dtype=float)
        shift = shift.reshape(-1)
        assert shift.size==3
        for i in self.partsData[partInd]:
            self.atom[i] += shift

    def export_xyz_string(self, cellSize=1):
        f = io.StringIO()
        format = lambda a, an: an.rjust(2) + '  ' + str(a[0] * cellSize).rjust(10) + '  ' + str(a[1] * cellSize).rjust(10) + '  ' + str(a[2] * cellSize).rjust(10) + "\n"
        f.write(str(self.atom.shape[0]) + '\n')
        f.write('\n')
        for i in range(self.atom.shape[0]):
            a = self.atom[i]
            an = atom_names[self.atomNumber[i]]
            f.write(format(a, an))
        if self.latticeVectors is not None:
            for i in range(3):
                a = self.latticeVectors[i]
                an = f'VEC{i + 1}'
                f.write(format(a, an))
        f.seek(0)
        return f.read()

    def export_xyz(self, file, cellSize=1):
        d = os.path.split(file)[0]
        if d != '':  os.makedirs(d, exist_ok=True)
        with open(file, 'w') as f:
            f.write(self.export_xyz_string(cellSize))
            f.close()

    def export_struct(self, file, a, b, c, alpha, beta, gamma):
        sorted_coords = self.atom
        sorted_names = self.atomName
        lattice_type = 'P'
        isplit = 0
        NPT = 781
        R0 = 0.0001
        RMT = 2
        nsym = 0

        m = self.copy()

        heaviest_atom_index = np.argmax(m.az)
        tmp = m.atom[0]
        m.atom[0] = m.atom[heaviest_atom_index]
        m.atom[heaviest_atom_index] = tmp

        tmp = m.az[0]
        m.az[0] = m.az[heaviest_atom_index]
        m.az[heaviest_atom_index] = tmp

        tmp = m.atomName[heaviest_atom_index]
        m.atomName[0] = m.atomName[heaviest_atom_index]
        m.atomName[heaviest_atom_index] = tmp

        tmp = m.atomNumber[heaviest_atom_index]
        m.atomNumber[0] = m.atomNumber[heaviest_atom_index]
        m.atomNumber[heaviest_atom_index] = tmp

        m.atom[:,0] /= a
        m.atom[:,1] /= b
        m.atom[:,2] /= c

        shift = [0.5-m.atom[0,0], 0.5-m.atom[0,1], 0.5-m.atom[0,2]]
        m.atom = m.atom+shift

        with open(file, 'w') as f:
            f.write('{:<80}\n'.format('Title: ' + str(file.split('.')[0])))
            f.write(lattice_type + ' '*(4-len(lattice_type)) +'LATTICE,NONEQUIV.ATOMS:' + '{0:3}'.format(len(sorted_names)) + '\n')
            f.write('MODE OF CALC=RELA\n')
            f.write('{0:10.6f}{1:10.6f}{2:10.6f}{3:10.6f}{4:10.6f}{5:10.6f}\n'.format(a/0.529177, b/0.529177, c/0.529177, alpha, beta, gamma))

            for i in range(m.atom.shape[0]):
                f.write('ATOM{0:4}: X={1:10.8f} Y={2:10.8f} Z={3:10.8f}\n'.format(-1*(i+1), m.atom[i, 0], m.atom[i, 1], m.atom[i, 2]))
                f.write(' '*10 + 'MULT= 1' + ' '*10 + 'ISPLIT={0:2}\n'.format(isplit))
                f.write('{0:10}'.format(m.atomName[i]) + ' NPT={0:5}'.format(NPT) +
                        '  R0={0:10.8f} RMT={1:10.5f}   Z:{2:10.5f}\n'.format(R0, RMT, atom_proton_numbers[m.atomName[i]]))
                f.write('LOCAL ROT MATRIX:   {0:10.7f}{1:10.7f}{2:10.7f}\n'.format(1,0,0))
                f.write(' '*20 + '{0:10.7f}{1:10.7f}{2:10.7f}\n'.format(0,1,0))
                f.write(' '*20 + '{0:10.7f}{1:10.7f}{2:10.7f}\n'.format(0,0,1))
            f.write('{0:4}\n'.format(nsym))
            f.close()
        m.export_xyz('after_transofrm.xyz')

    def checkInteratomicDistance(self, minDist=0.8):
        dist = scipy.spatial.distance.cdist(self.atom, self.atom)
        np.fill_diagonal(dist, np.max(dist))
        # print(np.min(dist))
        return np.min(dist) >= minDist

    def rdf(self, r, sigma, atoms):
        if atoms is None:
            # assert np.array_equal(self.atom[0], [0, 0, 0])
            ind = range(1, self.atom.shape[0])
        elif isinstance(atoms, str):
            atomName = atoms
            atomNumber = atom_proton_numbers[atomName]
            ind = np.where(self.atomNumber == atomNumber)[0]
            if ind.size == 0:
                raise Exception("There are no atoms with name " + atomName)
        else:
            ind = np.array(atoms)
            assert len(np.unique(self.az[ind])) == 1, 'All atoms must be of the same type!'
        rdf = np.zeros(r.shape)
        for i in ind:
            r0 = np.linalg.norm(self.atom[i] - self.atom[0])
            rdf += utils.gauss(r,r0,sigma)/r**2
        return rdf

    def adf(self, angle, sigma, atoms=None, maxDist=2.5):
        if atoms is None:
            # assert np.array_equal(self.atom[0], [0, 0, 0])
            ind = []
            for i in range(1, self.atom.shape[0]):
                if np.linalg.norm(self.atom[i] - self.atom[0]) > maxDist:
                    continue
                ind.append(i)
        elif isinstance(atoms, str):
            atomName = atoms
            atomNumber = atom_proton_numbers[atomName]
            ind = np.where(self.atomNumber == atomNumber)[0]
            if ind.size == 0:
                raise Exception("There are no atoms with name " + atomName)
        else:
            assert len(atoms) > 1, "You should specify more than 1 atom"
            ind = np.array(atoms)
            assert len(np.unique(self.az[ind])) == 1, 'All atoms must be of the same type!'
        adf = np.zeros(angle.shape)
        for i in ind:
            v1_u = geometry.normalize(self.atom[i]-self.atom[0])
            for j in ind:
                if j == i: continue
                v2_u = geometry.normalize(self.atom[j]-self.atom[0])
                adf0 = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))/np.pi*180
                adf += utils.gauss(angle, adf0, sigma)
        return adf

    def getSortedDists(self, atoms=None):
        """
        Returns sorted distances to atom[0]
        """
        if atoms is None:
            ind = np.arange(start=0, stop=len(self.az))
        elif isinstance(atoms, str):
            atomName = atoms
            atomNumber = atom_proton_numbers[atomName]
            ind = np.where(self.atomNumber == atomNumber)[0]
            if ind.size == 0:
                raise Exception("There are no atoms with name " + atomName)
        else:
            ind = np.array(atoms)
            assert len(np.unique(self.az[ind])) == 1, 'All atoms must be of the same type!'
        dists = np.linalg.norm(self.atom[ind,:]-self.atom[0], axis=1)
        return np.sort(dists)

    def deleteAtomAt(self, index):
        assert isinstance(index, int), 'Expecting index to be integer'
        assert (index <= self.atom.shape[0]) and (index > 0), 'Trying to delete atom out of indexing bounds'

        # remove from coordinates array, atom numbers & names
        self.atom = np.delete(self.atom, index, 0)
        self.atomNumber = np.delete(self.atomNumber, index, 0)
        self.atomName = np.delete(self.atomName, index, 0)
        self.az = self.atomNumber

        # delete given index from parts & shift accordingly
        for partIndices in self.partsData:
            for i in range(len(partIndices)):
                if i >= len(partIndices): continue
                if partIndices[i] == index:
                    del partIndices[i]
                elif partIndices[i] > index:
                    partIndices[i] = partIndices[i] - 1

    def deleteAtomsAt(self, *args):
        assert len(args) == len(set(args)), 'Duplicate atom indices in input'
        atomsToDelete = sorted(args) # needed to decrease atom index after every deleted atom
        for i in range(len(atomsToDelete)):
            self.deleteAtomAt(atomsToDelete[i] - i)

    def unionWith(self, otherMolecule):
        # append atoms from other molecule to self atoms
        mergedAtoms = np.append(self.atom, otherMolecule.atom, axis=0)
        mergedAtomNames = np.append(self.atomName, otherMolecule.atomName)
        mergedAtomNumbers = np.append(self.atomNumber, otherMolecule.atomNumber)

        # the same goes for part data
        mergedParts = copy.deepcopy(self.partsData) + copy.deepcopy(otherMolecule.partsData)

        # add offset for each index of partData related to the second molecule
        secondPartsStartIndex = len(self.partsData)
        firstMoleculeAtomCount = self.atom.shape[0]
        for i in range(secondPartsStartIndex, len(mergedParts)):
            for atomIndex in range(len(mergedParts[i])):
                mergedParts[i][atomIndex] += firstMoleculeAtomCount

        # save merged data
        self.atom = mergedAtoms
        self.partsData = mergedParts
        self.atomName = mergedAtomNames
        self.atomNumber = mergedAtomNumbers
        self.az = self.atomNumber

        # re-assign because of new parts
        self.assignParts()


class MoleculePart:
    def __init__(self, molecule, partIndex):
        self.molecule = molecule
        self.partIndex = partIndex
        self.atom = self # alias
        self.atomNumber = AtomInfo(molecule, partIndex)
        self.az = self.atomNumber # alias

    def __getitem__(self, key):
        assert isinstance(key, int)
        return self.molecule.atom[self.molecule.partsData[self.partIndex][key]]

    def __setitem__(self, key, value):
        assert isinstance(key, int)
        assert isinstance(value, np.ndarray) or isinstance(value, list)
        if isinstance(value, list): value = np.array(value)
        self.molecule.atom[self.molecule.partsData[self.partIndex][key]] = value

    def __len__(self):
        return len(self.molecule.partsData[self.partIndex])

    def rotate(self, axis, center, angle):
        self.molecule.rotate__Impl(axis, center, angle, self.molecule.partsData[self.partIndex])

    def rotate_xyz_axes(self, center, angles):
        self.molecule.rotate_xyz_axes__Impl(center, angles, self.molecule.partsData[self.partIndex])

    def shift(self, shift):
        self.molecule.shift__Impl(shift, self.partIndex)


class AtomInfo:
    def __init__(self, molecule, partIndex):
        self.molecule = molecule
        self.partIndex = partIndex

    def __getitem__(self, key):
        assert isinstance(key, int)
        return self.molecule.atomNumber[self.molecule.partsData[self.partIndex][key]]


def turn_mol_fast(mol, axis, center, angle):
    # поворачивает часть молекулы вокруг оси, заданной вектором axis. Положительное направление поворота определяется по правилу закручивающегося в направлении оси буравчика
    newMol = np.copy(mol)
    cphi = math.cos(angle)
    sphi = math.sin(angle)
    if isinstance(axis, list): axis = np.array(axis, dtype=float)
    if isinstance(center, list): center = np.array(center, dtype=float)
    axis = axis / np.linalg.norm(axis)
    # https://ru.wikipedia.org/wiki/%D0%9C%D0%B0%D1%82%D1%80%D0%B8%D1%86%D0%B0_%D0%BF%D0%BE%D0%B2%D0%BE%D1%80%D0%BE%D1%82%D0%B0
    x = axis[0]; y = axis[1]; z = axis[2]
    M = np.matrix([[cphi+(1-cphi)*x*x, (1-cphi)*x*y-sphi*z, (1-cphi)*x*z+sphi*y],\
        [(1-cphi)*y*x+sphi*z, cphi+(1-cphi)*y*y, (1-cphi)*y*z-sphi*x],\
        [(1-cphi)*z*x-sphi*y, (1-cphi)*z*y+sphi*x, cphi+(1-cphi)*z*z]])
    for i in range(mol.shape[0]):
        newMol[i,:3] = np.ravel(np.dot(M, newMol[i,:3] - center)) + center
    return newMol

def turn_mol_fast_xyzAxes(atoms, angles, center):
    """Turn molecule around coordinate axes"""

    inds_ = np.array([
        [1, 2],
        [0, 2],
        [0, 1]
    ])
    coords = atoms.copy()
    for icoord in range(3):
        # вращаем вокруг оси № icoord
        phi = angles[icoord]
        cphi = math.cos(phi)
        sphi = math.sin(phi)
        inds = inds_[icoord]
        c = center[inds]
        sign = 1-(icoord//2)*2

        centered = coords[:, inds] - c[None, :]
        M = np.array([
            [cphi, sphi*sign],
            [-sphi*sign, cphi]
        ])
        rotated = np.dot(centered, M) + c[None, :]
        coords[:, inds] = rotated
    return coords

def find_mol_dist(angles, coords1, coords2):
    """Find distance between molecules
    when second molecule is turned by specified angle"""

    coords2_turn = turn_mol_fast_xyzAxes(coords2, angles, coords2[0,:3])
    p = 5
    w = np.array([1.0, 1.0, 1.0, 1000.0])[None, None, :]
    cdist = (np.sum(w * np.abs(coords1[:, None, :] - coords2_turn[None, :, :]) ** p, axis=2)) ** (1 / p)
    if coords1.shape[0] > coords2.shape[0]:
        dist = np.min(cdist, axis=0).sum()
    else:
        dist = np.min(cdist, axis=1).sum()
    # print(angles, dist)
    return dist

def rotate_to_make_close(atoms1, atoms2, closeAtomCount):
    """Rotate second molecule so it is close to the first one"""
    dists1 = np.linalg.norm(atoms1[:,:3]-atoms1[0,:3], axis=1)
    dists2 = np.linalg.norm(atoms2[:,:3]-atoms2[0,:3], axis=1)
    ind1 = np.argsort(dists1); ind2 = np.argsort(dists2)
    selected_atoms1 = atoms1[ind1[:closeAtomCount+1], ...]
    selected_atoms2 = atoms2[ind2[:closeAtomCount+1], ...]
    selected_atoms1[:, :3] -= atoms1[0, :3]
    selected_atoms2[:, :3] -= atoms2[0, :3]
    # print('find_mol_dist(x0) =', find_mol_dist([0.0, 0.0, 0.0], selected_atoms1, selected_atoms2))
    # print(selected_atoms1)
    # print(selected_atoms2)
    res = scipy.optimize.minimize(
        fun=find_mol_dist,
        x0=[0.0, 0.0, 0.0],
        args=(selected_atoms1, selected_atoms2),
    )

    x = res['x']
    turned_molecule2 = turn_mol_fast_xyzAxes(atoms2, x, atoms2[0,:3])
    # print(selected_atoms1-selected_atoms1[0])
    # print(selected_atoms2-selected_atoms2[0])
    # print(turned_molecule2[ind2[:closeAtomCount+1], ...]-turned_molecule2[0])
    metric_value = res['fun']
    # print(metric_value)
    return x, turned_molecule2, metric_value

def compare(mol1, mol2, distThr, referenceMol):
    atoms0 = referenceMol.atom
    atoms1 = mol1.atom
    atoms2 = mol2.atom
    atoms0 = atoms0-atoms0[0]
    atoms1 = atoms1-atoms1[0]
    atoms2 = atoms2-atoms2[0]
    dists0 = np.linalg.norm(atoms0, axis=1)
    # dists1 = np.linalg.norm(atoms1, axis=1)
    # dists2 = np.linalg.norm(atoms2, axis=1)
    ind1 = np.argsort(dists0)
    ind1 = ind1[dists0[ind1] < distThr]
    ind2 = ind1 #!!!!!!!!!!!
    oldDist = np.linalg.norm(atoms1[ind1]-atoms2[ind2], axis=1).mean()
    return oldDist
    # _, opt_atoms2, dist = rotate_to_make_close(atoms1, atoms2, closeAtomCount)
    # newDist = np.linalg.norm(atoms1[ind1, :3]-opt_atoms2[ind2, :3], axis=1).mean()
    # return newDist

# return mol1, mol2 with atoms, sorted in matching order. Atoms of the mol1 are sorted by distance to the first atom
def matchAtomsHelper(mol1, mol2, closeAtomCountForAngleOptimizing, verbose=True):
    m1 = np.hstack((mol1.atom, mol1.az[:, None]))
    m2 = np.hstack((mol2.atom, mol2.az[:, None]))
    a1 = m1[1:,:3]-m1[0,:3]; a2 = m2[1:,:3]-m2[0,:3]
    dist1 = np.linalg.norm(a1,axis=1)
    dist2 = np.linalg.norm(a2,axis=1)
    ind1 = np.argsort(dist1)
    ind2 = np.argsort(dist2)
    results = []
    for i1 in range(1): # cycle by close to 0 atoms of mol1
        atom1 = a1[ind1[i1]]
        # print('atom1 ind =',ind1[i1])
        # take orthogonal atom
        tmp = np.dot(a1, atom1.reshape(-1,1)).reshape(-1)/dist1/np.linalg.norm(atom1)
        tmp[np.abs(tmp)>1] = 1
        atomAngles1 = np.arccos(tmp)
        den = 4
        ind = np.where(np.abs(atomAngles1[ind1]-np.pi/2) <= np.pi/2-np.pi/den)[0]
        while ind.size==0:
            den += 1
            ind = np.where(np.abs(atomAngles1[ind1]-np.pi/2) <= np.pi/2-np.pi/den)[0]
        orthAtom1 = a1[ind1][ind[0]]; orthAtomAngle1 = atomAngles1[ind1][ind[0]]; orthAtomDist1 = dist1[ind1][ind[0]]
        # print('orthAtom1 ind =',ind1[ind[0]])
        grOrthAtom1 = geometry.gramShmidt(atom1, orthAtom1)
        r = np.linalg.norm(atom1)
        closeAtomCount = np.sum(dist2<=r+0.5)
        # print('closeAtomCount =', closeAtomCount)
        if closeAtomCount==0: closeAtomCount=1
        for i2 in range(closeAtomCount): # cycle by close to 0 atoms of mol2
            # put a1[ind1[i1]] and a2[ind2[i2]] on the same axis
            atom2 = a2[ind2[i2]]
            # print('atom2 ind =',ind2[i2])
            rotationAxis = np.cross(atom1, atom2)
            if np.linalg.norm(rotationAxis)>1e-6:
                angle = -geometry.calcAngle(atom1, atom2)
                newM2 = turn_mol_fast(m2, rotationAxis, m2[0,:3], angle)
            else: newM2 = m2
            newA2 = newM2[1:,:3]-newM2[0,:3]
            newAtom2 = newA2[ind2[i2]]

            # find in mol2 atoms close to orthAtom1
            tmp = np.dot(newA2, newAtom2.reshape(-1,1)).reshape(-1)/dist2/np.linalg.norm(newAtom2)
            tmp[np.abs(tmp)>1] = 1
            atomAngles2 = np.arccos(tmp)
            orthogonalDistance = np.abs(orthAtomAngle1*orthAtomDist1 - atomAngles2*dist2) + np.abs(orthAtomDist1-dist2)
            ind = np.argsort( orthogonalDistance )
            closeOrthCount = np.sum(orthogonalDistance<=2)
            # print('closeOrthCount =', closeOrthCount)
            if closeOrthCount == 0: closeOrthCount=1
            for j2 in range(closeOrthCount):
                orthAtom2 = newA2[ind[j2]]
                # print('orthAtom2 ind =',ind[j2])
                grOrthAtom2 = geometry.gramShmidt(newAtom2, orthAtom2)
                # turn mol2 to match orthAtom planes
                rotationAxis = np.cross(grOrthAtom1, grOrthAtom2)
                if np.linalg.norm(rotationAxis)>1e-6:
                    angle = -geometry.calcAngle(grOrthAtom1, grOrthAtom2)
                    newM2_2 = turn_mol_fast(newM2, rotationAxis, newM2[0,:3], angle)
                else: newM2_2 = newM2
                # fit rotation angles
                _, newM2_3, dist = rotate_to_make_close(m1, newM2_2, closeAtomCountForAngleOptimizing)
                # print('dist =', dist)
                results.append([dist, newM2_3])
    ind = np.argmin([el[0] for el in results])
    bestM2 = results[ind][1]
    bestDist = results[ind][0]
    m1_ = np.copy(m1); m1_[:,:3]-=m1[0,:3]
    m2_ = np.copy(bestM2); m2_[:,:3]-=m2_[0,:3]
    cdist = scipy.spatial.distance.cdist(m1_, m2_, metric='wminkowski', w=[1.0, 1.0, 1.0, 1000.0], p=5)
    dist1_ = np.insert(dist1,0,0).reshape(-1,1); dist2_ = np.insert(dist2,0,0).reshape(1,-1)
    dist_ = (dist1_ + dist2_)/2
    assert (dist_.shape[0] == dist1_.size) and (dist_.shape[1] == dist2_.size)
    dist_[0][0] = 1
    # cdist /= dist_**2
    ind, indExtra = wbm.wbm(cdist)
    if verbose:
        print(ind)
    assert ind[0] == 0
    resMol1 = copy.deepcopy(mol1); resMol2 = copy.deepcopy(mol2)
    resMol2.atom = bestM2[:,:3]
    resMol1.atom = resMol1.atom[ind>=0]
    resMol1.setParts('0-'+str(len(resMol1.atom)-1))
    resMol2.atom = resMol2.atom[indExtra[indExtra>=0]]
    resMol2.atom += m1[0,:3]-m2[0,:3]
    resMol2.setParts('0-'+str(len(resMol2.atom)-1))
    notMatched = np.where(ind<0)[0]
    if verbose:
        if notMatched.size>0: print('Not matched atoms in mol1:')
        for i in notMatched:
            print(i,'proton_number =', '%.0f' % m1[i,3], 'x =', '%.4f' % m1[i,0], 'dist =', '%.2f' % np.linalg.norm(m1[i,:3]-m1[0,:3]))
            print('------------------------')
    return resMol1, resMol2


# return mol1, mol2 with atoms, sorted in matching order. Atoms of the mol1 are sorted by distance to the first atom
def matchAtoms(mol1, mol2, closeAtomCountForAngleOptimizing, distThrForSymmetryChoose):
    # match original molecules
    matched = matchAtomsHelper(
        mol1, mol2, closeAtomCountForAngleOptimizing,
        verbose=False)
    assert mol1.atom.shape[0] == matched[0].atom.shape[0]
    assert mol2.atom.shape[0] == matched[1].atom.shape[0]
    dist = compare(*matched, distThrForSymmetryChoose, mol1)

    # match mirrored molecules
    mol2_mir = copy.deepcopy(mol2)
    mol2_mir.atom[:, 0] *= -1
    matched_mir = matchAtomsHelper(
        mol1, mol2_mir, closeAtomCountForAngleOptimizing,
        verbose=False)
    dist_mir = compare(*matched_mir, distThrForSymmetryChoose, mol1)

    # print(dist, dist_mir)
    if dist < dist_mir:
        return matched[0], matched[1], dist
    return matched_mir[0], matched_mir[1], dist_mir

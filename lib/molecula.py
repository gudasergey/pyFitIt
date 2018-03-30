import copy
import tempfile
import os
import pandas as pd
import numpy as np
import math
import geometry

atom_proton_numbers = {'H':1, 'He':2, \
    'Li':3, 'Be':4, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Ne':10, \
    'Na':11, 'Mg':12, 'Al':13, 'Si':14, 'P':15, 'S':16, 'Cl':17, 'Ar':18,\
    'K':19, 'Ca':20, 'Sc':21, 'Ti':22, 'V':23, 'Cr':24, 'Mn':25, 'Fe':26, 'Co':27, 'Ni':28,\
    'Cu':29, 'Zn':30, 'Ga':31, 'Ge':32, 'As':33, 'Se':34, 'Br':35, 'Kr':36,\
    'Rb':37, 'Sr':38, 'Y':39, 'Zr':40, 'Nb':41, 'Mo':42, 'Tc':43, 'Ru':44, 'Rh':45, 'Pd':46,\
    'Ag':47, 'Cd':48, 'In':49, 'Sn':50, 'Sb':51, 'Te':52, 'I':53, 'Xe':54,\
    'Cs':55, 'Ba':56, 'Nd':60, 'Hf':72, 'Ta':73, 'W':74, 'Re':75, 'Os':76, 'Ir':77, 'Pt':78,\
    'Au':79, 'Hg':80, 'Tl':81, 'Pb':82, 'Bi':83, 'Po':84, 'At':85, 'Rn':86,\
    'Fr':87, 'Ra':88, 'U':92, 'Rf':104, 'Db':105, 'Sg':106, 'Bh':107, 'Hn':108, 'Mt':109\
    } # TODO: дозаполнить

class MoleculaPart:
    ind = None # индексы атомов в PandasMol2 dataframe (только несимметричная часть)
    center = None # координаты центра вращения (не обязательно позиция атома)
    def __init__(self, ind, center):
        self.ind = ind
        self.center = center

def dist(r1, r2):
    return math.sqrt((r1['x'] - r2['x'])**2 + (r1['y'] - r2['y'])**2 + (r1['z'] - r2['z'])**2)

class Molecula:
    mol = None # PandasMol2 dataframe (связи не использовать! они нарушаются после применения symmetry т.к. индексы атомов меняются!)
    parts = [] # список классов MoleculaPart (только несимметричная часть)
    # функция, которая для координат атома из nonSymmetryAtomInds применяет к ним нужно число раз отображение (повороты, отражения)
    # и возвращает список симметричных атомов. Может работать и для произвольно точки (не обязательно в которой есть атом)
    symmetry = None
    minDistance = 0.9 #минимальное расстояние сближания атомов. Если после применения преобразования атомы сближаются ближе, то преобразование не применяем
    maxDistEqualAtoms = 1e-3 # если атомы ближе данного расстояния, то считаем что они в одной точке

    # достаточно задавать только нессимметричные атомы, но для проверки можно задать все
    # части не должны пересекаться и объединение всех частей должно равняться нессимметричной части молекулы
    # symmetry - отображение, которое точке [x,y,z] ставит в соответствие список симметричных точек
    # конструктор может распарсить .mol2 или .xyz-файл с пустыми строками-разделителями частей. В последнем случае массив частей конструируется автоматически
    # lastGroupIsPart - если парсится файл .xyz, то последняя часть обычно используется для проверки преобразования симметрии предыдущих частей.
    # Если этой части нет, то нужно установить lastGroupIsPart = True
    # Способ выбора partCenter (mean или first atom) - используется только при парсинге частей из файла xyz
    def __init__(self, filename, symmetry, parts = [], checkSymmetry = False, expandBySymmetry = True, lastGroupIsPart = False, partCenter = 'mean'):
        if isinstance(filename, str):
            ext = '.mol2'
            if filename[-len(ext):] == ext: self.mol, _ = Molecula.parse_mol2(filename)
            else:
                ext = '.xyz'
                if filename[-len(ext):] == ext: self.mol, self.parts = Molecula.parse_xyz_parts(filename, lastGroupIsPart)
                else: raise Exception('Unknown file extension')
        else:
            self.mol = filename
        if len(parts)>0: self.parts = parts
        else:
            for part in self.parts:
                if partCenter == 'mean': part.center = self.mol.loc[part.ind,['x','y','z']].mean().values
                elif partCenter == 'zero': part.center = np.zeros(3)
                else:
                    assert partCenter == 'first atom', 'Unknown partCenter value'
                    part.center = self.mol.loc[part.ind[0],['x','y','z']].values
        self.symmetry = symmetry
        if checkSymmetry:
            self.checkSymmetryOK()
        if expandBySymmetry:
            self.expandBySymmetry()
        # self.export_xyz('check.xyz')

    def copy(self):
        return Molecula(copy.deepcopy(self.mol), copy.deepcopy(self.symmetry), \
            parts = copy.deepcopy(self.parts), checkSymmetry = False, expandBySymmetry = False)

    def assign(self, otherMol):
        self.mol = otherMol.mol
        self.parts = otherMol.parts

    # связи из mol2 формата генерируются Discavery, но на самом деле это условно, нам нужны другие связи (сейчас связи разрушаются из-за перенумеровки атомов в expandBySymmetry)
    # bond types: 1 = single, 2 = double, 3 = triple, am = amide, ar = aromatic, du = dummy, un = unknown (cannot be determined from the parameter tables), nc = not connected
    @staticmethod
    def parse_mol2(filename):
        pmol = PandasMol2().read_mol2(filename)
        s = pmol.mol2_text
        i = s.find('@<TRIPOS>BOND'); i = s.find("\n",i)+1
        j = s.find("@",i)
        mol2_bonds = pd.read_csv(StringIO(s[i:j]), sep='\s+', names=['bondid', 'atom1', 'atom2', 'bondtype'])
        proton_map = {'F':26, 'N':7, 'C':6, 'H':1}
        atom_names = pmol.df['atom_type'].apply(lambda n: n[0:n.find('.')] if n.find('.')>=0 else n)
        pmol.df['proton_number'] = atom_names.map(proton_map)
        return pmol.df, mol2_bonds

    # Столбцы: буква атома, координаты. Пустые строки - разделители частей. Последняя часть - это все атомы, которые получаются из предыдущих отображениями симметрии
    @staticmethod
    def parse_xyz_parts(filename, lastGroupIsPart):
        mol = pd.DataFrame(columns=['proton_number','x','y','z','atom_name'])
        s = open(filename, 'r').read().strip()
        partsLines = s.split("\n\n")
        parts = []
        def parsePart(ps):
            lines = ps.split("\n")
            atoms = []
            for line in lines:
                if line[0] == '#': continue
                words = line.strip().split()
                assert len(words)==4, "Wrong line "+line+" in file "+filename
                atoms.append([atom_proton_numbers[words[0]], float(words[1]), float(words[2]), float(words[3]), words[0]])
            return atoms
        k = 0
        for partStr in partsLines:
            atoms = parsePart(partStr)
            inds = []
            maxCharge = -1
            center = np.zeros(3)
            for atom in atoms:
                mol.loc[k] = atom
                inds.append(k)
                if atom[0] > maxCharge:
                    maxCharge = atom[0]
                    center[:] = atom[1:4]
                k += 1
            if (partStr != partsLines[-1]) or lastGroupIsPart: parts.append(MoleculaPart(inds, center))
        return mol, parts

    def __str__(self):
        s = ''
        for part in self.parts:
            for i in part.ind:
                r = self.mol.loc[i]
                s += r['atom_name'].rjust(2)+'  '+str(r['x']).rjust(10)+'  '+str(r['y']).rjust(10)+'  '+str(r['z']).rjust(10)+"\n"
            s += '\n'
        return s

    def export_xyz(self, file, full = False, cellSize=1):
        f = open(file, 'w')
        f.write(str(self.mol.shape[0])+'\n')
        f.write('molecula\n')
        if full:
            for i in range(self.mol.shape[0]):
                r = self.mol.loc[i]
                f.write(r['atom_name'].rjust(2)+'  '+str(r['x']*cellSize).rjust(10)+'  '+str(r['y']*cellSize).rjust(10)+'  '+str(r['z']*cellSize).rjust(10)+"\n")
        else:
            for part in self.parts:
                for i in part.ind:
                    r = self.mol.loc[i]
                    f.write(r['atom_name'].rjust(2)+'  '+str(r['x']*cellSize).rjust(10)+'  '+str(r['y']*cellSize).rjust(10)+'  '+str(r['z']*cellSize).rjust(10)+"\n")
                f.write('\n')
        f.close()

    def unionWith(self, other0):
        other = other0.copy()
        n0 = self.mol.shape[0]
        partCount0 = len(self.parts)
        self.mol = pd.concat([self.mol, other.mol], ignore_index=True)
        self.parts.extend(other.parts)
        for i in range(partCount0, len(self.parts)):
            p = self.parts[i]
            for j in range(len(p.ind)): p.ind[j] += n0
        self.expandBySymmetry()

    def move_mol_part(self, partInd, shift):
        # сдвигает часть молекулы на вектор shift
        shift = np.array(shift,dtype=float)
        newMol = self.copy()
        part = newMol.parts[partInd]
        for i in part.ind:
            newMol.mol.loc[i,['x','y','z']] += shift
        part.center += shift
        newMol.expandBySymmetry()
        ok,_,_ = newMol.checkDistanceOK()
        if ok: self.assign(newMol)
        return ok

    def turn_mol_part(self, partInd, axis, angle):
        # поворачивает часть молекулы вокруг оси, заданной вектором axis. Положительное направление поворота определяется по правилу закручивающегося в направлении оси буравчика
        newMol = self.copy()
        cphi = math.cos(angle)
        sphi = math.sin(angle)
        if isinstance(axis, list): axis = np.array(axis,dtype=float)
        axis /= np.linalg.norm(axis)
        # https://ru.wikipedia.org/wiki/%D0%9C%D0%B0%D1%82%D1%80%D0%B8%D1%86%D0%B0_%D0%BF%D0%BE%D0%B2%D0%BE%D1%80%D0%BE%D1%82%D0%B0
        x = axis[0]; y = axis[1]; z = axis[2]
        M = np.matrix([[cphi+(1-cphi)*x*x, (1-cphi)*x*y-sphi*z, (1-cphi)*x*z+sphi*y],\
            [(1-cphi)*y*x+sphi*z, cphi+(1-cphi)*y*y, (1-cphi)*y*z-sphi*x],\
            [(1-cphi)*z*x-sphi*y, (1-cphi)*z*y+sphi*x, cphi+(1-cphi)*z*z]])
        part = newMol.parts[partInd]
        center = part.center
        for i in part.ind:
            atom = newMol.mol.loc[i,['x','y','z']].values
            atom = np.ravel(np.dot(M, atom - center)) + center
            newMol.mol.loc[i, ['x','y','z']] = atom
        newMol.expandBySymmetry()
        ok,_,_ = newMol.checkDistanceOK()
        if ok: self.assign(newMol)
        return ok

    def turn_mol_part_xyzAxis(self, partInd, angles):
        # поворачивает часть молекулы вокруг осей координат и ее центра. angles - в радианах
        newMol = self.copy()
        part = newMol.parts[partInd]
        center = part.center
        #print(moleculas[imol])
        for icoord in range(3):
            # вращаем вокруг оси № icoord
            phi =angles[icoord]
            #phi=0
            cphi = math.cos(phi)
            sphi = math.sin(phi)
            inds = np.arange(3)
            inds = np.delete(inds,icoord)
            c = center[inds]
            for i in part.ind:
                atom = newMol.mol.loc[i,['x','y','z']].values
                sign = 1-(icoord//2)*2
                atom[inds] = geometry.turnCoords(atom[inds]-c, cphi, sphi*sign) + c
                newMol.mol.loc[i,['x','y','z']] = atom
        newMol.expandBySymmetry()
        ok,_,_ = newMol.checkDistanceOK()
        if ok: self.assign(newMol)
        return ok

    @staticmethod
    def findAtom(mol, atom, eps):
        for i, row in mol.iterrows():
            if (dist(row, atom) < eps) and (row['proton_number'] == atom['proton_number']): return i
        return -1

    def expandBySymmetry(self):
        if self.symmetry is None:
            return
        newMol = pd.DataFrame(columns = self.mol.columns)
        newParts = []
        k = 0

        for part in self.parts:
            newPart = []
            for i in part.ind:
                newAtom = self.mol.loc[i]
                newMol.loc[k] = newAtom
                newPart.append(k)
                #print('Adding atom '+str(newAtom['proton_number'])+' in pos', newAtom[['x','y','z']].values)
                k += 1
            newParts.append(MoleculaPart(newPart, part.center))

        for part in self.parts:
            newPart = []
            for i in part.ind:
                newAtom = self.mol.loc[i]
                symAtoms = self.symmetry(newAtom[['x','y','z']].values)
                for a in symAtoms:
                    atom = newAtom.copy(deep=True)
                    atom[['x','y','z']] = a
                    findRes = Molecula.findAtom(newMol, atom, self.maxDistEqualAtoms)
                    if findRes < 0:
                        newMol.loc[k] = atom
                        #print('Adding atom '+str(atom['proton_number'])+' in pos', atom[['x','y','z']].values)
                        k += 1
                    #else:
                        #print('Atom '+str(atom['proton_number'])+' already added in pos', atom[['x','y','z']].values)
                        #print(newMol.loc[findRes])
        self.mol = newMol
        self.parts = newParts

    def checkDistanceOK(self):
        for i in range(self.mol.shape[0]):
            for j in range(self.mol.shape[0]):
                if i == j: continue
                if dist(self.mol.loc[i], self.mol.loc[j]) < self.minDistance: return False, i, j
        return True, 0, 0

    def checkSymmetryOK(self):
        dup = self.copy()
        ok, i1, i2 = dup.checkDistanceOK()
        if not ok: raise Exception("Atoms "+str(i1)+" and "+str(i2)+" too close to each other. Check molecule")
        dup.expandBySymmetry()
        ok, i1, i2 = dup.checkDistanceOK()
        if not ok:
            print(dup.mol.loc[i1])
            print(dup.mol.loc[i2])
            raise Exception("After expandBySymmetry atoms "+str(i1)+" and "+str(i2)+" become too close to each other")
        notFoundCount = 0
        for i in range(self.mol.shape[0]):
            found = False
            for j in range(dup.mol.shape[0]):
                if dist(self.mol.loc[i], dup.mol.loc[j]) < self.maxDistEqualAtoms:
                    found = True
                    break
            #if not found: notFoundCount+=1; print("Can't find atom after expandBySymmetry:"); print(self.mol.loc[i]); print(dist(self.mol.loc[i], self.mol.loc[0]))
        #if notFoundCount>0: print("Not found", notFoundCount, "atoms")
        #if self.mol.shape[0] != dup.mol.shape[0]:
            #print('Warning: after expandBySymmetry atom counts are not equal: old=', self.mol.shape[0], 'new=', dup.mol.shape[0])
        notFoundCount = 0
        for i in range(dup.mol.shape[0]):
            found = False
            for j in range(self.mol.shape[0]):
                if dist(dup.mol.loc[i], self.mol.loc[j]) < self.maxDistEqualAtoms:
                    found = True
                    break
            #if not found: notFoundCount+=1; print("Extra atom after expandBySymmetry:"); print(self.mol.loc[i]); print(dist(dup.mol.loc[i], self.mol.loc[0]))
        #if notFoundCount>0: print("Extra found", notFoundCount, "atoms")

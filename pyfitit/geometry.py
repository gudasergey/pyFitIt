import numpy as np
import math


def sign(x): return math.copysign(1, x)


def normalize(v): return v/np.linalg.norm(v)


def calcAngle(p1, p2):
    p1_u = normalize(p1)
    p2_u = normalize(p2)
    return np.arccos(np.clip(np.dot(p1_u, p2_u), -1.0, 1.0))


def turnCoords(xpair, cphi, sphi):
    M = np.array([
        [cphi, sphi],
        [-sphi, cphi]
    ])
    return np.dot(xpair, M)


def turnCoords_degrees(xpair, degrees):
    phi = degrees/180*math.pi
    sphi = math.sin(phi)
    cphi = math.cos(phi)
    return np.array([xpair[0]*cphi-xpair[1]*sphi, xpair[0]*sphi+xpair[1]*cphi])


# поворачивает точку вокруг осей координат и заданного центра. angles - в радианах
def turn_xyzAxis(point0, rotCenter, angles):
    if isinstance(rotCenter, list): rotCenter = np.array(rotCenter, dtype=float)
    point = np.copy(point0)
    for icoord in range(3):
        # вращаем вокруг оси № icoord
        phi =angles[icoord]
        cphi = math.cos(phi)
        sphi = math.sin(phi)
        inds = np.arange(3)
        inds = np.delete(inds,icoord)
        c = rotCenter[inds]
        sign = 1-(icoord//2)*2
        point[inds] = turnCoords(point[inds]-c, cphi, sphi*sign) + c
    return point


# поворачивает точку вокруг оси, заданной вектором axis и точкой rotCenter. Положительное направление поворота определяется по правилу закручивающегося в направлении оси буравчика
def turn_axis(point0, rotCenter, axis, angle):
    point = np.copy(point0)
    cphi = math.cos(angle)
    sphi = math.sin(angle)
    if isinstance(axis, list): axis = np.array(axis, dtype=float)
    if isinstance(rotCenter, list): rotCenter = np.array(rotCenter, dtype=float)
    axis /= np.linalg.norm(axis)
    # https://ru.wikipedia.org/wiki/%D0%9C%D0%B0%D1%82%D1%80%D0%B8%D1%86%D0%B0_%D0%BF%D0%BE%D0%B2%D0%BE%D1%80%D0%BE%D1%82%D0%B0
    x = axis[0]; y = axis[1]; z = axis[2]
    M = np.matrix([[cphi+(1-cphi)*x*x, (1-cphi)*x*y-sphi*z, (1-cphi)*x*z+sphi*y],\
        [(1-cphi)*y*x+sphi*z, cphi+(1-cphi)*y*y, (1-cphi)*y*z-sphi*x],\
        [(1-cphi)*z*x-sphi*y, (1-cphi)*z*y+sphi*x, cphi+(1-cphi)*z*z]])
    point = np.ravel(np.dot(M, point - rotCenter)) + rotCenter
    return point


# ортогонализует вектор b
def gramShmidt(a, b):
    if a.shape[0]==1: a = a.reshape(-1); b = b.reshape(-1)
    return  b - a*np.dot(a,b)/np.dot(a,a)


# для поворотов вокруг сначала z на угол phi, потом вокруг перпендикулярной оси y на угол psi
# возвращает угол alpha отклонения от оси x и радиальный угол beta поворота вокруг x
def transformAngles(phi, psi):
    phi = np.array(phi)/180*math.pi
    psi = np.array(psi)/180*math.pi
    projection = np.sqrt(1 - np.cos(phi)**2*np.cos(psi)**2)
    if np.isscalar(phi):
        if projection<1e-6: projection = 1e-6
        if projection>1: projection=1
    else:
        projection[projection<1e-6] = 1e-6
        projection[projection>1] = 1
    alpha = math.pi/2 - np.arccos(projection)
    cosbeta = np.sin(phi)/projection
    if np.isscalar(phi):
        if cosbeta>1: cosbeta=1
    else: cosbeta[cosbeta>1]=1
    beta = np.arccos(cosbeta)
    if np.isscalar(beta):
        if psi<0: beta = 2*math.pi - beta
    else:
        ind = psi<0
        beta[ind] = 2*math.pi - beta[ind]
    return alpha/math.pi*180, beta/math.pi*180


def transformAnglesInv(alpha, beta):
    alpha = np.array(alpha)/180*math.pi
    beta = np.array(beta)/180*math.pi
    c = np.sin(alpha)*np.cos(beta)
    if np.isscalar(c):
        if c>1: c = 1
        if c<-1: c=-1
    else: c[c>1]=1; c[c<-1]=-1
    phi = math.pi/2 - np.arccos(c)
    c = np.cos(alpha)/np.cos(phi)
    if np.isscalar(c):
        if c>1: c = 1
        if c<-1: c=-1
    else: c[c>1]=1; c[c<-1]=-1
    psi = np.arccos(c)
    return phi/math.pi*180, psi/math.pi*180

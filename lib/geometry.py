import numpy as np
import math

def sign(x): return math.copysign(1, x)

def turnCoords(xpair, cphi, sphi):
    return np.array([xpair[0]*cphi-xpair[1]*sphi, xpair[0]*sphi+xpair[1]*cphi])

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
    return  b - a*np.dot(a,b)/np.dot(a,a)

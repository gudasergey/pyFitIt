import copy, math, scipy
import numpy as np
from numba import jit


def sign(x): return math.copysign(1, x)


def normalize(v):
    n = np.linalg.norm(v)
    if n > 0:
        return v/np.linalg.norm(v)
    else: return np.zeros(v.size)


def length(x):
    if hasattr(x, "__len__"): return len(x)
    else: return 1   # x is scalar


def relDist(a, b):
    assert len(a.shape) <= 2 and len(b.shape) <= 2
    if len(a.shape) == 1: a = a.reshape(1, -1)
    if len(b.shape) == 1: b = b.reshape(1, -1)
    sum = np.linalg.norm(a, axis=1) + np.linalg.norm(b, axis=1)
    sum[sum == 0] = 1
    return np.linalg.norm(a-b, axis=1)/sum


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


def get_line_by_2_points(x1, y1, x2, y2):
    a = y2-y1
    b = -(x2-x1)
    c = -x1*a - y1*b
    assert np.abs(a * x1 + b * y1 + c) < 1e-10
    assert np.abs(a * x2 + b * y2 + c) < 1e-10
    return a,b,c


def get_line_by_2_points_kb(x1, y1, x2, y2):
    """
    Return b,k for the model y=kx+b
    """
    a,b,c = get_line_by_2_points(x1, y1, x2, y2)
    assert np.abs(b) > max(np.abs(a), np.abs(c))*1e-10
    return -c/b,-a/b


def get_line_intersection_kb(b1, k1, b2, k2):
    return get_line_intersection(k1, -1, b1, k2, -1, b2)


def get_line_intersection(a1, b1, c1, a2, b2, c2):
    d = a1*b2 - b1*a2
    if d == 0:
        n1 = np.linalg.norm([a1,b1])
        n2 = np.linalg.norm([a2,b2])
        if n1 == 0 and c1 != 0: return np.nan, np.nan
        if n2 == 0 and c2 != 0: return np.nan, np.nan
        if n1 == 0 and n2 == 0: return 0,0
        if n1 == 0: return a2/n2**2*c2, b2/n2**2*c2
        if n2 == 0: return a1/n1**2*c1, b1/n1**2*c1
        if np.isclose(c1/n1, c2/n2, rtol=1e-6): return a1/n1**2*c1, b1/n1**2*c1
        else: return np.nan, np.nan
    z = np.linalg.solve(np.array([[a1,b1], [a2,b2]]), [-c1,-c2])
    x = z[0]; y = z[1]
    assert np.abs(a1*x+b1*y+c1)<1e-6, f'{np.abs(a1*x+b1*y+c1)}>1e-6. a1={a1} b1={b1} c1={c1}   a2={a2} b2={b2} c2={c2}'
    assert np.abs(a2*x+b2*y+c2)<1e-6, f'{np.abs(a2*x+b2*y+c2)}>1e-6. a1={a1} b1={b1} c1={c1}   a2={a2} b2={b2} c2={c2}'
    return x, y


def get_dist_to_line(a,b,c,x0,y0):
    return np.abs(a*x0+b*y0+c)/np.sqrt(a**2+b**2)

def getNNdistsStable(x1, x2):
    dists = scipy.spatial.distance.cdist(x1, x2)
    M = np.max(dists)
    mean = np.mean(dists)
    np.fill_diagonal(dists, M)
    NNdists = np.min(dists, axis=1)
    return NNdists, mean


def getMinDist(x1, x2):
    NNdists, mean = getNNdistsStable(x1, x2)
    # take min element with max index (we want to throw last points first)
    min_dist_ind = np.max(np.where(NNdists == np.min(NNdists))[0])
    min_dist = NNdists[min_dist_ind]
    return min_dist_ind, min_dist, mean


def unique_mulitdim(p, rel_err=1e-6):
    """
    Remove duplicate points from array p

    :param p: points (each row is one point)
    :param rel_err: max difference between points to be equal i.e. dist < np.quantile(all_dists, 0.1) * rel_err
    :returns: new_p, good_ind
    """
    p = copy.deepcopy(p)
    assert len(p.shape) == 2, 'p must be 2-dim (each row is one point). For 1 dim we can\'t determine whether it is row or column'
    good_ind = np.arange(len(p))
    min_dist_ind, min_dist, med_dist = getMinDist(p,p)
    while min_dist <= med_dist * rel_err:
        good_ind = np.delete(good_ind, min_dist_ind)
        min_dist_ind, min_dist, _ = getMinDist(p[good_ind, :], p[good_ind, :])
    return p[good_ind, :], good_ind


@jit(nopython=True)
def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


@jit(nopython=True)
def get_convex_hull(p: np.array) -> np.array:
    n = p.shape[0]
    k = 0
    if n == 1:
        return p
    hull = np.zeros((n * 2, 2))

    # sort points lexicographically
    x1 = np.sort(p[:, 0])
    dx1 = x1[1:] - x1[:-1]
    mdx1 = np.min(dx1[dx1 > 0])
    mnoj = 2 / mdx1
    ind = np.argsort(p[:, 0] * mnoj + p[:, 1])
    p = p[ind]

    # build lower hull
    for ind, point in enumerate(p):
        while k >= 2 and cross(hull[k - 2], hull[k - 1], point) <= 0:
            k -= 1
        hull[k] = point
        k += 1

    # build upper hull
    t = k + 1
    for ind in range(n - 2, -1, -1):
        while k >= t and cross(hull[k - 2], hull[k - 1], p[ind]) <= 0:
            k -= 1
        hull[k] = p[ind]
        k += 1

    # remove trailing zeros
    hull = hull[: (k - 1), :]
    return hull
import copy, scipy
import numpy as np
from . import utils

class Exafs:
    def __init__(self, k, chi):
        self.k = k
        self.chi = chi

    def toXanes(self, Efermi, k_power):
        me = 2 * 9.109e-31  # kg
        h = 1.05457e-34  # J*s
        J = 6.24e18  # 1 J = 6.24e18 eV
        e = self.k**2 * J / me * h**2 * 1e20 + Efermi
        k = copy.deepcopy(self.k)
        k[k==1] = 1
        return utils.Spectrum(e, self.chi/k**k_power + 1)

    def shift(self, dE, Efermi, inplace=False):
        xan = self.toXanes(Efermi, k_power=0)
        xan.energy += dE
        exafs = xan.toExafs(Efermi, k_power=0)
        if inplace:
            self.k, self.chi = exafs.k, exafs.chi
        else:
            return exafs

    def smooth(self, SO2, sigmaSquare, inplace=False):
        chi1 = self.chi * np.exp(-2 * self.k ** 2 * sigmaSquare) * SO2
        if inplace: self.chi = chi1
        else: return Exafs(self.k, chi1)


def ftwindow(x, xmin=None, xmax=None, dx=1, dx2=None, window='hanning', **kws):
    VALID_WINDOWS = ['han', 'fha', 'gau', 'kai', 'par', 'wel', 'sin', 'bes']
    if window is None:
        window = VALID_WINDOWS[0]
    nam = window.strip().lower()[:3]
    if nam not in VALID_WINDOWS:
        raise RuntimeError("invalid window name %s" % window)

    dx1 = dx
    if dx2 is None:  dx2 = dx1
    if xmin is None: xmin = min(x)
    if xmax is None: xmax = max(x)

    xstep = (x[-1] - x[0]) / (len(x) - 1)
    xeps = 1.e-4 * xstep
    x1 = max(min(x), xmin - dx1 / 2.0)
    x2 = xmin + dx1 / 2.0 + xeps
    x3 = xmax - dx2 / 2.0 - xeps
    x4 = min(max(x), xmax + dx2 / 2.0)

    if nam == 'fha':
        if dx1 < 0: dx1 = 0
        if dx2 > 1: dx2 = 1
        x2 = x1 + xeps + dx1 * (xmax - xmin) / 2.0
        x3 = x4 - xeps - dx2 * (xmax - xmin) / 2.0
    elif nam == 'gau':
        dx1 = max(dx1, xeps)

    def asint(val): return int((val + xeps) / xstep)

    i1, i2, i3, i4 = asint(x1), asint(x2), asint(x3), asint(x4)
    i1, i2 = max(0, i1), max(0, i2)
    i3, i4 = min(len(x) - 1, i3), min(len(x) - 1, i4)
    if i2 == i1: i1 = max(0, i2 - 1)
    if i4 == i3: i3 = max(i2, i4 - 1)
    x1, x2, x3, x4 = x[i1], x[i2], x[i3], x[i4]
    if x1 == x2: x2 = x2 + xeps
    if x3 == x4: x4 = x4 + xeps
    # initial window
    fwin = np.zeros(len(x))
    if i3 > i2:
        fwin[i2:i3] = np.ones(i3 - i2)

    # now finish making window
    if nam in ('han', 'fha'):
        fwin[i1:i2 + 1] = np.sin((np.pi / 2) * (x[i1:i2 + 1] - x1) / (x2 - x1)) ** 2
        fwin[i3:i4 + 1] = np.cos((np.pi / 2) * (x[i3:i4 + 1] - x3) / (x4 - x3)) ** 2
    elif nam == 'par':
        fwin[i1:i2 + 1] = (x[i1:i2 + 1] - x1) / (x2 - x1)
        fwin[i3:i4 + 1] = 1 - (x[i3:i4 + 1] - x3) / (x4 - x3)
    elif nam == 'wel':
        fwin[i1:i2 + 1] = 1 - ((x[i1:i2 + 1] - x2) / (x2 - x1)) ** 2
        fwin[i3:i4 + 1] = 1 - ((x[i3:i4 + 1] - x3) / (x4 - x3)) ** 2
    elif nam in ('kai', 'bes'):
        cen = (x4 + x1) / 2
        wid = (x4 - x1) / 2
        arg = 1 - (x - cen) ** 2 / (wid ** 2)
        arg[arg < 0] = 0
        if nam == 'bes':  # 'bes' : ifeffit 1.0 implementation of kaiser-bessel
            fwin = scipy.special.i0(dx * np.sqrt(arg)) / scipy.special.i0(dx)
            fwin[x <= x1] = 0
            fwin[x >= x4] = 0
        else:  # better version
            scale = max(1.e-10, scipy.special.i0(dx) - 1)
            fwin = (scipy.special.i0(dx * np.sqrt(arg)) - 1) / scale
    elif nam == 'sin':
        fwin[i1:i4 + 1] = np.sin(np.pi * (x4 - x[i1:i4 + 1]) / (x4 - x1))
    elif nam == 'gau':
        cen = (x4 + x1) / 2
        fwin = np.exp(-(((x - cen) ** 2) / (2 * dx1 * dx1)))
    return fwin


def FT_Transform(k,chi):
    ZF=2048 #2^11
    npt=ZF/2
    pask=1/(k[1]-k[0])
    freq=(1/ZF)*(np.arange(npt))*pask
    omega=2*(np.pi)*freq
    tff=np.fft.fft(chi,ZF,norm='ortho')
    ft=tff[0:int(npt)]
    return omega/2, ft


def convert2RSpace(k, chi, kmin=3, kmax=11, dk=1, dk2=None, window='kaiser'):
    """
    Doesn't multiply spectrum by k**2. You should do it manually
    """
    return FT_Transform(k, ftwindow(k, xmin=kmin, xmax=kmax, dx=dk, dx2=dk2, window=window) * chi)


def waveletTransform(k, chi, interval=None, sigma=1, eta=5, kmin=2.4, kmax=10.5, Rmin=0.5, Rmax=4.5, knum=200, Rnum=200, plotFilename=None):
    """
    :param interval: [k1, k2] - k-space for the wavelet
    :param sigma: don't touch this parameter
    :param eta: put this value equal to two time the distance where you suppose to have your scattering
    :param kmin: kmin for the apodization window (rectangular)
    :param kmax: kmax for the apodization window (rectangular)
    :param Rmin: R-space for the Wavelet
    :param Rmax: R-space for the Wavelet
    :returns: arra of k values, array of r values, w matrix (rows - k, cols - r)
    """
    if interval is None: interval = [k[0], k[-1]]
    # range of the apodization window (rectangular)
    i = (interval[0]<=k) & (k<=interval[1])
    k_redW, chi_redW = k[i], chi[i]
    kw = np.linspace(*interval, num=knum)  # k-points of the Wavelet
    rw = np.linspace(Rmin, Rmax, num=Rnum)  # R-points of the Wavelet

    def fz(k, eta, sigma):
        I = complex(0, 1)
        return np.exp(k * eta * I) * np.exp((-k ** 2) / (2 * sigma ** 2))

    # interpolation of the experimental signal
    def chi1(omega):
        return np.interp(omega, k_redW, chi_redW * ftwindow(k_redW, xmin=kmin, xmax=kmax, dx=1, dx2=None, window='hanning'))

    # function experimental wavelet
    def W_exp(omega, r):
        a = eta / (2 * r)
        k2 = k_redW
        a1 = (k2[:, None] - omega)
        return (1 / np.sqrt(a)) * np.trapz(((chi1(k2) * np.conj(fz(a1[..., None] / a, eta, sigma)).T).T), k2, axis=0)

    def plot(kw, rw, w, fileName):
        from . import plotting
        X_exp, Y_exp = np.meshgrid(kw, rw)
        fig, ax = plotting.createfig()
        C = ax.contourf(X_exp, Y_exp, w.T)
        fig.colorbar(C, ax=ax)
        ax.set_xlabel('k')
        ax.set_ylabel('R')
        plotting.savefig(fileName, fig)

    w = np.abs(W_exp(kw,rw))
    if plotFilename is not None: plot(kw, rw, w, plotFilename)
    return kw, rw, w

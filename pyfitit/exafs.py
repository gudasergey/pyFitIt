import copy, scipy
import numpy as np
from . import utils, larch
from .larch import xafs


class Exafs(utils.Spectrum):
    def __init__(self, k, chi):
        super().__init__(k, chi, xName='k', yName='chi')

    def toXanes(self, Efermi, k_power):
        me = 2 * 9.109e-31  # kg
        h = 1.05457e-34  # J*s
        J = 6.24e18  # 1 J = 6.24e18 eV
        e = np.sign(self.x)*self.x**2 * J / me * h**2 * 1e20 + Efermi
        k = copy.deepcopy(self.x)
        k[k==0] = 1
        return utils.Spectrum(e, self.y/k**k_power + 1)

    def shiftEnergy(self, shift, inplace=False):
        """
        shift - energy shift in eV
        """
        # the result doesn't depend on Efermi
        Efermi = self.x[0]
        xan = self.toXanes(Efermi, k_power=0)
        xan.energy += shift
        res = xanes2exafs(xan, Efermi, k_power=0, preserveGrid=True)
        if inplace: self.x[:], self.y[:] = res.x, res.y
        else: return res

    def smooth(self, SO2, sigmaSquare, inplace=False):
        exp = np.exp(-2 * self.x ** 2 * sigmaSquare) * SO2
        if inplace: self.y *= exp
        else: return Exafs(self.x, self.y*exp)


def xanes2exafs(xanes, Efermi, k_power, preserveGrid=False):
    me = 2 * 9.109e-31  # kg
    h = 1.05457e-34  # J*s
    J = 6.24e18  # 1 J = 6.24e18 eV
    e, s = xanes.x, xanes.y
    i = e >= Efermi
    k = np.sqrt((e[i] - Efermi) / J * me / h ** 2 / 1e20)  # J * kg /  (J*s)^2 = kg / (J * s^2) = kg / ( kg*m^2 ) = 1/m^2 = 1e-20 / A^2
    intSqr = (s[i] - 1) * k ** k_power
    if preserveGrid:
        k1 = np.sign(e-Efermi)*np.sqrt(np.abs(e-Efermi) / J * me / h ** 2 / 1e20)
        intSqr1 = np.zeros(len(e))
        intSqr1[i] = intSqr
    else:
        k1 = np.linspace(k[0], k[-1], int((k[-1]-k[0])*100))
        intSqr1 = np.interp(k1, k, intSqr)
    return Exafs(k1, intSqr1)


def ftwindow(x, xmin=None, xmax=None, dx=1, dx2=None, window='han', **kws):
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
    nfft = 2048 #2^11
    kstep = k[1]-k[0]
    omega = 2*np.pi/(kstep*nfft) * np.arange(int(nfft/2))
    ft = np.fft.fft(chi,nfft)[0:int(nfft/2)] *kstep/np.sqrt(np.pi)
    # ft = scipy.fftpack.fft(chi,nfft)[0:int(nfft/2)] *kstep/np.sqrt(np.pi)
    return omega/2, ft
    # print('kstep =', kstep)
    # ft = scipy.fftpack.fft(chi,nfft)[:int(nfft/2)] *kstep/np.sqrt(np.pi)
    # print('Andrea: out.shape =',ft.shape, ft[100:102])
    # return np.pi/(kstep*nfft)*np.arange(int(nfft/2)), ft


def convert2RSpace(k, chi, kmin=3, kmax=11, dk=1, dk2=None, window='kaiser', kweight=0, rmax_out=10, kstep=0.05):
    if dk2 is None: dk2 = dk
    k_max = max(max(k), kmax+dk2)
    k_   = kstep * np.arange(int(1.01+k_max/kstep), dtype='float64')
    k, chi = k_, np.interp(k_,k,chi)
    if kweight != 0: chi = chi*k**kweight
    r,chi = FT_Transform(k, ftwindow(k, xmin=kmin, xmax=kmax, dx=dk, dx2=dk2, window=window) * chi)
    i = (0<=r) & (r<=rmax_out)
    return r[i], chi[i]


def convert2RSpace_larch(k,chi,kmin=3, kmax=11, dk=1, dk2=None, window='kaiser', kweight=0, rmax_out=10):
    group = larch.Group(name='tmp')
    larch.xafs.xftf(k, chi=chi, group=group, kmin=kmin, kmax=kmax, kweight=kweight, dk=dk, dk2=dk2, with_phase=False, window=window, rmax_out=rmax_out, nfft=2048, kstep=0.05, _larch=None)
    return group.r, group.chir


def plotWavelet(kw, rw, w, fileName, levels=None):
    from . import plotting
    X_exp, Y_exp = np.meshgrid(kw, rw)
    fig, ax = plotting.createfig()
    C = ax.contourf(X_exp, Y_exp, w.T, levels=levels)
    fig.colorbar(C, ax=ax)
    ax.set_xlabel('k')
    ax.set_ylabel('R')
    plotting.savefig(fileName, fig)


def waveletTransform(k, chi, interval=None, sigma=1, eta=5, kmin=2.4, kmax=10.5, Rmin=0.5, Rmax=4.5, knum=200, Rnum=200, plotFilename=None, levels=None):
    """
    doi.org/10.1016/j.radphyschem.2019.05.023
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
    k_redW, chi_redW = k, chi*ftwindow(k, xmin=interval[0], xmax=interval[-1], window='han')
    # range of the apodization window (rectangular)
    # i = (interval[0]<=k) & (k<=interval[1])
    # k_redW, chi_redW = k[i], chi[i]
    kw = np.linspace(*interval, num=knum)  # k-points of the Wavelet
    rw = np.linspace(Rmin, Rmax, num=Rnum)  # R-points of the Wavelet

    def fz(k, eta, sigma):
        I = complex(0, 1)
        return 1/np.sqrt(2*np.pi)/sigma * np.exp(k * eta * I) * np.exp(-k**2 / (2 * sigma**2))

    # interpolation of the experimental signal
    def chi1(omega):
        return np.interp(omega, k_redW, chi_redW)

    # function experimental wavelet (The formula (1) and the Morlet function are used in article)
    def W_exp(omega, r):
        a = eta / (2 * r) # a.shape = Rnum
        k2 = k_redW
        a1 = k2[:, None] - omega[None,:]   # k2[:, None].shape = (len(k), 1); omega.shape = knum; a1.shape = (len(k), knum)
        if 2*Rnum*knum*Rnum < 10e6:
            pod_ind = (chi1(k2) * np.conj(fz(a1[..., None] / a, eta, sigma)).T).T
            integral = np.trapz(pod_ind, k2, axis=0)
        else:
            integral = np.zeros((knum,Rnum), dtype=np.complex64)
            for j in range(Rnum):
                # chi1(k2).shape = len(k)
                # a1.shape = (len(k), knum)
                # np.conj(fz(a1 / a[j], eta, sigma)).shape = (len(k), knum)
                # np.conj(fz(a1 / a[j], eta, sigma)).T.shape = (knum, len(k))
                # (chi1(k2) * np.conj(fz(a1 / a[j], eta, sigma)).T).shape = (knum, len(k))
                # pod_ind = (len(k), knum)
                pod_ind = (chi1(k2)[None,:] * np.conj(fz(a1 / a[j], eta, sigma)).T).T
                integral[:,j] = np.trapz(pod_ind, k2, axis=0) # integral.shape = (knum,Rnum)
        return (1 / np.sqrt(a)) * integral

    w = W_exp(kw,rw)
    if plotFilename is not None: plotWavelet(kw, rw, np.abs(w), plotFilename, levels=levels)
    return kw, rw, w


def waveletTransform_larch(k, chi, interval=None, Rmin=0.5, Rmax=4.5, plotFilename=None, levels=None):
    if interval is None: interval = [k[0], k[-1]]
    group = larch.Group(name='tmp')
    if interval is None: chi1 = chi
    else: chi1 = chi*ftwindow(k, xmin=interval[0], xmax=interval[-1], window='han')
    larch.xafs.cauchy_wavelet(k, chi1, group, kweight=0, rmax_out=Rmax, nfft=2048)
    kw,rw,w = group.wcauchy_k, group.wcauchy_r, group.wcauchy.T
    i = (Rmin<=rw) & (rw<=Rmax)
    rw, w = rw[i], w[:,i]
    i = (interval[0]<=k) & (k<=interval[1])
    kw, w = kw[i], w[i,:]
    if plotFilename is not None: plotWavelet(kw, rw, np.abs(w), plotFilename, levels=levels)
    return kw, rw, w


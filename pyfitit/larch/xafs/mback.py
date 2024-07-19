#!/usr/bin/env python
"""
  XAFS MBACK normalization algorithms.
"""
import copy

from ...utils import importXrayDB
xraydb = importXrayDB()

import numpy as np
from scipy.special import erfc
from time import time
# from xraydb import (xray_edge, xray_line, xray_lines,
                    # f1_chantler, f2_chantler, guess_edge,
                    # atomic_number, atomic_symbol)
from lmfit import Parameter, Parameters, minimize

from ...larch import Group, isgroup, parse_group_args, Make_CallArgs

from ...larch.math import index_of, index_nearest, remove_dups, remove_nans2

from .xafsutils import set_xafsGroup
from .pre_edge import find_e0, preedge, pre_edge


MAXORDER = 6


def find_xray_line(z, edge):
    """
    Finds most intense X-ray emission line energy for a given element and edge.
    """
    intensity = 0
    line      = ''
    for key, value in xraydb.xray_lines(z).items() :
        if value.initial_level == edge.upper():
            if value.intensity > intensity:
                intensity = value.intensity
                line      = key
    return xraydb.xray_line(z, line[:-1])


def linearReg(x,y):
    p = np.polyfit(x, y, deg=1)
    return [p[1],p[0]]


def mback(energy, mu=None, group=None, z=None, edge='K', e0=None, pre1=None, pre2=-50,
          norm1=100, norm2=None, order=3, leexiang=False, tables='chantler', fit_erfc=False, preliminary_coarse_normalization='auto',
          return_f1=False, _larch=None):
    """
    Match mu(E) data for tabulated f''(E) using the MBACK algorithm and,
    optionally, the Lee & Xiang extension

    Arguments
    ----------
      energy:     array of x-ray energies, in eV.
      mu:         array of mu(E).
      group:      output group.
          z:          atomic number of the absorber.
          edge:       x-ray absorption edge (default 'K')
      e0:         edge energy, in eV.  If None, it will be determined here.
      pre1:       low E range (relative to e0) for pre-edge region.
      pre2:       high E range (relative to e0) for pre-edge region.
      norm1:      low E range (relative to e0) for post-edge region.
      norm2:      high E range (relative to e0) for post-edge region.
      order:      order of the legendre polynomial for normalization.
                      (default=3, min=0, max=5).
      leexiang:   boolean (default False)  to use the Lee & Xiang extension.
      tables:     tabulated scattering factors: 'chantler' [deprecated]
      fit_erfc:   boolean (default False) to fit parameters of error function.
      return_f1:  boolean (default False) to include the f1 array in the group.


    Returns
    -------
      None

    The following attributes will be written to the output group:
      group.f2:            tabulated f2(E).
      group.f1:            tabulated f1(E) (if 'return_f1' is True).
      group.fpp:           mback atched spectrum.
          group.edge_step:     edge step of spectrum.
          group.norm:          normalized spectrum.
      group.mback_params:  group of parameters for the minimization.

    Notes:
        Chantler tables is now used, with Cromer-Liberman no longer supported.
    References:
      * MBACK (Weng, Waldo, Penner-Hahn): http://dx.doi.org/10.1086/303711
      * Lee and Xiang: http://dx.doi.org/10.1088/0004-637X/702/2/970
      * Cromer-Liberman: http://dx.doi.org/10.1063/1.1674266
      * Chantler: http://dx.doi.org/10.1063/1.555974
    """
    assert order >= 0
    assert np.all(np.isfinite(mu))
    if preliminary_coarse_normalization != 'auto':
        assert preliminary_coarse_normalization in [True, False]
    order = max(min(order, MAXORDER), 0)

    ### implement the First Argument Group convention
    energy, mu, group = parse_group_args(energy, members=('energy', 'mu'),
                                         defaults=(mu,), group=group,
                                         fcn_name='mback')
    if len(energy.shape) > 1:
        energy = energy.squeeze()
    if len(mu.shape) > 1:
        mu = mu.squeeze()

    if _larch is not None:
        group = set_xafsGroup(group, _larch=_larch)

    energy = remove_dups(energy)
    sigma = 10
    energy_uniform = np.linspace(energy[0]-sigma, energy[-1]+sigma, len(energy)*4)
    gauss_e = np.arange(-5*sigma, 5*sigma, energy_uniform[1]-energy_uniform[0])
    gauss = 1/sigma/np.sqrt(2*np.pi)*np.exp(-gauss_e**2/2/sigma**2)
    if energy.size <= 1:
        raise ValueError("energy array must have at least 2 points")
    if e0 is None or e0 < energy[1] or e0 > energy[-2]:
        e0 = find_e0(energy, mu, group=group)

    ie0 = index_nearest(energy, e0)
    e0 = energy[ie0]

    if pre1 is None:  pre1  = min(energy) - e0
    if norm2 is None: norm2 = max(energy) - e0
    if norm2 < 0:     norm2 = max(energy) - e0 - norm2
    pre1  = max(pre1,  (min(energy) - e0))
    norm2 = min(norm2, (max(energy) - e0))

    if pre1 > pre2:
        pre1, pre2 = pre2, pre1
    if norm1 > norm2:
        norm1, norm2 = norm2, norm1

    p1 = index_of(energy, pre1+e0)
    p2 = index_nearest(energy, pre2+e0)
    n1 = index_nearest(energy, norm1+e0)
    n2 = index_of(energy, norm2+e0)
    if p2 - p1 < 2:
        p2 = min(len(energy), p1 + 2)
    if n2 - n1 < 2:
        p2 = min(len(energy), p1 + 2)

    ## theta is a boolean array indicating the
        ## energy values considered for the fit.
    ## theta=1 for included values, theta=0 for excluded values.
    theta            = np.zeros_like(energy, dtype='int')
    theta[p1:(p2+1)] = 1
    theta[n1:(n2+1)] = 1

    ## weights for the pre- and post-edge regions, as defined in the MBACK paper (?)
    weight            = np.ones_like(energy, dtype=float)
    weight[p1:(p2+1)] = np.sqrt(np.sum(weight[p1:(p2+1)]))
    weight[n1:(n2+1)] = np.sqrt(np.sum(weight[n1:(n2+1)]))
    assert np.all(weight != 0)

    ## get the f'' function from CL or Chantler
    f1 = xraydb.f1_chantler(z, energy)
    f2 = xraydb.f2_chantler(z, energy)
    group.f2 = f2
    if return_f1:
        group.f1 = f1

    WL = energy[np.argmax(np.diff(f2))]
    f2[energy <= WL] = 0
    f2[energy > WL] = 1

    if preliminary_coarse_normalization == True:
        mu = manual_normalizing(energy, mu, [pre1 + e0, pre2 + e0], [norm1 + e0, norm2 + e0], f2)

    # determine initial value for s
    ind = (norm1+e0 <= energy) & (energy <= min(norm2+e0, norm1+e0+200))
    if order >= 1:
        b2, a2 = linearReg(energy[ind], mu[ind])
        assert np.all(np.isfinite([b2,a2]))
        s_initial = (a2 * energy[p2] + b2) - mu[p2]
    else:
        b2 = np.mean(mu[ind])
        s_initial = b2 - mu[p2]
    if s_initial < 0: s_initial = 0

    params0 = Parameters()
    params0.add(name='s',  value=s_initial,  vary=False)  # scale of data
    params0.add(name='cc1', value=0, vary=False)
    params0.add(name='cc2',  value=0, vary=False)
    if fit_erfc:
        params0['cc1'].vary = True
        params0['cc2'].vary = True

    for i in range(order+1):  # polynomial coefficients
        params0.add(name='c%d' % i, value=0, vary=True)
    params0['c0'].value = b2 - s_initial
    if order >= 1: params0['c1'].value = a2

    def build_norm(p):
        eoff = energy - e0
        norm = p['c0']*np.ones(energy.shape)
        for i in range(order):  # successive orders of polynomial
            attr = 'c%d' % (i + 1)
            if attr in p:
                norm += p[attr] * eoff ** (i + 1)
        # correction instead of erfc
        if fit_erfc:
            erfc = p['cc1'] * (energy_uniform-WL) + p['cc2'] * (energy_uniform-WL) ** 2
            erfc[energy_uniform>=WL] = 0
            erfc = np.convolve(erfc, gauss, mode="same")
            norm += np.interp(energy, energy_uniform, erfc)
        return norm

    def match_f2_helper(p):
        func = ( mu - (p['s']*f2 + build_norm(p)) ) * theta / weight
        return func

    def match_f2(p):
        if method == 'leastsq':
            return np.abs(match_f2_helper(p))**0.5
        else:
            return np.sum(np.abs(match_f2_helper(p)))

    def getError(p):
        ind_pre = (pre1+e0<=energy) & (energy<=pre2+e0)
        ind_post = (norm1+e0<=energy) & (energy<=norm2+e0)
        y = match_f2_helper(p)*weight
        y[~(ind_pre | ind_post)] = 0
        return np.max(np.abs(y[ind_pre]))

    method = 'Powell'  # leastsq Powell
    if method == 'leastsq':
        minimize_params = dict(gtol=1.e-5, ftol=1.e-5, xtol=1.e-5, epsfcn=1.e-5)
    else:
        minimize_params = {'options': dict(ftol=1e-5, xtol=1e-5)}
    out = minimize(match_f2, params0, method=method, **minimize_params)
    opars0 = out.params.valuesdict()
    error0 = getError(opars0)
    # print('error0 =', error0)
    params1 = copy.deepcopy(params0)
    for name in params1: params1[name].value = opars0[name]
    params1['s'].vary = True
    out = minimize(match_f2, params1, method=method, **minimize_params)
    opars1 = out.params.valuesdict()
    error1 = getError(opars1)
    # print('error1 =', error1)
    if error1 > error0:
        print(f'MBACK failed to fit s. Rollback to manual s. Error0 = {error0}, error1 = {error1}')
        error1 = error0
        params1 = params0
        opars1 = opars0
    opars = opars1
    max_error = 0.07
    if error1 > max_error and order >= 1 and preliminary_coarse_normalization=='auto':
        print(f'MBACK failed: error = {error1}. Try coarse normalization and then MBACK once more')
        mu1 = mu
        mu = manual_normalizing(energy, mu, [pre1+e0,pre2+e0], [norm1+e0,norm2+e0], f2)
        params2 = copy.deepcopy(params0)
        params2['c0'].value = 0
        params2['c1'].value = 0
        out = minimize(match_f2, params2, method=method, **minimize_params)
        opars2 = out.params.valuesdict()
        error2 = getError(opars2)
        if error2 < error1 and error2 < max_error:
            opars = opars2
            print(f'Coarse normalization succeeded. Old error = {error1}, new error = {error2}')
        else:
            mu = mu1
            print(f'Coarse normalization failed. Old error = {error1}, new error = {error2}')

    norm_function = build_norm(opars)
    group.e0 = e0
    if opars['s'] != 0:
        group.fpp = (mu - norm_function)/opars['s']
    else:
        group.fpp = mu - norm_function
    # calculate edge step and normalization from f2 + norm_function
    pre_f2 = preedge(energy, group.f2+norm_function, e0=e0, pre1=pre1,
                 pre2=pre2, norm1=norm1, norm2=norm2, nnorm=2, nvict=0)
    if opars['s'] != 0:
        group.edge_step = pre_f2['edge_step'] / opars['s']
    else:
        group.edge_step = pre_f2['edge_step']
    group.norm = (opars['s']*mu -  pre_f2['pre_edge']) / pre_f2['edge_step']
    group.mback_details = Group(params=opars, pre_f2=pre_f2,
                                f2_scaled=opars['s']*f2,
                                norm_function=norm_function,
                                target=mu-opars['s']*f2)
    group.loss = np.linalg.norm(match_f2_helper(opars)[theta==1], ord=1)


def manual_normalizing(e, y, pre, post, f2_init):
    i2 = (post[0] <= e) & (e <= post[1])
    b2, a2 = linearReg(e[i2], y[i2])
    y = y - (a2 * e + b2)
    i1 = index_nearest(e, pre[1])
    a1 = y[i1]
    if a1 != 0: y = (y - a1) / np.abs(a1)
    b2, a2 = linearReg(e[i2], f2_init[i2])
    f2 = f2_init - (a2 * e + b2)
    i1 = (pre[0] <= e) & (e <= pre[1])
    a1 = np.min(f2[i1])
    y = y * np.abs(a1) + a1
    y = y + (a2 * e + b2)
    return y


def f2norm(params, en=1, mu=1, f2=1, weights=1):

    """
    Objective function for matching mu(E) data to tabulated f''(E)
    """
    p = params.valuesdict()
    model = (p['offset'] + p['slope']*en + f2) * p['scale']
    return weights * (model - mu)

@Make_CallArgs(["energy","mu"])
def mback_norm(energy, mu=None, group=None, z=None, edge='K', e0=None,
               pre1=None, pre2=None, norm1=None, norm2=None, nnorm=None, nvict=1,
               _larch=None):
    """
    simplified version of MBACK to Match mu(E) data for tabulated f''(E)
    for normalization

    Arguments:
      energy, mu:  arrays of energy and mu(E)
      group:       output group (and input group for e0)
      z:           Z number of absorber
      e0:          edge energy
      pre1:        low E range (relative to E0) for pre-edge fit
      pre2:        high E range (relative to E0) for pre-edge fit
      norm1:       low E range (relative to E0) for post-edge fit
      norm2:       high E range (relative to E0) for post-edge fit
      nnorm:       degree of polynomial (ie, nnorm+1 coefficients will be
                   found) for post-edge normalization curve fit to the
                   scaled f2. Default=1 (linear)

    Returns:
      group.norm_poly:     normalized mu(E) from pre_edge()
      group.norm:          normalized mu(E) from this method
      group.mback_mu:      tabulated f2 scaled and pre_edge added to match mu(E)
      group.mback_params:  Group of parameters for the minimization

    References:
      * MBACK (Weng, Waldo, Penner-Hahn): http://dx.doi.org/10.1086/303711
      * Chantler: http://dx.doi.org/10.1063/1.555974
    """
    ### implement the First Argument Group convention
    energy, mu, group = parse_group_args(energy, members=('energy', 'mu'),
                                         defaults=(mu,), group=group,
                                         fcn_name='mback')
    if len(energy.shape) > 1:
        energy = energy.squeeze()
    if len(mu.shape) > 1:
        mu = mu.squeeze()

    if _larch is not None:
        group = set_xafsGroup(group, _larch=_larch)
    group.norm_poly = group.norm*1.0

    if z is not None:              # need to run find_e0:
        e0_nominal = xraydb.xray_edge(z, edge).energy
    if e0 is None:
        e0 = getattr(group, 'e0', None)
        if e0 is None:
            find_e0(energy, mu, group=group)
            e0 = group.e0

    atsym = None
    if z is None or z < 2:
        atsym, edge = xraydb.guess_edge(group.e0)
        z = xraydb.atomic_number(atsym)
    if atsym is None and z is not None:
        atsym = xraydb.atomic_symbol(z)

    if getattr(group, 'pre_edge_details', None) is None:  # pre_edge never run
        pre_edge(group, pre1=pre1, pre2=pre2, nvict=nvict,
                norm1=norm1, norm2=norm2, e0=e0, nnorm=nnorm)
    if pre1 is None:
        pre1 = group.pre_edge_details.pre1
    if pre2 is None:
        pre2 = group.pre_edge_details.pre2
    if nvict is None:
        nvict = group.pre_edge_details.nvict
    if norm1 is None:
        norm1 = group.pre_edge_details.norm1
    if norm2 is None:
        norm2 = group.pre_edge_details.norm2
    if nnorm is None:
        nnorm = group.pre_edge_details.nnorm

    mu_pre = mu - group.pre_edge
    f2 = xraydb.f2_chantler(z, energy)

    weights = np.ones(len(energy))*1.0

    if norm2 is None:
        norm2 = max(energy) - e0
    if norm2 < 0:
        norm2 = (max(energy) - e0)  - norm2

    # avoid l2 and higher edges
    if edge.lower().startswith('l'):
        if edge.lower() == 'l3':
            e_l2 = xraydb.xray_edge(z, 'L2').energy
            norm2 = min(norm2,  e_l2-e0)
        elif edge.lower() == 'l2':
            e_l1 = xraydb.xray_edge(z, 'L1').energy
            norm2 = min(norm2,  e_l1-e0)

    ipre2 = index_of(energy, e0+pre2)
    inor1 = index_of(energy, e0+norm1)
    inor2 = index_of(energy, e0+norm2) + 1



    weights[ipre2:] = 0.0
    weights[inor1:inor2] = np.linspace(0.1, 1.0, inor2-inor1)

    params = Parameters()
    params.add(name='slope',   value=0.0,    vary=True)
    params.add(name='offset',  value=-f2[0], vary=True)
    params.add(name='scale',   value=f2[-1], vary=True)

    out = minimize(f2norm, params, method='leastsq',
                   gtol=1.e-5, ftol=1.e-5, xtol=1.e-5, epsfcn=1.e-5,
                   kws = dict(en=energy, mu=mu_pre, f2=f2, weights=weights))

    p = out.params.valuesdict()

    model = (p['offset'] + p['slope']*energy + f2) * p['scale']

    group.mback_mu = model + group.pre_edge

    pre_f2 = preedge(energy, model, nnorm=nnorm, nvict=nvict, e0=e0,
                     pre1=pre1, pre2=pre2, norm1=norm1, norm2=norm2)

    step_new = pre_f2['edge_step']

    group.edge_step_poly  = group.edge_step
    group.edge_step_mback = step_new
    group.norm_mback = mu_pre / step_new


    group.mback_params = Group(e0=e0, pre1=pre1, pre2=pre2, norm1=norm1,
                               norm2=norm2, nnorm=nnorm, fit_params=p,
                               fit_weights=weights, model=model, f2=f2,
                               pre_f2=pre_f2, atsym=atsym, edge=edge)

    if (abs(step_new - group.edge_step)/(1.e-13+group.edge_step)) > 0.75:
        print("Warning: mback edge step failed....")
    else:
        group.edge_step = step_new
        group.norm       = group.norm_mback

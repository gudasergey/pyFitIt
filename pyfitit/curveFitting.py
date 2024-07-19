import scipy, scipy.special, copy, warnings, lmfit, os, scipy.interpolate
import numpy as np
from lmfit.models import ExpressionModel, PolynomialModel
from . import utils, molecule, larch, geometry, plotting, smoothLib
xraydb = utils.importXrayDB()
from .larch import xafs
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def linearReg2(y_true, f1, f2, weights=None):
    """
    returns [w1,w2] from the model y = w1*f1 + w2*f2. If de is not None, than model: y(e) = w1*f1(e) + w2*f2(e), and weights = e[1:]-e[:-1], we minimize integral of squared error
    :param y_true:
    :param f1: feature1
    :param f2: feature2
    :param weights: point weights
    :return: [w1,w2] from the model y = w1*f1 + w2*f2
    """
    assert len(y_true) == len(f1) and len(f1) == len(f2)
    if weights is None: weights = np.ones(len(y_true))
    matr_11 = np.sum(f1*f1*weights)
    matr_12 = np.sum(f1*f2*weights)
    matr_21 = matr_12
    matr_22 = np.sum(f2*f2*weights)
    rhs1 = np.sum(y_true*f1*weights)
    rhs2 = np.sum(y_true*f2*weights)
    det = matr_11*matr_22 - matr_21*matr_12
    if det == 0:
        norm_f1 = np.sqrt(np.sum(f1**2*weights))
        norm_f2 = np.sqrt(np.sum(f2**2*weights))
        norm_y = np.sqrt(np.sum(y_true ** 2 * weights))
        if norm_f1 != 0:
            return [norm_y/norm_f1, 0]
        elif norm_f2 != 0:
            return [0, norm_y / norm_f2]
        else: return [0,0]
    return (rhs1*matr_22-rhs2*matr_12)/det, (matr_11*rhs2-matr_21*rhs1)/det


def linearReg(x,y,de=None):
    """
    returns [b,a] from the model y=ax+b. If de is not None, than model: y(e) = a*x(e) + b, and de = e[1:]-e[:-1], we minimize integral of squared error
    :param x:
    :param y:
    :param de: point weights
    :return: [b,a] from the model y=ax+b
    """
    return linearReg2(y, np.ones(len(x)), x, weights=de)


def linearReg_mult_only(x,y,de):
    sumX2 = np.sum(x*x*de)
    sumXY = np.sum(x*y*de)
    if (sumX2 == 0) or np.isnan(sumX2) or np.isnan(sumXY): return 0
    return sumXY/sumX2


def fit_by_regression(exp_e, exp_xanes, fdmnes_xan, fitEnergyInterval, normType='multOnly'):
    assert normType in ['multOnly', 'linearMult', 'mult and add']
    ind = (fitEnergyInterval[0]<=exp_e) & (exp_e<=fitEnergyInterval[1])
    e = exp_e[ind]
    ex = exp_xanes[ind]
    fx = fdmnes_xan[ind]
    mex = (ex[1:]+ex[:-1])/2
    mfx = (fx[1:]+fx[:-1])/2
    de = e[1:]-e[:-1]
    me = (e[1:] + e[:-1]) / 2
    if normType == 'multOnly':
        w = linearReg_mult_only(mfx, mex, de)
        norm = 1/w
        return fdmnes_xan/norm, norm
    elif normType == 'mult and add':
        [b, a] = linearReg(mfx, mex, de)
        return a*fdmnes_xan + b, {'a':a, 'b':b}
    else:
        assert normType == 'linearMult'
        [a, b] = linearReg2(mex, mfx*me, mfx, weights=de)
        return (a*exp_e + b)*fdmnes_xan, {'a':a, 'b':b}


def fit_to_experiment_by_norm_or_regression(exp_e, exp_xanes, fit_interval, fdmnes_en, fdmnes_xan, shift, norm=None, normType='multOnly'):
    fdmnes_en = fdmnes_en + shift
    fdmnes_xan = np.interp(exp_e, fdmnes_en, fdmnes_xan)
    if norm is None:
        fdmnes_xan1, norm = fit_by_regression(exp_e, exp_xanes, fdmnes_xan, fit_interval, normType=normType)
        return fdmnes_xan1, norm
    else:
        if normType == 'multOnly':
            return fdmnes_xan/norm, norm
        elif normType == 'mult and add':
            return fdmnes_xan*norm['a'] + norm['b'], norm
        else:
            assert normType == 'linearMult'
            return fdmnes_xan * (norm['a']*exp_e + norm['b']), norm


def fit_by_polynom_old(e1, xan1, fit_interval):
    e, xan = utils.limit(fit_interval, e1, xan1)

    def model(t, e0, a):
        ind = t < e0
        res = np.zeros(t.shape)
        res[~ind] = a*(t[~ind]-e0)**2
        return 1 + res
    mod = lmfit.Model(model, independent_vars=['t'])
    params = mod.make_params(e0=5200, a=-1e-7)  # - стартовые
    params['e0'].set(min=fit_interval[0], max=fit_interval[1])
    params['a'].set(min=-1e-4, max=1e-4)
    result = mod.fit(xan, params, t=e)
    e0 = result.best_values['e0']
    a = result.best_values['a']
    app = model(e1,e0,a)
    return app


def fit_by_polynom(e1, xan1, fit_interval, deg):
    e, xan = utils.limit(fit_interval, e1, xan1)
    assert len(e)>0, f"All energies is out of fit interval {fit_interval}. Energy = " + str(e1)
    p = np.polyfit(e, xan, deg)
    return np.polyval(p, e1)


def findExpEfermiSimple(exp_e, exp_xanes, search_shift_level):
    ind = np.where(exp_xanes >= search_shift_level)[0][0]
    exp_Efermi_left = exp_e[ind]
    i = ind
    while i + 1 < len(exp_xanes) and exp_xanes[i] <= exp_xanes[i + 1]:
        i += 1
    exp_Efermi_peak = exp_e[i]
    while i + 1 < len(exp_xanes) and exp_xanes[i] >= exp_xanes[i + 1]:
        i += 1
    exp_Efermi_right = exp_e[i]
    return exp_Efermi_left, exp_Efermi_peak, exp_Efermi_right


def findExpEfermi(exp_e, exp_xanes, search_shift_level):
    xz, ind, monot = findZeros(exp_e, exp_xanes - search_shift_level)
    if xz is None: return findExpEfermiSimple(exp_e, exp_xanes, search_shift_level)[0]
    ind_inc = np.where(monot == 1)[0]
    if len(ind_inc) == 0: return findExpEfermiSimple(exp_e, exp_xanes, search_shift_level)[0]
    ind_inc = ind[ind_inc[-1]]
    return exp_e[ind_inc]


def findEfermiFast(e, xanes):
    max_v = np.mean(xanes[-5:])
    min_v = np.min(xanes)
    search_level = (max_v+min_v)/2
    ind = np.where(xanes<=search_level)[0][-1]
    return e[ind]


def findEfermiByArcTan(energy, intensity, maxlevel=None):
    """
    Searches Efermi energy by fitting xanes by arctan.
    :param energy:
    :param intensity:
    :return: best_params = {'a':..., 'x0':...}, arctan_y
    """
    assert len(energy) == len(intensity), f'{len(energy)} != {len(intensity)} ' + str(energy.shape) + ' ' + str(intensity.shape)
    intensity = copy.deepcopy(intensity)
    if maxlevel is not None:
        e_level = findExpEfermi(energy, intensity, maxlevel)
        intensity[energy > e_level] = np.interp([e_level], energy, intensity)[0]
    last = np.max(intensity)
    efermi0 = findExpEfermi(energy, intensity, 0.5*last)
    a0 = np.max((intensity[1:]-intensity[:-1])/(energy[1:]-energy[:-1]))
    mod = ExpressionModel('2*b/(1+exp(-(a/b)*(x - x0)))+c')
    params = mod.make_params(a=a0, x0=efermi0, b=last, c=0)  # - start
    params['a'].set(min=0)
    params['b'].set(min=0)
    result = mod.fit(intensity, params, x=energy)
    # print(result.best_values['x0'])
    return result.best_values, result.best_fit


def findEfermiByArcTan1(energy, intensity, a0):
    """
    Searches Efermi energy by fitting xanes by arctan.
    :param energy:
    :param intensity:
    :return: best_params = {'a':..., 'x0':...}, arctan_y
    """
    assert len(energy) == len(intensity), f'{len(energy)} != {len(intensity)} ' + str(energy.shape) + ' ' + str(intensity.shape)
    intensity = copy.deepcopy(intensity)
    c0 = intensity[0]
    b0 = (intensity[-1]-c0)/2
    mod = ExpressionModel(f'2*b/(1+exp(-(a/b)*(x - x0)))+c')
    params = mod.make_params(a=a0, x0=(energy[0]+energy[-1])/2, b=b0, c=c0)  # - start
    # result = mod.fit(intensity, params, x=energy)
    # return result.best_values, result.best_fit
    return params, mod.eval(params, x=energy)


def subtractLinearBase(x, y, baseFitInterval, debug=False):
    ind = (baseFitInterval[0] <= x) & (x <= baseFitInterval[1])
    xb,yb = x[ind],y[ind]
    yb = (yb - np.min(yb)) / (np.max(yb) - np.min(yb))
    # pre-edge is near the intersection point of the left and right linear envelopes of the spectrum
    left_abc = None
    i1 = 0
    for i2_ in range(len(xb)-1):
        i2 = len(xb)-i2_-1
        a,b,c = geometry.get_line_by_2_points(xb[i1], yb[i1], xb[i2], yb[i2])
        k,b1 = -a/b, -c/b
        y_plus = yb-(k*xb+b1)
        if np.all(y_plus >= -1e-10):
            left_abc = [a,b,c]
            break
    right_abc = None
    i2 = len(xb)-1
    for i1 in range(len(xb)-1):
        a,b,c = geometry.get_line_by_2_points(xb[i1], yb[i1], xb[i2], yb[i2])
        k,b1 = -a/b, -c/b
        y_plus = yb-(k*xb+b1)
        if np.all(y_plus >= -1e-10):
            right_abc = [a,b,c]
            break
    assert left_abc is not None
    assert right_abc is not None
    corner_x, corner_y = geometry.get_line_intersection(*left_abc, *right_abc)

    # all the significant intervals of spectrum decrease should be inside pre-edge region
    decr_intervals = []
    for i in range(len(xb)-1):
        if yb[i+1] <= yb[i]:
            last = decr_intervals[-1] if len(decr_intervals)>0 else [-100,-100]
            if  last[1] == i: last[1] = i+1
            else: decr_intervals.append([i,i+1])
    # print('decr_intervals =', decr_intervals)
    if len(decr_intervals) > 0:
        decr_intervals_vals = np.array([yb[interval[0]]-yb[interval[1]] for interval in decr_intervals])
        assert np.all(decr_intervals_vals>=0)
        decr_intervals_y = np.array([yb[interval[0]] for interval in decr_intervals])
        # print('decr_intervals_vals =', decr_intervals_vals)
        big_ind = np.where( (decr_intervals_vals >= np.max(decr_intervals_vals)*0.3) & (decr_intervals_y >= np.max(yb)*0.03) )[0]
        if len(big_ind) == 0: decr_intervals_union = [1e15,-1e15]
        else:
            decr_intervals = [decr_intervals[i] for i in big_ind]
            # print('big decr_intervals =', decr_intervals)
            decr_intervals_union = [decr_intervals[0][0], decr_intervals[-1][1]]
            # print('decr_intervals_union =', decr_intervals_union)
    else: decr_intervals_union = [1e15,-1e15]
    if debug:
        if decr_intervals_union[1]>decr_intervals_union[0]:
            print('decr_intervals_union =', xb[decr_intervals_union[0]], xb[decr_intervals_union[1]])
        else: print('Spectrum has no decrease intervals')

    n = len(xb)
    # in descending order of priority
    s_all_decr = np.zeros((n,n)) - 1e10 # contains the significant intervals of spectrum decrease
    s_pos_small_y_corner = np.zeros((n, n)) - 1e10 # positive pre-edge, small y, near the corner
    s_pos_small_y = np.zeros((n, n)) - 1e10 # positive pre-edge, small y
    s_pos = np.zeros((n,n)) - 1e10 # positive pre-edge
    # we iterate through all pairs of points, connect them with a line and see if the spectrum is located in the interval between the points above the line
    for i1 in range(n-1):
        for i2 in range(i1+1, n):
            b,k = geometry.get_line_by_2_points_kb(xb[i1], yb[i1], xb[i2], yb[i2])
            y_plus = yb[i1:i2+1] - (k*xb[i1:i2+1]+b)
            integral = utils.integral(xb[i1:i2+1], y_plus)
            if i1<=decr_intervals_union[0] and decr_intervals_union[1]<=i2:
                s_all_decr[i1,i2] = integral
            if len(y_plus)>0 and np.all(y_plus>=-1e-10):
                s_pos[i1,i2] = integral
                if yb[i2] < 0.5:
                    s_pos_small_y[i1,i2] = s_pos[i1,i2]
                    if abs(corner_x-xb[i1]) <= 20:
                        s_pos_small_y_corner[i1,i2] = s_pos[i1,i2]
    def chooseMax(s):
        i1, i2 = np.unravel_index(s.argmax(), s.shape)
        if i1 == i2: return None, None
        else: return i1,i2

    i1, i2 = chooseMax(s_all_decr)
    # print('chooseMax =', i1,i2, s_all_decr[i1,i2])
    # for iii in range(i1,i2): print(s_all_decr[iii,i2], iii)
    if i1 is None:
        if debug: print('subtractLinearBase failed to find peaks containing major decrease intervals')
        i1, i2 = chooseMax(s_pos_small_y_corner)
    if i1 is None:
        if debug: print('subtractLinearBase failed to find close to edge peaks')
        i1, i2 = chooseMax(s_pos_small_y)
    if i1 is None:
        if debug: print('subtractLinearBase failed to find close to edge peaks and low peaks')
        i1, i2 = chooseMax(s_pos)
    if i1 is None:
        if debug: print('subtractLinearBase failed to find any peaks')
        i1,i2 = 0,n-1
    a,b,c = geometry.get_line_by_2_points(x[i1], y[i1], x[i2], y[i2])
    k, b = -a / b, -c / b
    x_peak = xb[i1:i2+1]
    y_peak = yb[i1:i2+1]
    return x_peak, y_peak - (k * x_peak + b), [k, b]


def subtractBase_old(x, y, peakInterval, baseFitInterval, model='arctan', weights=None, slopeCorrection=None, extrapolate=None, useStartParams=None, plotFileName=None, sepFolders=False):
    """
    Fit base by arctan function and subtract from y.
    :param x: argument
    :param y: function values
    :param peakInterval: interval of peak search (do not included in base fitting)
    :param baseFitInterval: interval of base fit. It should include peakInterval
    :param model: 'cauchy' or 'bezier' or 'arctan' or 'rbf
    :param weights: dict(left, middle, right, slope) - use to fit base, middle of the base if fitted to the left edge of the peak. Default: dict(left=1, middle=0, right=1, slope=1)
    :param slopeCorrection: addition or subtraction from the derivative of the base in the right peak edge
    :param extrapolate: {'left':percent_dx_left, 'right':percent_dx_right}
    :param useStartParams: start param dict for optimization
    :param plotFileName: filename to save debug plot
    :param sepFolders: save graphs in different folders
    :return: dict with keys: 'peak' - spectrum of peak with subtracted base (on interval peakInterval); 'peak full' - the same on interval baseFitInterval, 'base' - spectrum of base on peakInterval; 'base full' - the same on baseFitInterval, 'part' - initial spectrum on peakInterval, 'part full' - the same on baseFitInterval; 'info' - optimization info
    """
    assert model in ['cauchy', 'bezier', 'arctan', 'rbf', 'triangle']
    assert len(x) == len(y)
    assert utils.intervalIntersection(x[[0,-1]], peakInterval) is not None
    assert utils.intervalIntersection(x[[0,-1]], baseFitInterval) is not None
    assert peakInterval[0] < peakInterval[1]
    assert baseFitInterval[0] < baseFitInterval[1]
    assert baseFitInterval[0] < peakInterval[0]
    assert peakInterval[1] < baseFitInterval[1]
    defaultWeights = dict(left=1, middle=0, right=1, slope=1)
    if weights is None: weights = defaultWeights
    assert set(weights.keys()) <= {'left','middle','right','slope'}, str(set(weights.keys()))
    for n in ['left','middle','right','slope']:
        if n not in weights: weights[n] = defaultWeights[n]
    spectrum = utils.Spectrum(x,y)
    if extrapolate is None: extrapolate = {}
    ind_peak = (x >= peakInterval[0]) & (x <= peakInterval[1])
    ind_full = (x >= baseFitInterval[0]) & (x <= baseFitInterval[1])
    ind_fit = ind_full & ~ind_peak
    ind_fit_left = (baseFitInterval[0] <= x) & (x < peakInterval[0])
    ind_fit_right = (peakInterval[1] < x) & (x <= baseFitInterval[1])
    assert np.sum(ind_fit_left)>=1, f'Too few points in the left part of the baseFitInterval {baseFitInterval}. peakInterval={peakInterval}'
    assert np.sum(ind_fit_right)>=1, f'Too few points in the right part of the baseFitInterval {baseFitInterval}. peakInterval = {peakInterval}'
    x_peak = x[ind_peak]; y_peak = y[ind_peak]
    x_fit = x[ind_fit]; y_fit = y[ind_fit]
    x_full = x[ind_full]
    y_full = y[ind_full]
    x_fit_left = x[ind_fit_left]; y_fit_left = y[ind_fit_left]
    x_fit_right = x[ind_fit_right]; y_fit_right = y[ind_fit_right]

    # make x_fit y_fit by extrapolating base inside peak interval (linear extrapolation from both ends)
    x1 = x_fit[0]; x2 = x_fit[-1]
    if 'left' in extrapolate:
        n = np.where(x_fit <= x1+(x2-x1)/10)[0][-1] + 1
        if n<2: n = 2
        slope, intercept,_,_,_ = scipy.stats.linregress(x_fit[:n], y_fit[:n])
        percent = extrapolate['left']; count = np.round(len(x_fit)*percent);
        first = x1 - (x2-x1)*percent; last = x1-(x2-x1)/count
        new_x = np.linspace(first, last, count)
        x_fit = np.insert(x_fit,0,new_x)
        y_fit = np.insert(y_fit,0,new_x*slope+intercept)
    if 'right' in extrapolate:
        n = np.where(x_fit >= x2-(x2-x1)/10)[0][-1] + 1
        if n<2: n = 2
        slope, intercept,_,_,_ = scipy.stats.linregress(x_fit[-n:], y_fit[-n:])
        percent = extrapolate['right']; count = np.round(len(x_fit)*percent);
        last = x2 + (x2-x1)*percent; first = x2+(x2-x1)/count
        new_x = np.linspace(first, last, count)
        x_fit = np.append(x_fit,new_x)
        y_fit = np.append(y_fit,new_x*slope+intercept)
    assert (len(x_peak) >= 2) and (len(x_fit) >= 2), 'len(x_peak) = '+str(len(x_peak))+' len(x_fit) = '+str(len(x_fit))

    maxy = np.max(y)
    b_left, a_left = linearReg(x_fit_left, y_fit_left)
    modelFunc = lambda x, a, b, c, x0, d: 2*b/(1+np.exp(-(a/b)*(x - x0)))+c+d*(x-x_fit[0]) + a_left*x+b_left
    mod = ExpressionModel(f'2*b/(1+exp(-(a/b)*(x - x0)))+c+d*(x-({x_fit[0]})) + ({a_left})*x+({b_left})')

    # prepare func for findEfermiByArcTan
    y1 = copy.deepcopy(y)
    y1[ind_fit_left] = y[ind_fit_left]-(a_left*x[ind_fit_left]+b_left)
    x1,y1 = utils.expandByReflection(x[ind_full], y1[ind_full], side='right', reflType='odd', stableReflMeanCount=1)

    max_dy = np.max((y1[1:]-y1[:-1])/(x1[1:]-x1[:-1]))
    efp, findEfermiByArcTan_res = findEfermiByArcTan1(x1, y1, a0=max_dy)
    findEfermiByArcTan_res = utils.Spectrum(x1, findEfermiByArcTan_res)
    plotting.plotToFile(x1,y1,'expand', findEfermiByArcTan_res.x, findEfermiByArcTan_res.y,'atan', fileName='debug')
    if efp['x0'] < x_peak[0]: efp['x0'] = x_peak[0]
    a0 = efp['a'].value
    b0 = efp['b'].value
    c0 = efp['c'].value
    x00 = efp['x0'].value
    d0 = 0
    params = mod.make_params(a=a0, b=b0, c=c0, x0=x00, d=d0)
    plotting.plotToFile(x[ind_full],y[ind_full],'xanes', x[ind_full], mod.eval(params, x=x[ind_full]),'model', fileName='debug2')
    param_order = {'a':0, 'b':1, 'c':2, 'x0':3, 'd':4}
    heuristicStart = [params['a'].value, params['b'].value, params['c'].value, params['x0'].value, params['d'].value]
    assert modelFunc(*([x_peak[0]]+heuristicStart)) == mod.eval(params,x=x_peak[0]), f'{modelFunc(*([x_peak[0]] + heuristicStart))} != {mod.eval(params, x=x_peak[0])}'
    assert np.all(x[1:]-x[:-1] > 0), str(x)
    params['a'].set(min=0); params['a'].set(max=max_dy*2)
    params['b'].set(min=0)
    params['x0'].set(min=x_peak[0])
    a_linreg = linearReg(x_fit_left, y_fit_left)[1]
    params['d'].set(min=0); params['d'].set(max=3*abs(a_linreg))
    left_dy = np.abs((y_peak[0]-y_fit[0])/(x_peak[0]-x_fit[0]))
    # do not expand x0 bound - optimize can't find minimum
    bounds = [[0,a0*100], [0,maxy*10], [-maxy,maxy], [x_fit_right[0],max(x_fit_right[0]+2*(x00-x_fit_right[0]), x_fit_right[-1])], [0, 3*left_dy]]
    for i in range(len(bounds)):
        if heuristicStart[i] < bounds[i][0]: heuristicStart[i] = bounds[i][0]
        if heuristicStart[i] > bounds[i][1]: heuristicStart[i] = bounds[i][1]
    scale_x = x_fit_right[-1] - x_fit_left[0]
    scale_y = np.max(y_fit_right) - np.min(y_fit_left)
    def getDerivative(modelFunc, params):
        assert len(x_fit_right) > 0
        xfr = x_fit_right
        if len(xfr) == 1:
            xfr = [x_fit_right[0], x_fit_right[0] + 0.1]
        y_app_right = modelFunc(np.array(xfr), *params)
        approxDerivative = (y_app_right[-1] - y_app_right[0]) / (xfr[-1] - xfr[0])
        return approxDerivative

    def getTargetFunc(modelFunc, derivative=None, debug=False):
        def func(params):
            y_app_left = modelFunc(x_fit_left, *params)
            y_app_right = modelFunc(x_fit_right, *params)
            # norm to equalize visual areas
            left = utils.integral(x_fit_left/scale_x, np.abs(y_app_left - y_fit_left)/scale_y)
            right = utils.integral(x_fit_right/scale_x, np.abs(y_app_right - y_fit_right)/scale_y)
            middle = 0
            if weights['middle'] != 0:
                y_app_peak = modelFunc(x_peak, *params)
                y_sub = y_peak-y_app_peak
                y_sub[y_sub<0] = 0
                if np.sum(y_sub) > 0:
                    middle_x = np.sum(x_peak*y_sub)/np.sum(y_sub)
                    assert x_peak[0] <= middle_x <= x_peak[-1], f'{x_peak[0]} {middle_x} {x_peak[-1]}'
                    middle_y = y_peak[0]
                    middle_app = np.interp(middle_x, x_peak, y_app_peak)
                    middle = (middle_app - middle_y)/scale_y
                    if middle<0: middle = 0
            # print(left, middle, right)
            # try to optimize max(left,right)
            if derivative is None or weights['slope'] == 0: derivError = 0
            else:
                approxDerivative = getDerivative(modelFunc, params)
                derivError = abs(approxDerivative - derivative)/(a0*b0)

            if debug: print(left, middle, right, derivError)
            # print(weights)
            edges = np.linalg.norm([left*weights['left'], right*weights['right']], ord=10)
            return np.linalg.norm([edges, middle*weights['middle'], derivError*weights['slope']], ord=2)
            # return max(left, right)
        return func

    def optimize(func, start, **params):
        res = scipy.optimize.minimize(func, start, **params)
        x = res.x if res.success else start
        return dict(optimParam=x, optimVal=func(x), optimResult=res)

    def fixParam(names, values, func, start, bounds):
        ind_exclude = [param_order[name] for name in names]
        ind_keep = [i for i in range(len(x)) if i not in ind_exclude]
        squeeze = lambda x: [x[i] for i in range(len(x)) if i in ind_keep]
        def expand(x1):
            x = [None]*len(param_order)
            for i,v in zip(ind_keep,x1): x[i] = v
            for i,v in zip(ind_exclude,values):
                assert x[i] is None
                x[i] = v
            return x
        func1 = lambda params1: func(expand(params1))
        return func1, squeeze(start), squeeze(bounds), expand
    
    def buildResult(info):
        info = copy.deepcopy(info)
        approx_peak = modelFunc(x_peak, *info['optimParam'])
        approx_full = modelFunc(x_full, *info['optimParam'])
        y_sub = y_peak-approx_peak
        y_sub_full = y_full - approx_full
        info['peakInterval'] = peakInterval
        info['baseFitInterval'] = baseFitInterval
        info['spectrum'] = spectrum
        info['findEfermiByArcTan'] = findEfermiByArcTan_res
        return {'peak':utils.Spectrum(x_peak, y_sub), 'peak full':utils.Spectrum(x_full, y_sub_full),
                'base': utils.Spectrum(x_peak, approx_peak), 'base full':utils.Spectrum(x_full, approx_full),
                'part':utils.Spectrum(x_peak, y_peak), 'part full':utils.Spectrum(x_full, y_full),
                'info':info}

    debug = True
    if debug: print('bounds:',bounds)
    start = heuristicStart if useStartParams is None else useStartParams
    info = {'optimParam':start, 'optimVal':getTargetFunc(modelFunc)(start)}
    if debug: print('start: ', info['optimParam'], 'F =', info['optimVal'])
    result0_start = buildResult(info)
    plotting.plotToFile(x[ind_full],y[ind_full],'xanes', x[ind_full], result0_start['base full'].y,'base', fileName='debug3')

    func = getTargetFunc(modelFunc)

    method = 'Powell'  # None 'Powell' 'trust-constr'
    if useStartParams is None:
        # model(a,b,c,x0,d): 2*b/(1+np.exp(-(a/b)*(x - x0)))+c+d*(x-x_fit[0]) + a_left*x+b_left
        func1, start1, bounds1, expander = fixParam(['a','b','d'], [a0,b0,0], func, info['optimParam'], bounds)
        info1 = optimize(func1, start1, bounds=bounds1, method=method)
        info1['optimParam'] = expander(info1['optimParam'])
        result1_fix_abd = buildResult(info1)
        if debug: print('fix_abd: ', info1['optimParam'], 'F =', info1['optimVal'])
        if info1['optimVal'] < info['optimVal']: info = info1

        func1, start1, bounds1, expander = fixParam(['b','d'], [b0,0], func, info['optimParam'], bounds)
        info1 = optimize(func1, start1, bounds=bounds1, method=method)
        info1['optimParam'] = expander(info1['optimParam'])
        result2_fix_bd = buildResult(info1)
        if debug: print('fix_bd: ', info1['optimParam'], 'F =', info1['optimVal'])
        if info1['optimVal'] < info['optimVal']: info = info1

        func1, start1, bounds1, expander = fixParam(['d'], [0], func, info['optimParam'], bounds)
        info1 = optimize(func1, start1, bounds=bounds1, method=method)
        info1['optimParam'] = expander(info1['optimParam'])
        result2_fix_d = buildResult(info1)
        if debug: print('fix_d: ', info1['optimParam'], 'F =', info1['optimVal'])
        if info1['optimVal'] < info['optimVal']: info = info1

    info1 = optimize(func, info['optimParam'], bounds=bounds, method=method)
    result3_wo_slopeCorr = buildResult(info1)
    if debug: print('wo_slopeCorr: ', info1['optimParam'], 'F =', info1['optimVal'])
    if info1['optimVal'] < info['optimVal']: info = info1

    if slopeCorrection is not None:
        info = optimize(getTargetFunc(modelFunc, derivative=getDerivative(modelFunc,info['optimParam'])*slopeCorrection), info['optimParam'], bounds=bounds, method=method)
        info['optimVal'] = func(info['optimParam'])

    approx_peak = modelFunc(x_peak, *info['optimParam'])
    approx_full = modelFunc(x_full, *info['optimParam'])
    y_sub = y_peak-approx_peak
    y_sub_full = y_full - approx_full
    info['peakInterval'] = peakInterval
    info['baseFitInterval'] = baseFitInterval
    info['spectrum'] = spectrum
    info['findEfermiByArcTan'] = findEfermiByArcTan_res
    info['result0_start'] = result0_start
    if useStartParams is None:
        info['result1_fix_abd'] = result1_fix_abd
        info['result2_fix_bd'] = result2_fix_bd
        info['result2_fix_d'] = result2_fix_d
    info['result3_wo_slopeCorr'] = result3_wo_slopeCorr
    result = {'peak':utils.Spectrum(x_peak, y_sub), 'peak full':utils.Spectrum(x_full, y_sub_full),
            'base': utils.Spectrum(x_peak, approx_peak), 'base full':utils.Spectrum(x_full, approx_full),
            'part':utils.Spectrum(x_peak, y_peak), 'part full':utils.Spectrum(x_full, y_full),
            'info':info}
    if plotFileName is not None:
        plotPreedge(result, plotFileName, sepFolders)
    return result


def plotPreedge(result, plotFileName, sepFolders=False, only='all'):
    assert only in ['base', 'full', 'sub', 'all']
    def safeGetSpectrum(result, name):
        if name in result['info']:
            s = result['info'][name]
            if isinstance(s, utils.Spectrum):
                return s.x,s.y,name
            else:
                assert isinstance(s, dict)
                assert 'peak' in s
                s = s['base full']
                if name.startswith('result'): name = name[6:]
                return s.x,s.y,name
        else: return tuple()
    x, y = result['info']['spectrum'].toTuple()
    x_full, approx_full = result['base full'].toTuple()
    x_peak, y_peak = result['part'].toTuple()
    y_sub = result['peak'].y
    y_sub_full = result['peak full'].y
    info = result['info']
    arctan = info['findEfermiByArcTan'] if findEfermiByArcTan in info else None
    if only != 'all': filename = plotFileName
    else:
        folder = os.path.split(plotFileName)[0]
        fn = os.path.split(plotFileName)[-1]
        name, ext = os.path.splitext(fn)
    if only == 'all':
        filename = f'{folder}/base/{fn}' if sepFolders else f'{folder}/{name}-base{ext}'
    if only in ['all', 'base']:
        toPlot = ([x_full[0], x_full[-1]], [0, 0], 'zero', x_full, y_sub_full, 'sub full', x, y, 'init', x_full, approx_full, 'base', x_peak, y_peak, {'label': 'peak', 'lw': 3}, x_peak, y_sub, {'label':'sub','lw':3})
        for name in ['pseudo_approx_full', 'Cauchy_fit', 'Cauchy_sub']:
            toPlot += safeGetSpectrum(result, name)
        plotting.plotToFile(*toPlot, fileName=filename, xlim=[x_full[0], x_full[-1]])
        toPlot = (x, y, 'init')
        if arctan is not None: toPlot += (*arctan.toTuple(), {'label':'findEfermiByArcTan', 'alpha':0.5})
        for name in ['result0_start', 'result1_fix_abd', 'result2_fix_bd', 'result2_fix_d', 'result3_wo_slopeCorr']:
            toPlot += safeGetSpectrum(result, name)
        if 'result0_start' in info: plotting.plotToFile(*toPlot, fileName=os.path.splitext(filename)[0]+'_SBres.png', xlim=[x_full[0], x_full[-1]])
    if only == 'all':
        filename = f'{folder}/full/{fn}' if sepFolders else f'{folder}/{name}-full{ext}'
    if only in ['all', 'full']:
        plotting.plotToFile(x, y, 'init', x_full, approx_full, 'base', x_peak, y_peak, 'peak', fileName=filename)
        toPlot = (x, y, 'init')
        if arctan is not None: toPlot += (*arctan.toTuple(), {'label':'findEfermiByArcTan', 'alpha':0.5})
        for name in ['result0_start', 'result1_fix_abd', 'result2_fix_bd', 'result2_fix_d', 'result3_wo_slopeCorr']:
            toPlot += safeGetSpectrum(result, name)
        if 'result0_start' in info: plotting.plotToFile(*toPlot, fileName=os.path.splitext(filename)[0]+'_SBres.png')
    if only == 'all':
        filename = f'{folder}/sub/{fn}' if sepFolders else f'{folder}/{name}-sub{ext}'
    if only in ['all', 'sub']:
        plotting.plotToFile(x_full, y_sub_full, 'sub full', x_peak, y_sub, 'sub', fileName=filename)


def findZeros(x, y):
    b = (y>0).astype(int)
    dy = b[1:]-b[:-1]
    ind = np.where(dy != 0)[0]
    # print('x[ind] =', x[ind], x[ind[-1]], x[ind[-1]+1], y[ind[-1]], y[ind[-1]+1] )
    if len(ind) == 0: return None, None, None
    xz = np.zeros(len(ind))
    monot = dy[ind]
    for ii,i in enumerate(ind):
        a, b, c = geometry.get_line_by_2_points(x[i], y[i], x[i + 1], y[i + 1])
        xz[ii], _ = geometry.get_line_intersection(a, b, c, 0, 1, 0)
    return xz, ind, monot


def buildPreedgeResult(peakInterval, baseFitInterval, x, y, approx_full):
    """
    :param approx_full: monotone lower envelope of the spectrum on the baseFitInterval
    """
    assert peakInterval[0] <= peakInterval[1], f'peakInterval = {peakInterval} baseFitInterval = {baseFitInterval}'
    assert baseFitInterval[0] <= baseFitInterval[1], f'peakInterval = {peakInterval} baseFitInterval = {baseFitInterval}'
    assert baseFitInterval[0] <= peakInterval[0], f'peakInterval = {peakInterval} baseFitInterval = {baseFitInterval}'
    assert baseFitInterval[1] >= peakInterval[1], f'peakInterval = {peakInterval} baseFitInterval = {baseFitInterval}'
    ind_full = (x >= baseFitInterval[0]) & (x <= baseFitInterval[1])
    assert np.sum(ind_full) == len(approx_full), f'{np.sum(ind_full)} != {len(approx_full)}'
    x_full, y_full = x[ind_full], y[ind_full]
    ind_peak = (x >= peakInterval[0]) & (x <= peakInterval[1])
    x_peak, y_peak = x[ind_peak], y[ind_peak]
    ind_peak_in_full = (x_full >= peakInterval[0]) & (x_full <= peakInterval[1])
    approx_peak = approx_full[ind_peak_in_full]
    y_sub = y_peak-approx_peak
    y_sub_full = y_full - approx_full

    ind_fit = ind_full & ~ind_peak
    ind_fit_left = (baseFitInterval[0] <= x) & (x < peakInterval[0])
    ind_fit_right = (peakInterval[1] < x) & (x <= baseFitInterval[1])

    info = dict(peakInterval=peakInterval, baseFitInterval=baseFitInterval, spectrum=utils.Spectrum(x,y), ind_full=ind_full, ind_peak=ind_peak, ind_peak_in_full=ind_peak_in_full, ind_fit=ind_fit, ind_fit_left=ind_fit_left, ind_fit_right=ind_fit_right)
    return {'peak':utils.Spectrum(x_peak, y_sub), 'peak full':utils.Spectrum(x_full, y_sub_full),
                'base': utils.Spectrum(x_peak, approx_peak), 'base full':utils.Spectrum(x_full, approx_full),
                'part':utils.Spectrum(x_peak, y_peak), 'part full':utils.Spectrum(x_full, y_full),
                'info':info}


def plotPreedgeResult(result, plotFileName, postfix, final):
    folder = os.path.split(plotFileName)[0]
    fn = os.path.split(plotFileName)[-1]
    name, ext = os.path.splitext(fn)
    def subfolder_filename(subfold, postfix):
        return f'{folder}/{subfold}/{name}{postfix}{ext}'
    plotPreedge(result, subfolder_filename('base',postfix), sepFolders=True, only='base')
    if final:
        plotPreedge(result, subfolder_filename('base_final',''), sepFolders=True, only='base')
        plotPreedge(result, subfolder_filename('full_final',''), sepFolders=True, only='full')


def fitCauchy(x_left, y_left, center_x, center_y, x_peak0):
    # print(x_left)
    # print(y_left)
    # print(center_x, center_y, x_peak0)
    y_left = np.copy(y_left)
    if x_left[-1] < center_x:
        x_left = np.append(x_left, center_x)
        y_left = np.append(y_left, center_y)
    y_left[y_left < 0] = 0
    area = 2*utils.integral(x_left, y_left)
    if area == 0: return 0,1
    x,y = x_left, y_left/area
    i1, i2 = x<x_peak0, x>=x_peak0
    w1, w2 = 1,1
    def optimFunc(sigma):
        diff = np.abs(smoothLib.kernelCauchy(x-center_x,0,sigma) - y)**2
        return utils.integral(x[i1], diff[i1]) * w1 + utils.integral(x[i2], diff[i2]) * w2
    sigmaGauss = np.sqrt(utils.integral(x, (x-center_x)**2*y))
    # print('sigmaGauss =', sigmaGauss)
    # print('center_x =', center_x, 'sigmaGauss =', sigmaGauss, 'width/2 =', center_x-x_peak0)
    res = scipy.optimize.minimize_scalar(optimFunc, bounds=(sigmaGauss*0.1, sigmaGauss*10))
    sigma = res.x
    # print('best sigma =', sigma, 'area =', area)
    return area, sigma


def fitPreedgeByCauchy(result, stepName, plotFileName, debug):
    result = copy.deepcopy(result)
    peakInterval, baseFitInterval = result['info']['peakInterval'], result['info']['baseFitInterval']
    x_peak, y_sub = result['peak'].toTuple()
    x_full, y_full = result['part full'].toTuple()
    y_sub = np.copy(y_sub)
    y_sub[y_sub<0] = 0
    if np.all(y_sub == 0): return np.zeros(len(x_full))
    area = utils.integral(x_peak, y_sub)
    center = utils.integral(x_peak, y_sub * x_peak) / area
    assert peakInterval[0] <= center <= peakInterval[1]
    sp = result['info']['spectrum']
    i2 = x_full <= center
    # print(x_full[i2], y_full[i2], center, peakInterval)
    cauchy_area, cauchy_sigma = fitCauchy(x_full[i2], y_full[i2], center, sp.val(center), peakInterval[0])
    cauchy_full = smoothLib.kernelCauchy(x_full, center, cauchy_sigma) * cauchy_area
    result['info']['Cauchy_fit'] = utils.Spectrum(x_full, result['base full'].y+cauchy_full)
    result['info']['Cauchy_sub'] = utils.Spectrum(x_full, cauchy_full)
    if 'pseudo_approx_full' in result['info']: del result['info']['pseudo_approx_full']
    plotPreedgeResult(result, plotFileName, f'_{stepName}_0', final=False)
    return cauchy_full


def refineSubtractBaseResultHelper(result, cauchy_sub, stepName, plotFileName, final, debug):

    def fix_negative_sub(x_peak,y_peak,b,k,x0,y0):
        """
        If line kx+b goes somewhere upper y_peak, we replace it by lower envelope line goes through (x0,y0).
        :returns: new_b,new_k
        """
        assert np.all(x_peak >= x0) or np.all(x_peak <= x0)
        if np.all(y_peak >= k*x_peak+b): return b,k
        tang = (y_peak-y0) / (x_peak-x0)
        if np.all(x0 <= x_peak):  new_k = np.min(tang)
        else: new_k = np.max(tang)
        new_b = y0-new_k*x0
        return new_b, new_k


    def fix_negative_approx(x_approx,y_approx,b,k,x0,y0):
        """
        If line kx+b goes everywhere upper y_approx, we replace it by lower line goes through (x0,y0).
        :returns: new_b,new_k
        """
        assert np.all(x_approx >= x0) or np.all(x_approx <= x0)
        if np.any(y_approx > k*x_approx+b): return b,k
        tang = (y_approx-y0) / (x_approx-x0)
        ind = np.argsort(tang)
        if np.all(x0 <= x_approx):
            if len(tang) >= 2:
                i = ind[-2]
                new_k = tang[i]
            else: new_k = tang[0]*0.9
        else:
            if len(tang) >= 2:
                i = ind[1]
                new_k = tang[i]
            else: new_k = tang[0]*1.1
        new_b = y0-new_k*x0
        return new_b, new_k

    def smooth_corner(x0, y0, cx0, cy0):
        # print(x0,cx0)
        scale_x = x0[-1] - x0[0]
        scale_y = y0[-1] - y0[0]
        dx, dy = scale_x/10, scale_y/5
        # scale
        x,y = (x0-cx0)/dx, (y0-cy0)/dy
        cx, cy = 0, 0
        assert np.any(x<-1), f'Decrease left edge of the base fit interval or increase the left edge of the peak fit interval'
        assert np.any(y>+1), f'Increase right edge of the base fit interval or decrease the right edge of the peak fit interval'
        lx, ly = x[x< -1][-1], y[x< -1][-1]
        # dist to corner must be equal, or we can't find correct circle
        dc = np.sqrt((lx-cx)**2 + (ly-cy)**2)
        v = np.array([ x[y>+1][0], y[y>+1][0] ])
        v = geometry.normalize(v)
        rx, ry = cx+v[0]*dc, cy+v[1]*dc
        # print('l =', lx, ly)
        # print('r =', rx, ry)

        # find the circle, which touches the corner lines in the points (lx,ly) (rx,ry)
        a1,b1,c1 = geometry.get_line_by_2_points(cx, cy, lx, ly)
        a2,b2,c2 = geometry.get_line_by_2_points(cx, cy, rx, ry)
        orth1 = (b1, -a1, -b1*lx+a1*ly)
        orth2 = (b2, -a2, -b2*rx+a2*ry)
        # print(orth1, orth2)
        circ_x,circ_y = geometry.get_line_intersection(*orth1, *orth2)
        # print('circ =', circ_x, circ_y)
        # print('dist to line:', geometry.get_dist_to_line(a1,b1,c1,circ_x,circ_y), geometry.get_dist_to_line(a2,b2,c2,circ_x,circ_y))
        r = np.sqrt((lx-circ_x)**2 + (ly-circ_y)**2)
        r_check = np.sqrt((rx-circ_x)**2 + (ry-circ_y)**2)
        assert np.abs(r-r_check)/r < 1e-6, f'{r} != {r_check}'

        i = (lx<=x) & (x<=rx)
        new_y = np.copy(y)
        new_y[i] = circ_y - np.sqrt( np.abs(r**2 - (x[i]-circ_x)**2) )
        return new_y*dy+cy0

    x_full, y_full = result['part full'].toTuple()
    x_peak, y_peak = result['part'].toTuple()
    peakInterval, baseFitInterval = result['info']['peakInterval'], result['info']['baseFitInterval']
    pseudo_approx_full = y_full - cauchy_sub
    pseudo_approx_full_spectrum = utils.Spectrum(x_full, pseudo_approx_full)
    ind_fit_left_in_full = x_full < peakInterval[0]
    ind_fit_right_in_full = x_full > peakInterval[1]
    # fit linear model y=kx+b on the left and right of the base. Returns [b,k]
    # linear base is the most stable. Moreover - due to Cauchy subtraction we do not think on choosing intervals for linear fit! It is the best algorithm, I've done!
    approx_left_params = linearReg(x_full[ind_fit_left_in_full], pseudo_approx_full[ind_fit_left_in_full])
    approx_right_params = linearReg(x_full[ind_fit_right_in_full], pseudo_approx_full[ind_fit_right_in_full])

    if approx_left_params[1] < 0: approx_left_params = [np.mean(pseudo_approx_full[ind_fit_left_in_full]), 0]

    approx_left_params1 = fix_negative_sub(x_peak,y_peak,*approx_left_params, np.mean(x_full[ind_fit_left_in_full]), np.mean(pseudo_approx_full[ind_fit_left_in_full]))
    if debug and approx_left_params1 != approx_left_params:
        print('Left part linear fit failed to make y_sub positive')
    approx_left_params = approx_left_params1

    approx_left_params1 = fix_negative_approx(x_full[ind_fit_right_in_full], pseudo_approx_full[ind_fit_right_in_full],*approx_left_params, np.mean(x_full[ind_fit_left_in_full]), np.mean(pseudo_approx_full[ind_fit_left_in_full]))
    if debug and approx_left_params1 != approx_left_params:
        print('Left part linear fit failed to make right part of the pseudo_approx_full positive')
    approx_left_params = approx_left_params1

    approx_right_params1 = fix_negative_sub(x_peak,y_peak,*approx_right_params, np.mean(x_full[ind_fit_right_in_full]), np.mean(pseudo_approx_full[ind_fit_right_in_full]))
    if debug and approx_right_params1 != approx_right_params:
        print('Right part linear fit failed to make y_sub positive')
    approx_right_params = approx_right_params1

    approx_right_params1 = fix_negative_approx(x_full[ind_fit_left_in_full], pseudo_approx_full[ind_fit_left_in_full],*approx_right_params, np.mean(x_full[ind_fit_right_in_full]), np.mean(pseudo_approx_full[ind_fit_right_in_full]))
    if debug and approx_right_params1 != approx_right_params:
        print('Right part linear fit failed to make left part of the pseudo_approx_full positive')
    approx_right_params = approx_right_params1

    corner_x, corner_y = geometry.get_line_intersection_kb(*approx_left_params, *approx_right_params)
    new_approx_full = approx_left_params[1]*x_full + approx_left_params[0]
    i = x_full >= corner_x
    new_approx_full[i] = approx_right_params[1]*x_full[i] + approx_right_params[0]
    ind_peak_in_full = result['info']['ind_peak_in_full']
    y_sub_not_smoothed = y_full[ind_peak_in_full] - new_approx_full[ind_peak_in_full]
    if not np.all(y_sub_not_smoothed<=0):
        new_approx_full = smooth_corner(x_full, new_approx_full, corner_x, corner_y)
    new_result = buildPreedgeResult(peakInterval, baseFitInterval, *result['info']['spectrum'].toTuple(), new_approx_full)
    new_result['info']['pseudo_approx_full'] = utils.Spectrum(x_full,pseudo_approx_full)
    plotPreedgeResult(new_result, plotFileName, f'_{stepName}_1', final=final)
    return new_result


def refineSubtractBaseResult(result, stepName, plotFileName, final, debug):
    cauchy_sub = fitPreedgeByCauchy(result, stepName, plotFileName, debug=debug)
    return refineSubtractBaseResultHelper(result, cauchy_sub=cauchy_sub, stepName=stepName, plotFileName=plotFileName, final=final, debug=debug)


def subtractBase(x, y, peakInterval, baseFitInterval, plotFileName=None, debug=False):
    """
    Subtract base from y
    :param x: argument
    :param y: function values
    :param peakInterval: interval of peak search (do not included in base fitting)
    :param baseFitInterval: interval of base fit. It should include peakInterval
    :param plotFileName: filename to save debug plot
    :return: dict with keys: 'peak' - spectrum of peak with subtracted base (on interval peakInterval); 'peak full' - the same on interval baseFitInterval, 'base' - spectrum of base on peakInterval; 'base full' - the same on baseFitInterval, 'part' - initial spectrum on peakInterval, 'part full' - the same on baseFitInterval; 'info' - optimization info
    """
    assert len(x) == len(y)
    assert utils.intervalIntersection(x[[0,-1]], peakInterval) is not None
    assert utils.intervalIntersection(x[[0,-1]], baseFitInterval) is not None
    assert peakInterval[0] < peakInterval[1]
    assert baseFitInterval[0] < baseFitInterval[1]
    assert baseFitInterval[0] < peakInterval[0]
    assert peakInterval[1] < baseFitInterval[1]
    ind_fit_left = (baseFitInterval[0] <= x) & (x < peakInterval[0])
    ind_fit_right = (peakInterval[1] < x) & (x <= baseFitInterval[1])
    assert np.sum(ind_fit_left)>=1, f'Too few points in the left part of the baseFitInterval {baseFitInterval}. peakInterval={peakInterval}'
    assert np.sum(ind_fit_right)>=1, f'Too few points in the right part of the baseFitInterval {baseFitInterval}. peakInterval = {peakInterval}'
    result = buildPreedgeResult(peakInterval, baseFitInterval, x, y, approx_full=y[(baseFitInterval[0] <= x) & (x <= baseFitInterval[1])])
    result = refineSubtractBaseResultHelper(result, np.zeros(len(result['peak full'].x)), '1_init', plotFileName, final=False, debug=debug)
    result = refineSubtractBaseResult(result, stepName='2_refine', plotFileName=plotFileName, final=True, debug=debug)
    return result


def subtractBaseAuto(x, y, fitBaseInterval=None, edgeLevel=None, plotFileName=None, debug=False):
    """
    :param x: spectrum argument values
    :param y: spectrum intensity values
    :param fitBaseInterval: [x1,x2] - interval for pre-edge base fit (should be wider than pre-edge region)
    :param edgeLevel: in (0,1), relative level to search peak (right min of the peak should be < edgeLevel*y[-1]). fitBaseInterval is chosen left to the edgeLevel energy point
    :param plotFileName: file name for plot save
    :param debug: enable debug output
    """
    edgeLevel1 = 0.5 if edgeLevel is None else edgeLevel
    sp = utils.Spectrum(x,y)

    # Step 1: find edge
    # 'b/(1+exp(-a*(x - x0))) + c'
    params, arctan_y = findEfermiByArcTan(x, y)
    mid = params['c'] + params['b'] * edgeLevel1
    rightBaseBorder = x[-1]
    if edgeLevel is not None:
        rightBaseBorder = sp.inverse(params['b'] * edgeLevel, select='max')
    if fitBaseInterval is not None:
        rightBaseBorder = fitBaseInterval[-1]
    xz, ind, monot = findZeros(x, y-mid)
    assert xz is not None, 'findEfermiByArcTan fails - no intersection with mean line'
    ind_inc = np.where(monot==1)[0]
    assert len(ind_inc)>0, 'No intersection with mean line with func increase?'
    ind_inc = ind[ind_inc[-1]]

    # Step 2: linear peak search
    baseFitInterval = [x[0], x[ind_inc]] if fitBaseInterval is None else fitBaseInterval
    baseFitInterval[1] = min(baseFitInterval[1], rightBaseBorder)
    if debug: print('before linear peak search baseFitInterval =', baseFitInterval)
    x_peak, y_sub, ab = subtractLinearBase(x, y, baseFitInterval, debug=debug)
    peakInterval = [x_peak[0], x_peak[-1]]
    assert np.sum((peakInterval[0]<=x) & (x<=peakInterval[1])) == len(x_peak)
    assert np.all(x[(peakInterval[0]<=x) & (x<=peakInterval[1])] == x_peak)
    if debug: print('subtractLinearBase gives peakInterval =', peakInterval)
    approx_full = copy.deepcopy(y)
    approx_full[(peakInterval[0]<=x) & (x<=peakInterval[1])] = ab[0]*x_peak + ab[1]
    result = buildPreedgeResult(peakInterval, baseFitInterval, x, y, approx_full[(baseFitInterval[0]<=x) & (x<=baseFitInterval[1])])
    plotPreedgeResult(result, plotFileName, '_0_linear', final=False)

    # Step 3: interval correction: pre-edge should occupy 1/4 of the baseFitInterval width and 1/2 of the xanes height on baseFitInterval
    def intervalCorrection(peakInterval, baseFitInterval):
        peakInterval = copy.deepcopy(peakInterval)
        baseFitInterval = copy.deepcopy(baseFitInterval)
        min_y_preedge = np.min(y[(peakInterval[0]<=x) & (x<=peakInterval[1])])
        max_y_preedge = np.max(y[(peakInterval[0]<=x) & (x<=peakInterval[1])])
        # print('max_y_preedge =', max_y_preedge)
        ymax_baseFitInterval = max_y_preedge + (max_y_preedge-min_y_preedge)
        # print('ymax_baseFitInterval0 =', ymax_baseFitInterval)
        ymax_baseFitInterval0 = ymax_baseFitInterval
        my = np.min(y)
        ymax_baseFitInterval = np.min([ymax_baseFitInterval, (np.max(y)-my)*0.7+my, (y[-1]-my)*0.7+my])
        # print('before if: ymax_baseFitInterval =', ymax_baseFitInterval)
        if ymax_baseFitInterval != ymax_baseFitInterval0:
            tmp = (sp.val(peakInterval[1]) + ymax_baseFitInterval) / 2
            ymax_baseFitInterval = max(ymax_baseFitInterval, tmp)
        # print('after if: ymax_baseFitInterval =', ymax_baseFitInterval)
        xmax_baseFitInterval =x[(y>=ymax_baseFitInterval) & (x>=peakInterval[1])][0]
        # print('xmax_baseFitInterval =', xmax_baseFitInterval)
        preedge_width = peakInterval[1] - peakInterval[0]
        xmin_baseFitInterval = peakInterval[0] - 2*preedge_width
        xmin_baseFitInterval = max(xmin_baseFitInterval, x[0])
        if np.sum((baseFitInterval[0]<=x) & (x<=peakInterval[0])) < 2:
            peakInterval[0] = x[x>=peakInterval[0]][2]
        baseFitInterval = [xmin_baseFitInterval, xmax_baseFitInterval]
        if debug: print('After interval relative size correction: baseFitInterval =', baseFitInterval, 'peakInterval =', peakInterval)
        return peakInterval, baseFitInterval
    peakInterval, baseFitInterval = intervalCorrection(peakInterval, baseFitInterval)
    approx_full = approx_full[(baseFitInterval[0]<=x) & (x<=baseFitInterval[1])]
    result = buildPreedgeResult(peakInterval, baseFitInterval, x, y, approx_full)
    result = refineSubtractBaseResult(result, stepName='2_refine1', plotFileName=plotFileName, final=False, debug=debug)
    result = refineSubtractBaseResult(result, stepName='3_refine2', plotFileName=plotFileName, final=True, debug=debug)
    # expand peakInterval if peak edges are too high
    def expandPeakInterval(result):
        M = np.max(result['peak'].y)
        pf = result['peak full']
        i = np.where(pf.x == result['peak'].x[-1])[0][0]
        while pf.y[i] >= M*0.05:
            i = i+1
            if i>=len(pf.y):
                i = len(pf.y)-1
                break
        right = pf.x[i]
        i = np.where(pf.x == result['peak'].x[0])[0][0]
        while pf.y[i] >= M*0.05:
            i = i-1
            if i<0:
                i = 0
                break
        left = pf.x[i]
        return [left, right]
    peakInterval = expandPeakInterval(result)
    if peakInterval != result['info']['peakInterval']:
        if debug: print('After expandPeakInterval: peakInterval =', peakInterval)
        approx_full = copy.deepcopy(y)
        approx_full[result['info']['ind_full']] = result['base full'].y
        peakInterval, baseFitInterval = intervalCorrection(peakInterval, baseFitInterval)
        approx_full = approx_full[(baseFitInterval[0]<=x) & (x<=baseFitInterval[1])]
        result = buildPreedgeResult(peakInterval, baseFitInterval, x, y, approx_full)
        result = refineSubtractBaseResult(result, stepName='4_refine1', plotFileName=plotFileName, final=False, debug=debug)
        result = refineSubtractBaseResult(result, stepName='5_refine2', plotFileName=plotFileName, final=True, debug=debug)
    return result


def subtractBaseAuto_old(x, y, fitBaseInterval=None, edgeLevel=None, weights=None, slopeCorrection=None, plotFileName=None, sepFolders=False, debug=False):
    """
    :param fitBaseInterval: [x1,x2] - interval for base
    :param edgeLevel: in (0,1), relative level to search peak (right min of the peak should be < edgeLevel*y[-1]). baseInterval is chosen left to the edgeLevel energy point
    :param weights: dict(left, middle, right, slope) - use to fit base, middle of the base if fitted to the left edge of the peak. Default: dict(left=1, middle=0, right=1, slope=1)
    :param slopeCorrection: addition or subtraction from the derivative of the base in the right peak edge
    """
    edgeLevel1 = 0.5 if edgeLevel is None else edgeLevel
    sp = utils.Spectrum(x,y)
    refinePeakLevel = 0.5
    refinePeakLevel2 = 0.2
    model = 'arctan'
    if plotFileName is not None:
        folder = os.path.split(plotFileName)[0]
        fn = os.path.split(plotFileName)[-1]
        name, ext = os.path.splitext(fn)
        filename = lambda postfix: f'{folder}/{name}_{postfix}{ext}'
        def subfolder_filename(subfold, postfix):
            if postfix != '': postfix = '_'+postfix
            return f'{folder}/{subfold}/{name}{postfix}{ext}' if sepFolders else filename(postfix)
        basefn = lambda postfix: subfolder_filename('base', postfix) if sepFolders else filename(postfix)
    else:
        filename = lambda postfix: None
        basefn = filename

    # Step 1: find edge
    # 'b/(1+exp(-a*(x - x0))) + c'
    params, arctan_y = findEfermiByArcTan(x, y)
    mid = params['c'] + params['b'] * edgeLevel1
    rightBaseBorder = x[-1]
    if edgeLevel is not None:
        rightBaseBorder = sp.inverse(params['b'] * edgeLevel, select='max')
    if fitBaseInterval is not None:
        rightBaseBorder = fitBaseInterval[-1]
    xz, ind, monot = findZeros(x, y-mid)
    assert xz is not None, 'findEfermiByArcTan fails - no intersection with mean line'
    ind_inc = np.where(monot==1)[0]
    assert len(ind_inc)>0, 'No intersection with mean line with func increase?'
    ind_inc = ind[ind_inc[-1]]

    # Step 2: linear peak search
    baseFitInterval = [x[0], x[ind_inc]] if fitBaseInterval is None else fitBaseInterval
    baseFitInterval[1] = min(baseFitInterval[1], rightBaseBorder)
    if debug: print('before linear peak search baseFitInterval =', baseFitInterval)
    x_peak, y_sub, ab = subtractLinearBase(x, y, baseFitInterval, debug=debug)
    newPeakInterval = [x_peak[0], x_peak[-1]]
    if debug: print('subtractLinearBase3 gives peakInterval =', newPeakInterval)

    # корректирум интервалы: предкрай должен занимать 1/4 экрана по высоте и 1/4 по ширине
    preedgeHeight = np.max(y_sub)
    min_y = np.min(sp.val(newPeakInterval))
    ymax_baseFitInterval = min_y + 3*preedgeHeight
    ymax_baseFitInterval = min(ymax_baseFitInterval, np.max(y))
    xmax_baseFitInterval = sp.inverse(ymax_baseFitInterval, select='min')
    preedge_width = newPeakInterval[1] - newPeakInterval[0]
    xmin_baseFitInterval = newPeakInterval[0] - 2*preedge_width
    baseFitInterval = [xmin_baseFitInterval, xmax_baseFitInterval]
    if debug: print('After interval relative size correction: baseFitInterval =', baseFitInterval)

    y1 = sp.val(newPeakInterval[1])
    if mid - y1 < y1 - y[0]:
        # pre-edge looks like shoulder
        new_x1 = sp.limit([newPeakInterval[1], x[-1]]).inverse((y1 + params['c']+params['b']) / 2, select='min')
        if new_x1 is None: new_x1 = newPeakInterval[1] + (newPeakInterval[1]-newPeakInterval[0])
        baseFitInterval[1] = new_x1
        if debug: print('Pre-edge looks like shoulder. Corrected baseFitInterval =', baseFitInterval)
    dy = params['b']*params['a']/2  # model derivative at x=x0: diff (b/(1+exp(-a*(x - x0))) + c)
    dx = (sp.val(baseFitInterval[1]) - sp.val(baseFitInterval[0])) / dy
    # print('dx =', dx)
    # correct too large left part
    if dy>0 and dx < (baseFitInterval[1] - baseFitInterval[0])/20:
        k = dx / ((baseFitInterval[1] - baseFitInterval[0])/20)
        # print('k =', k)
        baseFitInterval[0] = baseFitInterval[1] - (baseFitInterval[1] - baseFitInterval[0])*k
        # print(baseFitInterval[0])
        if newPeakInterval[0] - baseFitInterval[0] < baseFitInterval[1] - newPeakInterval[1]:
            baseFitInterval[0] = newPeakInterval[0] - (baseFitInterval[1] - newPeakInterval[1])
        # if baseFitInterval[1] - baseFitInterval[0] < 10: baseFitInterval[0] = baseFitInterval[1]-10
        if debug: print('Too large left part. Corrected baseFitInterval =', baseFitInterval)
    # correct too small left part (it results in wrong inclination of the base)
    if baseFitInterval[1]-baseFitInterval[0] < 30:
        baseFitInterval[0] = baseFitInterval[1]-30
        if debug: print('Too small left part (it results in wrong inclination of the base). Corrected baseFitInterval =', baseFitInterval)
    baseFitInterval_tmp = copy.deepcopy(baseFitInterval)
    if baseFitInterval[0] < x[0]: baseFitInterval[0] = x[0]
    baseFitInterval[1] = min(baseFitInterval[1], rightBaseBorder)
    if debug and baseFitInterval_tmp!=baseFitInterval: print('Correct baseFitInterval to match spectrum energy borders:', baseFitInterval)
    newPeakInterval_tmp = copy.deepcopy(newPeakInterval)
    if np.sum((newPeakInterval[1] < x) & (x < baseFitInterval[1])) <= 1:
        newPeakInterval[1] = baseFitInterval[1]-1
    if np.sum((baseFitInterval[0] < x) & (x < newPeakInterval[0])) <= 1:
        newPeakInterval[0] = baseFitInterval[0]+1
    if debug and newPeakInterval_tmp!=newPeakInterval: print('Corrected newPeakInterval to match baseFitInterval:', newPeakInterval)
    if plotFileName is not None:
        a,b = ab
        plotting.plotToFile(x, y, 'init', x_peak, y_sub, 'y_sub', x_peak,a*x_peak+b,'ax+b', fileName=basefn('step2_linear'), xlim=baseFitInterval)

    # Step 3: peak search
    result = subtractBase(x, y, newPeakInterval, baseFitInterval, model=model, weights=weights, slopeCorrection=slopeCorrection, extrapolate=None, useStartParams=None, plotFileName=filename('step3_subtractBase'), sepFolders=sepFolders)
    exit(0)
    def takeSignificantPart(y, level):
        res = copy.deepcopy(y)
        res[y < level] = level
        return res

    def calcErrorByXandY(x, exact_y, approx_y, corner):
        # calc error (by y and x simultaneously)
        assert len(x) == len(exact_y) and len(x) == len(approx_y)
        sp = utils.Spectrum(x, exact_y)
        scale_x = (x[-1] - x[0])/2  # /2 because width of graph is usually twice larger then height
        err = exact_y - approx_y
        scale_y = np.max(err)
        for i in range(len(exact_y)):
            # if x[i] < corner[0]: continue
            x_inv = sp.inverse(approx_y[i], select='max')
            if x_inv is None: continue
            inv_err = abs(x[i] - x_inv) / scale_x * scale_y
            if np.abs(inv_err) < err[i]: err[i] = inv_err
        return err

    def getCorner(x_full, approx_full, peakInterval, baseFitInterval):
        i = (peakInterval[1]<x_full) & (x_full<baseFitInterval[1])
        b2,a2 = linearReg(x_full[i], approx_full[i])
        i = (baseFitInterval[0]<x_full) & (x_full<peakInterval[0])
        b1,a1 = linearReg(x_full[i], approx_full[i])
        xc,yc = geometry.get_line_intersection(a1,-1,b1, a2,-1,b2)
        return xc,yc

    def getErrCorner(subtractBaseResult):
        x_full, y_full = subtractBaseResult['part full'].toTuple()
        y_sub_full = subtractBaseResult['peak full'].y
        approx_full = subtractBaseResult['base full'].y
        scale_x = (x_full[-1] - x_full[0])/2  # /2 because width of graph is usually twice larger then height
        scale_y = np.max(y_sub_full)
        corner = getCorner(x_full, approx_full, subtractBaseResult['info']['peakInterval'], subtractBaseResult['info']['baseFitInterval'])
        if debug: print('corner =',corner)
        distToCorner = lambda x,y: np.sqrt( ((x-corner[0])/scale_x)**2 + ((y-corner[1])/scale_y)**2 )
        err = calcErrorByXandY(x_full, y_full, approx_full, corner)
        dist = distToCorner(x_full, y_full)
        errCorner = err * (1 - dist / np.max(dist))
        return errCorner, x_full, y_full, y_sub_full, approx_full, err

    errCorner, x_full, y_full, y_sub_full, approx_full, err = getErrCorner(result)
    MerrCorner = np.max(errCorner)
    errCornerM10 = takeSignificantPart(errCorner, MerrCorner * 0.1)

    # Step 4: peak interval refinement
    def refineIntervals(preedgeInterval0, iter):
        preedgeInterval = copy.deepcopy(preedgeInterval0)
        def finalCorrection(preedgeInterval):
            preedgeInterval = copy.deepcopy(preedgeInterval)
            ind = (baseFitInterval[0] <= x_full) & (x_full < preedgeInterval[0])
            if np.sum(ind) <= 1: preedgeInterval[0] = x_full[baseFitInterval[0] <= x_full][2]
            ind = (preedgeInterval[1] < x_full) & (x_full <= baseFitInterval[1])
            if np.sum(ind) <= 1: preedgeInterval[1] = x_full[x_full <= baseFitInterval[1]][-3]
            return preedgeInterval

        def findPeaks(level):
            xz, ind, monot = findZeros(x_full, errCorner - level)
            # print('xz =', xz)
            if xz is None: return preedgeInterval
            # find high peaks
            peaks = []
            for i in range(len(xz)-1):
                if not (monot[i] == +1 and monot[i+1] == -1): continue
                height = np.max(errCorner[ind[i]:ind[i+1]])
                width = xz[i+1]-xz[i]
                area = height*width
                im1 = utils.findNextMinimum(errCornerM10, ind[i], direction=-1)
                im2 = utils.findNextMinimum(errCornerM10, ind[i+1], direction=+1)
                peaks.append({'height':height, 'width':width, 'area':area, 'pos':(xz[i+1]+xz[i])/2, 'region':[x_full[im1], x_full[im2]]})
            return peaks
        peaks = findPeaks(MerrCorner*refinePeakLevel)
        intersect = lambda s,d: not (s[1]<d[0] or d[1]<s[0])
        preedge_intersect_peaks_ind = [i for i,p in enumerate(peaks) if intersect(p['region'], preedgeInterval0)]
        if len(preedge_intersect_peaks_ind) == 0: return finalCorrection(preedgeInterval)
        i_best = np.argmax([peaks[i]['area'] for i in preedge_intersect_peaks_ind])
        i_best = preedge_intersect_peaks_ind[i_best]
        preedgePeaksInd = [i_best]
        largestPeakWidth = peaks[i_best]['width']

        # add neighbour high peaks
        def addPeak(direction, preedgePeaksInd):
            newPreedgePeaksInd = preedgePeaksInd
            borderPeakInd = preedgePeaksInd[-1] if direction == +1 else preedgePeaksInd[0]
            nextPeakInd = borderPeakInd+direction
            if nextPeakInd<0 or nextPeakInd>len(peaks)-1: return newPreedgePeaksInd
            dir01 = (direction+1)//2
            if np.abs(peaks[nextPeakInd]['region'][1-dir01] - peaks[borderPeakInd]['region'][dir01]) < largestPeakWidth:
                if direction == +1: newPreedgePeaksInd = preedgePeaksInd + [nextPeakInd]
                else: newPreedgePeaksInd = [nextPeakInd] + preedgePeaksInd
            # print(direction, preedgePeaksInd, newPreedgePeaksInd)
            # print(np.abs(peaks[nextPeakInd]['region'][1-dir01] - peaks[borderPeakInd]['region'][dir01]), largestPeakWidth)
            return newPreedgePeaksInd

        def addMany(preedgePeaksInd):
            while True:
                newPreedgePeaksInd = addPeak(+1, preedgePeaksInd)
                if len(newPreedgePeaksInd) == len(preedgePeaksInd): break
                preedgePeaksInd = newPreedgePeaksInd
            while True:
                newPreedgePeaksInd = addPeak(-1, preedgePeaksInd)
                if len(newPreedgePeaksInd) == len(preedgePeaksInd): break
                preedgePeaksInd = newPreedgePeaksInd
            return preedgePeaksInd

        preedgePeaksInd = addMany(preedgePeaksInd)
        preedgeInterval = [peaks[preedgePeaksInd[0]]['region'][0], peaks[preedgePeaksInd[-1]]['region'][1]]
        if debug: print('preedgeInterval after large peaks analysis:', preedgeInterval)

        # estimate noise level
        ind = (x_full < preedgeInterval[0]) & (preedgeInterval[1] < x_full)
        noise = errCorner[ind]
        noiseLevel = np.median(np.abs(noise))

        newLevel = max(MerrCorner*refinePeakLevel2, noiseLevel*3)
        if debug: print('old level =', MerrCorner*refinePeakLevel, ' new level =', newLevel)
        if newLevel < MerrCorner*refinePeakLevel:
            peaks = findPeaks(max(MerrCorner*refinePeakLevel2, noiseLevel*3))
            if len(peaks) == 0: return finalCorrection(preedgeInterval)
            intersectPreedge = lambda peak: np.any(preedgeInterval[0] <= peak['region']) and np.any(peak['region']<=preedgeInterval[1])
            preedgePeaksInd = [i for i in range(len(peaks)) if intersectPreedge(peaks[i])]
            if len(preedgePeaksInd) == 0: return finalCorrection(preedgeInterval)
            preedgePeaksInd = list(range(preedgePeaksInd[0], preedgePeaksInd[-1]+1))
            preedgePeaksInd = addMany(preedgePeaksInd)
            preedgeInterval = [peaks[preedgePeaksInd[0]]['region'][0], peaks[preedgePeaksInd[-1]]['region'][1]]
            if debug: print('preedgeInterval after all peaks analysis:', preedgeInterval)
        preedgeInterval = finalCorrection(preedgeInterval)

        if plotFileName is not None:
            getIntervalPlot = lambda d, name: ([d[0], d[0], d[1], d[1]], [y_full[-1], 0, 0, y_full[-1]], name)
            plotting.plotToFile([x_full[0],x_full[-1]],[0,0],'zero', x, y, 'init', x_full, approx_full, 'base', x_full, y_sub_full, 'y_sub_full', x_full, y_full, 'base', x_full, errCorner, 'errCorner', x_full, err, 'err', *getIntervalPlot(preedgeInterval0, 'old interval'), *getIntervalPlot(preedgeInterval, 'new interval'), fileName=basefn(f'step4_refine{iter}'), xlim=[x_full[0], x_full[-1]],)
        return preedgeInterval

    newPeakInterval2 = refineIntervals(newPeakInterval, 1)
    if debug: print('refineIntervals1 =', newPeakInterval2)
    newPeakInterval3 = refineIntervals(newPeakInterval2, 2)
    if debug: print('refineIntervals2 =', newPeakInterval3)
    newPeakInterval = newPeakInterval3

    # Step 5: peak search after refinement
    result = subtractBase(x, y, newPeakInterval, baseFitInterval, model=model, weights=weights, slopeCorrection=slopeCorrection, extrapolate=None, useStartParams=None, plotFileName=filename('step5_subtractBase'), sepFolders=sepFolders)

    # Step 6: enlarge the peak
    def fixNegative(result):
        x_full = result['peak full'].x
        y_sub = result['peak'].y
        y_sub_full = result['peak full'].y
        peakInterval = result['info']['peakInterval']
        ind = y_sub < 0
        ind_full = (y_sub_full < 0) & (x_full>=peakInterval[0]) & (x_full<=peakInterval[1])
        y_sub[ind] = 0
        y_sub_full[ind_full] = 0

    def enlargePositivePeak(result, peakInterval, tryCount):
        peakInterval0 = peakInterval
        peakInterval = copy.deepcopy(peakInterval)

        def getErrCornerExtra(result):
            errCorner, x_full, y_full, y_sub_full, approx_full, _ = getErrCorner(result)
            MerrCorner = np.max(errCorner)
            errCornerM20 = takeSignificantPart(errCorner, MerrCorner*0.05)
            errCorner0 = takeSignificantPart(errCorner, 0)
            i1_0 = utils.findNearest(x_full, peakInterval[0], returnInd=True)
            i2_0 = utils.findNearest(x_full, peakInterval[1], returnInd=True)
            pew = utils.integral(result['peak'].x, result['peak'].y) / (np.max(result['peak'].y)/2)
            return errCorner, errCorner0, errCornerM20, MerrCorner, i1_0, i2_0, pew, x_full, y_full, y_sub_full, approx_full

        def check(new_i, old_i, y):
            if abs(x_full[new_i] - x_full[old_i]) < pew/2 and (y[old_i] - y[new_i]) > MerrCorner * 0.05:
                if new_i >= 2 and new_i <= len(y)-3: return new_i
            return old_i

        errCorner, errCorner0, errCornerM20, MerrCorner, i1_0, i2_0, pew, x_full, y_full, y_sub_full, approx_full = getErrCornerExtra(result)
        i1_M20 = check(utils.findNextMinimum(errCornerM20, i1_0, direction=-1), i1_0, errCorner)
        i2_M20 = check(utils.findNextMinimum(errCornerM20, i2_0, direction=+1), i2_0, errCorner)
        i1_M0 = check(utils.findNextMinimum(errCorner0, i1_M20, direction=-1), i1_M20, errCorner)
        i2_M0 = check(utils.findNextMinimum(errCorner0, i2_M20, direction=+1), i2_M20, errCorner)
        # check very negative minimums
        i1_min = check(utils.findNextMinimum(y_sub_full, i1_M0, direction=-1), i1_M0, y_sub_full)
        i2_min = check(utils.findNextMinimum(y_sub_full, i2_M0, direction=+1), i2_M0, y_sub_full)
        peakInterval = [x_full[i1_min], x_full[i2_min]]

        def plotResult(result):
            if plotFileName is None: return
            plotPreedge(result, subfolder_filename('base', ''), sepFolders=sepFolders, only='base')
            plotPreedge(result, subfolder_filename('base_final', ''), sepFolders=sepFolders, only='base')
            plotPreedge(result, subfolder_filename('full_final', ''), sepFolders=sepFolders, only='full')

        # if negative pit is too large, run subtractBase with positive constrains once more
        if min(errCorner[i1_min], errCorner[i2_min]) < -MerrCorner*0.1 and peakInterval0 != peakInterval:
            if debug:
                print('Negative edge values detected. Run subtractBase once more. peakInterval =', peakInterval)
            oldTol = result['info']['optimVal']
            result1 = subtractBase(x, y, peakInterval, baseFitInterval, model=model, weights=weights, slopeCorrection=slopeCorrection, extrapolate=None, useStartParams=None, plotFileName=filename(f'step6_enlarge{tryCount+1}'), sepFolders=sepFolders)
            if debug: print(f'step6_enlarge{tryCount+1}: oldTol =', oldTol, 'new =', result1['info']['optimVal'])
            if result1['info']['optimVal'] > oldTol:
                result1 = subtractBase(x, y, peakInterval, baseFitInterval, model=model, weights=weights, slopeCorrection=slopeCorrection, extrapolate=None, useStartParams=result['info']['optimParam'], plotFileName=filename(f'step6_pos{tryCount+1}sp'), sepFolders=sepFolders)
                if debug: print(f'step6_enlarge{tryCount+1} with startParams: oldTol =', oldTol, 'new =', result1['info']['optimVal'])
                if result1['info']['optimVal'] <= oldTol:
                    result = result1
                    result = enlargePositivePeak(result, peakInterval, tryCount+1)
            else:
                result = result1
                result = enlargePositivePeak(result, peakInterval, tryCount + 1)

        errCorner, errCorner0, errCornerM20, MerrCorner, i1_0, i2_0, pew, x_full, y_full, y_sub_full, approx_full = getErrCornerExtra(result)
        # bad results for zeolite Ti TP0469
        # if errCorner[i1_min] < 0: i1_min = check(utils.findNextMinimum(errCorner0, i1_0, direction=-1), i1_0, errCorner)
        # if errCorner[i2_min] < 0: i2_min = check(utils.findNextMinimum(errCorner0, i2_0, direction=+1), i2_0, errCorner)
        peakInterval = [x_full[i1_min], x_full[i2_min]]
        if debug: print('enlargePositivePeak finishes. peakInterval =', peakInterval)
        x_peak = x_full[i1_min:i2_min+1]; y_peak = y_full[i1_min:i2_min+1]
        y_sub = y_sub_full[i1_min:i2_min+1]
        result['peak'] = utils.Spectrum(x_peak, y_sub)
        result['base'] = utils.Spectrum(x_peak, approx_full[i1_min:i2_min+1])
        result['part'] = utils.Spectrum(x_peak, y_peak)
        result['info']['peakInterval'] = peakInterval
        result['info']['baseFitInterval'] = baseFitInterval
        fixNegative(result)
        plotResult(result)
        return result
    result = enlargePositivePeak(result, newPeakInterval, 0)
    return result


def calculateRFactor(exp_e, exp_xanes, predictionXanes, energyRange):
    ind = (exp_e >= energyRange[0]) & (exp_e <= energyRange[1])
    return utils.integral(exp_e[ind], (exp_xanes[ind] - predictionXanes[ind]) ** 2) / \
           utils.integral(exp_e[ind], exp_xanes[ind] ** 2)


def microWaves(energy, intensity, maxWaveLength):
    """Find microwaves in spectrum.

    :param energy: [description]
    :param intensity: [description]
    :param maxWaveLength: [description]
    """
    polyDeg = 2; C = 1
    intervalSize = 3 # must be odd number, real interval size = maxWaveLength*intervalSize
    overlap = maxWaveLength/3
    assert intervalSize % 2 == 1
    assert energy.size == intensity.size
    a = energy[0]; b = energy[-1]
    xc0 = a
    result = []
    while xc0<b:
        xc1 = min(xc0 + maxWaveLength, b)
        d = ((intervalSize-1)//2)*maxWaveLength
        x0 = max(xc0 - d, a)
        x1 = min(x0 + maxWaveLength*intervalSize, b)
        if x1-x0 < maxWaveLength*intervalSize*0.9: x0 = max(x1 - maxWaveLength*intervalSize, a)
        i_inner = (xc0<=energy) & (energy<=xc1)
        assert np.sum(i_inner) > 0, 'No spectrum points on interval ['+str(xc0)+', '+str(xc1)+']'
        # model = sklearn.svm.SVR(kernel='poly', degree=polyDeg, C=C, gamma='scale', max_iter=1000)
        model = make_pipeline(PolynomialFeatures(polyDeg), Ridge())
        i = (x0<=energy) & (energy<=x1)
        model.fit(energy[i].reshape(-1,1), intensity[i])
        # if len(result) == 0:
        #     print(energy[i], intensity[i])
        new_intensity = model.predict(energy[i_inner].reshape(-1,1))
        # if model.fit_status_ != 0 or np.linalg.norm(intensity[i_inner]-new_intensity) > np.std(intensity[i]):
        #     model = make_pipeline(PolynomialFeatures(polyDeg), Ridge())
        #     model.fit(energy[i].reshape(-1,1), intensity[i])
        #     new_intensity = model.predict(energy[i_inner].reshape(-1,1))
        # print(intensity[i_inner])
        # print(new_intensity)
        # print('=======================')
        result.append({'inner':[xc0, xc1], 'outer':[x0,x1], 'spectrum':new_intensity})
        xc0 = xc0 + maxWaveLength-overlap

    mean_check = np.zeros(intensity.shape)
    for i in range(len(result)):
        xc0, xc1 = result[i]['inner']
        ind = (xc0<=energy) & (energy<=xc1)
        mean_check[ind] = result[i]['spectrum']

    # merge overlaps
    def kernel(x, a, b):
        result = np.zeros(x.shape)
        i_left = x<=a
        result[i_left] = 1
        i = (a<=x) & (x<=b)
        result[i] = (np.cos((x[i]-a)/(b-a)*np.pi)+1) / 2
        i_right = x>=b
        assert result[i_right].size==0 or np.all(result[i_right]) == 0
        return result

    for i in range(len(result)-1):
        xc0_prev, xc1_prev = result[i]['inner']
        xc0_next, xc1_next = result[i+1]['inner']
        assert xc0_next<xc1_prev, str(result[i]['inner'])+'  '+str(result[i+1]['inner'])
        sp_prev = result[i]['spectrum']; sp_prev0 = copy.deepcopy(sp_prev)
        sp_next = result[i+1]['spectrum']; sp_next0 = copy.deepcopy(sp_next)
        e_prev = energy[(xc0_prev<=energy) & (energy<=xc1_prev)]
        e_next = energy[(xc0_next<=energy) & (energy<=xc1_next)]
        ind_common_in_prev = (xc0_next<=e_prev) & (e_prev<xc1_prev)
        ind_common_in_next = (xc0_next<=e_next) & (e_next<xc1_prev)
        e_common = e_prev[ind_common_in_prev]
        assert np.all(e_common == e_next[ind_common_in_next])
        k = kernel(e_common, xc0_next, xc1_prev)
        new_sp = k*sp_prev[ind_common_in_prev] + (1-k)*sp_next[ind_common_in_next]
        sp_prev[ind_common_in_prev] = new_sp
        sp_next[ind_common_in_next] = new_sp
        # print(e_prev)
        # print(sp_prev0)
        # print(result[i]['spectrum'])
        # print("====================================")
        # print(e_next)
        # print(sp_next0)
        # print(result[i+1]['spectrum'])
        # exit(0)

    # compose one spectrum - result
    mean = np.zeros(intensity.shape)
    for i in range(len(result)):
        xc0, xc1 = result[i]['inner']
        ind = (xc0<=energy) & (energy<=xc1)
        mean[ind] = result[i]['spectrum']

    # if np.max(np.abs(mean-mean_check)) >= np.std(mean_check):
    #     print(mean)
    #     print(mean_check)
    #     print(mean-mean_check)
    assert np.max(np.abs(mean-mean_check)) < np.std(mean_check)

    return intensity-mean, mean


def interpExtrap(x, xp, yp, min_dx=None, min_n=3):
    """
    Do interpolation and linear extrapolation
    :param x:
    :param xp:
    :param yp:
    :param min_dx: min edge interval to use for linear extrapolation
    :param min_n: min number of edge points to use for linear extrapolation
    :return: y values in x
    """
    if np.isscalar(x): x = np.array([x])
    if isinstance(x, list): x = np.array(x)
    y = np.zeros(x.shape)
    check = np.zeros(x.shape)
    ind = (xp[0] <= x) & (x <= xp[-1])
    y[ind] = np.interp(x[ind], xp, yp)
    check[ind] += 1
    if np.sum(~ind) == 0:
        assert np.all(check == 1)
        return y
    # edges
    for edge in [-1, +1]:
        edge_x = xp[-(edge+1)//2]
        # indexes of xp edge points
        ind = np.where((xp - edge_x)*edge + min_dx > 0)[0]
        if len(ind) < min_n:
            ind = np.arange(min_n)
            if edge == 1: ind = -ind-1
        b,a = linearReg(xp[ind], yp[ind])
        ind_x = (x - edge_x) * edge > 0
        y[ind_x] = a * x[ind_x] + b
        check[ind_x] += 1
    assert np.all(check == 1)
    return y


def smooth_spectrum(energy, mu):
    n = len(mu)
    mu_smooth = np.zeros(n)
    mu_smooth[-1] = 0.5 * (mu[-2] + mu[-1])
    mu_smooth[0] = 0.5 * (mu[0] + mu[1])
    h1 = energy[1] - energy[0]
    for i in range(1, n - 1):
        h0 = h1
        h1 = energy[i + 1] - energy[i]
        h00 = h0 ** 2
        h01 = h0 * h1
        h11 = h1 ** 2
        mu_smooth[i] = 0.5 * ((h01 + h11) * mu[i - 1] + (h00 + h11) * mu[i] + (h00 + h01) * mu[i + 1]) \
                       / (h00 + h01 + h11)
    return mu_smooth


def detect_pre_post(element, edge, energy, mu, elements=None):

    n = len(energy)

    e0_base = xraydb.xray_edge(element, edge, True)

    # Detect the nearest edges (for known elements)
    if elements is not None and energy[0] < e0_base < energy[-1]:
        # TODO compare with 0.003*e0_base and 15.0*xafs.core_width(element, edge)
        E_SAFE_AREA = 15.0
        # TODO compare with 0.001*e0_base and 5.0*xafs.core_width(element, edge)
        E_EDGE_AREA = 5.0
        # debug_print('input:', element, edge, e0_base, energy[0], energy[-1])
        edges = [energy[0], energy[-1]]
        for el in elements:
            for el_edge, el_data in xraydb.xray_edges(el).items():
                if el_edge in ['K', 'L1', 'L2', 'L3'] and \
                        not (el == element and el_edge == edge) and \
                        abs(el_data[0] - e0_base) > E_SAFE_AREA:  # move to constant
                    # debug_print(el, el_edge, el_data[0])
                    edges.append(el_data[0])
        # debug_print('edges=', edges)
        e_prev_edges = [e for e in edges if e < e0_base]
        i_min = 0 if not e_prev_edges else max(0, np.searchsorted(energy, max(e_prev_edges) + E_EDGE_AREA) - 1)
        # debug_print('e_prev_edge=', i_min, energy[i_min], e_prev_edges)
        e_next_edges = [e for e in edges if e > e0_base]
        i_max = n if not e_next_edges else np.searchsorted(energy, min(e_next_edges) - E_EDGE_AREA)
        # debug_print('e_next_edge=', i_max, energy[i_max - 1], e_next_edges)
        energy = energy[i_min:i_max]
        mu = mu[i_min:i_max]
        n = len(energy)

    # Calculate fluctuations
    dmu = np.zeros((n - 2, 2))
    for i in range(1, n - 1):
        dmu[i - 1, 0] = 0.5 * (mu[i - 1] + mu[i + 1]) - mu[i]
        dmu[i - 1, 1] = energy[i + 1] - energy[i - 1]
    # dmu = dmu[dmu[:, 0].argsort()]
    dmu.view('f8,f8').sort(axis=0, order='f0')
    for i in range(1, n - 2):
        dmu[i, 1] += dmu[i - 1, 1]
    total = dmu[-1, 1]
    i1 = np.searchsorted(dmu[:, 1], 0.25 * total)
    i2 = np.searchsorted(dmu[:, 1], 0.75 * total)
    # difference for 50% of data (25% to 75% quartiles)
    noise = dmu[i2, 0] - dmu[i1, 0]
    threshold = 10.0 * noise

    # Detect extrema
    mu_smooth = smooth_spectrum(energy, mu)
    dmu = np.gradient(mu_smooth, energy)
    # dmu_median = np.median(dmu)

    minmax_pos = []
    last_min = 0
    last_max = 0
    for i in range(1, n - 1):
        if mu_smooth[i] >= mu_smooth[i - 1] and mu_smooth[i] > mu_smooth[i + 1]:
            if np.min(mu_smooth[last_max:i]) < min(mu_smooth[last_max], mu_smooth[i]) - threshold:
                if last_max > 0:
                    minmax_pos.append((last_max, True))
                last_max = i
            elif mu_smooth[i] >= mu_smooth[last_max]:
                last_max = i
        elif mu_smooth[i] <= mu_smooth[i - 1] and mu_smooth[i] < mu_smooth[i + 1]:
            if np.max(mu_smooth[last_min:i]) > max(mu_smooth[last_min], mu_smooth[i]) + threshold:
                if last_min > 0:
                    minmax_pos.append((last_min, False))
                last_min = i
            elif mu_smooth[i] <= mu_smooth[last_min]:
                last_min = i
    if last_max > 0 and np.min(mu_smooth[last_max:]) < mu_smooth[last_max] - threshold:
        minmax_pos.append((last_max, True))
    if last_min > 0 and np.max(mu_smooth[last_min:]) > mu_smooth[last_min] + threshold:
        minmax_pos.append((last_min, False))
    minmax_pos.sort(key=lambda i: i[0])
    # debug_print('extrema=', [energy[i[0]] for i in minmax_pos])

    # Detect edges
    edges_pos = []
    if minmax_pos and minmax_pos[0][1]:
        pos_max = minmax_pos[0][0]
        if mu_smooth[pos_max] - mu_smooth[0] > 5.0 * threshold:
            edges_pos.append((0, pos_max))
    for i in range(0, len(minmax_pos) - 1):
        if not minmax_pos[i][1]:
            pos_min = minmax_pos[i][0]
            pos_max = minmax_pos[i + 1][0]
            if mu_smooth[pos_max] - mu_smooth[pos_min] > 5.0 * threshold:
                edges_pos.append((pos_min, pos_max))
    # debug_print('edges=', edges_pos)

    # Detect e0
    i1 = np.searchsorted(energy, min(0.995 * e0_base, e0_base - 20.))
    i2 = np.searchsorted(energy, max(1.001 * e0_base, e0_base + 20.))
    e0 = larch.xafs.find_e0(energy[i1:i2], mu[i1:i2])
    i_e0 = np.searchsorted(energy, e0)

    # Detect pre-minimum
    de_approx = min(0.05 * e0, max(10.0, 0.001 * e0))
    # debug_print('e0=', e0, e0_base, de_approx)
    emin = min(e0, e0_base, e0 - de_approx)
    if emin <= energy[0]:
        i_emin = 0
    else:
        i_emin = np.searchsorted(energy, emin)
        i_prev_maximum = [i[0] for i in minmax_pos if i[0] < i_emin and i[1]]
        i_prev_maximum = None if not i_prev_maximum else max(i_prev_maximum)
        i_prev_minimum = [i[0] for i in minmax_pos if i[0] < i_emin and not i[1]]
        i_prev_minimum = None if not i_prev_minimum else max(i_prev_minimum)
        if i_prev_minimum is not None and (i_prev_maximum is None or i_prev_minimum > i_prev_maximum):
            i_emin = i_prev_minimum
    emin = energy[i_emin]
    # debug_print('emin=', emin)

    i_start = 0
    i_maxdmu = np.argmax(dmu[:i_emin + 1])
    dmu_mean = np.mean(dmu[:i_emin + 1])
    dmu_sigma = np.std(dmu[:i_emin + 1])
    if dmu[i_maxdmu] - dmu_mean > 3.0 * dmu_sigma and any((i[1] for i in minmax_pos if i_maxdmu < i[0] < i_emin)):
        e_start = energy[i_maxdmu] + min(50.0, 0.2 * e0)
        i_start = np.searchsorted(energy, e_start)
        # debug_print('i_start=', i_start)

    # Detect energy scale (de)
    i_emax = np.searchsorted(energy, max(e0, e0_base) + 50.0)
    emax = energy[i_e0 + np.argmax(mu_smooth[i_e0:i_emax])]
    i_emax = np.searchsorted(energy, emax)
    mu_jump = mu_smooth[i_emax] - mu_smooth[i_emin]
    de = max(np.sqrt(abs((e0 - emin) * (emax - e0))), 2.0 * abs(e0 - e0_base), 0.002 * e0, 10.0)
    de = min(de, 350.0)

    # Detect a safe pre-area with small derivatives
    maxdmu = np.max(dmu[max(i_e0 - 10, 0):min(i_e0 + 11, n)])
    # debug_print('maxdmu=', maxdmu, maxdmu * 0.05)
    # debug_print('dmu=', dmu[i_start:i_emin])
    # debug_print('where=', np.where(dmu[i_start:i_emin] < maxdmu * 0.05))
    idx = i_start + np.where(dmu[i_start:i_emin] < maxdmu * 0.05)[0]
    # debug_print([(energy[i], dmu[i]) for i in range(i_start, i_emin)])
    # debug_print('idx=', idx)
    i, nn = 0, len(idx)
    pre1 = pre2 = 0.0
    while i < nn:
        j = i
        while j < nn - 1 and idx[j + 1] - idx[j] == 1:
            j += 1
        # debug_print('try:', i, j, energy[idx[i]], energy[idx[j]])
        if j > i and energy[j] - energy[i] > pre2 - pre1:
            pre1 = energy[idx[i]]
            pre2 = energy[idx[j]]
        i = j + 1
    if pre1 == 0.0:
        pre1 = energy[i_start]
        pre2 = emin if i_emin > i_start else pre1 + 0.3 * (e0 - pre1)
    # debug_print('pre=', pre1, pre2, e0, de)
    pre2 = min(pre2, e0 - 1.25 * de)
    pre1 = min(pre1, emin)
    if pre1 >= pre2:
        pre2 = 0.5 * (pre1 + emin)
    # debug_print('pre#=', pre1, pre2)

    # Detect start of post-area
    post1 = max(emax + de, e0 + 2. * de, e0 + 50.0)
    post_e = [e0] + [energy[i[0]] for i in minmax_pos if energy[i[0]] > e0]
    for i in range(1, len(post_e)):
        if post_e[i] > post1 and post_e[i] - post_e[i - 1] > 4. * de:
            break
        post1 = max(post1, post_e[i])
    post1 += 0.5 * de

    # Detect end of post-area
    i_post1 = np.searchsorted(energy, post1)
    if i_post1 >= n:
        post1 = 0.5 * (e0 + energy[-1])
        post2 = energy[-1]
    else:
        post_edges = [i[0] for i in edges_pos
                      if energy[i[0]] > post1 and mu_smooth[i[1]] - mu_smooth[i[0]] > 0.2 * mu_jump]
        i_post2 = min(n - 1 if not post_edges else post_edges[0], i_post1 + np.argmin(mu[i_post1:]))
        post2 = energy[i_post2]
        if i_post2 < n - 1:
            post2 -= 0.5 * de
    post2 = min(post2, e0 + 1500.0)
    if post1 >= post2:
        post1 = 0.5 * (e0 + post2)
    # debug_print('post=', post1, post2)

    # Require at least MIN_POINTS points and MIN_DE interval
    MIN_NPOINTS = 14
    MIN_DE = min(0.05 * e0, 10.0)
    i_pre1 = np.searchsorted(energy, pre1)
    i_pre2 = np.searchsorted(energy, pre2)
    i_post1 = np.searchsorted(energy, post1)
    i_post2 = np.searchsorted(energy, post2)
    if i_pre2 - i_pre1 < MIN_NPOINTS:
        i_pre1 -= (MIN_NPOINTS - (i_pre2 - i_pre1)) // 2
        i_pre1 = max(0, i_pre1)
        i_pre2 = min(i_pre1 + MIN_NPOINTS, n - 1)
        pre1 = energy[i_pre1]
        pre2 = energy[i_pre2]
    if pre2 - pre1 < MIN_DE:
        pre1 = max(energy[0], pre1 - (MIN_DE - (pre2 - pre1)))
        pre2 = min(energy[-1], pre1 + MIN_DE)
    if i_post2 - i_post1 < MIN_NPOINTS:
        i_post2 += (MIN_NPOINTS - (i_post2 - i_post1)) // 2
        i_post2 = min(n - 1, i_post2)
        i_post1 = max(0, i_post2 - MIN_NPOINTS)
        post1 = energy[i_post1]
        post2 = energy[i_post2]
    if post2 - post1 < MIN_DE:
        post2 = min(energy[-1], post2 + (MIN_DE - (post2 - post1)))
        post1 = max(energy[0], post2 - MIN_DE)
    # Detect overlapping (just to be sure, it never happens in the wild)
    if post1 < pre2:
        pre2 = post1 = 0.5 * (pre2 + post1)

    # debug_print('result=', pre1, pre2, e0, post1, post2, de)

    return pre1, pre2, e0, post1, post2


def setUnknownMbackParams(spectrum, pre, post, e0, element, edge):
    if pre is None or post is None or e0 is None or element is None or edge is None:
        e = spectrum.x
        e00 = e0
        if e0 is None: e0 = larch.xafs.find_e0(e, spectrum.y)
        if element is None or edge is None: element, edge = xraydb.guess_edge(e0)
        pre1, pre2, e0_guessed, post1, post2 = detect_pre_post(element, edge, e, spectrum.y)
        if e00 is None: e0 = e0_guessed
        if pre is None: pre = [pre1, pre2]
        if post is None: post = [post1, post2]
    return pre, post, e0, element, edge


def mback(spectrum, pre=None, post=None, e0=None, element=None, edge=None, deg=None, fit_erfc=True, returnExtra=False, debug=True):
    """
    Important: mback polynomial must be the same from left and from right of the edge. Otherwise the edge would be corrupted
    If the spectrum is not so, than we can't normalize it
    """
    if isinstance(deg, float):
        assert deg-int(deg) == 0
        deg = int(deg)
    spectrum = spectrum.clone()
    tmp = spectrum.limit([pre[0], post[1]]).y if pre is not None and post is not None else spectrum.y
    spectrum.y = (spectrum.y - tmp.min()) / (tmp.max() - tmp.min())
    pre, post, e0, element, edge = setUnknownMbackParams(spectrum, pre, post, e0, element, edge)
    # print(pre, post, e0, element, edge)
    e = spectrum.x

    def mbackHelper(spectrum):
        # leexiang - turn on correcting weights, divides it by spectrum values (if spectrum is nearly normalized, correction breaks optimization)
        leexiang = False  # if True then mback make wrong base on left from e0 for nearly normalized spectra
        degs = [deg] if deg is not None else [0,1,2,3]
        group = [None]*len(degs)
        for id,d in enumerate(degs):
            group[id] = larch.Group(name='tmp')
            larch.xafs.mback(e, spectrum.y, group=group[id], z=molecule.atom_proton_numbers[element], edge=edge, order=d, leexiang=leexiang, fit_erfc=fit_erfc, e0=e0, pre1=pre[0] - e0, pre2=pre[1] - e0, norm1=post[0] - e0, norm2=post[1] - e0)
            group[id].deg = d
        sgroups = sorted(group, key=lambda g: g.loss)
        group = sgroups[0]
        if debug and deg is None:
            print(f'MBACK best deg = {sgroups[0].deg}')
        return group

    group = mbackHelper(spectrum)
    # if len(sgroups) > 1: print('best deg =',group.deg)
    flat = group.fpp

    # print(dir(group)) # 'e0', 'edge_step', 'f2', 'fpp', 'mback_details', 'norm'
    # print(dir(group.mback_details)) # 'f2_scaled', 'norm_function', 'params', 'pre_f2'
    # print(group.mback_details.params) # dict with keys: s, xi, a, c0, c1, c2, c3, c4, c5
    # print(list(group.mback_details.pre_f2.keys())) # 'e0', 'edge_step', 'norm', 'pre_edge', 'post_edge', 'norm_coefs', 'nvict', 'nnorm', 'norm1', 'norm2', 'pre1', 'pre2', 'precoefs'
    # print(group.mback_details.pre_f2)

    # mu – исходный кривой спектр
    # group.f2 - tabulated f2(E) (наклонная ступенька)
    # group.edge_step – коэффициент, который показывает величину скачка спектра около E0
    # opars['s'] – коэффициент, на который домножается спектр, чтобы подогнать его к group.f2
    # norm_function (fitted background over whole energy region = Erf + POLYNOMIAL) – фон, который прибавляется к mu*opars['s'], чтобы подогнать его к group.f2
    # group.fpp = opars['s']*mu - norm_function – спектр с вычтенным фоном, подогнанный к group.f2

    flat = utils.Spectrum(e, flat)
    if returnExtra:
        s = group.mback_details.params['s']
        extra = {'e':e, 'fpp':group.fpp, 'f2':group.f2, 'norm':group.norm, 'mu*s':s*spectrum.y, 'norm_function':group.mback_details.norm_function, 'target':group.mback_details.target, 'mu':spectrum.y, 'params':{'pre':pre, 'post':post, 'e0':e0, 'element':element, 'edge':edge}}
        return flat, extra
    return flat


def autobk(spectrum, **larchAutobkParams):
    """
    :returns: flat spectrum, chi(k) spectrum and extra info dict with the keys: 'chie' - chi(e), 'bkg', 'delta_bkg', 'pre_edge', 'post_edge', 'norm', 'kraw'
    """
    group = larch.Group(name='tmp')
    larch.xafs.autobk(spectrum.x, spectrum.y, group, **larchAutobkParams)
    # print(dir(group))  # atsym, autobk_details, bkg, callargs, chi, chie, d2mude, delta_bkg, delta_chi, dmude, e0, edge, edge_step, edge_step_poly, flat, journal, k, norm, norm_poly, post_edge, pre_edge, pre_edge_details
    # print(dir(group.autobk_details))  # aic, bic, chisqr, init_bkg, init_chi, init_knots_y, irbkg, kmax, kmin, knots_k, knots_y, nfev, nknots, nspl, params, redchi, report
    # print(group.norm)
    if hasattr(group, 'flat'):
        flat_fixed = crossfade(spectrum.x, group.flat, group.chie+1, group.e0+10, group.e0+50)
    else:
        flat_fixed = copy.deepcopy(spectrum.y)
    extra = {'chie':group.chie, 'bkg':group.bkg, 'delta_bkg':group.delta_bkg}
    for n in ['pre_edge', 'post_edge', 'norm', 'kraw']:
        if hasattr(group, n): extra[n] = getattr(group, n)
    return utils.Spectrum(spectrum.x, flat_fixed), utils.Spectrum(group.k, group.chi), extra


def fluo_corr_my(s, alpha=None):
    import pyfitit.larch as larch
    import pyfitit.larch.xafs, xraydb
    group = larch.Group(name='tmp')
    # larch.xafs.fluo_corr(s.x, s.y, formula='TiO2', elem='Ti', group=group, edge='K', line='Ka', anginp=45, angout=45)
    energy, mu = s.x, s.y
    if alpha is None:
        anginp, angout = 45, 45
        elem = 'Ti'
        edge = 'K'
        line = 'Ka'
        formula = 'TiO2'

        ang_corr = (np.sin(max(1.e-7, np.deg2rad(anginp))) / np.sin(max(1.e-7, np.deg2rad(angout))))
        print(ang_corr)

        # find edge energies and fluorescence line energy
        e_edge = xraydb.xray_edge(elem, edge).energy
        e_fluor = xraydb.xray_line(elem, line).energy

        # calculate mu(E) for fluorescence energy, above, below edge
        muvals = xraydb.material_mu(formula, np.array([e_fluor, e_edge - 10.0, e_edge + 10.0]), density=1)
        print(muvals)

        alpha = (muvals[0] * ang_corr + muvals[1]) / (muvals[2] - muvals[1])
        print('alpha =', alpha)
        # alpha = 10
    mu_corr = mu * alpha / (alpha + 1 - mu)
    return utils.Spectrum(s.x, mu_corr)


def crossfade(x,y1,y2,a,b,form='erf'):
    assert form in ['erf', 'linear']
    assert len(x) == len(y1)
    assert len(x) == len(y2)
    assert a<b
    if form == 'linear':
        left = x[0] if x[0]<a else a-1
        right = x[-1] if x[-1]>b else b+1
        w = np.interp(x, [left,a,b,right], [0,0,1,1])
    else:
        c = (a+b)/2
        r = (b-a)/2
        w = (scipy.special.erf((x-c)/r)+1)/2 # x=a,b => arg = 1.5
    return y1*(1-w) + y2*w

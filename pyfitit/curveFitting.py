import scipy, sklearn, copy
import numpy as np
from lmfit.models import ExpressionModel, PolynomialModel
from . import utils
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# возвращает [b,a] из модели y=ax+b
def linearReg(x,y,de=None):
    if de is None: de = np.ones(len(x))
    N = np.sum(de)
    sumX = np.sum(x*de)
    sumX2 = np.sum(x*x*de)
    sumY = np.sum(y*de)
    sumXY = np.sum(x*y*de)
    det = N*sumX2-sumX*sumX
    if det == 0:
        nn = abs(sumX)+abs(N)
        return [-sumX/nn, N/nn]
    return [(sumY*sumX2-sumXY*sumX)/det, (N*sumXY-sumX*sumY)/det]


def linearReg_mult_only(x,y,de):
    sumX2 = np.sum(x*x*de)
    sumXY = np.sum(x*y*de)
    if (sumX2 == 0) or np.isnan(sumX2) or np.isnan(sumXY): return 0
    return sumXY/sumX2


def fit_by_regression_mult_only(exp_e, exp_xanes, fdmnes_xan, fitEnergyInterval):
    ind = (fitEnergyInterval[0]<=exp_e) & (exp_e<=fitEnergyInterval[1])
    e = exp_e[ind]
    ex = exp_xanes[ind]
    fx = fdmnes_xan[ind]
    mex = (ex[1:]+ex[:-1])/2
    mfx = (fx[1:]+fx[:-1])/2
    de = e[1:]-e[:-1]
    w = linearReg_mult_only(mfx, mex, de)
    return w*fdmnes_xan


def fit_to_experiment_by_norm_or_regression_mult_only(exp_e, exp_xanes, fit_interval, fdmnes_en, fdmnes_xan, shift, norm = None):
    fdmnes_en = fdmnes_en + shift
    fdmnes_xan = np.interp(exp_e, fdmnes_en, fdmnes_xan)
    if norm is None:
        fdmnes_xan1 = fit_by_regression_mult_only(exp_e, exp_xanes, fdmnes_xan, fit_interval)
        s = np.sum(fdmnes_xan1)
        if s != 0: norm = np.sum(fdmnes_xan)/s
        else: norm = 0
        # print(norm)
        return fdmnes_xan1, norm
    else: return fdmnes_xan/norm, norm


def findExpEfermi(exp_e, exp_xanes, search_shift_level):
    ind = np.where(exp_xanes>=search_shift_level)[0][0]
    exp_Efermi_left = exp_e[ind]
    i = ind
    while i+1<len(exp_xanes) and exp_xanes[i]<=exp_xanes[i+1]: 
        i += 1
    exp_Efermi_peak = exp_e[i]
    while i+1<len(exp_xanes) and exp_xanes[i]>=exp_xanes[i+1]: 
        i += 1
    exp_Efermi_right = exp_e[i]
    return exp_Efermi_left, exp_Efermi_peak, exp_Efermi_right


def findEfermiByArcTan(energy, intensity):
    """
    Searches Efermi energy by fitting xanes by arctan.
    :param energy:
    :param intensity:
    :return: best_params = {'a':..., 'x0':...}, arctan_y
    """
    last = np.mean(intensity[-5:])
    efermi0, _, _ = findExpEfermi(energy, intensity, 0.5*last)
    mod = ExpressionModel('b/(1+exp(-a*(x - x0)))+c')
    params = mod.make_params(a=0.3, x0=efermi0, b=last, c=0)  # - стартовые
    params['a'].set(min=0)
    params['b'].set(min=0)
    result = mod.fit(intensity, params, x=energy)
    return result.best_values, result.best_fit


def substractBase(x, y, peakInterval, baseFitInterval, model, usePositiveConstrains, extrapolate, useStartParams=None):
    """
    Fit base by Cauchy function and substract from y.
    :param x: argument
    :param y: function values
    :param peakInterval: interval of peak search (do not included in base fitting)
    :param baseFitInterval: interval of base fit. Usually it includes peakInterval
    :param model: 'cauchy' or 'bezier' or 'arctan'
    :param usePositiveConstrains: add constrain y_base <= y
    :param extrapolate: {'left':percent_dx_left, 'right':percent_dx_right}
    :return: x_peak, y_peak - peak with substracted base (on interval peakInterval); x_base, y_base - base on baseFitInterval
    """
    assert model in ['cauchy', 'bezier', 'arctan']
    assert len(x) == len(y)
    ind_peak = (x >= peakInterval[0]) & (x <= peakInterval[1])
    ind_base_full = (x >= baseFitInterval[0]) & (x <= baseFitInterval[1])
    ind_base = ind_base_full & ~ind_peak
    x_peak = x[ind_peak]; y_peak = y[ind_peak]
    x_base = x[ind_base]; y_base = y[ind_base]
    x_base_full = x[ind_base_full]
    y_base_full = y[ind_base_full]
    
    # make x_fit y_fit by extrapolating base inside peak interval (linear extrapolation from both ends)
    if usePositiveConstrains:
        ind_base_left = (x<peakInterval[0]) & ind_base_full
        b1, a1 = linearReg(x[ind_base_left], y[ind_base_left])
        ind_base_right = (x>peakInterval[1]) & ind_base_full
        b2, a2 = linearReg(x[ind_base_right], y[ind_base_right])
        y_gap = np.max([a1*x_peak+b1, a2*x_peak+b2], axis=0).reshape(-1)
        assert len(y_gap) == len(x_peak)
        x_fit = x_base_full
        y_fit = np.concatenate((y[ind_base_left], y_gap, y[ind_base_right]))
        assert len(x_fit) == len(y_fit), str(len(x_fit))+" "+str(len(y_fit))
    else:
        x_fit = x_base
        y_fit = y_base

    x1 = x_base[0]; x2 = x_base[-1]
    y1 = y_base[0]; y2 = y_base[-1]
    if 'left' in extrapolate:
        n = np.where(x_base <= x1+(x2-x1)/10)[0][-1] + 1
        if n<2: n = 2
        slope, intercept,_,_,_ = scipy.stats.linregress(x_base[:n], y_base[:n])
        percent = extrapolate['left']; count = np.round(len(x_base)*percent);
        first = x1 - (x2-x1)*percent; last = x1-(x2-x1)/count
        new_x = np.linspace(first, last, count)
        x_base = np.insert(x_base,0,new_x)
        y_base = np.insert(y_base,0,new_x*slope+intercept)
    if 'right' in extrapolate:
        n = np.where(x_base >= x2-(x2-x1)/10)[0][-1] + 1
        if n<2: n = 2
        slope, intercept,_,_,_ = scipy.stats.linregress(x_base[-n:], y_base[-n:])
        percent = extrapolate['right']; count = np.round(len(x_base)*percent);
        last = x2 + (x2-x1)*percent; first = x2+(x2-x1)/count
        new_x = np.linspace(first, last, count)
        x_base = np.append(x_base,new_x)
        y_base = np.append(y_base,new_x*slope+intercept)
    assert (len(x_peak) >= 2) and (len(x_base) >= 2), 'len(x_peak) = '+str(len(x_peak))+' len(x_base) = '+str(len(x_base))

    minx = np.min(x); maxx = np.max(x); maxy = np.max(y)
    if model == 'cauchy':
        fff = lambda x, a, b, g, d: a / ((x - b) ** 2 + g) + d
        mod = ExpressionModel('a/((x-b)**2+g) + d')
        b0 = x2 + x2 - x1
        g0 = 1
        a0 = (y2 - y1) / (1 / ((x2 - b0) ** 2 + g0) - 1 / ((x1 - b0) ** 2 + g0))
        d0 = y1 - a0 / ((x1 - b0) ** 2 + g0)
        params = mod.make_params(a=a0, b=b0, g=g0, d=d0)
        param_order = {'a': 0, 'b': 1, 'g': 2, 'd': 3}
        start0 = [params['a'].value, params['b'].value, params['g'].value, params['d'].value]
        result = mod.fit(y_fit, params, x=x_fit)
        start = [result.params['a'].value, result.params['b'].value, result.params['g'].value, result.params['d'].value]
        bounds = [[0,1e3*maxy],[minx,maxx+(maxx-minx)*10],[0,(maxx-minx)*10],[-maxy,maxy]]
    elif model == 'arctan':
        fff = lambda x, a, b, c, x0, d: b/(1+np.exp(-a*(x - x0)))+c+d*(x-x_base[0])
        mod = ExpressionModel('b/(1+exp(-a*(x - x0)))+c+d*(x-'+str(x_base[0])+')')
        efermi0, _, _ = findExpEfermi(x, y, 0.5*np.mean(y[-5:]))
        if efermi0<x_peak[0]: efermi0 = x_peak[0]
        a0 = 1; b0 = y[-1]-y[0]; c0 = y[0]; x00 = efermi0; d0 = (y_peak[0]-y_base[0])/(x_peak[0]-x_base[0])
        params = mod.make_params(a=a0, b=b0, c=c0, x0=x00, d=d0)
        param_order = {'a':0, 'b':1, 'c':2, 'x0':3, 'd':4}
        start0 = [params['a'].value, params['b'].value, params['c'].value, params['x0'].value, params['d'].value]
        max_dy = np.max((y[1:]-y[:-1])/(x[1:]-x[:-1]))
        params['a'].set(min=0); params['a'].set(max=max_dy/(np.max(y)-np.min(y))*10)
        params['b'].set(min=0)
        params['x0'].set(min=x_peak[0])
        params['d'].set(min=0); params['d'].set(max=3*(y_peak[0]-y_base[0])/(x_peak[0]-x_base[0]))
        dist = np.max([abs(x00-minx), abs(x00-maxx), maxx-minx])
        bounds = [[0,a0*100], [0,maxy*10], [-maxy,maxy], [minx-dist,maxx+dist*10], [0,3*(y_peak[0]-y_base[0])/(x_peak[0]-x_base[0])]]
        # TODO: remove lmfit, because scipy.optimize.minimize works better
        if useStartParams is None:
            result = mod.fit(y_fit, params, x=x_fit)
            # result.plot()
            # plt.show()
            # print(result.fit_report())
            start = [result.params['a'].value, result.params['b'].value, result.params['c'].value, result.params['x0'].value, result.params['d'].value]
        else:
            start = useStartParams
    else:
        Mtk = lambda n, t, k: t**k * (1-t)**(n-k) * scipy.misc.comb(n,k)
        BezierCoeff = lambda ts: [[Mtk(3,t,k) for k in range(4)] for t in ts]
        t = np.linspace(0,1,len(x_base))
        Pseudoinverse = np.linalg.pinv(BezierCoeff(t))
        data = np.column_stack((x_base, y_base))
        control_points = Pseudoinverse.dot(data)
        Bezier = np.array(BezierCoeff(tPlot)).dot(control_points)
        assert not usePositiveConstrains
        
        return x_peak, y_peak-app_y_base_inside_peak, x_base_full, app_y_base_full, y_peak, y_base_full - app_y_base_full

    def func(params):
        y_app = fff(x_base, *params)
        return np.linalg.norm(y_app - y_base)
    if useStartParams is None:
        res = scipy.optimize.minimize(func, start0, bounds=bounds)
        # print(func(start), res.fun)
        if res.fun < func(start):
            for name in result.params:
                # print(f'Setting {name} = ',res.x[param_order[name]])
                result.params[name].set(res.x[param_order[name]])
            # print(result.params)
            start = res.x
    info = {'optimParam':start, 'optimVal':func(start)}
    if usePositiveConstrains:
        #while True:
            #if np.all(fff(x_peak,*start)<=y_peak): break
            #dx = np.max(x_peak)-np.min(x_peak)
            #dy = np.max(y_peak)-np.min(y_peak)
            #start[1] += dx*0.01
            #start[3] -= dy*0.01
            
        constrains = tuple()
        for i in range(len(x_peak)):
            cons_fun = lambda params,i=i: fff(x_peak[i], *params)
            constrains += (scipy.optimize.NonlinearConstraint(cons_fun, -maxy, y_peak[i]),)
        # print(bounds)
        res = scipy.optimize.minimize(func, start, bounds=bounds, constraints=constrains)
        params = res.x
        app_y_base_inside_peak = fff(x_peak, *params)
        app_y_base_full = fff(x_base_full, *params)
        info = {'optimParam': params, 'optimVal': res.fun}
    else:
        app_y_base_inside_peak = mod.eval(result.params, x=x_peak)
        app_y_base_full = mod.eval(result.params, x=x_base_full)
    return x_peak, y_peak-app_y_base_inside_peak, x_base_full, app_y_base_full, y_peak, y_base_full - app_y_base_full, info


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
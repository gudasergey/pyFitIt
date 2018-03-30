import numpy as np
import pandas as pd
import os
from experiment import Experiment
from smoothLib import DefaultSmoothParams
from minimize import param
from molecula import Molecula
from utils import Xanes

def moleculaConstructor(params):
    folder = os.path.dirname(os.path.realpath(__file__))
    m = Molecula(folder+'/Febpy_XRDstruct.xyz', None, checkSymmetry = False, lastGroupIsPart = True, partCenter = 'first atom')
    return m

def parse():
    folder = os.path.dirname(os.path.realpath(__file__))
    experiment_data = pd.read_csv(folder+'/exp_Febpy_lowspin-ground.txt', sep="\t", decimal=",", header=1).values
    exp_e = experiment_data[:, 0];
    ind = (exp_e>=7100) & (exp_e<=7350)
    exp_e = exp_e[ind]
    exp_xanes = experiment_data[ind, 1]
    exp_xanes /= np.mean(exp_xanes[-3:])
    fit_intervals = {'norm':[exp_e[0], exp_e[-1]], 'smooth':[exp_e[0], exp_e[-1]], 'geometry':[exp_e[0], exp_e[-1]]}
    exp = Experiment('Febpy', Xanes(exp_e, exp_xanes), fit_intervals)
    exp.defaultSmoothParams = DefaultSmoothParams(7113)
    exp.moleculaConstructor = moleculaConstructor
    return exp
